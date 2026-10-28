"""Slot scheduler with absolute completion times and synchronous padded batches."""

from dataclasses import asdict
from math import fsum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .allocation import TPBandwidth
from .config import Config
from .energy import EnergyLedger, NoEnergy
from .physics import batch_cost, channel_trace, full_band_rates
from .workload import ClosedLoopWorkload


class EdgeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config, *, energy=True, allocator=None):
        self.config = config
        self.record_energy = energy
        self.allocator = TPBandwidth() if allocator is None else allocator
        n, j = config.users, config.nodes
        self.action_space = spaces.MultiDiscrete(np.full(n, j + 1))
        # q=(input, output cap, SLO), independent uplink/downlink gains, idle flags.
        self.observation_space = spaces.Box(0, np.inf, shape=(3 * n + 2 * n * j + j,), dtype=np.float64)

    @property
    def decision_s(self):
        return (self.slot + 1) * self.config.slot_s

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31))
        scenario = options.get("scenario") if options is not None else None
        if scenario is not None:
            assert scenario.config == self.config
            episode_seed = scenario.episode_seed
        self.episode_seed = episode_seed
        self.action_space.seed(episode_seed)
        self.slot = 0
        if scenario is None:
            self.up_channels, self.down_channels = channel_trace(self.config, episode_seed)
            self.workload = ClosedLoopWorkload(self.config, episode_seed)
        else:
            self.up_channels, self.down_channels = scenario.up, scenario.down
            self.workload = scenario.workload(options["workload_mode"])
        j = self.config.nodes
        self.node_free_at = np.zeros(j)
        self.node_gpu_start = np.zeros(j)
        self.node_gpu_end = np.zeros(j)
        self.node_peak_bytes = np.full(j, 2 * self.config.model.parameters, dtype=float)
        self.ledger = EnergyLedger(self.config) if self.record_energy else NoEnergy()
        self.records = []
        self.batch_records = []
        self.rewards = []
        self.current_requests = self.workload.arrivals(0, self.config.slot_s)
        return self.observation(), {"action_mask": self.action_masks().reshape(self.config.users, -1)}

    def observation(self):
        c = self.config
        q = np.zeros((3, c.users))
        for u, r in self.current_requests.items():
            q[:, u] = r.input_tokens, r.output_cap, r.slo_s
        return np.concatenate((q.ravel(), self.up_channels[self.slot].ravel() * 1e9,
                               self.down_channels[self.slot].ravel() * 1e9,
                               (self.node_free_at <= self.decision_s).astype(float)))

    def action_masks(self):
        c = self.config
        mask = np.zeros((c.users, c.nodes + 1), dtype=bool)
        mask[:, 0] = True
        users = list(self.current_requests)
        mask[users, 1:] = self.node_free_at <= self.decision_s
        return mask.ravel()  # SB3-Contrib MultiDiscrete contract.

    def memory_bytes(self, time_s):
        weights = 2 * self.config.model.parameters
        during_gpu = (self.node_gpu_start <= time_s) & (time_s < self.node_gpu_end)
        return np.where(during_gpu, self.node_peak_bytes, weights)

    def _record(self, request, node, status, response_s, *, good=False, inferred=False,
                delivered=False, upload_s=None, batch_wait_s=None, prefill_s=None,
                decode_s=None, downlink_s=None, gpu_tpot_s=None):
        r = request
        self.records.append({
            **asdict(r), "slot": self.slot, "node": node, "status": status,
            "decision_s": self.decision_s, "response_s": float(response_s),
            "ap_wait_s": self.decision_s - r.arrival_s, "upload_s": upload_s,
            "batch_wait_s": batch_wait_s, "gpu_ttft_s": prefill_s,
            "gpu_decode_s": decode_s, "gpu_tpot_s": gpu_tpot_s, "downlink_s": downlink_s,
            "e2e_s": float(response_s - r.arrival_s) if delivered else None,
            "inferred": inferred, "delivered": delivered, "good": bool(good),
        })
        self.workload.finish(r.user, response_s)

    def _execute_batch(self, node, requests):
        c, r, now = self.config, self.config.radio, self.decision_s
        users = np.array([x.user for x in requests])
        input_bits = np.array([x.input_tokens for x in requests]) * c.bits_per_token
        output_bits = np.array([x.output_tokens for x in requests]) * c.bits_per_token
        up_rates = full_band_rates(self.up_channels[self.slot, users, node], r.uplink_hz, r.uplink_w, r.noise_up_w)
        down_rates = full_band_rates(self.down_channels[self.slot, users, node], r.downlink_hz,
                                    r.downlink_per_link_w, r.noise_down_w)
        up_beta = np.asarray(self.allocator.uplink(input_bits, up_rates))
        assert up_beta.shape == input_bits.shape and np.all(up_beta > 0)
        # A single representable rounding step is numerical equality, not resource headroom.
        assert fsum(up_beta) <= np.nextafter(1.0, np.inf)
        t_up = input_bits / (up_beta * up_rates)
        upload_end = now + float(t_up.max())
        for t in t_up:
            self.ledger.record("uplink_j", now, now + t, r.uplink_w, node)

        cost = batch_cost(c.model, [x.input_tokens for x in requests],
                          [x.output_tokens for x in requests], c.flops_s)
        batch_row = {"slot": self.slot, "node": node, "users": users.tolist(),
                     "upload_start_s": now, "upload_end_s": upload_end,
                     **asdict(cost), "oom": cost.peak_bytes > c.capacity_bytes,
                     "up_beta": up_beta.tolist()}
        if cost.peak_bytes > c.capacity_bytes:
            # Analytical reservation failure after upload; no partial GPU execution.
            for request, t in zip(requests, t_up):
                self._record(request, node, "oom", upload_end, upload_s=float(t),
                             batch_wait_s=float(t_up.max() - t))
            self.node_free_at[node] = upload_end
            self.batch_records.append({**batch_row, "release_s": upload_end, "down_beta": [0.0] * len(users)})
            return

        prefill_end = upload_end + cost.prefill_s
        inference_end = prefill_end + cost.decode_s
        self.node_gpu_start[node] = upload_end
        self.node_gpu_end[node] = inference_end
        self.node_peak_bytes[node] = cost.peak_bytes
        p = c.energy
        self.ledger.record("gpu_prefill_increment_j", upload_end, prefill_end,
                           c.gpus_per_node * (p.prefill_w_per_gpu - p.idle_w_per_gpu), node)
        self.ledger.record("gpu_decode_increment_j", prefill_end, inference_end,
                           c.gpus_per_node * (p.decode_w_per_gpu - p.idle_w_per_gpu), node)

        deadlines = np.array([x.arrival_s + x.slo_s for x in requests])
        margins = deadlines - inference_end
        down_beta = np.asarray(self.allocator.downlink(output_bits, margins, down_rates))
        assert down_beta.shape == input_bits.shape and np.all(down_beta >= 0)
        assert fsum(down_beta) <= np.nextafter(1.0, np.inf)
        t_down = np.zeros(len(users))
        delivered = down_beta > 0
        t_down[delivered] = output_bits[delivered] / (down_beta[delivered] * down_rates[delivered])
        response = np.full(len(users), inference_end)
        response[delivered] = inference_end + t_down[delivered]
        # Equivalent SLO test in bandwidth space preserves P3 equality without
        # subtract/add roundoff or a discretionary latency tolerance.
        required = np.full(len(users), np.inf)
        positive_margin = margins > 0
        required[positive_margin] = output_bits[positive_margin] / (margins[positive_margin] * down_rates[positive_margin])
        for i, request in enumerate(requests):
            success = bool(delivered[i] and down_beta[i] >= required[i])
            status = ("delivered" if success else "slo_miss") if delivered[i] else "downlink_drop"
            self._record(request, node, status, response[i], good=success, inferred=True,
                         delivered=bool(delivered[i]), upload_s=float(t_up[i]),
                         batch_wait_s=float(t_up.max() - t_up[i]), prefill_s=cost.prefill_s,
                         decode_s=cost.decode_s, downlink_s=float(t_down[i]) if delivered[i] else None,
                         gpu_tpot_s=cost.tpot_s)
            self.ledger.record("downlink_j", inference_end, response[i],
                               r.downlink_per_link_w if delivered[i] else 0.0, node)
        self.node_free_at[node] = response.max()
        self.batch_records.append({**batch_row, "inference_start_s": upload_end,
                                   "inference_end_s": inference_end, "release_s": float(response.max()),
                                   "down_beta": down_beta.tolist()})

    def step(self, action):
        assert self.slot < self.config.slots, "Call reset after a terminated/truncated episode."
        action = np.asarray(action)
        assert action.shape == (self.config.users,) and np.issubdtype(action.dtype, np.integer)
        assert np.all((0 <= action) & (action <= self.config.nodes))
        first_record = len(self.records)
        mask = self.action_masks().reshape(self.config.users, -1)
        batches = [[] for _ in range(self.config.nodes)]
        for u, request in self.current_requests.items():
            choice = int(action[u])
            if choice == 0:
                self._record(request, -1, "rejected", self.decision_s)
            elif not mask[u, choice]:
                # Out-of-mask decisions fail explicitly; they are never silently rerouted.
                self._record(request, choice - 1, "invalid", self.decision_s)
            else:
                batches[choice - 1].append(request)
        for node, requests in enumerate(batches):
            if requests:
                self._execute_batch(node, requests)
        outcomes = self.records[first_record:]
        count = len(outcomes)
        gp = sum(x["good"] for x in outcomes)
        oom = sum(x["status"] == "oom" for x in outcomes)
        reward = (self.config.good_weight * gp - self.config.oom_weight * oom) / count if count else 0.0
        self.rewards.append(float(reward))
        self.slot += 1
        self.current_requests = self.workload.arrivals(self.slot * self.config.slot_s, self.decision_s)
        info = {"requests": count, "goodput": gp, "oom": oom,
                "action_mask": self.action_masks().reshape(self.config.users, -1)}
        return self.observation(), float(reward), False, self.slot == self.config.slots, info

    def summary(self):
        rows, horizon = self.records, self.config.horizon_s
        n = len(rows)
        good = sum(x["good"] for x in rows)
        delivered = [x for x in rows if x["delivered"]]
        inferred = [x for x in rows if x["inferred"]]
        tpots = [x["gpu_tpot_s"] for x in inferred if x["gpu_tpot_s"] is not None]
        settled_at = max([horizon] + [x["response_s"] for x in rows])
        return {
            "requests": n, "goodput": good, "throughput": len(delivered),
            "inference_completed": len(inferred), "goodput_ratio": good / n if n else 0.0,
            "reward": float(sum(self.rewards)),
            "oom_requests": sum(x["status"] == "oom" for x in rows),
            "rejected": sum(x["status"] == "rejected" for x in rows),
            "invalid": sum(x["status"] == "invalid" for x in rows),
            "downlink_dropped": sum(x["status"] == "downlink_drop" for x in rows),
            "slo_missed": sum(x["status"] == "slo_miss" for x in rows),
            "goodput_within_window": sum(x["good"] and x["response_s"] <= horizon for x in rows),
            "throughput_within_window": sum(x["delivered"] and x["response_s"] <= horizon for x in rows),
            "pending_at_horizon": sum(x["response_s"] > horizon for x in rows),
            "e2e_s": float(np.mean([x["e2e_s"] for x in delivered])) if delivered else None,
            "gpu_ttft_s": float(np.mean([x["gpu_ttft_s"] for x in inferred])) if inferred else None,
            "gpu_tpot_s": float(np.mean(tpots)) if tpots else None,
            "horizon_s": horizon, "cohort_settled_at_s": settled_at,
            "energy_window": self.ledger.summary(0, horizon),
            "energy_tail": self.ledger.summary(horizon, settled_at),
            "energy_source": self.config.energy.source if self.record_energy else None,
        }

