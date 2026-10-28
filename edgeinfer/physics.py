"""SI units throughout. Formulae implement TP-MPPO.md II-A and II-B."""

from dataclasses import dataclass
import numpy as np
from .config import Config, Model


@dataclass(frozen=True)
class BatchCost:
    prefill_s: float
    decode_s: float
    peak_bytes: float
    output_tokens: int

    @property
    def inference_s(self):
        return self.prefill_s + self.decode_s

    @property
    def tpot_s(self):
        return self.decode_s / (self.output_tokens - 1) if self.output_tokens > 1 else None


def batch_cost(model: Model, inputs, outputs, flops_s: float) -> BatchCost:
    inputs, outputs = np.asarray(inputs), np.asarray(outputs)
    assert inputs.ndim == outputs.ndim == 1 and inputs.size == outputs.size > 0
    assert np.all(inputs > 0) and np.all(outputs >= 1) and flops_s > 0
    b, s, o = inputs.size, int(inputs.max()), int(outputs.max())
    h, L, ff = model.hidden, model.layers, model.ffn_hidden
    pre = L * b * (6 * s * h**2 + 4 * s**2 * h + 2 * s * h**2 + 4 * s * h * ff)
    dec = L * b * (o - 1) * (6 * h**2 + 4 * (s + o / 2) * h + 2 * h**2 + 4 * h * ff)
    # FP16 MHA approximation from the paper, including output KV growth.
    memory = 2 * model.parameters + 4 * h * L * b * (s + o)
    return BatchCost(float(pre / flops_s), float(dec / flops_s), float(memory), o)


def full_band_rates(gains, bandwidth_hz: float, power_w: float, noise_w: float):
    gains = np.asarray(gains)
    assert np.all(gains > 0) and min(bandwidth_hz, power_w, noise_w) > 0
    # Paper noise is total power, not a spectral density. beta multiplies rate.
    return bandwidth_hz * np.log2(1 + power_w * gains / noise_w)


def disk_positions(rng, size, radius):
    angle = rng.uniform(0, 2 * np.pi, size)
    r = radius * np.sqrt(rng.uniform(size=size))
    return np.column_stack((r * np.cos(angle), r * np.sin(angle)))


def channel_trace(config: Config, seed: int):
    deployment = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    users = disk_positions(deployment, config.users, config.radius_m)
    nodes = disk_positions(deployment, config.nodes, config.radius_m)
    distances = np.linalg.norm(users[:, None] - nodes[None], axis=-1)
    assert np.all(distances > 0)
    r = config.radio
    average = (r.antenna_gain * 3e8 / (4 * np.pi * r.carrier_hz * distances)) ** r.path_exponent
    shape = (config.slots + 1, config.users, config.nodes)
    up_rng = np.random.default_rng(np.random.SeedSequence([seed, 12]))
    down_rng = np.random.default_rng(np.random.SeedSequence([seed, 13]))
    return up_rng.exponential(size=shape) * average, down_rng.exponential(size=shape) * average

