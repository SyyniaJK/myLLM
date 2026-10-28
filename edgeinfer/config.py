"""Typed configuration. Assertions express input contracts; values are never repaired."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Model:
    name: str = "Llama-7B"
    parameters: int = 7_000_000_000
    hidden: int = 4096
    layers: int = 32
    ffn_hidden: int = 11008


MODELS = {
    "Llama-7B": Model(),
    "TinyLlama-1.1B": Model("TinyLlama-1.1B", 1_100_000_000, 2048, 22, 5632),
    "Qwen1.5-4B": Model("Qwen1.5-4B", 4_000_000_000, 2560, 40, 6912),
    "Llama-30B": Model("Llama-30B", 30_000_000_000, 6656, 60, 17920),
}


@dataclass(frozen=True)
class Device:
    name: str = "paper_L2_analytical"
    memory_gb: float = 24.0
    tflops: float = 96.5


@dataclass(frozen=True)
class Radio:
    uplink_hz: float = 5e6
    downlink_hz: float = 5e6
    uplink_w: float = 1.0
    downlink_per_link_w: float = 1.0
    noise_up_w: float = 1e-14
    noise_down_w: float = 2e-14
    antenna_gain: float = 3.0
    carrier_hz: float = 915e6
    path_exponent: float = 3.0


@dataclass(frozen=True)
class Power:
    source: str = "illustrative_phase_power_not_hardware_measurement"
    idle_w_per_gpu: float = 15.0
    prefill_w_per_gpu: float = 60.0
    decode_w_per_gpu: float = 45.0


@dataclass(frozen=True)
class PPOConfig:
    hidden: int = 128
    rollout_steps: int = 256
    minibatch: int = 128
    epochs: int = 10
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    max_grad_norm: float = 0.5
    reward_scaling: bool = True


@dataclass(frozen=True)
class Config:
    name: str = "paper_aligned"
    users: int = 15
    nodes: int = 3
    gpus_per_node: int = 1
    slots: int = 200
    slot_s: float = 1.0
    radius_m: float = 300.0
    mean_think_s: float = 2.0
    input_tokens: tuple[int, ...] = (128, 256, 512)
    output_caps: tuple[int, ...] = (128, 256, 512, 1024)
    output_fractions: tuple[float, ...] = (0.5, 0.75, 1.0)
    slo_s: tuple[float, ...] = (3.0, 5.0, 9.0)
    bits_per_token: float = 8.0
    good_weight: float = 1.0
    oom_weight: float = 0.5
    model: Model = field(default_factory=Model)
    device: Device = field(default_factory=Device)
    radio: Radio = field(default_factory=Radio)
    energy: Power = field(default_factory=Power)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        assert self.users > 0 and self.nodes > 0 and self.gpus_per_node > 0
        assert self.slots > 0 and self.slot_s > 0 and self.mean_think_s > 0
        assert len(self.input_tokens) == len(self.slo_s)
        assert min(self.input_tokens) > 0 and min(self.output_caps) >= 1
        assert all(0 < x <= 1 for x in self.output_fractions)
        assert all(float(c * f).is_integer() for c in self.output_caps for f in self.output_fractions)
        assert min(self.slo_s) > 0 and self.bits_per_token > 0 and self.radius_m > 0
        assert min(asdict(self.model)[k] for k in ("parameters", "hidden", "layers", "ffn_hidden")) > 0
        assert self.device.memory_gb > 0 and self.device.tflops > 0
        assert all(v > 0 for v in asdict(self.radio).values())
        assert 0 <= self.energy.idle_w_per_gpu <= self.energy.prefill_w_per_gpu
        assert self.energy.idle_w_per_gpu <= self.energy.decode_w_per_gpu
        assert self.ppo.rollout_steps % self.ppo.minibatch == 0
        assert self.ppo.epochs > 0 and 0 <= self.ppo.gae_lambda <= 1
        assert 0 <= self.ppo.gamma <= 1 and 0 < self.ppo.clip < 1

    @property
    def horizon_s(self):
        return self.slots * self.slot_s

    @property
    def capacity_bytes(self):
        return self.gpus_per_node * self.device.memory_gb * 1e9

    @property
    def flops_s(self):
        return self.gpus_per_node * self.device.tflops * 1e12

    def resized(self, *, users=None, nodes=None, gpus=None, model=None):
        values = {"users": users, "nodes": nodes, "gpus_per_node": gpus}
        values = {k: v for k, v in values.items() if v is not None}
        if model is not None:
            values["model"] = MODELS[model]
        return replace(self, **values)


def load_config(path: str | Path) -> Config:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(data)


def config_from_dict(data) -> Config:
    data = dict(data)
    for key, cls in (("model", Model), ("device", Device), ("radio", Radio), ("energy", Power), ("ppo", PPOConfig)):
        data[key] = cls(**data[key])
    for key in ("input_tokens", "output_caps", "output_fractions", "slo_s"):
        data[key] = tuple(data[key])
    return Config(**data)

