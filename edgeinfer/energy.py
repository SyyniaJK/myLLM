"""Passive energy observer; no random numbers or scheduling-state writes."""

from dataclasses import asdict, dataclass
import numpy as np
from .config import Config


@dataclass(frozen=True)
class PowerInterval:
    component: str
    start_s: float
    end_s: float
    power_w: float
    node: int

    def energy(self, start_s, end_s):
        overlap = max(0.0, min(end_s, self.end_s) - max(start_s, self.start_s))
        return overlap * self.power_w


class EnergyLedger:
    def __init__(self, config: Config):
        self.config = config
        self.intervals: list[PowerInterval] = []

    def record(self, component, start_s, end_s, power_w, node):
        assert end_s >= start_s and power_w >= 0
        self.intervals.append(PowerInterval(component, float(start_s), float(end_s), float(power_w), int(node)))

    def summary(self, start_s, end_s):
        assert 0 <= start_s <= end_s
        values = dict.fromkeys(("uplink_j", "downlink_j", "gpu_prefill_increment_j", "gpu_decode_increment_j"), 0.0)
        for interval in self.intervals:
            values[interval.component] += interval.energy(start_s, end_s)
        c = self.config
        values["gpu_idle_baseline_j"] = c.nodes * c.gpus_per_node * c.energy.idle_w_per_gpu * (end_s - start_s)
        values["total_j"] = sum(values.values())
        return values

    def rows(self):
        return [asdict(x) for x in self.intervals]


class NoEnergy:
    def record(self, component, start_s, end_s, power_w, node):
        pass

    def summary(self, start_s, end_s):
        return None

    def rows(self):
        return []


def integrate_power_samples(time_s, power_w):
    """Integrate a measured trace using linear interpolation between samples."""
    t, p = np.asarray(time_s, dtype=float), np.asarray(power_w, dtype=float)
    assert t.ndim == p.ndim == 1 and len(t) == len(p) >= 2
    assert np.all(np.isfinite(t)) and np.all(np.isfinite(p))
    assert np.all(np.diff(t) > 0) and np.all(p >= 0)
    return float(np.trapezoid(p, t))
