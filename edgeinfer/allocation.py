"""The resource allocator is an algorithm component, separate from physical models."""

from typing import Protocol
from math import fsum
import numpy as np


class BandwidthAllocator(Protocol):
    def uplink(self, bits: np.ndarray, full_rates: np.ndarray) -> np.ndarray: ...
    def downlink(self, bits: np.ndarray, margins: np.ndarray, full_rates: np.ndarray) -> np.ndarray: ...


class TPBandwidth:
    """Paper P2 equal-completion upload and P3 equal-value greedy download."""

    def uplink(self, bits, full_rates):
        assert np.all(bits > 0) and np.all(full_rates > 0)
        weights = bits / full_rates
        return weights / fsum(weights)

    def downlink(self, bits, margins, full_rates):
        assert np.all(bits > 0) and np.all(full_rates > 0)
        requirements = np.full(len(bits), np.inf)
        possible = margins > 0
        requirements[possible] = bits[possible] / (margins[possible] * full_rates[possible])
        order = np.argsort(requirements, kind="stable")
        selected = order[np.cumsum(requirements[order]) <= 1]
        beta = np.zeros(len(bits))
        if selected.size:
            # Teacher code's tie-break: give unused bandwidth proportionally to
            # the same selected users. P3's selected set/cardinality is unchanged.
            beta[selected] = requirements[selected] / fsum(requirements[selected])
        return beta


class EqualBandwidth:
    """Simple explicit allocator for diagnostics and alternative policies."""

    def uplink(self, bits, full_rates):
        return np.full(len(bits), 1 / len(bits))

    def downlink(self, bits, margins, full_rates):
        return np.full(len(bits), 1 / len(bits))

