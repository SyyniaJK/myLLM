from dataclasses import replace
from itertools import product
from math import fsum
import numpy as np
import pytest
from edgeinfer.allocation import TPBandwidth
from edgeinfer.config import Model, load_config
from edgeinfer.physics import batch_cost, full_band_rates, channel_trace


def test_paper_formula_hand_calculation():
    result = batch_cost(Model("test", 100, 2, 1, 4), [2], [2], 100)
    assert result.prefill_s == 1.6
    assert result.decode_s == 0.88
    assert result.peak_bytes == 200 + 4 * 2 * 1 * (2 + 2)


def test_one_token_output_has_no_decode():
    result = batch_cost(Model(), [128], [1], 1e14)
    assert result.decode_s == 0
    assert result.tpot_s is None


def test_batch_uses_max_input_and_max_output():
    a = batch_cost(Model(), [128, 512], [1024, 128], 1e14)
    b = batch_cost(Model(), [512, 512], [1024, 1024], 1e14)
    assert a == b


def test_kv_includes_output_growth():
    a = batch_cost(Model(), [128], [128], 1e14)
    b = batch_cost(Model(), [128], [512], 1e14)
    assert b.peak_bytes - a.peak_bytes == 4 * 4096 * 32 * (512 - 128)


def test_config_7b_fits_and_old_14b_does_not():
    c = load_config("configs/paper_aligned.toml")
    assert batch_cost(c.model, [128], [128], c.flops_s).peak_bytes < c.capacity_bytes
    old = Model("old", 14_000_000_000, 5120, 40, 13824)
    assert batch_cost(old, [128], [128], c.flops_s).peak_bytes > c.capacity_bytes


def test_si_memory_units_and_multigpu_scale(tiny_config):
    c = tiny_config.resized(gpus=3)
    assert c.capacity_bytes == 3 * c.device.memory_gb * 1e9
    assert c.flops_s == 3 * c.device.tflops * 1e12


def test_radio_formula_and_direction_specific_noise(tiny_config):
    gain = np.array([2e-14])
    assert full_band_rates(gain, 100, 1, 2e-14)[0] == 100
    a, b = channel_trace(tiny_config, 42)
    aa, bb = channel_trace(tiny_config, 42)
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)
    assert not np.array_equal(a, b)


def test_uplink_equalizes_completion_times():
    bits, rates = np.array([1.0, 4.0]), np.array([2.0, 2.0])
    beta = TPBandwidth().uplink(bits, rates)
    np.testing.assert_allclose(beta, [0.2, 0.8])
    np.testing.assert_allclose(bits / (rates * beta), [2.5, 2.5])


@pytest.mark.parametrize("seed", range(12))
def test_downlink_cardinality_against_exact_subset_enumeration(seed):
    rng = np.random.default_rng(seed)
    requirements = rng.uniform(0.01, 1.2, 6)
    result = TPBandwidth().downlink(requirements, np.ones(6), np.ones(6))
    maximum = max(sum(subset) for subset in product((0, 1), repeat=6)
                  if np.dot(subset, requirements) <= 1)
    assert np.count_nonzero(result) == maximum
    assert result.sum() <= np.nextafter(1.0, np.inf)
    assert np.all(result[result > 0] >= requirements[result > 0])


def test_downlink_nonpositive_margin_is_infeasible():
    beta = TPBandwidth().downlink(np.ones(3), np.array([-1.0, 0.0, 2.0]), np.ones(3))
    np.testing.assert_array_equal(beta, [0, 0, 1])


def test_boundary_exactly_one_bandwidth_unit():
    beta = TPBandwidth().downlink(np.array([0.25, 0.75]), np.ones(2), np.ones(2))
    np.testing.assert_array_equal(beta, [0.25, 0.75])


def test_invalid_formula_domain_is_an_explicit_contract():
    with pytest.raises(AssertionError):
        batch_cost(Model(), [], [], 1e14)


def test_bandwidth_partition_roundoff_regression():
    rng = np.random.default_rng(1)
    allocator = TPBandwidth()
    for _ in range(10000):
        bits, rates = rng.uniform(1, 10000, 15), rng.uniform(1, 1e8, 15)
        beta = allocator.uplink(bits, rates)
        assert fsum(beta) <= np.nextafter(1.0, np.inf)
        np.testing.assert_allclose(bits / (beta * rates), fsum(bits / rates))

