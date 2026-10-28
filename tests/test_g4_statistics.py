import numpy as np
import pytest
from edgeinfer.benchmark_stats import (bootstrap_indices, bootstrap_means, conditional_latency,
                                      energy_per_good, score_matrix, summarize_matrix)


def test_complete_crossed_cells_required():
    rows = [{"replicate": 1, "scenario_seed": 7, "score": 1}]
    with pytest.raises(AssertionError):
        score_matrix(rows, "score", [1, 2], [7])
    with pytest.raises(AssertionError):
        score_matrix(rows + rows, "score", [1], [7])


def test_bootstrap_is_reproducible_and_preserves_pairing():
    a = np.array([[1., 2., 3.], [5., 6., 7.]])
    b = a - 4
    indices = bootstrap_indices(2, 3, 500, 71)
    second = bootstrap_indices(2, 3, 500, 71)
    np.testing.assert_array_equal(indices[0], second[0])
    np.testing.assert_array_equal(indices[1], second[1])
    paired = summarize_matrix(a - b, indices)
    assert paired["mean"] == paired["ci_low"] == paired["ci_high"] == 4


def test_deterministic_duplicate_rows_do_not_increase_scenario_sample_size():
    values = np.array([[1., 4., 10.]])
    columns = np.array([[0, 1, 2], [0, 0, 0], [2, 2, 2]])
    first = bootstrap_means(values, (np.zeros((3, 1), dtype=int), columns))
    repeated = np.repeat(values, 3, axis=0)
    second = bootstrap_means(repeated, (np.array([[0, 1, 2], [2, 2, 2], [1, 0, 1]]), columns))
    np.testing.assert_array_equal(first, second)


def test_efficiency_uses_ratio_of_means_and_never_zero_floor():
    idx = bootstrap_indices(1, 2, 50, 1)
    energy = np.array([[20., 60.]])
    goodput = np.array([[1., 9.]])
    assert energy_per_good(energy, goodput, idx)["mean"] == 8
    undefined = energy_per_good(energy, np.zeros_like(goodput), idx)
    assert undefined["mean"] is None and undefined["ci_low"] is None
    assert undefined["undefined_resamples"] == 50


def test_completed_latency_is_request_weighted_not_mean_of_episode_means():
    rows = [{"throughput_window": 1, "e2e_mean_s": 1.},
            {"throughput_window": 9, "e2e_mean_s": 3.},
            {"throughput_window": 0, "e2e_mean_s": None}]
    result = conditional_latency(rows)
    assert result["e2e_mean_s"] == 2.8
    assert result["latency_completed_requests"] == 10
    assert result["episodes_without_completed_output"] == 1
