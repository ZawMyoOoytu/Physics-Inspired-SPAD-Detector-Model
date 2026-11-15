
import copy

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    simulate_physical_spad,
)


def run_small_simulation(config=None, seed=12345):

    test_config = copy.deepcopy(
        CONFIG if config is None else config
    )

    test_config["pulses"] = 5_000

    return simulate_physical_spad(
        distance_km=10,
        dark_count_rate_hz=1_000,
        config=test_config,
        seed=seed,
    )


def test_simulator_is_reproducible():

    first = run_small_simulation(seed=12345)
    second = run_small_simulation(seed=12345)

    assert first == second


def test_avalanche_classification_is_consistent():

    result = run_small_simulation()

    classified = (
        result["signal_only_avalanches"]
        + result["dark_only_avalanches"]
        + result["afterpulse_only_avalanches"]
        + result["mixed_source_avalanches"]
    )

    assert result["registered_avalanches"] == classified


def test_afterpulse_accounting_is_consistent():

    result = run_small_simulation()

    assert (
        result["afterpulse_candidates_scheduled"]
        ==
        result["afterpulse_release_attempts"]
        + result["afterpulse_pending_at_end"]
    )

    assert (
        result["afterpulse_blocked_by_dead_time"]
        <= result["afterpulse_release_attempts"]
    )


def test_detection_and_key_counters_are_valid():

    result = run_small_simulation()

    assert (
        result["detected_events"]
        <= result["registered_avalanches"]
    )

    assert (
        result["registered_avalanches"]
        <= 2 * result["pulses"]
    )

    assert (
        result["errors"]
        <= result["sifted_key_length"]
    )

    if result["sifted_key_length"] > 0:

        assert result["qber"] is not None

        assert 0 <= result["qber"] <= 1


def test_zero_detection_produces_undefined_qber():

    config = copy.deepcopy(CONFIG)

    config["pulses"] = 500
    config["mean_photon_number"] = 0.0
    config["dark_count_rate_hz"] = 0
    config["afterpulse_probability_per_avalanche"] = 0.0

    result = simulate_physical_spad(
        distance_km=10,
        dark_count_rate_hz=0,
        config=config,
        seed=12345,
    )

    assert result["detected_events"] == 0
    assert result["sifted_key_length"] == 0
    assert result["errors"] == 0
    assert result["qber"] is None
