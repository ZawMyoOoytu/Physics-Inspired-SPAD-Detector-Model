
import copy

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    simulate_physical_spad,
)


def run_validation_simulation(seed=20261115):
    config = copy.deepcopy(CONFIG)
    config["pulses"] = 10_000

    return simulate_physical_spad(
        distance_km=10,
        dark_count_rate_hz=1_000,
        config=config,
        seed=seed,
    )


def test_qber_matches_error_fraction():
    result = run_validation_simulation()

    if result["sifted_key_length"] == 0:
        assert result["qber"] is None
    else:
        expected = (
            result["errors"] / result["sifted_key_length"]
        )
        assert result["qber"] == expected


def test_double_click_counter_bounds():
    result = run_validation_simulation()

    assert 0 <= result["double_click_events"]
    assert (
        result["double_click_events"]
        <= result["detected_events"]
    )


def test_sifted_key_counter_bounds():
    result = run_validation_simulation()

    assert (
        0 <= result["sifted_key_length"]
        <= result["detected_events"]
    )

    assert 0 <= result["errors"] <= result["sifted_key_length"]


def test_afterpulse_accounting_identity():
    result = run_validation_simulation()

    assert (
        result["afterpulse_candidates_scheduled"]
        ==
        result["afterpulse_release_attempts"]
        + result["afterpulse_pending_at_end"]
    )

    assert (
        0
        <= result["afterpulse_blocked_by_dead_time"]
        <= result["afterpulse_release_attempts"]
    )


def test_registered_avalanche_upper_bound():
    result = run_validation_simulation()

    assert (
        result["registered_avalanches"]
        <= 2 * result["pulses"]
    )


def test_rate_calculations_are_consistent():
    result = run_validation_simulation()

    expected_duration = (
        result["pulses"]
        / CONFIG["pulse_repetition_rate_hz"]
    )

    assert result["duration_seconds"] == expected_duration

    assert result["detection_rate"] == (
        result["detected_events"] / result["pulses"]
    )

    assert result["sifted_rate"] == (
        result["sifted_key_length"] / result["pulses"]
    )

    assert result["sifted_bits_per_second"] == (
        result["sifted_key_length"] / expected_duration
    )

    assert result["detection_rate_per_second"] == (
        result["detected_events"] / expected_duration
    )
