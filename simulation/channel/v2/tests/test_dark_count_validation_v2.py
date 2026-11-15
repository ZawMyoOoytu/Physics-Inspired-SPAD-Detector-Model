
import copy

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    simulate_physical_spad,
)


def run_dark_only_simulation(seed=20261115):
    config = copy.deepcopy(CONFIG)

    # Large pulse count to obtain sufficient dark-count events.
    config["pulses"] = 500_000

    # Disable optical signal and afterpulsing.
    config["mean_photon_number"] = 0.0
    config["afterpulse_probability_per_avalanche"] = 0.0

    # High dark-count rate for a controlled statistical test.
    config["dark_count_rate_hz"] = 1_000_000

    return simulate_physical_spad(
        distance_km=10,
        dark_count_rate_hz=1_000_000,
        config=config,
        seed=seed,
    )


def test_dark_only_simulation_has_no_signal_attempts():
    result = run_dark_only_simulation()

    assert result["signal_attempts"] == 0


def test_dark_only_simulation_has_no_afterpulse_candidates():
    result = run_dark_only_simulation()

    assert result["afterpulse_candidates_scheduled"] == 0
    assert result["afterpulse_release_attempts"] == 0


def test_dark_only_simulation_produces_detection_events():
    result = run_dark_only_simulation()

    assert result["detected_events"] > 0
    assert result["registered_avalanches"] > 0


def test_dark_only_qber_is_approximately_unbiased():
    result = run_dark_only_simulation()

    assert result["sifted_key_length"] >= 100

    qber = result["qber"]

    assert qber is not None

    # Statistical tolerance for a finite simulation.
    # This is a model-level consistency check, not a
    # physical detector calibration test.
    assert 0.40 <= qber <= 0.60
