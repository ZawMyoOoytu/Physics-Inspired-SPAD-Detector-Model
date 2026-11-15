import copy

import simulation.channel.v2.physical_spad_detector_v2 as detector


class ZeroRNG:
    """Deterministic RNG: every random draw returns zero."""

    def random(self):
        return 0.0


def make_config(pulses, dead_time_us=1.0):

    config = copy.deepcopy(detector.CONFIG)

    config["pulses"] = pulses
    config["dead_time_us"] = dead_time_us

    config["afterpulse_probability_per_avalanche"] = 1.0
    config["trap_lifetimes_us"] = [5.0]
    config["trap_population_weights"] = [1.0]
    config["afterpulse_window_us"] = 4.0

    config["misalignment_probability"] = 0.0

    return config


def prepare_deterministic_simulation(monkeypatch):

    # Make every pulse produce a signal attempt.
    monkeypatch.setattr(
        detector,
        "calculate_signal_probability",
        lambda distance_km, config, timing: {
            "fiber_loss_db": 0.0,
            "transmission": 1.0,
            "total_efficiency": 1.0,
            "signal_probability": 1.0,
        },
    )

    # Make all random draws deterministic.
    monkeypatch.setattr(
        detector.random,
        "Random",
        lambda seed: ZeroRNG(),
    )


def run_simulation(config, seed=12345):

    return detector.simulate_physical_spad(
        distance_km=10,
        dark_count_rate_hz=0,
        config=config,
        seed=seed,
    )


def assert_afterpulse_accounting_identity(result):

    assert (
        result["afterpulse_candidates_scheduled"]
        ==
        result["afterpulse_release_attempts"]
        + result["afterpulse_pending_at_end"]
    )


def test_afterpulse_queue_is_detector_specific(monkeypatch):

    prepare_deterministic_simulation(monkeypatch)

    config = make_config(pulses=4, dead_time_us=1.0)

    # Every avalanche schedules an afterpulse one gate later.
    monkeypatch.setattr(
        detector,
        "sample_afterpulse_gate",
        lambda rng, config, distributions: 1,
    )

    original_resolver = detector.resolve_detector_gate

    resolver_calls = []

    def recording_resolver(**kwargs):

        resolver_calls.append(
            kwargs["afterpulse_release_count"]
        )

        return original_resolver(**kwargs)

    monkeypatch.setattr(
        detector,
        "resolve_detector_gate",
        recording_resolver,
    )

    result = run_simulation(config)

    # Calls are ordered:
    # pulse 0: detector 0, detector 1
    # pulse 1: detector 0, detector 1
    # and so on.
    assert resolver_calls == [
        0, 0,
        1, 0,
        1, 0,
        1, 0,
    ]

    assert result["afterpulse_candidates_scheduled"] == 4
    assert result["afterpulse_release_attempts"] == 3
    assert result["afterpulse_pending_at_end"] == 1

    assert_afterpulse_accounting_identity(result)


def test_afterpulse_release_during_dead_time_is_blocked(monkeypatch):

    prepare_deterministic_simulation(monkeypatch)

    config = make_config(pulses=3, dead_time_us=2.0)

    monkeypatch.setattr(
        detector,
        "sample_afterpulse_gate",
        lambda rng, config, distributions: 1,
    )

    result = run_simulation(config)

    # Pulse 0 schedules one afterpulse for pulse 1.
    # Detector 0 is still inside its dead time at pulse 1.
    assert result["afterpulse_candidates_scheduled"] == 2
    assert result["afterpulse_release_attempts"] == 1
    assert result["afterpulse_blocked_by_dead_time"] == 1
    assert result["afterpulse_pending_at_end"] == 1

    assert_afterpulse_accounting_identity(result)


def test_multiple_afterpulse_releases_register_one_avalanche(
    monkeypatch
):

    prepare_deterministic_simulation(monkeypatch)

    config = make_config(pulses=3, dead_time_us=1.0)

    # Pulse 0 schedules release at pulse 2.
    # Pulse 1 also schedules release at pulse 2.
    # Pulse 2 therefore receives two afterpulse releases
    # in the same detector gate.
    delays = iter([2, 1, 2])

    monkeypatch.setattr(
        detector,
        "sample_afterpulse_gate",
        lambda rng, config, distributions: next(delays),
    )

    original_resolver = detector.resolve_detector_gate

    resolver_results = []

    def recording_resolver(**kwargs):

        result = original_resolver(**kwargs)

        resolver_results.append(
            (
                kwargs["afterpulse_release_count"],
                result,
            )
        )

        return result

    monkeypatch.setattr(
        detector,
        "resolve_detector_gate",
        recording_resolver,
    )

    result = run_simulation(config)

    # Pulse 2, detector 0 is the fifth resolver call.
    release_count, resolution = resolver_results[4]

    assert release_count == 2
    assert resolution["registered"] is True
    assert resolution["registered_avalanches"] == 1
    assert resolution["source_class"] == "mixed_source"

    assert result["afterpulse_candidates_scheduled"] == 3
    assert result["afterpulse_release_attempts"] == 2
    assert result["afterpulse_pending_at_end"] == 1

    assert result["registered_avalanches"] == 3
    assert result["mixed_source_avalanches"] == 1

    assert_afterpulse_accounting_identity(result)
