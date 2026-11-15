
import math
import random

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    calculate_timing,
    build_trap_gate_distributions,
    sample_afterpulse_gate,
)


def build_test_distributions():
    timing = calculate_timing(CONFIG)

    distributions = build_trap_gate_distributions(
        CONFIG,
        timing,
    )

    return timing, distributions


class FixedRandom:
    """Deterministic RNG for testing mixture selection."""

    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def test_distribution_count_matches_trap_lifetimes():
    _, distributions = build_test_distributions()

    assert len(distributions) == len(
        CONFIG["trap_lifetimes_us"]
    )


def test_gate_probabilities_sum_to_one():
    _, distributions = build_test_distributions()

    for distribution in distributions:
        probabilities = distribution["gate_probabilities"]

        assert math.isclose(
            sum(probabilities),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_gate_probabilities_are_nonnegative():
    _, distributions = build_test_distributions()

    for distribution in distributions:
        assert all(
            probability >= 0
            for probability in distribution["gate_probabilities"]
        )


def test_cumulative_distribution_is_monotonic():
    _, distributions = build_test_distributions()

    for distribution in distributions:
        cumulative = distribution["cumulative"]

        assert all(
            cumulative[index] <= cumulative[index + 1]
            for index in range(len(cumulative) - 1)
        )

        assert math.isclose(
            cumulative[-1],
            1.0,
            abs_tol=1e-12,
        )


def test_release_gate_count_matches_observation_window():
    timing, distributions = build_test_distributions()

    expected_gates = timing["afterpulse_window_gates"]

    for distribution in distributions:
        assert len(
            distribution["gate_probabilities"]
        ) == expected_gates


def test_sampled_release_delay_is_within_window():
    timing, distributions = build_test_distributions()

    rng = random.Random(20261115)

    for _ in range(10_000):
        delay = sample_afterpulse_gate(
            rng,
            CONFIG,
            distributions,
        )

        assert 1 <= delay <= timing["afterpulse_window_gates"]


def test_shorter_trap_lifetime_has_earlier_expected_release():
    _, distributions = build_test_distributions()

    lifetimes = [
        distribution["lifetime_us"]
        for distribution in distributions
    ]

    shortest_index = min(
        range(len(lifetimes)),
        key=lambda index: lifetimes[index],
    )

    longest_index = max(
        range(len(lifetimes)),
        key=lambda index: lifetimes[index],
    )

    assert lifetimes[shortest_index] < lifetimes[longest_index]

    def expected_gate(distribution):
        return sum(
            (index + 1) * probability
            for index, probability in enumerate(
                distribution["gate_probabilities"]
            )
        )

    short_mean = expected_gate(
        distributions[shortest_index]
    )

    long_mean = expected_gate(
        distributions[longest_index]
    )

    assert short_mean < long_mean


def test_mixture_sampling_selects_first_trap():
    config = {
        "trap_population_weights": [0.7, 0.3]
    }

    distributions = [
        {"cumulative": [0.2, 1.0]},
        {"cumulative": [0.8, 1.0]},
    ]

    rng = FixedRandom([0.1, 0.5])

    delay = sample_afterpulse_gate(
        rng,
        config,
        distributions,
    )

    assert delay == 2


def test_mixture_sampling_selects_second_trap():
    config = {
        "trap_population_weights": [0.7, 0.3]
    }

    distributions = [
        {"cumulative": [0.2, 1.0]},
        {"cumulative": [0.8, 1.0]},
    ]

    rng = FixedRandom([0.9, 0.5])

    delay = sample_afterpulse_gate(
        rng,
        config,
        distributions,
    )

    assert delay == 1
