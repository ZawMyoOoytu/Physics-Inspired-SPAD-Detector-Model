
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from scipy import stats


# ============================================================
# PRACTICAL 17
# TRAP LIFETIME SENSITIVITY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.channel.physical_spad_detector import (
    CONFIG as BASE_CONFIG,
    simulate_physical_spad,
)


# ------------------------------------------------------------
# EXPERIMENT DESIGN
# ------------------------------------------------------------

TRAP_LIFETIMES_US = [
    1.0,
    5.0,
    10.0,
    50.0,
    100.0,
]

AFTERPULSE_PROBABILITIES = [
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
]

DISTANCES_KM = [
    10,
    50,
]

REPETITIONS = 50

PULSES_PER_RUN = 200_000

BASE_SEED = 20263000

CONFIDENCE_LEVEL = 0.95


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "results" / "channel"

RAW_FILE = (
    OUTPUT_DIR
    / "trap_lifetime_sensitivity_raw.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "trap_lifetime_sensitivity_summary.csv"
)

STATISTICS_FILE = (
    OUTPUT_DIR
    / "trap_lifetime_sensitivity_statistics.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

def build_experiment_config(
    lifetime_us,
    afterpulse_probability,
):

    config = deepcopy(BASE_CONFIG)

    config["pulses"] = PULSES_PER_RUN

    config["trap_lifetimes_us"] = [
        lifetime_us
    ]

    # Test one trap lifetime at a time.
    config["trap_population_weights"] = [
        1.0
    ]

    config[
        "afterpulse_probability_per_avalanche"
    ] = afterpulse_probability

    return config


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def calculate_confidence_interval(values):

    n = len(values)

    mean_value = statistics.mean(values)

    if n < 2:

        return {
            "mean": mean_value,
            "sd": 0.0,
            "ci_lower": mean_value,
            "ci_upper": mean_value,
        }

    standard_deviation = statistics.stdev(values)

    standard_error = (
        standard_deviation / math.sqrt(n)
    )

    critical_value = stats.t.ppf(
        1 - (1 - CONFIDENCE_LEVEL) / 2,
        df=n - 1,
    )

    margin = (
        critical_value * standard_error
    )

    return {
        "mean": mean_value,
        "sd": standard_deviation,
        "ci_lower": mean_value - margin,
        "ci_upper": mean_value + margin,
    }


# ============================================================
# EFFECT SIZES
# ============================================================

def calculate_hedges_g(
    baseline,
    comparison,
):

    n1 = len(baseline)
    n2 = len(comparison)

    if n1 < 2 or n2 < 2:
        return 0.0

    variance1 = statistics.variance(baseline)
    variance2 = statistics.variance(comparison)

    pooled_variance = (
        ((n1 - 1) * variance1)
        + ((n2 - 1) * variance2)
    ) / (n1 + n2 - 2)

    if pooled_variance <= 0:
        return 0.0

    pooled_sd = math.sqrt(pooled_variance)

    cohen_d = (
        statistics.mean(comparison)
        - statistics.mean(baseline)
    ) / pooled_sd

    degrees_of_freedom = n1 + n2 - 2

    correction = (
        1 - 3 / (4 * degrees_of_freedom - 1)
    )

    return cohen_d * correction


def calculate_cliffs_delta(
    baseline,
    comparison,
):

    greater = 0
    smaller = 0

    for value_a in baseline:

        for value_b in comparison:

            if value_b > value_a:
                greater += 1

            elif value_b < value_a:
                smaller += 1

    total = len(baseline) * len(comparison)

    if total == 0:
        return 0.0

    return (greater - smaller) / total


# ============================================================
# HOLM MULTIPLE-TEST CORRECTION
# ============================================================

def holm_adjust(p_values):

    count = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [1.0] * count

    running_maximum = 0.0

    for rank, (original_index, p_value) in enumerate(
        indexed
    ):

        multiplier = count - rank

        corrected = min(
            1.0,
            multiplier * p_value,
        )

        running_maximum = max(
            running_maximum,
            corrected,
        )

        adjusted[original_index] = (
            running_maximum
        )

    return adjusted


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiments():

    raw_results = []

    total_runs = (
        len(TRAP_LIFETIMES_US)
        * len(AFTERPULSE_PROBABILITIES)
        * len(DISTANCES_KM)
        * REPETITIONS
    )

    run_number = 0

    print("\n================================================")
    print(" PRACTICAL 17")
    print(" TRAP LIFETIME SENSITIVITY ANALYSIS")
    print("================================================")

    print(
        f"Trap lifetimes: {TRAP_LIFETIMES_US}"
    )

    print(
        f"Afterpulse probabilities: "
        f"{AFTERPULSE_PROBABILITIES}"
    )

    print(
        f"Distances: {DISTANCES_KM}"
    )

    print(
        f"Repetitions per condition: {REPETITIONS}"
    )

    print(
        f"Pulses per run: {PULSES_PER_RUN:,}"
    )

    print(
        f"Total simulation runs: {total_runs:,}"
    )

    print(
        f"Total simulated pulses: "
        f"{total_runs * PULSES_PER_RUN:,}"
    )

    print("\nStarting simulation...\n")

    for distance in DISTANCES_KM:

        for lifetime in TRAP_LIFETIMES_US:

            for ap_probability in AFTERPULSE_PROBABILITIES:

                config = build_experiment_config(
                    lifetime,
                    ap_probability,
                )

                for repetition in range(REPETITIONS):

                    run_number += 1

                    seed = (
                        BASE_SEED + run_number
                    )

                    result = simulate_physical_spad(
                        distance_km=distance,
                        dark_count_rate_hz=(
                            config["dark_count_rate_hz"]
                        ),
                        config=config,
                        seed=seed,
                    )

                    raw_row = {
                        "distance_km": distance,
                        "trap_lifetime_us": lifetime,
                        "afterpulse_probability": (
                            ap_probability
                        ),
                        "repetition": repetition + 1,
                        "seed": seed,

                        "signal_clicks": (
                            result["signal_clicks"]
                        ),

                        "dark_clicks": (
                            result["dark_clicks"]
                        ),

                        "afterpulse_candidates_scheduled": (
                            result[
                                "afterpulse_candidates_scheduled"
                            ]
                        ),

                        "afterpulse_release_attempts": (
                            result[
                                "afterpulse_release_attempts"
                            ]
                        ),

                        "afterpulse_registered_attempts": (
                            result[
                                "afterpulse_registered_attempts"
                            ]
                        ),

                        "afterpulse_blocked_by_dead_time": (
                            result[
                                "afterpulse_blocked_by_dead_time"
                            ]
                        ),

                        "afterpulse_pending_at_end": (
                            result[
                                "afterpulse_pending_at_end"
                            ]
                        ),

                        "dead_time_lost_signal": (
                            result["dead_time_lost_signal"]
                        ),

                        "detected_events": (
                            result["detected_events"]
                        ),

                        "double_click_events": (
                            result["double_click_events"]
                        ),

                        "sifted_key_length": (
                            result["sifted_key_length"]
                        ),

                        "errors": result["errors"],

                        "qber": result["qber"],

                        "sifted_bits_per_second": (
                            result["sifted_bits_per_second"]
                        ),
                    }

                    raw_results.append(raw_row)

                    if (
                        run_number % 25 == 0
                        or run_number == total_runs
                    ):

                        print(
                            f"Progress: "
                            f"{run_number}/{total_runs} "
                            f"({100 * run_number / total_runs:.1f}%)"
                        )

    return raw_results


# ============================================================
# SAVE RAW RESULTS
# ============================================================

def save_raw_results(raw_results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(raw_results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(raw_results)

    print("\nRaw results saved:")
    print(RAW_FILE)


# ============================================================
# GROUP RESULTS
# ============================================================

def group_results(raw_results):

    grouped = defaultdict(list)

    for row in raw_results:

        key = (
            row["distance_km"],
            row["trap_lifetime_us"],
            row["afterpulse_probability"],
        )

        grouped[key].append(row)

    return grouped


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def generate_summary(raw_results):

    grouped = group_results(raw_results)

    summary_rows = []

    for key in sorted(grouped):

        distance, lifetime, ap_probability = key

        rows = grouped[key]

        metrics = [
            "qber",
            "sifted_bits_per_second",
            "afterpulse_candidates_scheduled",
            "afterpulse_release_attempts",
            "afterpulse_registered_attempts",
            "afterpulse_blocked_by_dead_time",
            "dead_time_lost_signal",
            "sifted_key_length",
        ]

        output = {
            "distance_km": distance,
            "trap_lifetime_us": lifetime,
            "afterpulse_probability": ap_probability,
            "repetitions": len(rows),
        }

        for metric in metrics:

            values = [
                float(row[metric])
                for row in rows
            ]

            ci = calculate_confidence_interval(
                values
            )

            output[f"{metric}_mean"] = ci["mean"]

            output[f"{metric}_sd"] = ci["sd"]

            output[f"{metric}_ci95_lower"] = (
                ci["ci_lower"]
            )

            output[f"{metric}_ci95_upper"] = (
                ci["ci_upper"]
            )

        summary_rows.append(output)

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSummary saved:")
    print(SUMMARY_FILE)

    return summary_rows


# ============================================================
# STATISTICAL COMPARISONS
# ============================================================

def generate_statistical_tests(raw_results):

    grouped = group_results(raw_results)

    test_rows = []

    # AP = 0 is the baseline for each distance/lifetime.
    for distance in DISTANCES_KM:

        for lifetime in TRAP_LIFETIMES_US:

            baseline_key = (
                distance,
                lifetime,
                0.0,
            )

            baseline_rows = grouped[baseline_key]

            baseline_qber = [
                float(row["qber"])
                for row in baseline_rows
            ]

            baseline_throughput = [
                float(row["sifted_bits_per_second"])
                for row in baseline_rows
            ]

            for ap_probability in (
                value
                for value in AFTERPULSE_PROBABILITIES
                if value > 0
            ):

                comparison_key = (
                    distance,
                    lifetime,
                    ap_probability,
                )

                comparison_rows = grouped[
                    comparison_key
                ]

                comparison_qber = [
                    float(row["qber"])
                    for row in comparison_rows
                ]

                comparison_throughput = [
                    float(row["sifted_bits_per_second"])
                    for row in comparison_rows
                ]

                # --------------------------------------------
                # QBER: primary endpoint
                # --------------------------------------------

                welch_qber = stats.ttest_ind(
                    baseline_qber,
                    comparison_qber,
                    equal_var=False,
                )

                mw_qber = stats.mannwhitneyu(
                    baseline_qber,
                    comparison_qber,
                    alternative="two-sided",
                )

                test_rows.append({
                    "distance_km": distance,
                    "trap_lifetime_us": lifetime,
                    "afterpulse_probability": ap_probability,
                    "metric": "qber",
                    "baseline_mean": statistics.mean(
                        baseline_qber
                    ),
                    "comparison_mean": statistics.mean(
                        comparison_qber
                    ),
                    "mean_difference": (
                        statistics.mean(comparison_qber)
                        - statistics.mean(baseline_qber)
                    ),
                    "welch_t_statistic": (
                        welch_qber.statistic
                    ),
                    "welch_p_value": (
                        welch_qber.pvalue
                    ),
                    "mann_whitney_u": (
                        mw_qber.statistic
                    ),
                    "mann_whitney_p_value": (
                        mw_qber.pvalue
                    ),
                    "hedges_g": calculate_hedges_g(
                        baseline_qber,
                        comparison_qber,
                    ),
                    "cliffs_delta": calculate_cliffs_delta(
                        baseline_qber,
                        comparison_qber,
                    ),
                })

                # --------------------------------------------
                # Throughput: secondary endpoint
                # --------------------------------------------

                welch_rate = stats.ttest_ind(
                    baseline_throughput,
                    comparison_throughput,
                    equal_var=False,
                )

                mw_rate = stats.mannwhitneyu(
                    baseline_throughput,
                    comparison_throughput,
                    alternative="two-sided",
                )

                test_rows.append({
                    "distance_km": distance,
                    "trap_lifetime_us": lifetime,
                    "afterpulse_probability": ap_probability,
                    "metric": "sifted_bits_per_second",
                    "baseline_mean": statistics.mean(
                        baseline_throughput
                    ),
                    "comparison_mean": statistics.mean(
                        comparison_throughput
                    ),
                    "mean_difference": (
                        statistics.mean(comparison_throughput)
                        - statistics.mean(baseline_throughput)
                    ),
                    "welch_t_statistic": (
                        welch_rate.statistic
                    ),
                    "welch_p_value": (
                        welch_rate.pvalue
                    ),
                    "mann_whitney_u": (
                        mw_rate.statistic
                    ),
                    "mann_whitney_p_value": (
                        mw_rate.pvalue
                    ),
                    "hedges_g": calculate_hedges_g(
                        baseline_throughput,
                        comparison_throughput,
                    ),
                    "cliffs_delta": calculate_cliffs_delta(
                        baseline_throughput,
                        comparison_throughput,
                    ),
                })

    # --------------------------------------------------------
    # Holm correction is performed separately for each metric
    # and each statistical test.
    # --------------------------------------------------------

    for metric in [
        "qber",
        "sifted_bits_per_second",
    ]:

        metric_indices = [
            index
            for index, row in enumerate(test_rows)
            if row["metric"] == metric
        ]

        welch_p_values = [
            test_rows[index]["welch_p_value"]
            for index in metric_indices
        ]

        mw_p_values = [
            test_rows[index]["mann_whitney_p_value"]
            for index in metric_indices
        ]

        welch_adjusted = holm_adjust(
            welch_p_values
        )

        mw_adjusted = holm_adjust(
            mw_p_values
        )

        for position, original_index in enumerate(
            metric_indices
        ):

            test_rows[original_index][
                "welch_holm_adjusted_p"
            ] = welch_adjusted[position]

            test_rows[original_index][
                "mann_whitney_holm_adjusted_p"
            ] = mw_adjusted[position]

            test_rows[original_index][
                "welch_significant_005"
            ] = (
                welch_adjusted[position] < 0.05
            )

            test_rows[original_index][
                "mann_whitney_significant_005"
            ] = (
                mw_adjusted[position] < 0.05
            )

    with STATISTICS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(test_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(test_rows)

    print("\nStatistical tests saved:")
    print(STATISTICS_FILE)

    return test_rows


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_summary(summary_rows):

    print("\n================================================")
    print(" PRACTICAL 17 SUMMARY")
    print("================================================")

    print(
        f"{'Distance':>8} "
        f"{'Lifetime':>9} "
        f"{'AP %':>7} "
        f"{'QBER %':>10} "
        f"{'Throughput':>13}"
    )

    for row in summary_rows:

        print(
            f"{row['distance_km']:>7}km "
            f"{row['trap_lifetime_us']:>8.1f} "
            f"{row['afterpulse_probability'] * 100:>6.2f} "
            f"{row['qber_mean'] * 100:>9.4f} "
            f"{row['sifted_bits_per_second_mean']:>13.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    raw_results = run_experiments()

    save_raw_results(raw_results)

    summary_rows = generate_summary(
        raw_results
    )

    generate_statistical_tests(
        raw_results
    )

    print_summary(summary_rows)

    print("\n================================================")
    print(" PRACTICAL 17 COMPLETED")
    print("================================================")

    print("\nGenerated files:")

    print(RAW_FILE)
    print(SUMMARY_FILE)
    print(STATISTICS_FILE)


if __name__ == "__main__":

    main()
