
import csv
import math
import statistics
from pathlib import Path

from scipy import stats

from time_gated_detector import (
    CONFIG,
    simulate_detector
)


# ============================================================
# PRACTICAL 15
# 50-REPETITION AFTERPULSE VALIDATION
# ============================================================

DISTANCES_KM = [10, 50]

AFTERPULSE_PROBABILITIES = [
    0.000,
    0.005,
    0.010,
    0.020,
    0.050
]

REPETITION_RATE_HZ = 1_000_000

DEAD_TIME_US = 10.0

AFTERPULSE_WINDOW_US = 50.0

DARK_COUNT_RATE_HZ = 1_000

REPEATS = 50

BASE_SEED = 20262000

ALPHA = 0.05

OUTPUT_DIR = Path("results/channel")

RAW_FILE = (
    OUTPUT_DIR
    / "afterpulse_sensitivity_v2_raw.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "afterpulse_sensitivity_v2_summary.csv"
)

STATISTICS_FILE = (
    OUTPUT_DIR
    / "afterpulse_sensitivity_v2_statistics.csv"
)


# ============================================================
# METRICS
# ============================================================

SUMMARY_METRICS = [
    "signal_clicks",
    "dark_clicks",
    "afterpulse_clicks",
    "dead_time_lost_signal",
    "detected_events",
    "double_click_events",
    "sifted_key_length",
    "errors",
    "qber",
    "sifted_bits_per_second",
    "detection_rate_per_second",
    "afterpulse_clicks_per_second"
]

STATISTICAL_METRICS = [
    "qber",
    "sifted_bits_per_second",
    "afterpulse_clicks",
    "dead_time_lost_signal"
]


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_configuration():

    if CONFIG["pulses"] <= 0:

        raise ValueError(
            "CONFIG['pulses'] must be positive."
        )

    if REPEATS < 2:

        raise ValueError(
            "REPEATS must be at least 2."
        )

    if REPETITION_RATE_HZ <= 0:

        raise ValueError(
            "Repetition rate must be positive."
        )

    if AFTERPULSE_WINDOW_US <= 0:

        raise ValueError(
            "Afterpulse window must be positive."
        )

    if not 0 <= DEAD_TIME_US:

        raise ValueError(
            "Dead time cannot be negative."
        )

    if "afterpulse_probability" not in CONFIG:

        raise ValueError(
            "The detector CONFIG must contain "
            "'afterpulse_probability'."
        )

    for probability in AFTERPULSE_PROBABILITIES:

        if not 0 <= probability <= 1:

            raise ValueError(
                f"Invalid afterpulse probability: "
                f"{probability}"
            )


# ============================================================
# CONFIDENCE INTERVAL
# Two-sided Student t confidence interval
# ============================================================

def calculate_statistics(values):

    n = len(values)

    mean_value = statistics.mean(values)

    std_value = (
        statistics.stdev(values)
        if n > 1
        else 0.0
    )

    if n > 1:

        standard_error = (
            std_value / math.sqrt(n)
        )

        critical_value = stats.t.ppf(
            1 - ALPHA / 2,
            n - 1
        )

        margin = (
            critical_value * standard_error
        )

    else:

        margin = 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "ci_low": mean_value - margin,
        "ci_high": mean_value + margin
    }


# ============================================================
# HOLM-BONFERRONI CORRECTION
# ============================================================

def holm_adjust(p_values):

    count = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1]
    )

    adjusted = [1.0] * count

    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(
        indexed
    ):

        candidate = (
            (count - rank) * p_value
        )

        running_max = max(
            running_max,
            candidate
        )

        adjusted[original_index] = min(
            1.0,
            running_max
        )

    return adjusted


# ============================================================
# WELCH TEST AND 95% CI OF MEAN DIFFERENCE
#
# Difference = comparison - baseline
# ============================================================

def welch_analysis(baseline, comparison):

    n1 = len(baseline)
    n2 = len(comparison)

    if n1 < 2 or n2 < 2:

        raise ValueError(
            "Welch test requires at least two "
            "observations in each group."
        )

    mean1 = statistics.mean(baseline)
    mean2 = statistics.mean(comparison)

    variance1 = statistics.variance(baseline)
    variance2 = statistics.variance(comparison)

    difference = mean2 - mean1

    component1 = variance1 / n1
    component2 = variance2 / n2

    se_squared = component1 + component2

    standard_error = math.sqrt(
        se_squared
    )

    if standard_error == 0:

        p_value = (
            1.0 if difference == 0
            else 0.0
        )

        return {
            "welch_t": (
                0.0 if difference == 0
                else None
            ),
            "welch_df": None,
            "welch_p": p_value,
            "mean_difference": difference,
            "difference_ci_low": difference,
            "difference_ci_high": difference
        }

    denominator = (
        component1 ** 2 / (n1 - 1)
        +
        component2 ** 2 / (n2 - 1)
    )

    degrees_of_freedom = (
        se_squared ** 2 / denominator
    )

    t_statistic = (
        difference / standard_error
    )

    p_value = (
        2 * stats.t.sf(
            abs(t_statistic),
            degrees_of_freedom
        )
    )

    critical_value = stats.t.ppf(
        1 - ALPHA / 2,
        degrees_of_freedom
    )

    margin = (
        critical_value * standard_error
    )

    return {
        "welch_t": t_statistic,
        "welch_df": degrees_of_freedom,
        "welch_p": p_value,
        "mean_difference": difference,
        "difference_ci_low": difference - margin,
        "difference_ci_high": difference + margin
    }


# ============================================================
# MANN-WHITNEY U TEST
# ============================================================

def mann_whitney_analysis(baseline, comparison):

    result = stats.mannwhitneyu(
        baseline,
        comparison,
        alternative="two-sided",
        method="auto"
    )

    return {
        "mann_whitney_u": float(
            result.statistic
        ),
        "mann_whitney_p": float(
            result.pvalue
        )
    }


# ============================================================
# EFFECT SIZE
# HEDGES' G AND CLIFF'S DELTA
# ============================================================

def calculate_effect_sizes(baseline, comparison):

    n1 = len(baseline)
    n2 = len(comparison)

    mean1 = statistics.mean(baseline)
    mean2 = statistics.mean(comparison)

    variance1 = statistics.variance(baseline)
    variance2 = statistics.variance(comparison)

    difference = mean2 - mean1

    pooled_df = n1 + n2 - 2

    pooled_variance = (
        (
            (n1 - 1) * variance1
            +
            (n2 - 1) * variance2
        )
        / pooled_df
    )

    pooled_sd = math.sqrt(
        pooled_variance
    )

    if pooled_sd == 0:

        hedges_g = (
            0.0 if difference == 0
            else math.copysign(
                float("inf"),
                difference
            )
        )

    else:

        cohens_d = (
            difference / pooled_sd
        )

        correction = (
            1 - 3 / (4 * pooled_df - 1)
        )

        hedges_g = (
            correction * cohens_d
        )

    # Cliff's delta
    greater = 0
    less = 0
    ties = 0

    for comparison_value in comparison:

        for baseline_value in baseline:

            if comparison_value > baseline_value:

                greater += 1

            elif comparison_value < baseline_value:

                less += 1

            else:

                ties += 1

    total_pairs = n1 * n2

    cliffs_delta = (
        (greater - less) / total_pairs
    )

    return {
        "hedges_g": hedges_g,
        "cliffs_delta": cliffs_delta,
        "pairwise_ties": ties
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(file_path, rows):

    if not rows:

        raise ValueError(
            f"No rows available for {file_path}"
        )

    fieldnames = list(dict.fromkeys(
        key
        for row in rows
        for key in row.keys()
    ))

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    validate_configuration()

    raw_results = []
    summary_results = []

    pulses = CONFIG["pulses"]

    duration_seconds = (
        pulses / REPETITION_RATE_HZ
    )

    afterpulse_window_gates = math.ceil(
        AFTERPULSE_WINDOW_US
        * 1e-6
        * REPETITION_RATE_HZ
    )

    total_experiments = (
        len(DISTANCES_KM)
        * len(AFTERPULSE_PROBABILITIES)
        * REPEATS
    )

    experiment = 0

    print("\n================================================")
    print(" PRACTICAL 15")
    print(" 50-REPETITION AFTERPULSE VALIDATION")
    print("================================================")

    print(
        f"Pulses per experiment: {pulses:,}"
    )

    print(
        f"Repetitions per condition: {REPEATS}"
    )

    print(
        f"Total experiments: {total_experiments}"
    )

    print(
        f"PRR: {REPETITION_RATE_HZ / 1e6:.2f} MHz"
    )

    print(
        f"Dead time: {DEAD_TIME_US:.2f} us"
    )

    print(
        f"Afterpulse window: {AFTERPULSE_WINDOW_US:.2f} us"
    )

    print(
        f"Afterpulse window gates: "
        f"{afterpulse_window_gates}"
    )

    print(
        f"Dark count rate: {DARK_COUNT_RATE_HZ} cps"
    )

    print(
        f"Experiment duration: "
        f"{duration_seconds:.4f} seconds"
    )

    print(
        f"Base random seed: {BASE_SEED}"
    )

    print(
        "\nStarting independent-seed experiments..."
    )

    # --------------------------------------------------------
    # DISTANCE SWEEP
    # --------------------------------------------------------

    for distance in DISTANCES_KM:

        # ----------------------------------------------------
        # AFTERPULSE PROBABILITY SWEEP
        # ----------------------------------------------------

        for ap_probability in AFTERPULSE_PROBABILITIES:

            config = CONFIG.copy()

            config["pulse_repetition_rate_hz"] = (
                REPETITION_RATE_HZ
            )

            config["dead_time_us"] = (
                DEAD_TIME_US
            )

            config["afterpulse_window_gates"] = (
                afterpulse_window_gates
            )

            config["afterpulse_probability"] = (
                ap_probability
            )

            group = []

            print("\n--------------------------------------------")

            print(
                f"Distance: {distance} km"
            )

            print(
                f"Afterpulse probability: "
                f"{ap_probability * 100:.2f}%"
            )

            # ------------------------------------------------
            # REPEATED EXPERIMENTS
            # ------------------------------------------------

            for repeat in range(REPEATS):

                experiment += 1

                seed = (
                    BASE_SEED + experiment
                )

                result = simulate_detector(
                    distance_km=distance,
                    dark_count_rate_hz=(
                        DARK_COUNT_RATE_HZ
                    ),
                    config=config,
                    seed=seed
                )

                sifted_bps = (
                    result["sifted_key_length"]
                    / duration_seconds
                )

                detection_cps = (
                    result["detected_events"]
                    / duration_seconds
                )

                afterpulse_cps = (
                    result["afterpulse_clicks"]
                    / duration_seconds
                )

                row = {
                    "experiment": experiment,
                    "repeat": repeat + 1,
                    "seed": seed,

                    "fiber_length_km": distance,

                    "afterpulse_probability": (
                        ap_probability
                    ),

                    "afterpulse_probability_percent": (
                        ap_probability * 100
                    ),

                    "pulse_repetition_rate_hz": (
                        REPETITION_RATE_HZ
                    ),

                    "dead_time_us": DEAD_TIME_US,

                    "afterpulse_window_us": (
                        AFTERPULSE_WINDOW_US
                    ),

                    "afterpulse_window_gates": (
                        afterpulse_window_gates
                    ),

                    "dark_count_rate_hz": (
                        DARK_COUNT_RATE_HZ
                    ),

                    "pulses": pulses,

                    "duration_seconds": (
                        duration_seconds
                    ),

                    "signal_clicks": (
                        result["signal_clicks"]
                    ),

                    "dark_clicks": (
                        result["dark_clicks"]
                    ),

                    "afterpulse_clicks": (
                        result["afterpulse_clicks"]
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
                        sifted_bps
                    ),

                    "detection_rate_per_second": (
                        detection_cps
                    ),

                    "afterpulse_clicks_per_second": (
                        afterpulse_cps
                    )
                }

                raw_results.append(row)

                group.append(row)

            # ------------------------------------------------
            # GROUP SUMMARY
            # ------------------------------------------------

            summary = {
                "fiber_length_km": distance,

                "afterpulse_probability": (
                    ap_probability
                ),

                "afterpulse_probability_percent": (
                    ap_probability * 100
                ),

                "pulse_repetition_rate_hz": (
                    REPETITION_RATE_HZ
                ),

                "dead_time_us": DEAD_TIME_US,

                "afterpulse_window_us": (
                    AFTERPULSE_WINDOW_US
                ),

                "afterpulse_window_gates": (
                    afterpulse_window_gates
                ),

                "dark_count_rate_hz": (
                    DARK_COUNT_RATE_HZ
                ),

                "pulses_per_experiment": pulses,

                "experiment_duration_seconds": (
                    duration_seconds
                ),

                "repetitions": REPEATS,

                "first_seed": group[0]["seed"],

                "last_seed": group[-1]["seed"]
            }

            for metric in SUMMARY_METRICS:

                values = [
                    item[metric]
                    for item in group
                ]

                metric_stats = (
                    calculate_statistics(values)
                )

                for field in [
                    "mean",
                    "std",
                    "ci_low",
                    "ci_high"
                ]:

                    summary[
                        f"{metric}_{field}"
                    ] = metric_stats[field]

            summary_results.append(summary)

            print(
                f"Completed repetitions: {REPEATS}"
            )

            print(
                f"Mean afterpulse clicks: "
                f"{summary['afterpulse_clicks_mean']:.2f}"
            )

            print(
                f"Mean sifted throughput: "
                f"{summary['sifted_bits_per_second_mean']:.2f} bit/s"
            )

            print(
                f"Throughput 95% CI: "
                f"[{summary['sifted_bits_per_second_ci_low']:.2f}, "
                f"{summary['sifted_bits_per_second_ci_high']:.2f}]"
            )

            print(
                f"Mean QBER: "
                f"{summary['qber_mean'] * 100:.4f}%"
            )

            print(
                f"QBER 95% CI: "
                f"[{summary['qber_ci_low'] * 100:.4f}%, "
                f"{summary['qber_ci_high'] * 100:.4f}%]"
            )

    # ========================================================
    # STATISTICAL COMPARISONS
    # ========================================================

    print("\n================================================")
    print(" STATISTICAL COMPARISONS")
    print("================================================")

    statistical_results = []

    for distance in DISTANCES_KM:

        for metric in STATISTICAL_METRICS:

            baseline = [
                row[metric]
                for row in raw_results
                if (
                    row["fiber_length_km"] == distance
                    and row["afterpulse_probability"]
                    == 0.0
                )
            ]

            for probability in AFTERPULSE_PROBABILITIES:

                if probability == 0.0:
                    continue

                comparison = [
                    row[metric]
                    for row in raw_results
                    if (
                        row["fiber_length_km"] == distance
                        and math.isclose(
                            row["afterpulse_probability"],
                            probability,
                            abs_tol=1e-12
                        )
                    )
                ]

                welch = welch_analysis(
                    baseline,
                    comparison
                )

                mann_whitney = (
                    mann_whitney_analysis(
                        baseline,
                        comparison
                    )
                )

                effect = calculate_effect_sizes(
                    baseline,
                    comparison
                )

                row = {
                    "fiber_length_km": distance,

                    "metric": metric,

                    "baseline_probability": 0.0,

                    "comparison_probability": (
                        probability
                    ),

                    "baseline_n": len(baseline),

                    "comparison_n": len(comparison),

                    "baseline_mean": (
                        statistics.mean(baseline)
                    ),

                    "comparison_mean": (
                        statistics.mean(comparison)
                    ),

                    **welch,

                    **mann_whitney,

                    **effect
                }

                statistical_results.append(row)

    # ========================================================
    # MULTIPLE TESTING CORRECTION
    # ========================================================

    # Global correction across all 32 comparisons.

    global_welch = holm_adjust([
        row["welch_p"]
        for row in statistical_results
    ])

    global_mw = holm_adjust([
        row["mann_whitney_p"]
        for row in statistical_results
    ])

    for index, row in enumerate(statistical_results):

        row["welch_p_holm_global_32"] = (
            global_welch[index]
        )

        row["mann_whitney_p_holm_global_32"] = (
            global_mw[index]
        )

    # Metric-specific family correction:
    # 8 comparisons per metric.

    for metric in STATISTICAL_METRICS:

        indices = [
            index
            for index, row in enumerate(
                statistical_results
            )
            if row["metric"] == metric
        ]

        metric_welch = holm_adjust([
            statistical_results[index]["welch_p"]
            for index in indices
        ])

        metric_mw = holm_adjust([
            statistical_results[index]["mann_whitney_p"]
            for index in indices
        ])

        for position, index in enumerate(indices):

            statistical_results[index][
                "welch_p_holm_metric_8"
            ] = metric_welch[position]

            statistical_results[index][
                "mann_whitney_p_holm_metric_8"
            ] = metric_mw[position]

    # Significance flags

    for row in statistical_results:

        row["welch_significant_global_32"] = (
            row["welch_p_holm_global_32"] < ALPHA
        )

        row["mann_whitney_significant_global_32"] = (
            row["mann_whitney_p_holm_global_32"] < ALPHA
        )

        row["welch_significant_metric_8"] = (
            row["welch_p_holm_metric_8"] < ALPHA
        )

        row["mann_whitney_significant_metric_8"] = (
            row["mann_whitney_p_holm_metric_8"] < ALPHA
        )

    # ========================================================
    # SAVE ALL OUTPUTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        RAW_FILE,
        raw_results
    )

    save_csv(
        SUMMARY_FILE,
        summary_results
    )

    save_csv(
        STATISTICS_FILE,
        statistical_results
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n================================================")
    print(" FINAL 50-REPETITION SUMMARY")
    print("================================================")

    print(
        f"{'Distance':>9} "
        f"{'AP %':>7} "
        f"{'AP clicks':>12} "
        f"{'Sifted/s':>13} "
        f"{'QBER':>10}"
    )

    for item in summary_results:

        print(
            f"{item['fiber_length_km']:>8}km "
            f"{item['afterpulse_probability_percent']:>6.2f}% "
            f"{item['afterpulse_clicks_mean']:>12.2f} "
            f"{item['sifted_bits_per_second_mean']:>13.2f} "
            f"{item['qber_mean'] * 100:>9.4f}%"
        )

    # ========================================================
    # STATISTICAL SIGNIFICANCE SUMMARY
    # ========================================================

    print("\n================================================")
    print(" STATISTICAL SIGNIFICANCE SUMMARY")
    print("================================================")

    for metric in STATISTICAL_METRICS:

        subset = [
            row
            for row in statistical_results
            if row["metric"] == metric
        ]

        welch_count = sum(
            row["welch_significant_metric_8"]
            for row in subset
        )

        mw_count = sum(
            row["mann_whitney_significant_metric_8"]
            for row in subset
        )

        print(
            f"{metric}: "
            f"Welch Holm-8 = {welch_count}/8, "
            f"Mann-Whitney Holm-8 = {mw_count}/8"
        )

    print("\nSaved files:")

    print(RAW_FILE)

    print(SUMMARY_FILE)

    print(STATISTICS_FILE)

    print(
        f"\nTotal experiments completed: "
        f"{len(raw_results)}"
    )

    print(
        "\nPractical 15 completed successfully."
    )


if __name__ == "__main__":

    main()
