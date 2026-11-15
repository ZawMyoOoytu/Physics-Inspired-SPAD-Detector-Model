
import argparse
import csv
import math
from pathlib import Path

from scipy import stats

from physical_spad_detector import (
    CONFIG as BASE_CONFIG,
    simulate_physical_spad,
)


# ============================================================
# PRACTICAL 17
# TRAP LIFETIME SENSITIVITY ANALYSIS
# ============================================================

DISTANCES_KM = [10, 50]

TRAP_LIFETIMES_US = [
    1.0,
    5.0,
    10.0,
    50.0,
    100.0,
]

AFTERPULSE_PROBABILITIES = [
    0.000,
    0.005,
    0.010,
    0.020,
    0.050,
]

DEFAULT_REPEATS = 20
DEFAULT_PULSES = 200_000

BASE_SEED = 20261700

CONFIDENCE_LEVEL = 0.95

METRICS = [
    "qber",
    "sifted_bits_per_second",
    "afterpulse_registered_attempts",
    "afterpulse_blocked_by_dead_time",
]


# ============================================================
# MULTIPLE TESTING CORRECTION
# ============================================================

def holm_adjust(p_values):

    p_values = [
        float(value)
        for value in p_values
    ]

    count = len(p_values)

    if count == 0:
        return []

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: (
            math.inf if math.isnan(item[1])
            else item[1]
        )
    )

    adjusted = [float("nan")] * count

    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(
        indexed
    ):

        if math.isnan(p_value):
            continue

        multiplier = count - rank

        corrected = min(
            1.0,
            multiplier * p_value
        )

        running_max = max(
            running_max,
            corrected
        )

        adjusted[original_index] = running_max

    return adjusted


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def mean_confidence_interval(values):

    count = len(values)

    mean_value = sum(values) / count

    if count < 2:

        return {
            "mean": mean_value,
            "sd": 0.0,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    standard_deviation = stats.tstd(values)

    standard_error = (
        standard_deviation / math.sqrt(count)
    )

    critical_value = stats.t.ppf(
        0.975,
        df=count - 1
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
# WELCH CONFIDENCE INTERVAL FOR DIFFERENCE IN MEANS
# ============================================================

def welch_difference_ci(
    baseline,
    comparison
):

    n1 = len(baseline)
    n2 = len(comparison)

    mean1 = sum(baseline) / n1
    mean2 = sum(comparison) / n2

    variance1 = stats.tvar(baseline)
    variance2 = stats.tvar(comparison)

    difference = mean2 - mean1

    component1 = variance1 / n1
    component2 = variance2 / n2

    standard_error = math.sqrt(
        component1 + component2
    )

    denominator = (
        (component1 ** 2) / (n1 - 1)
        +
        (component2 ** 2) / (n2 - 1)
    )

    if denominator == 0:

        return (
            difference,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    degrees_freedom = (
        (component1 + component2) ** 2
        / denominator
    )

    critical = stats.t.ppf(
        0.975,
        degrees_freedom
    )

    margin = (
        critical * standard_error
    )

    return (
        difference,
        degrees_freedom,
        difference - margin,
        difference + margin,
    )


# ============================================================
# EFFECT SIZE: HEDGES' G
# ============================================================

def hedges_g(
    baseline,
    comparison
):

    n1 = len(baseline)
    n2 = len(comparison)

    variance1 = stats.tvar(baseline)
    variance2 = stats.tvar(comparison)

    degrees = n1 + n2 - 2

    pooled_variance = (
        ((n1 - 1) * variance1)
        +
        ((n2 - 1) * variance2)
    ) / degrees

    if pooled_variance <= 0:
        return float("nan")

    pooled_sd = math.sqrt(
        pooled_variance
    )

    mean1 = sum(baseline) / n1
    mean2 = sum(comparison) / n2

    cohen_d = (
        mean2 - mean1
    ) / pooled_sd

    correction = (
        1 - 3 / (4 * degrees - 1)
    )

    return correction * cohen_d


# ============================================================
# EFFECT SIZE: CLIFF'S DELTA
# ============================================================

def cliffs_delta(
    baseline,
    comparison
):

    greater = 0
    lower = 0

    for comparison_value in comparison:

        for baseline_value in baseline:

            if comparison_value > baseline_value:

                greater += 1

            elif comparison_value < baseline_value:

                lower += 1

    total = (
        len(baseline)
        * len(comparison)
    )

    if total == 0:
        return float("nan")

    return (
        greater - lower
    ) / total


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

def make_experiment_config(
    distance_km,
    lifetime_us,
    afterpulse_probability,
    pulses
):

    config = dict(BASE_CONFIG)

    config["pulses"] = pulses

    config["trap_lifetimes_us"] = [
        lifetime_us
    ]

    config["trap_population_weights"] = [
        1.0
    ]

    config[
        "afterpulse_probability_per_avalanche"
    ] = afterpulse_probability

    return config


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_experiments(
    repeats,
    pulses
):

    raw_results = []

    total_conditions = (
        len(DISTANCES_KM)
        *
        len(TRAP_LIFETIMES_US)
        *
        len(AFTERPULSE_PROBABILITIES)
    )

    condition_number = 0

    total_runs = (
        total_conditions * repeats
    )

    run_number = 0

    print("\n==============================================")
    print(" PRACTICAL 17")
    print(" TRAP LIFETIME SENSITIVITY ANALYSIS")
    print("==============================================")

    print(
        f"Conditions: {total_conditions}"
    )

    print(
        f"Repeats per condition: {repeats}"
    )

    print(
        f"Total runs: {total_runs:,}"
    )

    print(
        f"Pulses per run: {pulses:,}"
    )

    print(
        f"Total simulated pulses: "
        f"{total_runs * pulses:,}"
    )

    for distance_index, distance in enumerate(
        DISTANCES_KM
    ):

        for lifetime_index, lifetime in enumerate(
            TRAP_LIFETIMES_US
        ):

            for probability_index, probability in enumerate(
                AFTERPULSE_PROBABILITIES
            ):

                condition_number += 1

                config = make_experiment_config(
                    distance_km=distance,
                    lifetime_us=lifetime,
                    afterpulse_probability=probability,
                    pulses=pulses,
                )

                print("\n----------------------------------------------")

                print(
                    f"Condition {condition_number}/"
                    f"{total_conditions}"
                )

                print(
                    f"Distance: {distance} km"
                )

                print(
                    f"Trap lifetime: {lifetime} us"
                )

                print(
                    f"Afterpulse probability: "
                    f"{probability * 100:.2f}%"
                )

                for repeat_index in range(repeats):

                    run_number += 1

                    seed = (
                        BASE_SEED
                        + distance_index * 1_000_000
                        + lifetime_index * 100_000
                        + probability_index * 1_000
                        + repeat_index
                    )

                    result = simulate_physical_spad(
                        distance_km=distance,
                        dark_count_rate_hz=1_000,
                        config=config,
                        seed=seed,
                    )

                    row = {
                        "distance_km": distance,
                        "trap_lifetime_us": lifetime,
                        "afterpulse_probability": probability,
                        "repeat": repeat_index + 1,
                        "seed": seed,
                        "pulses": pulses,

                        "qber": result["qber"],

                        "sifted_key_length": (
                            result["sifted_key_length"]
                        ),

                        "sifted_bits_per_second": (
                            result["sifted_bits_per_second"]
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

                        "dead_time_lost_signal": (
                            result[
                                "dead_time_lost_signal"
                            ]
                        ),

                        "detected_events": (
                            result["detected_events"]
                        ),
                    }

                    raw_results.append(row)

                    if (
                        repeat_index == 0
                        or
                        (repeat_index + 1) % 5 == 0
                        or
                        repeat_index + 1 == repeats
                    ):

                        print(
                            f"Run {run_number}/{total_runs} | "
                            f"Repeat {repeat_index + 1}/{repeats} | "
                            f"QBER: {result['qber'] * 100:.4f}% | "
                            f"AP registered: "
                            f"{result['afterpulse_registered_attempts']}"
                        )

    return raw_results


# ============================================================
# GROUP RESULTS
# ============================================================

def group_results(raw_results):

    groups = {}

    for row in raw_results:

        key = (
            row["distance_km"],
            row["trap_lifetime_us"],
            row["afterpulse_probability"],
        )

        groups.setdefault(
            key,
            []
        ).append(row)

    return groups


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary(raw_results):

    groups = group_results(raw_results)

    summary = []

    for key in sorted(groups):

        distance, lifetime, probability = key

        rows = groups[key]

        output = {
            "distance_km": distance,
            "trap_lifetime_us": lifetime,
            "afterpulse_probability": probability,
            "repeats": len(rows),
        }

        for metric in METRICS:

            values = [
                float(row[metric])
                for row in rows
            ]

            result = mean_confidence_interval(
                values
            )

            output[f"{metric}_mean"] = (
                result["mean"]
            )

            output[f"{metric}_sd"] = (
                result["sd"]
            )

            output[f"{metric}_ci_lower"] = (
                result["ci_lower"]
            )

            output[f"{metric}_ci_upper"] = (
                result["ci_upper"]
            )

        summary.append(output)

    return summary


# ============================================================
# STATISTICAL TESTING
# ============================================================

def calculate_statistics(raw_results):

    groups = group_results(raw_results)

    records = []

    for distance in DISTANCES_KM:

        for lifetime in TRAP_LIFETIMES_US:

            baseline_key = (
                distance,
                lifetime,
                0.0,
            )

            baseline_rows = groups[
                baseline_key
            ]

            for probability in AFTERPULSE_PROBABILITIES:

                if probability == 0:
                    continue

                comparison_key = (
                    distance,
                    lifetime,
                    probability,
                )

                comparison_rows = groups[
                    comparison_key
                ]

                for metric in METRICS:

                    baseline = [
                        float(row[metric])
                        for row in baseline_rows
                    ]

                    comparison = [
                        float(row[metric])
                        for row in comparison_rows
                    ]

                    welch_result = stats.ttest_ind(
                        comparison,
                        baseline,
                        equal_var=False,
                        alternative="two-sided",
                    )

                    mw_result = stats.mannwhitneyu(
                        comparison,
                        baseline,
                        alternative="two-sided",
                    )

                    (
                        difference,
                        welch_df,
                        ci_lower,
                        ci_upper,
                    ) = welch_difference_ci(
                        baseline,
                        comparison,
                    )

                    record = {
                        "distance_km": distance,

                        "trap_lifetime_us": lifetime,

                        "afterpulse_probability": (
                            probability
                        ),

                        "metric": metric,

                        "baseline_mean": (
                            sum(baseline) / len(baseline)
                        ),

                        "comparison_mean": (
                            sum(comparison) / len(comparison)
                        ),

                        "difference_comparison_minus_baseline": (
                            difference
                        ),

                        "welch_df": welch_df,

                        "welch_ci_lower": ci_lower,

                        "welch_ci_upper": ci_upper,

                        "welch_p": float(
                            welch_result.pvalue
                        ),

                        "mann_whitney_p": float(
                            mw_result.pvalue
                        ),

                        "hedges_g": hedges_g(
                            baseline,
                            comparison,
                        ),

                        "cliffs_delta": cliffs_delta(
                            baseline,
                            comparison,
                        ),
                    }

                    records.append(record)

    # --------------------------------------------------------
    # Holm correction within each metric family.
    # Each metric family contains 40 comparisons:
    # 2 distances x 5 lifetimes x 4 nonzero AP probabilities.
    # --------------------------------------------------------

    for metric in METRICS:

        indices = [
            index
            for index, row in enumerate(records)
            if row["metric"] == metric
        ]

        welch_values = [
            records[index]["welch_p"]
            for index in indices
        ]

        mw_values = [
            records[index]["mann_whitney_p"]
            for index in indices
        ]

        welch_adjusted = holm_adjust(
            welch_values
        )

        mw_adjusted = holm_adjust(
            mw_values
        )

        for position, index in enumerate(indices):

            records[index]["welch_holm_metric"] = (
                welch_adjusted[position]
            )

            records[index]["mw_holm_metric"] = (
                mw_adjusted[position]
            )

    # --------------------------------------------------------
    # QBER-specific family correction.
    # QBER has 40 comparisons.
    # --------------------------------------------------------

    qber_indices = [
        index
        for index, row in enumerate(records)
        if row["metric"] == "qber"
    ]

    qber_welch = holm_adjust([
        records[index]["welch_p"]
        for index in qber_indices
    ])

    qber_mw = holm_adjust([
        records[index]["mann_whitney_p"]
        for index in qber_indices
    ])

    for position, index in enumerate(qber_indices):

        records[index]["welch_holm_qber40"] = (
            qber_welch[position]
        )

        records[index]["mw_holm_qber40"] = (
            qber_mw[position]
        )

    # --------------------------------------------------------
    # Global correction across all metrics.
    # 4 metrics x 40 comparisons = 160 tests.
    # --------------------------------------------------------

    global_welch = holm_adjust([
        row["welch_p"]
        for row in records
    ])

    global_mw = holm_adjust([
        row["mann_whitney_p"]
        for row in records
    ])

    for index, row in enumerate(records):

        row["welch_holm_global160"] = (
            global_welch[index]
        )

        row["mw_holm_global160"] = (
            global_mw[index]
        )

    return records


# ============================================================
# CSV EXPORT
# ============================================================

def write_csv(
    output_path,
    rows
):

    if not rows:
        raise ValueError(
            f"No data available for {output_path}"
        )

    fieldnames = list(rows[0].keys())

    with output_path.open(
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
# SUMMARY REPORT
# ============================================================

def print_summary(summary):

    print("\n==============================================")
    print(" PRACTICAL 17 SUMMARY")
    print("==============================================")

    print(
        f"{'Dist':>5} "
        f"{'Tau':>7} "
        f"{'AP':>7} "
        f"{'AP reg.':>9} "
        f"{'QBER':>10} "
        f"{'Sifted/s':>12}"
    )

    for row in summary:

        print(
            f"{row['distance_km']:>4}km "
            f"{row['trap_lifetime_us']:>6.1f} "
            f"{row['afterpulse_probability'] * 100:>6.2f}% "
            f"{row['afterpulse_registered_attempts_mean']:>9.2f} "
            f"{row['qber_mean'] * 100:>9.4f}% "
            f"{row['sifted_bits_per_second_mean']:>12.2f}"
        )


def print_statistical_summary(records):

    print("\n==============================================")
    print(" STATISTICAL TEST SUMMARY")
    print("==============================================")

    for metric in METRICS:

        subset = [
            row
            for row in records
            if row["metric"] == metric
        ]

        welch_significant = sum(
            row["welch_holm_metric"] < 0.05
            for row in subset
        )

        mw_significant = sum(
            row["mw_holm_metric"] < 0.05
            for row in subset
        )

        print(f"\nMetric: {metric}")

        print(
            f"Welch significant after Holm: "
            f"{welch_significant}/{len(subset)}"
        )

        print(
            f"Mann-Whitney significant after Holm: "
            f"{mw_significant}/{len(subset)}"
        )

    qber_rows = [
        row
        for row in records
        if row["metric"] == "qber"
    ]

    print("\nQBER global correction:")

    print(
        "Welch Holm-160 significant: "
        f"{sum(row['welch_holm_global160'] < 0.05 for row in qber_rows)}"
        f"/{len(qber_rows)}"
    )

    print(
        "Mann-Whitney Holm-160 significant: "
        f"{sum(row['mw_holm_global160'] < 0.05 for row in qber_rows)}"
        f"/{len(qber_rows)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Physical SPAD trap lifetime sensitivity analysis"
        )
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
    )

    parser.add_argument(
        "--pulses",
        type=int,
        default=DEFAULT_PULSES,
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
    )

    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError(
            "At least 2 repeats are required."
        )

    if args.pulses < 1:
        raise ValueError(
            "Pulses must be positive."
        )

    raw_results = run_experiments(
        repeats=args.repeats,
        pulses=args.pulses,
    )

    summary = calculate_summary(
        raw_results
    )

    statistics_results = calculate_statistics(
        raw_results
    )

    output_dir = Path(
        "results/channel"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        f"_{args.suffix}"
        if args.suffix
        else ""
    )

    raw_file = (
        output_dir
        / f"physical_spad_sensitivity_raw{suffix}.csv"
    )

    summary_file = (
        output_dir
        / f"physical_spad_sensitivity_summary{suffix}.csv"
    )

    statistics_file = (
        output_dir
        / f"physical_spad_sensitivity_statistics{suffix}.csv"
    )

    write_csv(
        raw_file,
        raw_results
    )

    write_csv(
        summary_file,
        summary
    )

    write_csv(
        statistics_file,
        statistics_results
    )

    print_summary(summary)

    print_statistical_summary(
        statistics_results
    )

    print("\n==============================================")
    print(" FILES SAVED")
    print("==============================================")

    print(raw_file)
    print(summary_file)
    print(statistics_file)

    print("\nPractical 17 completed.")


if __name__ == "__main__":

    main()
