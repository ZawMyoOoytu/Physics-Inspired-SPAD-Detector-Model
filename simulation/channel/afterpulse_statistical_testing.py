
import csv
import math
import statistics
from pathlib import Path

from scipy import stats


# =========================================
# AFTERPULSE STATISTICAL SIGNIFICANCE TEST
# =========================================

INPUT_FILE = Path(
    "results/channel/afterpulse_sensitivity_raw.csv"
)

OUTPUT_DIR = Path("results/channel")

BASELINE_PROBABILITY = 0.0

COMPARISON_PROBABILITIES = [
    0.005,
    0.010,
    0.020,
    0.050
]

DISTANCES_KM = [10, 50]

METRICS = [
    "qber",
    "sifted_bits_per_second",
    "afterpulse_clicks",
    "dead_time_lost_signal"
]

ALPHA = 0.05


# =========================================
# LOAD DATA
# =========================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        data = []

        for row in reader:

            converted = {}

            for key, value in row.items():

                if key in [
                    "experiment",
                    "repeat",
                    "seed",
                    "fiber_length_km"
                ]:

                    converted[key] = int(value)

                else:

                    converted[key] = float(value)

            data.append(converted)

    return data


# =========================================
# HOLM-BONFERRONI CORRECTION
# =========================================

def holm_adjust(p_values):

    n = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1]
    )

    adjusted = [1.0] * n

    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(
        indexed
    ):

        adjusted_value = (
            (n - rank) * p_value
        )

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted[original_index] = min(
            1.0,
            running_max
        )

    return adjusted


# =========================================
# WELCH T-TEST AND CONFIDENCE INTERVAL
# =========================================

def welch_analysis(baseline, comparison):

    n1 = len(baseline)
    n2 = len(comparison)

    mean1 = statistics.mean(baseline)
    mean2 = statistics.mean(comparison)

    variance1 = statistics.variance(baseline)
    variance2 = statistics.variance(comparison)

    # Difference = comparison - baseline
    mean_difference = mean2 - mean1

    se_squared = (
        variance1 / n1
        + variance2 / n2
    )

    standard_error = math.sqrt(
        se_squared
    )

    if standard_error == 0:

        return {
            "baseline_mean": mean1,
            "comparison_mean": mean2,
            "mean_difference": mean_difference,
            "standard_error": 0.0,
            "degrees_of_freedom": None,
            "t_statistic": None,
            "p_value": 1.0 if mean_difference == 0 else 0.0,
            "ci_low": mean_difference,
            "ci_high": mean_difference
        }

    # Welch-Satterthwaite degrees of freedom
    numerator = se_squared ** 2

    denominator = (
        (variance1 / n1) ** 2 / (n1 - 1)
        +
        (variance2 / n2) ** 2 / (n2 - 1)
    )

    degrees_of_freedom = (
        numerator / denominator
    )

    t_statistic = (
        mean_difference / standard_error
    )

    p_value = 2 * stats.t.sf(
        abs(t_statistic),
        degrees_of_freedom
    )

    critical_value = stats.t.ppf(
        1 - ALPHA / 2,
        degrees_of_freedom
    )

    margin = (
        critical_value * standard_error
    )

    return {
        "baseline_mean": mean1,
        "comparison_mean": mean2,
        "mean_difference": mean_difference,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "ci_low": mean_difference - margin,
        "ci_high": mean_difference + margin
    }


# =========================================
# MAIN ANALYSIS
# =========================================

def main():

    data = load_data()

    print("\n======================================")
    print(" AFTERPULSE STATISTICAL TESTING")
    print("======================================")

    print(f"Input rows: {len(data)}")
    print(f"Baseline probability: {BASELINE_PROBABILITY:.3f}")
    print(f"Alpha: {ALPHA}")
    print("Test: Welch independent two-sample t-test")
    print("Correction: Holm-Bonferroni")

    results = []

    for distance in DISTANCES_KM:

        print("\n======================================")
        print(f"DISTANCE: {distance} km")
        print("======================================")

        for metric in METRICS:

            baseline = [
                row[metric]
                for row in data
                if (
                    row["fiber_length_km"] == distance
                    and
                    row["afterpulse_probability"]
                    == BASELINE_PROBABILITY
                )
            ]

            if len(baseline) < 2:

                raise ValueError(
                    f"Insufficient baseline observations "
                    f"for {distance} km, {metric}"
                )

            metric_results = []

            for probability in COMPARISON_PROBABILITIES:

                comparison = [
                    row[metric]
                    for row in data
                    if (
                        row["fiber_length_km"] == distance
                        and
                        math.isclose(
                            row["afterpulse_probability"],
                            probability,
                            abs_tol=1e-12
                        )
                    )
                ]

                if len(comparison) < 2:

                    raise ValueError(
                        f"Insufficient comparison observations "
                        f"for {distance} km, "
                        f"{probability}, {metric}"
                    )

                result = welch_analysis(
                    baseline,
                    comparison
                )

                result.update({
                    "fiber_length_km": distance,
                    "metric": metric,
                    "baseline_probability": (
                        BASELINE_PROBABILITY
                    ),
                    "comparison_probability": (
                        probability
                    ),
                    "baseline_n": len(baseline),
                    "comparison_n": len(comparison)
                })

                metric_results.append(result)

            adjusted_p_values = holm_adjust([
                item["p_value"]
                for item in metric_results
            ])

            for item, adjusted_p in zip(
                metric_results,
                adjusted_p_values
            ):

                item["holm_adjusted_p_value"] = (
                    adjusted_p
                )

                item["significant_after_holm"] = (
                    adjusted_p < ALPHA
                )

                results.append(item)

                print(
                    f"\nMetric: {metric}"
                )

                print(
                    f"Afterpulse: "
                    f"{item['comparison_probability'] * 100:.2f}%"
                )

                print(
                    f"Baseline mean: "
                    f"{item['baseline_mean']:.8f}"
                )

                print(
                    f"Comparison mean: "
                    f"{item['comparison_mean']:.8f}"
                )

                print(
                    f"Mean difference: "
                    f"{item['mean_difference']:.8f}"
                )

                print(
                    f"95% CI: "
                    f"[{item['ci_low']:.8f}, "
                    f"{item['ci_high']:.8f}]"
                )

                print(
                    f"Welch p-value: "
                    f"{item['p_value']:.8g}"
                )

                print(
                    f"Holm adjusted p-value: "
                    f"{item['holm_adjusted_p_value']:.8g}"
                )

                print(
                    "Statistically significant: "
                    f"{item['significant_after_holm']}"
                )

    # =====================================
    # SAVE CSV
    # =====================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "afterpulse_statistical_tests.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # =====================================
    # FINAL SUMMARY
    # =====================================

    significant_count = sum(
        item["significant_after_holm"]
        for item in results
    )

    print("\n======================================")
    print(" FINAL STATISTICAL SUMMARY")
    print("======================================")

    print(f"Total tests: {len(results)}")
    print(
        f"Significant after Holm correction: "
        f"{significant_count}"
    )

    print(f"Output: {output_file}")

    print(
        "\nStatistical testing completed."
    )


if __name__ == "__main__":
    main()
