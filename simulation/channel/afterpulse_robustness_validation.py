
import csv
import math
import statistics
from pathlib import Path

from scipy import stats


# ============================================================
# PRACTICAL 14
# ROBUSTNESS AND GLOBAL MULTIPLE-TESTING VALIDATION
# ============================================================

INPUT_FILE = Path(
    "results/channel/afterpulse_sensitivity_raw.csv"
)

OUTPUT_FILE = Path(
    "results/channel/afterpulse_robustness_validation.csv"
)

DISTANCES_KM = [10, 50]

BASELINE_PROBABILITY = 0.0

COMPARISON_PROBABILITIES = [
    0.005,
    0.010,
    0.020,
    0.050
]

METRICS = [
    "qber",
    "sifted_bits_per_second",
    "afterpulse_clicks",
    "dead_time_lost_signal"
]

ALPHA = 0.05


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

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

        required_columns = {
            "fiber_length_km",
            "afterpulse_probability",
            *METRICS
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:

            raise ValueError(
                "Missing CSV columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        data = []

        for row in reader:

            item = {
                "fiber_length_km": int(
                    row["fiber_length_km"]
                ),

                "afterpulse_probability": float(
                    row["afterpulse_probability"]
                )
            }

            for metric in METRICS:

                item[metric] = float(
                    row[metric]
                )

            data.append(item)

    if not data:

        raise ValueError(
            "Input CSV contains no observations."
        )

    return data


# ============================================================
# VALIDATE EXPERIMENT GROUPS
# ============================================================

def validate_groups(data):

    expected_probabilities = [
        BASELINE_PROBABILITY
    ] + COMPARISON_PROBABILITIES

    for distance in DISTANCES_KM:

        for probability in expected_probabilities:

            matching_rows = [
                row
                for row in data
                if (
                    row["fiber_length_km"] == distance
                    and math.isclose(
                        row["afterpulse_probability"],
                        probability,
                        abs_tol=1e-12
                    )
                )
            ]

            if len(matching_rows) < 2:

                raise ValueError(
                    f"Insufficient observations: "
                    f"distance={distance}, "
                    f"afterpulse={probability}, "
                    f"n={len(matching_rows)}"
                )

            print(
                f"Validated: {distance} km, "
                f"{probability * 100:.2f}% AP, "
                f"n={len(matching_rows)}"
            )


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
# WELCH INDEPENDENT TWO-SAMPLE T-TEST
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

    mean_difference = mean2 - mean1

    component1 = variance1 / n1
    component2 = variance2 / n2

    standard_error_squared = (
        component1 + component2
    )

    standard_error = math.sqrt(
        standard_error_squared
    )

    if standard_error == 0:

        p_value = (
            1.0 if mean_difference == 0
            else 0.0
        )

        return {
            "welch_t": (
                0.0 if mean_difference == 0
                else None
            ),
            "welch_df": None,
            "welch_p": p_value,
            "mean_difference": mean_difference,
            "difference_ci_low": mean_difference,
            "difference_ci_high": mean_difference
        }

    denominator = (
        component1 ** 2 / (n1 - 1)
        +
        component2 ** 2 / (n2 - 1)
    )

    degrees_of_freedom = (
        standard_error_squared ** 2
        / denominator
    )

    t_statistic = (
        mean_difference / standard_error
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
        "mean_difference": mean_difference,
        "difference_ci_low": (
            mean_difference - margin
        ),
        "difference_ci_high": (
            mean_difference + margin
        )
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

def effect_size_analysis(baseline, comparison):

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

        if difference == 0:
            cohens_d = 0.0
            hedges_g = 0.0

        else:
            cohens_d = math.copysign(
                float("inf"),
                difference
            )

            hedges_g = cohens_d

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
        "cohens_d": cohens_d,
        "hedges_g": hedges_g,
        "cliffs_delta": cliffs_delta,
        "pairwise_ties": ties
    }


# ============================================================
# EXTRACT EXPERIMENT GROUP
# ============================================================

def get_group(
    data,
    distance,
    probability,
    metric
):

    return [
        row[metric]
        for row in data
        if (
            row["fiber_length_km"] == distance
            and math.isclose(
                row["afterpulse_probability"],
                probability,
                abs_tol=1e-12
            )
        )
    ]


# ============================================================
# RUN ALL STATISTICAL COMPARISONS
# ============================================================

def run_comparisons(data):

    results = []

    for distance in DISTANCES_KM:

        for metric in METRICS:

            baseline = get_group(
                data,
                distance,
                BASELINE_PROBABILITY,
                metric
            )

            for probability in COMPARISON_PROBABILITIES:

                comparison = get_group(
                    data,
                    distance,
                    probability,
                    metric
                )

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

                effect = effect_size_analysis(
                    baseline,
                    comparison
                )

                result = {
                    "fiber_length_km": distance,
                    "metric": metric,

                    "baseline_probability": (
                        BASELINE_PROBABILITY
                    ),

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

                results.append(result)

    return results


# ============================================================
# APPLY MULTIPLE-TESTING CORRECTIONS
# ============================================================

def apply_corrections(results):

    # Global family: all 32 comparisons
    global_welch = holm_adjust([
        row["welch_p"]
        for row in results
    ])

    global_mw = holm_adjust([
        row["mann_whitney_p"]
        for row in results
    ])

    for index, row in enumerate(results):

        row["welch_p_holm_global_32"] = (
            global_welch[index]
        )

        row["mann_whitney_p_holm_global_32"] = (
            global_mw[index]
        )

    # Primary endpoint: QBER, 8 comparisons
    qber_indices = [
        index
        for index, row in enumerate(results)
        if row["metric"] == "qber"
    ]

    qber_welch = holm_adjust([
        results[index]["welch_p"]
        for index in qber_indices
    ])

    qber_mw = holm_adjust([
        results[index]["mann_whitney_p"]
        for index in qber_indices
    ])

    for position, index in enumerate(qber_indices):

        results[index]["qber_welch_p_holm_8"] = (
            qber_welch[position]
        )

        results[index]["qber_mw_p_holm_8"] = (
            qber_mw[position]
        )

    # Secondary endpoints: 24 comparisons
    secondary_indices = [
        index
        for index, row in enumerate(results)
        if row["metric"] != "qber"
    ]

    secondary_welch = holm_adjust([
        results[index]["welch_p"]
        for index in secondary_indices
    ])

    secondary_mw = holm_adjust([
        results[index]["mann_whitney_p"]
        for index in secondary_indices
    ])

    for position, index in enumerate(secondary_indices):

        results[index]["secondary_welch_p_holm_24"] = (
            secondary_welch[position]
        )

        results[index]["secondary_mw_p_holm_24"] = (
            secondary_mw[position]
        )

    # Significance indicators
    for row in results:

        row["qber_welch_significant_global_8"] = (
            row.get("qber_welch_p_holm_8", 1.0)
            < ALPHA
        )

        row["qber_mw_significant_global_8"] = (
            row.get("qber_mw_p_holm_8", 1.0)
            < ALPHA
        )

        row["welch_significant_global_32"] = (
            row["welch_p_holm_global_32"]
            < ALPHA
        )

        row["mw_significant_global_32"] = (
            row["mann_whitney_p_holm_global_32"]
            < ALPHA
        )

    return results


# ============================================================
# SAVE RESULTS
# FIXED: UNION OF ALL CSV FIELD NAMES
# ============================================================

def save_results(results):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Collect keys from every row.
    # This prevents missing-field errors when QBER and
    # secondary metrics contain different result columns.

    fieldnames = list(dict.fromkeys(
        key
        for row in results
        for key in row.keys()
    ))

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nCSV successfully saved: {OUTPUT_FILE}"
    )

    print(
        f"CSV columns: {len(fieldnames)}"
    )


# ============================================================
# PRINT PRIMARY QBER RESULTS
# ============================================================

def print_qber_results(results):

    print("\n================================================")
    print(" PRIMARY ENDPOINT: QBER")
    print(" Global Holm family: 8 comparisons")
    print("================================================")

    for row in results:

        if row["metric"] != "qber":
            continue

        print(
            f"\nDistance: "
            f"{row['fiber_length_km']} km"
        )

        print(
            f"Afterpulse: "
            f"{row['comparison_probability'] * 100:.2f}%"
        )

        print(
            f"Baseline mean: "
            f"{row['baseline_mean'] * 100:.5f}%"
        )

        print(
            f"Comparison mean: "
            f"{row['comparison_mean'] * 100:.5f}%"
        )

        print(
            f"Mean difference: "
            f"{row['mean_difference'] * 100:.5f} "
            f"percentage points"
        )

        print(
            f"95% CI: "
            f"[{row['difference_ci_low'] * 100:.5f}, "
            f"{row['difference_ci_high'] * 100:.5f}]"
        )

        print(
            f"Welch p-value: "
            f"{row['welch_p']:.8g}"
        )

        print(
            f"Welch global-32 Holm: "
            f"{row['welch_p_holm_global_32']:.8g}"
        )

        print(
            f"Welch QBER-8 Holm: "
            f"{row['qber_welch_p_holm_8']:.8g}"
        )

        print(
            f"Mann-Whitney p-value: "
            f"{row['mann_whitney_p']:.8g}"
        )

        print(
            f"Mann-Whitney QBER-8 Holm: "
            f"{row['qber_mw_p_holm_8']:.8g}"
        )

        print(
            f"Hedges' g: "
            f"{row['hedges_g']:.6f}"
        )

        print(
            f"Cliff's delta: "
            f"{row['cliffs_delta']:.6f}"
        )

        print(
            f"Welch significant after QBER Holm: "
            f"{row['qber_welch_significant_global_8']}"
        )

        print(
            f"Mann-Whitney significant after QBER Holm: "
            f"{row['qber_mw_significant_global_8']}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(results):

    qber_results = [
        row
        for row in results
        if row["metric"] == "qber"
    ]

    qber_welch_count = sum(
        row["qber_welch_significant_global_8"]
        for row in qber_results
    )

    qber_mw_count = sum(
        row["qber_mw_significant_global_8"]
        for row in qber_results
    )

    global_welch_count = sum(
        row["welch_significant_global_32"]
        for row in results
    )

    global_mw_count = sum(
        row["mw_significant_global_32"]
        for row in results
    )

    print("\n================================================")
    print(" FINAL ROBUSTNESS SUMMARY")
    print("================================================")

    print(
        f"Total comparisons: {len(results)}"
    )

    print(
        f"QBER significant — Welch, Holm 8: "
        f"{qber_welch_count}/8"
    )

    print(
        f"QBER significant — Mann-Whitney, Holm 8: "
        f"{qber_mw_count}/8"
    )

    print(
        f"All metrics significant — Welch, Holm 32: "
        f"{global_welch_count}/32"
    )

    print(
        f"All metrics significant — Mann-Whitney, Holm 32: "
        f"{global_mw_count}/32"
    )

    print(
        f"\nOutput file: {OUTPUT_FILE}"
    )

    print(
        "\nPractical 14 completed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n================================================")
    print(" PRACTICAL 14")
    print(" ROBUSTNESS VALIDATION")
    print("================================================")

    print(
        "Primary endpoint: QBER"
    )

    print(
        "Statistical tests: Welch + Mann-Whitney U"
    )

    print(
        "Effect sizes: Hedges' g + Cliff's delta"
    )

    print(
        "Multiple testing: Global Holm-Bonferroni"
    )

    data = load_data()

    print(
        f"\nInput observations: {len(data)}"
    )

    validate_groups(data)

    results = run_comparisons(data)

    results = apply_corrections(results)

    save_results(results)

    print_qber_results(results)

    print_summary(results)


if __name__ == "__main__":

    main()
