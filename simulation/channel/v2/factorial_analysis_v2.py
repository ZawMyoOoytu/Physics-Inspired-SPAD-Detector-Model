
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


# ============================================================
# SPAD DETECTOR V2 — TWO-FACTOR FACTORIAL ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "channel"
    / "v2"
    / "physical_spad_detector_v2_multiseed_pilot.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "channel" / "v2"

ANOVA_FILE = OUTPUT_DIR / "physical_spad_detector_v2_factorial_anova.csv"
PAIRWISE_FILE = OUTPUT_DIR / "physical_spad_detector_v2_factorial_pairwise.csv"
DIAGNOSTICS_FILE = OUTPUT_DIR / "physical_spad_detector_v2_factorial_diagnostics.csv"

EXPECTED_DISTANCES = [10, 50, 100]
EXPECTED_DCRS = [100, 1000, 10000]
EXPECTED_REPLICATES = 5

OUTCOMES = {
    "qber": "Primary outcome",
    "sifted_bits_per_second": "Secondary outcome",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv_exclusive(path, rows):
    if not rows:
        raise ValueError(f"No output rows to write: {path}")

    if path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite:\n{path}"
        )

    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)

    require(
        np.all(np.isfinite(p_values)),
        "Non-finite p-value encountered."
    )

    require(
        np.all((p_values >= 0) & (p_values <= 1)),
        "P-values must be between 0 and 1."
    )

    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = (n - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def hedges_g(x, y):
    """Hedges' g using pooled within-group SD."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    df = nx + ny - 2

    pooled_variance = (
        (nx - 1) * np.var(x, ddof=1)
        + (ny - 1) * np.var(y, ddof=1)
    ) / df

    if pooled_variance <= 0:
        return float("nan")

    pooled_sd = math.sqrt(pooled_variance)
    d = (np.mean(x) - np.mean(y)) / pooled_sd

    correction = 1.0 - (3.0 / (4.0 * df - 1.0))

    return correction * d


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_data():
    require(INPUT_FILE.exists(), f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    require(len(rows) == 45, f"Expected 45 rows, found {len(rows)}")

    required_fields = {
        "random_seed",
        "fiber_length_km",
        "dark_count_rate_hz",
        "condition_id",
        "replicate_index",
        "qber",
        "sifted_bits_per_second",
    }

    available_fields = set(rows[0].keys())

    missing = required_fields - available_fields

    require(
        not missing,
        f"Missing required columns: {sorted(missing)}"
    )

    parsed = []
    seeds = set()
    condition_counts = {}

    for row_number, row in enumerate(rows, start=2):
        try:
            distance = int(row["fiber_length_km"])
            dcr = int(row["dark_count_rate_hz"])
            seed = int(row["random_seed"])
            replicate = int(row["replicate_index"])

            qber = float(row["qber"])
            throughput = float(row["sifted_bits_per_second"])

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value at CSV row {row_number}: {exc}"
            )

        require(seed not in seeds, f"Duplicate random seed: {seed}")
        seeds.add(seed)

        require(
            distance in EXPECTED_DISTANCES,
            f"Unexpected distance: {distance}"
        )

        require(
            dcr in EXPECTED_DCRS,
            f"Unexpected DCR: {dcr}"
        )

        require(
            math.isfinite(qber) and 0 <= qber <= 1,
            f"Invalid QBER at row {row_number}: {qber}"
        )

        require(
            math.isfinite(throughput) and throughput >= 0,
            f"Invalid throughput at row {row_number}: {throughput}"
        )

        expected_condition = f"D{distance}_DCR{dcr}"

        require(
            row["condition_id"] == expected_condition,
            f"Condition mismatch at row {row_number}"
        )

        require(
            1 <= replicate <= EXPECTED_REPLICATES,
            f"Invalid replicate index at row {row_number}"
        )

        key = (distance, dcr)

        condition_counts[key] = condition_counts.get(key, 0) + 1

        parsed.append({
            "distance": distance,
            "dcr": dcr,
            "seed": seed,
            "replicate": replicate,
            "condition_id": expected_condition,
            "qber": qber,
            "sifted_bits_per_second": throughput,
        })

    expected_cells = {
        (distance, dcr)
        for distance in EXPECTED_DISTANCES
        for dcr in EXPECTED_DCRS
    }

    require(
        set(condition_counts) == expected_cells,
        "Factorial design does not contain all 9 expected conditions."
    )

    for key, count in condition_counts.items():
        require(
            count == EXPECTED_REPLICATES,
            f"Condition {key} has {count} replicates."
        )

    require(len(seeds) == 45, "Expected 45 unique random seeds.")

    return parsed


# ============================================================
# BALANCED TWO-WAY ANOVA
# ============================================================

def factorial_anova(data, outcome):
    """
    Balanced fixed-effects two-way ANOVA with interaction.

    Model:
        Y = grand mean + Distance + DCR
            + Distance*DCR + residual

    Uses the balanced-design sums-of-squares decomposition.
    """

    distances = EXPECTED_DISTANCES
    dcrs = EXPECTED_DCRS

    a = len(distances)
    b = len(dcrs)
    n = EXPECTED_REPLICATES

    values = np.empty((a, b, n), dtype=float)

    for i, distance in enumerate(distances):
        for j, dcr in enumerate(dcrs):
            cell = [
                row[outcome]
                for row in data
                if row["distance"] == distance and row["dcr"] == dcr
            ]

            require(
                len(cell) == n,
                f"Incorrect cell size for distance={distance}, DCR={dcr}"
            )

            values[i, j, :] = cell

    grand_mean = np.mean(values)
    distance_means = np.mean(values, axis=(1, 2))
    dcr_means = np.mean(values, axis=(0, 2))
    cell_means = np.mean(values, axis=2)

    # Sum of squares
    ss_distance = b * n * np.sum((distance_means - grand_mean) ** 2)

    ss_dcr = a * n * np.sum((dcr_means - grand_mean) ** 2)

    interaction_component = (
        cell_means
        - distance_means[:, None]
        - dcr_means[None, :]
        + grand_mean
    )

    ss_interaction = n * np.sum(interaction_component ** 2)

    residuals = values - cell_means[:, :, None]
    ss_error = np.sum(residuals ** 2)

    ss_total = np.sum((values - grand_mean) ** 2)

    df_distance = a - 1
    df_dcr = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_error = a * b * (n - 1)
    df_total = a * b * n - 1

    ms_error = ss_error / df_error

    effects = [
        ("Distance", ss_distance, df_distance),
        ("DCR", ss_dcr, df_dcr),
        ("Distance_x_DCR", ss_interaction, df_interaction),
    ]

    results = []

    for effect, ss, df in effects:
        ms = ss / df

        if ms_error == 0:
            f_stat = float("nan")
            p_value = float("nan")
        else:
            f_stat = ms / ms_error
            p_value = float(stats.f.sf(f_stat, df, df_error))

        denominator = ss + ss_error

        partial_eta_squared = (
            ss / denominator if denominator > 0 else float("nan")
        )

        results.append({
            "outcome": outcome,
            "effect": effect,
            "sum_of_squares": ss,
            "df_effect": df,
            "mean_square": ms,
            "df_error": df_error,
            "f_statistic": f_stat,
            "p_value": p_value,
            "partial_eta_squared": partial_eta_squared,
            "residual_sum_of_squares": ss_error,
            "total_sum_of_squares": ss_total,
        })

    # Overall residual diagnostics
    flat_residuals = residuals.flatten()

    return results, flat_residuals, values


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

def run_diagnostics(data, outcome, residuals, values):
    # Shapiro-Wilk residual normality test
    if np.std(residuals, ddof=1) > 0:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
    else:
        shapiro_stat = float("nan")
        shapiro_p = float("nan")

    # Levene test across all nine experimental cells
    groups = [
        values[i, j, :]
        for i in range(len(EXPECTED_DISTANCES))
        for j in range(len(EXPECTED_DCRS))
    ]

    if all(np.std(group, ddof=1) > 0 for group in groups):
        levene_stat, levene_p = stats.levene(
            *groups,
            center="median"
        )
    else:
        levene_stat = float("nan")
        levene_p = float("nan")

    return [
        {
            "outcome": outcome,
            "diagnostic": "Shapiro_Wilk_residual_normality",
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "interpretation_threshold": 0.05,
            "note": (
                "Small p suggests departure from normality; "
                "low sample size limits diagnostic power."
            ),
        },
        {
            "outcome": outcome,
            "diagnostic": "Levene_median_variance_homogeneity",
            "statistic": levene_stat,
            "p_value": levene_p,
            "interpretation_threshold": 0.05,
            "note": (
                "Small p suggests unequal variances across the 9 cells."
            ),
        },
    ]


# ============================================================
# EXPLORATORY PAIRWISE COMPARISONS
# ============================================================

def pairwise_comparisons(data, outcome):
    """
    All 36 pairwise comparisons among 9 conditions.
    Welch's independent-samples t-test.
    Holm correction applied across the 36 comparisons per outcome.
    """

    condition_groups = {}

    for distance in EXPECTED_DISTANCES:
        for dcr in EXPECTED_DCRS:
            condition = f"D{distance}_DCR{dcr}"

            condition_groups[condition] = np.asarray([
                row[outcome]
                for row in data
                if row["condition_id"] == condition
            ], dtype=float)

    comparisons = []

    for condition_a, condition_b in combinations(
        condition_groups.keys(), 2
    ):
        x = condition_groups[condition_a]
        y = condition_groups[condition_b]

        test = stats.ttest_ind(
            x,
            y,
            equal_var=False,
            alternative="two-sided"
        )

        df = float(test.df)
        difference = float(np.mean(x) - np.mean(y))

        # Welch confidence interval
        se = math.sqrt(
            np.var(x, ddof=1) / len(x)
            + np.var(y, ddof=1) / len(y)
        )

        critical = float(stats.t.ppf(0.975, df))

        ci_lower = difference - critical * se
        ci_upper = difference + critical * se

        comparisons.append({
            "outcome": outcome,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "mean_a": float(np.mean(x)),
            "mean_b": float(np.mean(y)),
            "mean_difference_a_minus_b": difference,
            "welch_t": float(test.statistic),
            "welch_df": df,
            "p_raw": float(test.pvalue),
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
            "hedges_g": hedges_g(x, y),
        })

    adjusted = holm_adjust([
        row["p_raw"] for row in comparisons
    ])

    for row, adjusted_p in zip(comparisons, adjusted):
        row["p_holm_36_comparisons"] = float(adjusted_p)
        row["significant_after_holm_0_05"] = bool(adjusted_p < 0.05)

    return comparisons


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SPAD DETECTOR V2 — TWO-FACTOR FACTORIAL ANALYSIS")
    print("=" * 70)

    data = load_data()

    print("Input validation: PASSED")
    print(f"Dataset rows: {len(data)}")
    print("Distance levels:", EXPECTED_DISTANCES)
    print("DCR levels:", EXPECTED_DCRS)
    print(f"Replicates per condition: {EXPECTED_REPLICATES}")

    all_anova = []
    all_diagnostics = []
    all_pairwise = []

    for outcome, role in OUTCOMES.items():
        print("\n" + "=" * 70)
        print(f"OUTCOME: {outcome} ({role})")
        print("=" * 70)

        anova_rows, residuals, values = factorial_anova(
            data,
            outcome
        )

        all_anova.extend(anova_rows)

        for row in anova_rows:
            print(
                f"{row['effect']:<20} "
                f"F({row['df_effect']}, {row['df_error']})="
                f"{row['f_statistic']:.6f}, "
                f"p={row['p_value']:.6g}, "
                f"partial eta²={row['partial_eta_squared']:.6f}"
            )

        diagnostics = run_diagnostics(
            data,
            outcome,
            residuals,
            values
        )

        all_diagnostics.extend(diagnostics)

        print("\nDiagnostics:")

        for item in diagnostics:
            print(
                f"{item['diagnostic']}: "
                f"statistic={item['statistic']:.6f}, "
                f"p={item['p_value']:.6g}"
            )

        pairwise = pairwise_comparisons(data, outcome)
        all_pairwise.extend(pairwise)

        significant = [
            item for item in pairwise
            if item["significant_after_holm_0_05"]
        ]

        print("\nPairwise comparisons:")
        print(f"Total comparisons: {len(pairwise)}")
        print(f"Significant after Holm: {len(significant)}")

        for item in significant:
            print(
                f"{item['condition_a']} vs {item['condition_b']}: "
                f"p_Holm={item['p_holm_36_comparisons']:.6g}, "
                f"Hedges g={item['hedges_g']:.4f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv_exclusive(ANOVA_FILE, all_anova)
    write_csv_exclusive(PAIRWISE_FILE, all_pairwise)
    write_csv_exclusive(DIAGNOSTICS_FILE, all_diagnostics)

    print("\n" + "=" * 70)
    print("FACTORIAL ANALYSIS COMPLETED")
    print("=" * 70)

    print("ANOVA:", ANOVA_FILE)
    print("Pairwise:", PAIRWISE_FILE)
    print("Diagnostics:", DIAGNOSTICS_FILE)

    print("\nInterpretation cautions:")
    print("- This is a balanced fixed-effects factorial ANOVA.")
    print("- Each cell contains only five independent simulation runs.")
    print("- Residual tests have limited power with this sample size.")
    print("- Pairwise tests are exploratory and Holm-adjusted per outcome.")
    print("- QBER is bounded; ordinary ANOVA assumptions require scrutiny.")
    print("- Results describe this simulation model, not physical validation.")


if __name__ == "__main__":
    main()
