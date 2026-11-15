
"""
FACTORIAL SENSITIVITY ANALYSIS V2
=================================

Purpose
-------
1. Replicate-level simple-effects permutation tests:
   - DCR effects within each distance
   - Distance effects within each DCR level
2. Pairwise permutation comparisons within each simple-effect stratum.
3. Grouped-binomial sensitivity analysis using errors / sifted bits.
4. Holm multiple-testing correction.
5. Input integrity and statistical diagnostics.

Important
---------
- The original simulation CSV is read-only.
- No simulation parameters or source files are modified.
- Permutation tests operate on replicate-level observations.
- Grouped-binomial tests assume independent Bernoulli trials.
  This assumption may be violated by detector dead time and afterpulsing.
- Results are exploratory sensitivity analyses, not experimental validation.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact


# ============================================================
# PATHS AND ANALYSIS SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "channel"
    / "v2"
    / "physical_spad_detector_v2_multiseed_pilot.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "channel" / "v2"

OUTPUT_SIMPLE = (
    OUTPUT_DIR / "physical_spad_detector_v2_simple_effects_permutation.csv"
)

OUTPUT_PAIRWISE = (
    OUTPUT_DIR / "physical_spad_detector_v2_simple_effects_pairwise.csv"
)

OUTPUT_BINOMIAL = (
    OUTPUT_DIR / "physical_spad_detector_v2_grouped_binomial.csv"
)

OUTPUT_BINOMIAL_PAIRWISE = (
    OUTPUT_DIR / "physical_spad_detector_v2_grouped_binomial_pairwise.csv"
)

OUTPUT_DIAGNOSTICS = (
    OUTPUT_DIR / "physical_spad_detector_v2_sensitivity_diagnostics.csv"
)

N_PERMUTATIONS = 20_000
RANDOM_SEED = 20261115

EXPECTED_ROWS = 45
EXPECTED_CONDITIONS = 9
EXPECTED_REPLICATES = 5

DISTANCE_COLUMN = "fiber_length_km"
DCR_COLUMN = "dark_count_rate_hz"

OUTCOMES = {
    "qber": "qber",
    "throughput_bps": "sifted_bits_per_second",
}

DISTANCE_LEVELS = [10, 50, 100]
DCR_LEVELS = [100, 1000, 10000]


# ============================================================
# GENERAL UTILITIES
# ============================================================

def holm_adjust(p_values):
    """Return Holm-adjusted p-values in original input order."""

    p_values = np.asarray(p_values, dtype=float)

    if len(p_values) == 0:
        return np.array([], dtype=float)

    if not np.all(np.isfinite(p_values)):
        raise ValueError("Non-finite p-value encountered.")

    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("P-values must be between 0 and 1.")

    n = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, original_index in enumerate(order):
        multiplier = n - rank
        candidate = multiplier * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(n, dtype=float)

    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]

    return adjusted


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else float("nan")


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def one_way_f_statistic(groups):
    """Classical one-way ANOVA F statistic, used as a permutation statistic."""

    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(group) * (np.mean(group) - grand_mean) ** 2
        for group in groups
    )

    ss_within = sum(
        np.sum((group - np.mean(group)) ** 2)
        for group in groups
    )

    k = len(groups)
    n = len(all_values)

    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or ss_within <= 0:
        return 0.0 if ss_between == 0 else float("inf")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    return float(ms_between / ms_within)


def permutation_one_way(groups, rng):
    """
    Permutation one-way test.

    Values are pooled and labels are shuffled while group sizes
    remain fixed. This tests exchangeability under the null.
    """

    observed = one_way_f_statistic(groups)

    pooled = np.concatenate(groups)
    sizes = [len(group) for group in groups]

    extreme = 0

    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)

        permuted_groups = []
        start = 0

        for size in sizes:
            permuted_groups.append(shuffled[start:start + size])
            start += size

        statistic = one_way_f_statistic(permuted_groups)

        if statistic >= observed:
            extreme += 1

    p_value = (extreme + 1) / (N_PERMUTATIONS + 1)

    return observed, p_value


def permutation_two_group(a, b, rng):
    """Two-sided permutation test for difference in group means."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    observed = float(np.mean(a) - np.mean(b))

    pooled = np.concatenate([a, b])
    n_a = len(a)

    extreme = 0

    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(pooled)

        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]

        difference = np.mean(perm_a) - np.mean(perm_b)

        if abs(difference) >= abs(observed) - 1e-15:
            extreme += 1

    p_value = (extreme + 1) / (N_PERMUTATIONS + 1)

    return observed, p_value


def write_csv(path, rows):
    """Write dictionaries to CSV without modifying existing input data."""

    if not rows:
        print(f"WARNING: No rows to write: {path}")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {path}")


# ============================================================
# LOAD AND VALIDATE INPUT
# ============================================================

def load_dataset():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "model_version",
            "random_seed",
            DISTANCE_COLUMN,
            DCR_COLUMN,
            "errors",
            "sifted_key_length",
            "qber",
            "sifted_bits_per_second",
            "condition_id",
            "replicate_index",
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows; found {len(rows)}."
        )

    numeric_columns = [
        "random_seed",
        DISTANCE_COLUMN,
        DCR_COLUMN,
        "errors",
        "sifted_key_length",
        "qber",
        "sifted_bits_per_second",
        "replicate_index",
    ]

    for row in rows:
        for column in numeric_columns:
            row[column] = float(row[column])

        row["random_seed"] = int(row["random_seed"])
        row["fiber_length_km"] = int(row["fiber_length_km"])
        row["dark_count_rate_hz"] = int(row["dark_count_rate_hz"])
        row["errors"] = int(row["errors"])
        row["sifted_key_length"] = int(row["sifted_key_length"])
        row["replicate_index"] = int(row["replicate_index"])

    return rows


def validate_dataset(rows):
    problems = []

    condition_ids = {row["condition_id"] for row in rows}
    seeds = [row["random_seed"] for row in rows]

    if len(condition_ids) != EXPECTED_CONDITIONS:
        problems.append(
            f"Expected {EXPECTED_CONDITIONS} conditions; "
            f"found {len(condition_ids)}."
        )

    if len(set(seeds)) != len(seeds):
        problems.append("Duplicate random_seed values detected.")

    for row in rows:
        if row["fiber_length_km"] not in DISTANCE_LEVELS:
            problems.append("Unexpected distance level.")

        if row["dark_count_rate_hz"] not in DCR_LEVELS:
            problems.append("Unexpected DCR level.")

        if row["sifted_key_length"] <= 0:
            problems.append("Non-positive sifted_key_length.")

        if not (0 <= row["errors"] <= row["sifted_key_length"]):
            problems.append("Invalid errors / sifted_key_length.")

        if not (0 <= row["qber"] <= 1):
            problems.append("QBER outside [0, 1].")

        expected_qber = row["errors"] / row["sifted_key_length"]

        if not math.isclose(
            row["qber"],
            expected_qber,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            problems.append(
                f"QBER mismatch in {row['condition_id']} "
                f"replicate {row['replicate_index']}."
            )

    for condition_id in condition_ids:
        condition_rows = [
            row for row in rows
            if row["condition_id"] == condition_id
        ]

        if len(condition_rows) != EXPECTED_REPLICATES:
            problems.append(
                f"{condition_id}: expected {EXPECTED_REPLICATES} replicates; "
                f"found {len(condition_rows)}."
            )

    if problems:
        raise ValueError(
            "Dataset validation failed:\n- " + "\n- ".join(problems)
        )

    print("Dataset validation: PASSED")
    print(f"Rows: {len(rows)}")
    print(f"Conditions: {len(condition_ids)}")
    print(f"Unique seeds: {len(set(seeds))}")


# ============================================================
# SIMPLE-EFFECT GROUP DEFINITIONS
# ============================================================

def get_simple_effect_groups(rows, outcome, factor):
    """
    factor='DCR_within_distance':
        Compare DCR levels separately at each distance.

    factor='distance_within_DCR':
        Compare distance levels separately at each DCR.
    """

    results = []

    if factor == "DCR_within_distance":
        for distance in DISTANCE_LEVELS:
            groups = []

            for dcr in DCR_LEVELS:
                values = [
                    row[outcome]
                    for row in rows
                    if row[DISTANCE_COLUMN] == distance
                    and row[DCR_COLUMN] == dcr
                ]

                groups.append(np.asarray(values, dtype=float))

            results.append({
                "factor": factor,
                "stratum": f"distance_{distance}_km",
                "level_names": [str(x) for x in DCR_LEVELS],
                "groups": groups,
            })

    elif factor == "distance_within_DCR":
        for dcr in DCR_LEVELS:
            groups = []

            for distance in DISTANCE_LEVELS:
                values = [
                    row[outcome]
                    for row in rows
                    if row[DCR_COLUMN] == dcr
                    and row[DISTANCE_COLUMN] == distance
                ]

                groups.append(np.asarray(values, dtype=float))

            results.append({
                "factor": factor,
                "stratum": f"dcr_{dcr}_hz",
                "level_names": [str(x) for x in DISTANCE_LEVELS],
                "groups": groups,
            })

    else:
        raise ValueError(f"Unknown factor: {factor}")

    return results


# ============================================================
# REPLICATE-LEVEL SIMPLE EFFECTS
# ============================================================

def run_simple_effects(rows, rng):
    omnibus_rows = []
    pairwise_rows = []

    for outcome_name, column in OUTCOMES.items():
        for factor in [
            "DCR_within_distance",
            "distance_within_DCR",
        ]:
            strata = get_simple_effect_groups(rows, column, factor)

            for stratum in strata:
                groups = stratum["groups"]

                if any(len(group) != EXPECTED_REPLICATES for group in groups):
                    raise ValueError(
                        f"Unexpected replicate count in {stratum['stratum']}."
                    )

                f_stat, p_value = permutation_one_way(groups, rng)

                omnibus_rows.append({
                    "outcome": outcome_name,
                    "factor": factor,
                    "stratum": stratum["stratum"],
                    "test": "permutation_one_way_F",
                    "n_per_group": EXPECTED_REPLICATES,
                    "permutations": N_PERMUTATIONS,
                    "observed_F": f_stat,
                    "p_raw": p_value,
                    "mean_level_1": mean_or_nan(groups[0]),
                    "mean_level_2": mean_or_nan(groups[1]),
                    "mean_level_3": mean_or_nan(groups[2]),
                })

                for i in range(3):
                    for j in range(i + 1, 3):
                        difference, pair_p = permutation_two_group(
                            groups[i],
                            groups[j],
                            rng,
                        )

                        pairwise_rows.append({
                            "outcome": outcome_name,
                            "factor": factor,
                            "stratum": stratum["stratum"],
                            "level_1": stratum["level_names"][i],
                            "level_2": stratum["level_names"][j],
                            "n_1": len(groups[i]),
                            "n_2": len(groups[j]),
                            "mean_1": mean_or_nan(groups[i]),
                            "mean_2": mean_or_nan(groups[j]),
                            "mean_difference_1_minus_2": difference,
                            "permutations": N_PERMUTATIONS,
                            "p_raw": pair_p,
                        })

    # Holm correction:
    # Six omnibus simple-effect tests per outcome.
    for outcome_name in OUTCOMES:
        indices = [
            i for i, row in enumerate(omnibus_rows)
            if row["outcome"] == outcome_name
        ]

        adjusted = holm_adjust([
            omnibus_rows[i]["p_raw"] for i in indices
        ])

        for index, p_adjusted in zip(indices, adjusted):
            omnibus_rows[index]["p_holm_within_outcome"] = float(p_adjusted)

    # Holm correction:
    # Eighteen pairwise simple-effect comparisons per outcome.
    for outcome_name in OUTCOMES:
        indices = [
            i for i, row in enumerate(pairwise_rows)
            if row["outcome"] == outcome_name
        ]

        adjusted = holm_adjust([
            pairwise_rows[i]["p_raw"] for i in indices
        ])

        for index, p_adjusted in zip(indices, adjusted):
            pairwise_rows[index]["p_holm_within_outcome"] = float(p_adjusted)

    return omnibus_rows, pairwise_rows


# ============================================================
# GROUPED-BINOMIAL SENSITIVITY ANALYSIS
# ============================================================

def grouped_counts(rows, distance=None, dcr=None):
    selected = [
        row for row in rows
        if (distance is None or row[DISTANCE_COLUMN] == distance)
        and (dcr is None or row[DCR_COLUMN] == dcr)
    ]

    errors = sum(row["errors"] for row in selected)
    sifted = sum(row["sifted_key_length"] for row in selected)

    return errors, sifted, len(selected)


def run_grouped_binomial(rows):
    omnibus_rows = []
    pairwise_rows = []

    # --------------------------------------------------------
    # Omnibus 3 x 2 contingency tests
    # --------------------------------------------------------

    for factor in [
        "DCR_within_distance",
        "distance_within_DCR",
    ]:
        strata = []

        if factor == "DCR_within_distance":
            strata = [
                ("distance", distance)
                for distance in DISTANCE_LEVELS
            ]
        else:
            strata = [
                ("dcr", dcr)
                for dcr in DCR_LEVELS
            ]

        for stratum_type, stratum_value in strata:
            contingency = []
            level_names = []

            if stratum_type == "distance":
                for dcr in DCR_LEVELS:
                    errors, sifted, _ = grouped_counts(
                        rows,
                        distance=stratum_value,
                        dcr=dcr,
                    )

                    contingency.append([errors, sifted - errors])
                    level_names.append(str(dcr))

                stratum_name = f"distance_{stratum_value}_km"

            else:
                for distance in DISTANCE_LEVELS:
                    errors, sifted, _ = grouped_counts(
                        rows,
                        distance=distance,
                        dcr=stratum_value,
                    )

                    contingency.append([errors, sifted - errors])
                    level_names.append(str(distance))

                stratum_name = f"dcr_{stratum_value}_hz"

            table = np.asarray(contingency, dtype=int)

            chi2, p_value, dof, expected = chi2_contingency(
                table,
                correction=False,
            )

            omnibus_rows.append({
                "factor": factor,
                "stratum": stratum_name,
                "test": "grouped_binomial_pearson_chi_square",
                "levels": ";".join(level_names),
                "total_errors": int(np.sum(table[:, 0])),
                "total_sifted_bits": int(np.sum(table)),
                "chi_square": float(chi2),
                "degrees_of_freedom": int(dof),
                "p_raw": float(p_value),
                "minimum_expected_cell_count": float(np.min(expected)),
                "independence_assumption": (
                    "Assumes independent Bernoulli trials; "
                    "detector temporal dependence may violate this."
                ),
            })

    # Holm correction for six omnibus tests.
    adjusted = holm_adjust([row["p_raw"] for row in omnibus_rows])

    for row, p_adjusted in zip(omnibus_rows, adjusted):
        row["p_holm_six_tests"] = float(p_adjusted)

    # --------------------------------------------------------
    # Pairwise 2 x 2 Fisher exact comparisons
    # --------------------------------------------------------

    for factor in [
        "DCR_within_distance",
        "distance_within_DCR",
    ]:
        if factor == "DCR_within_distance":
            strata = [
                (
                    f"distance_{distance}_km",
                    [
                        (distance, dcr)
                        for dcr in DCR_LEVELS
                    ],
                    [str(x) for x in DCR_LEVELS],
                )
                for distance in DISTANCE_LEVELS
            ]
        else:
            strata = [
                (
                    f"dcr_{dcr}_hz",
                    [
                        (distance, dcr)
                        for distance in DISTANCE_LEVELS
                    ],
                    [str(x) for x in DISTANCE_LEVELS],
                )
                for dcr in DCR_LEVELS
            ]

        for stratum_name, conditions, level_names in strata:
            for i in range(3):
                for j in range(i + 1, 3):
                    distance_1, dcr_1 = conditions[i]
                    distance_2, dcr_2 = conditions[j]

                    errors_1, sifted_1, _ = grouped_counts(
                        rows,
                        distance=distance_1,
                        dcr=dcr_1,
                    )

                    errors_2, sifted_2, _ = grouped_counts(
                        rows,
                        distance=distance_2,
                        dcr=dcr_2,
                    )

                    table = np.asarray([
                        [errors_1, sifted_1 - errors_1],
                        [errors_2, sifted_2 - errors_2],
                    ])

                    odds_ratio, p_value = fisher_exact(
                        table,
                        alternative="two-sided",
                    )

                    qber_1 = errors_1 / sifted_1
                    qber_2 = errors_2 / sifted_2

                    pairwise_rows.append({
                        "factor": factor,
                        "stratum": stratum_name,
                        "level_1": level_names[i],
                        "level_2": level_names[j],
                        "errors_1": errors_1,
                        "sifted_1": sifted_1,
                        "pooled_qber_1": qber_1,
                        "errors_2": errors_2,
                        "sifted_2": sifted_2,
                        "pooled_qber_2": qber_2,
                        "pooled_qber_difference_1_minus_2": (
                            qber_1 - qber_2
                        ),
                        "odds_ratio": float(odds_ratio),
                        "p_raw": float(p_value),
                        "test": "two_sided_fisher_exact",
                        "independence_assumption": (
                            "Assumes independent Bernoulli trials; "
                            "detector temporal dependence may violate this."
                        ),
                    })

    # Holm correction across all 18 pairwise tests.
    adjusted = holm_adjust([
        row["p_raw"] for row in pairwise_rows
    ])

    for row, p_adjusted in zip(pairwise_rows, adjusted):
        row["p_holm_18_tests"] = float(p_adjusted)

    return omnibus_rows, pairwise_rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def run_diagnostics(rows):
    diagnostics = []

    for distance in DISTANCE_LEVELS:
        for dcr in DCR_LEVELS:
            selected = [
                row for row in rows
                if row[DISTANCE_COLUMN] == distance
                and row[DCR_COLUMN] == dcr
            ]

            errors, sifted, n_runs = grouped_counts(
                rows,
                distance=distance,
                dcr=dcr,
            )

            replicate_qber = [
                row["qber"] for row in selected
            ]

            replicate_sifted = [
                row["sifted_key_length"] for row in selected
            ]

            diagnostics.append({
                "condition_id": selected[0]["condition_id"],
                "distance_km": distance,
                "dcr_hz": dcr,
                "replicate_count": n_runs,
                "total_errors": errors,
                "total_sifted_bits": sifted,
                "pooled_qber": errors / sifted,
                "mean_replicate_qber": mean_or_nan(replicate_qber),
                "sd_replicate_qber": sample_sd(replicate_qber),
                "minimum_replicate_sifted_bits": min(replicate_sifted),
                "maximum_replicate_sifted_bits": max(replicate_sifted),
                "zero_error_replicates": sum(
                    row["errors"] == 0 for row in selected
                ),
                "interpretation_note": (
                    "Zero observed errors do not establish zero underlying "
                    "error probability. Replicate precision differs by "
                    "sifted-key sample size."
                ),
            })

    return diagnostics


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 68)
    print("FACTORIAL SENSITIVITY ANALYSIS V2")
    print("=" * 68)

    print(f"Input: {INPUT_CSV}")
    print(f"Permutations per test: {N_PERMUTATIONS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_dataset()
    validate_dataset(rows)

    rng = np.random.default_rng(RANDOM_SEED)

    simple_rows, pairwise_rows = run_simple_effects(rows, rng)

    binomial_rows, binomial_pairwise_rows = (
        run_grouped_binomial(rows)
    )

    diagnostics_rows = run_diagnostics(rows)

    write_csv(OUTPUT_SIMPLE, simple_rows)
    write_csv(OUTPUT_PAIRWISE, pairwise_rows)
    write_csv(OUTPUT_BINOMIAL, binomial_rows)
    write_csv(OUTPUT_BINOMIAL_PAIRWISE, binomial_pairwise_rows)
    write_csv(OUTPUT_DIAGNOSTICS, diagnostics_rows)

    print()
    print("=" * 68)
    print("SENSITIVITY ANALYSIS COMPLETED")
    print("=" * 68)

    print(f"Simple-effects omnibus tests: {len(simple_rows)}")
    print(f"Simple-effects pairwise tests: {len(pairwise_rows)}")
    print(f"Grouped-binomial omnibus tests: {len(binomial_rows)}")
    print(
        "Grouped-binomial pairwise tests: "
        f"{len(binomial_pairwise_rows)}"
    )
    print(f"Condition diagnostics: {len(diagnostics_rows)}")

    print()
    print("IMPORTANT INTERPRETATION:")
    print(
        "- Permutation tests use only five replicate observations "
        "per condition."
    )
    print(
        "- Exchangeability is assumed under each permutation null."
    )
    print(
        "- Grouped-binomial tests treat sifted bits as independent "
        "Bernoulli trials."
    )
    print(
        "- Detector dead time and afterpulsing may create temporal "
        "dependence."
    )
    print(
        "- These analyses are sensitivity checks, not independent "
        "physical validation."
    )


if __name__ == "__main__":
    main()
