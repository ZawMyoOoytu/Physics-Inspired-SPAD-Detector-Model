
import csv
import math
from pathlib import Path
from collections import defaultdict

from scipy.stats import t


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "channel"
    / "v2"
    / "physical_spad_detector_v2_multiseed_pilot.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "channel"
    / "v2"
    / "physical_spad_detector_v2_statistical_summary.csv"
)

EXPECTED_REPLICATES = 5
CONFIDENCE_LEVEL = 0.95


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    if len(values) < 2:
        return float("nan")

    avg = mean(values)

    variance = sum(
        (value - avg) ** 2
        for value in values
    ) / (len(values) - 1)

    return math.sqrt(variance)


def mean_confidence_interval(values, confidence=0.95):
    """
    Student's t confidence interval for the replicate mean.
    """

    n = len(values)
    avg = mean(values)

    if n < 2:
        return avg, float("nan"), float("nan")

    sd = sample_sd(values)
    standard_error = sd / math.sqrt(n)

    alpha = 1.0 - confidence

    critical_value = t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1,
    )

    margin = critical_value * standard_error

    return (
        avg,
        avg - margin,
        avg + margin,
    )


def wilson_interval(successes, trials, confidence=0.95):
    """
    Wilson score interval for pooled binomial QBER.

    successes = total error bits
    trials = total sifted bits
    """

    if trials <= 0:
        return float("nan"), float("nan")

    from scipy.stats import norm

    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)

    p = successes / trials

    denominator = 1.0 + (z * z / trials)

    center = (
        p + (z * z / (2.0 * trials))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p * (1.0 - p) / trials)
            + (z * z / (4.0 * trials * trials))
        )
        / denominator
    )

    return center - margin, center + margin


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        rows = list(csv.DictReader(file))

    if len(rows) != 45:
        raise ValueError(
            f"Expected 45 rows, found {len(rows)}"
        )

    required_fields = {
        "condition_id",
        "random_seed",
        "fiber_length_km",
        "dark_count_rate_hz",
        "qber",
        "errors",
        "sifted_key_length",
        "sifted_bits_per_second",
    }

    missing = required_fields - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    seeds = [int(row["random_seed"]) for row in rows]

    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate random seeds detected")

    groups = defaultdict(list)

    for row in rows:

        condition = row["condition_id"]

        row["random_seed"] = int(row["random_seed"])
        row["fiber_length_km"] = float(row["fiber_length_km"])
        row["dark_count_rate_hz"] = int(
            row["dark_count_rate_hz"]
        )

        row["errors"] = int(row["errors"])
        row["sifted_key_length"] = int(
            row["sifted_key_length"]
        )

        row["qber"] = float(row["qber"])

        row["sifted_bits_per_second"] = float(
            row["sifted_bits_per_second"]
        )

        if row["sifted_key_length"] <= 0:
            raise ValueError(
                f"Invalid sifted key length: {condition}"
            )

        if row["errors"] > row["sifted_key_length"]:
            raise ValueError(
                f"Errors exceed sifted key length: {condition}"
            )

        calculated_qber = (
            row["errors"] / row["sifted_key_length"]
        )

        if not math.isclose(
            row["qber"],
            calculated_qber,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"QBER inconsistency: {condition}"
            )

        groups[condition].append(row)

    if len(groups) != 9:
        raise ValueError(
            f"Expected 9 conditions, found {len(groups)}"
        )

    for condition, condition_rows in groups.items():

        if len(condition_rows) != EXPECTED_REPLICATES:
            raise ValueError(
                f"{condition}: expected "
                f"{EXPECTED_REPLICATES} replicates, "
                f"found {len(condition_rows)}"
            )

    return groups


# ============================================================
# CONDITION-LEVEL STATISTICAL ANALYSIS
# ============================================================

def analyze_condition(condition, rows):

    qber_values = [
        row["qber"]
        for row in rows
    ]

    throughput_values = [
        row["sifted_bits_per_second"]
        for row in rows
    ]

    total_errors = sum(
        row["errors"]
        for row in rows
    )

    total_sifted = sum(
        row["sifted_key_length"]
        for row in rows
    )

    pooled_qber = total_errors / total_sifted

    pooled_ci_low, pooled_ci_high = wilson_interval(
        total_errors,
        total_sifted,
        CONFIDENCE_LEVEL,
    )

    qber_mean, qber_ci_low, qber_ci_high = (
        mean_confidence_interval(
            qber_values,
            CONFIDENCE_LEVEL,
        )
    )

    throughput_mean, throughput_ci_low, throughput_ci_high = (
        mean_confidence_interval(
            throughput_values,
            CONFIDENCE_LEVEL,
        )
    )

    first = rows[0]

    return {
        "condition_id": condition,
        "distance_km": first["fiber_length_km"],
        "dark_count_rate_hz": first["dark_count_rate_hz"],
        "replicates": len(rows),

        "total_errors": total_errors,
        "total_sifted_bits": total_sifted,

        "mean_qber": qber_mean,
        "qber_sd": sample_sd(qber_values),
        "qber_ci95_lower": qber_ci_low,
        "qber_ci95_upper": qber_ci_high,

        "pooled_qber": pooled_qber,
        "pooled_qber_wilson95_lower": pooled_ci_low,
        "pooled_qber_wilson95_upper": pooled_ci_high,

        "mean_sifted_throughput_bps": throughput_mean,
        "throughput_sd_bps": sample_sd(throughput_values),
        "throughput_ci95_lower_bps": throughput_ci_low,
        "throughput_ci95_upper_bps": throughput_ci_high,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SPAD DETECTOR V2 — MULTI-SEED STATISTICAL ANALYSIS")
    print("=" * 65)

    groups = load_data()

    print("Input validation: PASSED")
    print(f"Conditions: {len(groups)}")
    print(f"Replicates per condition: {EXPECTED_REPLICATES}")

    summary = []

    for condition in sorted(groups):

        result = analyze_condition(
            condition,
            groups[condition],
        )

        summary.append(result)

        print()
        print("-" * 65)
        print(condition)
        print("-" * 65)

        print(
            f"Mean QBER: "
            f"{result['mean_qber']:.6f}"
        )

        print(
            f"QBER SD: "
            f"{result['qber_sd']:.6f}"
        )

        print(
            f"QBER 95% CI: "
            f"[{result['qber_ci95_lower']:.6f}, "
            f"{result['qber_ci95_upper']:.6f}]"
        )

        print(
            f"Pooled QBER: "
            f"{result['pooled_qber']:.6f}"
        )

        print(
            f"Mean throughput: "
            f"{result['mean_sifted_throughput_bps']:.2f} bps"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "Output already exists. "
            "No existing file was overwritten: "
            f"{OUTPUT_FILE}"
        )

    fieldnames = list(summary[0].keys())

    with OUTPUT_FILE.open(
        "x",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 65)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"Conditions analyzed: {len(summary)}")
    print(f"Output rows: {len(summary)}")
    print(f"Saved: {OUTPUT_FILE}")

    print()
    print("Interpretation note:")
    print(
        "Confidence intervals describe simulation replicate "
        "variability under the specified model."
    )
    print(
        "They do not establish physical detector accuracy "
        "or experimental validity."
    )


if __name__ == "__main__":
    main()
