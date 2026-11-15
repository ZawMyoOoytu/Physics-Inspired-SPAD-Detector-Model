
import csv
import math
import statistics
from pathlib import Path
from collections import defaultdict

from scipy.stats import norm


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
    / "physical_spad_detector_v2_diagnostic_summary.csv"
)

EXPECTED_REPLICATES = 5
CONFIDENCE_LEVEL = 0.95


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def wilson_interval(errors, trials):

    if trials <= 0:
        return float("nan"), float("nan")

    z = norm.ppf(
        1 - (1 - CONFIDENCE_LEVEL) / 2
    )

    p = errors / trials

    denominator = 1 + z**2 / trials

    center = (
        p + z**2 / (2 * trials)
    ) / denominator

    margin = (
        z * math.sqrt(
            p * (1 - p) / trials
            + z**2 / (4 * trials**2)
        ) / denominator
    )

    return center - margin, center + margin


def coefficient_of_variation(values):

    avg = statistics.mean(values)

    if avg == 0:
        return float("nan")

    return statistics.stdev(values) / abs(avg)


# ============================================================
# LOAD DATA
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

    groups = defaultdict(list)

    for row in rows:

        row["random_seed"] = int(row["random_seed"])
        row["errors"] = int(row["errors"])
        row["sifted_key_length"] = int(
            row["sifted_key_length"]
        )

        row["qber"] = float(row["qber"])

        row["sifted_bits_per_second"] = float(
            row["sifted_bits_per_second"]
        )

        if row["sifted_key_length"] <= 0:
            raise ValueError("Invalid sifted key length")

        if not 0 <= row["errors"] <= row["sifted_key_length"]:
            raise ValueError("Invalid error count")

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
                f"QBER mismatch: {row['condition_id']}"
            )

        groups[row["condition_id"]].append(row)

    if len(groups) != 9:
        raise ValueError("Expected 9 conditions")

    for condition, condition_rows in groups.items():

        if len(condition_rows) != EXPECTED_REPLICATES:
            raise ValueError(
                f"Invalid replicate count: {condition}"
            )

        seeds = [
            row["random_seed"]
            for row in condition_rows
        ]

        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"Duplicate seed in {condition}"
            )

    return groups


# ============================================================
# DIAGNOSTIC ANALYSIS
# ============================================================

def analyze_condition(condition, rows):

    qber = [r["qber"] for r in rows]

    errors = [r["errors"] for r in rows]

    key_lengths = [
        r["sifted_key_length"]
        for r in rows
    ]

    throughput = [
        r["sifted_bits_per_second"]
        for r in rows
    ]

    total_errors = sum(errors)
    total_bits = sum(key_lengths)

    pooled_qber = total_errors / total_bits

    ci_low, ci_high = wilson_interval(
        total_errors,
        total_bits,
    )

    return {

        "condition_id": condition,

        "distance_km": rows[0]["fiber_length_km"],

        "dark_count_rate_hz": rows[0]["dark_count_rate_hz"],

        "replicates": len(rows),

        "mean_qber": statistics.mean(qber),

        "median_qber": statistics.median(qber),

        "min_qber": min(qber),

        "max_qber": max(qber),

        "qber_sd": statistics.stdev(qber),

        "qber_cv": coefficient_of_variation(qber),

        "zero_error_replicates": sum(
            e == 0 for e in errors
        ),

        "total_errors": total_errors,

        "total_sifted_bits": total_bits,

        "mean_errors_per_1000_bits": (
            1000 * total_errors / total_bits
        ),

        "mean_sifted_key_length": statistics.mean(
            key_lengths
        ),

        "min_sifted_key_length": min(key_lengths),

        "max_sifted_key_length": max(key_lengths),

        "mean_throughput_bps": statistics.mean(
            throughput
        ),

        "pooled_qber": pooled_qber,

        "pooled_wilson95_lower": ci_low,

        "pooled_wilson95_upper": ci_high,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SPAD V2 — REPLICATE DIAGNOSTIC ANALYSIS")
    print("=" * 65)

    groups = load_data()

    summary = []

    for condition in sorted(groups):

        result = analyze_condition(
            condition,
            groups[condition],
        )

        summary.append(result)

        print()
        print(condition)

        print(
            f"Mean QBER: {result['mean_qber']:.6f}"
        )

        print(
            f"Median QBER: {result['median_qber']:.6f}"
        )

        print(
            f"QBER CV: {result['qber_cv']:.4f}"
        )

        print(
            f"Zero-error replicates: "
            f"{result['zero_error_replicates']}/5"
        )

        print(
            f"Mean key length: "
            f"{result['mean_sifted_key_length']:.1f}"
        )

        print(
            f"Errors per 1000 bits: "
            f"{result['mean_errors_per_1000_bits']:.3f}"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    with OUTPUT_FILE.open(
        "x",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary)

    print()
    print("=" * 65)
    print("DIAGNOSTIC ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"Conditions: {len(summary)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
