
import csv
import math
import statistics
from pathlib import Path

from time_gated_detector import (
    CONFIG,
    simulate_detector
)


# =========================================
# MULTI-SEED STATISTICAL VALIDATION
# =========================================

DISTANCES_KM = [10, 50]

DEAD_TIME_VALUES_US = [
    0,
    1,
    5,
    10,
    20
]

DARK_COUNT_RATE_HZ = 1_000

REPEATS = 20

BASE_SEED = 20261200

# Approximate two-sided 95% Student-t critical
# value for 19 degrees of freedom.
T_CRITICAL_95 = 2.093


def mean_ci(values):

    valid = [
        float(v)
        for v in values
        if v is not None
        and math.isfinite(float(v))
    ]

    if not valid:
        return {
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None
        }

    mean_value = statistics.mean(valid)

    if len(valid) < 2:
        return {
            "mean": mean_value,
            "std": None,
            "ci_low": None,
            "ci_high": None
        }

    std_value = statistics.stdev(valid)

    margin = (
        T_CRITICAL_95
        * std_value
        / math.sqrt(len(valid))
    )

    return {
        "mean": mean_value,
        "std": std_value,
        "ci_low": mean_value - margin,
        "ci_high": mean_value + margin
    }


def main():

    raw_results = []
    summary_results = []

    print("=== MULTI-SEED DETECTOR VALIDATION ===")

    print(
        f"Pulses per experiment: "
        f"{CONFIG['pulses']:,}"
    )

    print(f"Repetitions: {REPEATS}")

    print(
        f"Dark count rate: "
        f"{DARK_COUNT_RATE_HZ:,} cps"
    )

    experiment = 0

    for distance in DISTANCES_KM:

        for dead_time in DEAD_TIME_VALUES_US:

            group_results = []

            print("\n===================================")
            print(f"Distance: {distance} km")
            print(f"Dead time: {dead_time} us")

            for repeat in range(REPEATS):

                experiment += 1

                config = CONFIG.copy()

                config["dead_time_us"] = dead_time

                seed = BASE_SEED + experiment

                result = simulate_detector(
                    distance_km=distance,
                    dark_count_rate_hz=(
                        DARK_COUNT_RATE_HZ
                    ),
                    config=config,
                    seed=seed
                )

                row = {
                    "repeat": repeat + 1,
                    "seed": seed,
                    "fiber_length_km": distance,
                    "dead_time_us": dead_time,
                    "dark_count_rate_hz": (
                        DARK_COUNT_RATE_HZ
                    ),
                    "dead_time_gates": (
                        result["dead_time_gates"]
                    ),
                    "signal_clicks": (
                        result["signal_clicks"]
                    ),
                    "dead_time_lost_signal": (
                        result["dead_time_lost_signal"]
                    ),
                    "detected_events": (
                        result["detected_events"]
                    ),
                    "dark_only_events": (
                        result["dark_only_events"]
                    ),
                    "double_click_events": (
                        result["double_click_events"]
                    ),
                    "sifted_key_length": (
                        result["sifted_key_length"]
                    ),
                    "errors": result["errors"],
                    "qber": result["qber"],
                    "detection_rate": (
                        result["detection_rate"]
                    ),
                    "sifted_rate": (
                        result["sifted_rate"]
                    )
                }

                raw_results.append(row)
                group_results.append(row)

            # Statistical metrics
            metrics = [
                "signal_clicks",
                "dead_time_lost_signal",
                "detected_events",
                "sifted_key_length",
                "qber",
                "detection_rate",
                "sifted_rate"
            ]

            summary = {
                "fiber_length_km": distance,
                "dead_time_us": dead_time,
                "dark_count_rate_hz": (
                    DARK_COUNT_RATE_HZ
                ),
                "repetitions": REPEATS
            }

            for metric in metrics:

                values = [
                    item[metric]
                    for item in group_results
                ]

                stats = mean_ci(values)

                summary[f"{metric}_mean"] = (
                    stats["mean"]
                )

                summary[f"{metric}_std"] = (
                    stats["std"]
                )

                summary[f"{metric}_ci_low"] = (
                    stats["ci_low"]
                )

                summary[f"{metric}_ci_high"] = (
                    stats["ci_high"]
                )

            summary_results.append(summary)

            print(
                "Mean sifted key: "
                f"{summary['sifted_key_length_mean']:.2f}"
            )

            print(
                "Mean QBER: "
                f"{summary['qber_mean'] * 100:.4f}%"
            )

            print(
                "QBER SD: "
                f"{summary['qber_std'] * 100:.4f}%"
            )

            print(
                "QBER 95% CI: "
                f"["
                f"{summary['qber_ci_low'] * 100:.4f}%, "
                f"{summary['qber_ci_high'] * 100:.4f}%"
                f"]"
            )

            print(
                "Mean sifted rate: "
                f"{summary['sifted_rate_mean'] * 100:.4f}%"
            )

    # =====================================
    # SAVE RESULTS
    # =====================================

    output_dir = Path("results/channel")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_file = (
        output_dir / "multi_seed_raw.csv"
    )

    summary_file = (
        output_dir / "multi_seed_summary.csv"
    )

    # Raw experiment results
    with raw_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(raw_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(raw_results)

    # Summary results
    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(summary_results)

    # =====================================
    # FINAL SUMMARY
    # =====================================

    print("\n=== FINAL STATISTICAL SUMMARY ===")

    print(
        f"{'Distance':>9} "
        f"{'Dead(us)':>9} "
        f"{'Sifted Mean':>13} "
        f"{'QBER Mean':>12} "
        f"{'QBER SD':>10}"
    )

    for item in summary_results:

        print(
            f"{item['fiber_length_km']:>8}km "
            f"{item['dead_time_us']:>9} "
            f"{item['sifted_key_length_mean']:>13.2f} "
            f"{item['qber_mean'] * 100:>11.4f}% "
            f"{item['qber_std'] * 100:>9.4f}%"
        )

    print("\nSaved files:")

    print(raw_file)
    print(summary_file)

    print("\nMulti-seed validation completed.")


if __name__ == "__main__":
    main()
