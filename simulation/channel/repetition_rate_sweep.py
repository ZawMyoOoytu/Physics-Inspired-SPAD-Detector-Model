
import csv
import math
import statistics
from pathlib import Path

from time_gated_detector import (
    CONFIG,
    simulate_detector
)


# =========================================
# PULSE REPETITION RATE OPTIMIZATION
# =========================================

DISTANCES_KM = [10, 50]

REPETITION_RATES_HZ = [
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000
]

DEAD_TIME_US = 10.0

DARK_COUNT_RATE_HZ = 1_000

REPEATS = 10

BASE_SEED = 20261300

OUTPUT_DIR = Path("results/channel")


def calculate_statistics(values):

    mean_value = statistics.mean(values)

    std_value = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    if len(values) > 1:

        margin = (
            2.262
            * std_value
            / math.sqrt(len(values))
        )

    else:
        margin = 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "ci_low": mean_value - margin,
        "ci_high": mean_value + margin
    }


def main():

    raw_results = []
    summary_results = []

    print("=== PULSE REPETITION RATE SWEEP ===")

    print(
        f"Pulses per experiment: "
        f"{CONFIG['pulses']:,}"
    )

    print(f"Repetitions: {REPEATS}")

    print(
        f"Detector dead time: "
        f"{DEAD_TIME_US} us"
    )

    experiment = 0

    for distance in DISTANCES_KM:

        for repetition_rate in REPETITION_RATES_HZ:

            group = []

            print("\n===================================")
            print(f"Distance: {distance} km")
            print(
                f"Repetition rate: "
                f"{repetition_rate / 1_000_000:.2f} MHz"
            )

            for repeat in range(REPEATS):

                experiment += 1

                config = CONFIG.copy()

                config["pulse_repetition_rate_hz"] = (
                    repetition_rate
                )

                config["dead_time_us"] = (
                    DEAD_TIME_US
                )

                # Use the same seed for the same repeat
                # across configurations as a paired-seed
                # comparison. This is not a guarantee of
                # identical random-number paths.
                seed = BASE_SEED + repeat

                result = simulate_detector(
                    distance_km=distance,
                    dark_count_rate_hz=(
                        DARK_COUNT_RATE_HZ
                    ),
                    config=config,
                    seed=seed
                )

                pulses = config["pulses"]

                duration_seconds = (
                    pulses / repetition_rate
                )

                sifted_bits_per_second = (
                    result["sifted_key_length"]
                    / duration_seconds
                )

                detections_per_second = (
                    result["detected_events"]
                    / duration_seconds
                )

                row = {
                    "repeat": repeat + 1,
                    "seed": seed,
                    "fiber_length_km": distance,
                    "pulse_repetition_rate_hz": (
                        repetition_rate
                    ),
                    "dead_time_us": DEAD_TIME_US,
                    "dead_time_gates": (
                        result["dead_time_gates"]
                    ),
                    "detected_events": (
                        result["detected_events"]
                    ),
                    "signal_clicks": (
                        result["signal_clicks"]
                    ),
                    "dead_time_lost_signal": (
                        result["dead_time_lost_signal"]
                    ),
                    "sifted_key_length": (
                        result["sifted_key_length"]
                    ),
                    "errors": result["errors"],
                    "qber": result["qber"],
                    "sifted_bits_per_second": (
                        sifted_bits_per_second
                    ),
                    "detections_per_second": (
                        detections_per_second
                    )
                }

                raw_results.append(row)
                group.append(row)

            metrics = [
                "detected_events",
                "signal_clicks",
                "dead_time_lost_signal",
                "sifted_key_length",
                "qber",
                "sifted_bits_per_second",
                "detections_per_second"
            ]

            summary = {
                "fiber_length_km": distance,
                "pulse_repetition_rate_hz": (
                    repetition_rate
                ),
                "dead_time_us": DEAD_TIME_US,
                "repetitions": REPEATS
            }

            for metric in metrics:

                stats = calculate_statistics([
                    item[metric]
                    for item in group
                ])

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
                "Mean sifted rate: "
                f"{summary['sifted_bits_per_second_mean']:.2f} bits/s"
            )

            print(
                "Mean QBER: "
                f"{summary['qber_mean'] * 100:.4f}%"
            )

            print(
                "Mean detector count rate: "
                f"{summary['detections_per_second_mean']:.2f} counts/s"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_file = (
        OUTPUT_DIR / "repetition_rate_raw.csv"
    )

    summary_file = (
        OUTPUT_DIR / "repetition_rate_summary.csv"
    )

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

    print("\n=== REPETITION RATE SUMMARY ===")

    print(
        f"{'Distance':>9} "
        f"{'PRR(MHz)':>10} "
        f"{'Sifted(bits/s)':>16} "
        f"{'QBER':>10} "
        f"{'Dead Gates':>11}"
    )

    for item in summary_results:

        print(
            f"{item['fiber_length_km']:>8}km "
            f"{item['pulse_repetition_rate_hz'] / 1_000_000:>9.2f} "
            f"{item['sifted_bits_per_second_mean']:>16.2f} "
            f"{item['qber_mean'] * 100:>9.4f}% "
            f"{math.ceil(DEAD_TIME_US * 1e-6 * item['pulse_repetition_rate_hz']):>11}"
        )

    print("\nSaved files:")
    print(raw_file)
    print(summary_file)

    print("\nPulse repetition rate sweep completed.")


if __name__ == "__main__":
    main()
