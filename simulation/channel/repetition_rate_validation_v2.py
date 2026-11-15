
import csv
import math
import statistics
from pathlib import Path

from time_gated_detector import (
    CONFIG,
    simulate_detector
)


# =========================================
# PHYSICALLY CONSISTENT PRR VALIDATION
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

AFTERPULSE_WINDOW_US = 50.0

DARK_COUNT_RATE_HZ = 1_000

REPEATS = 10

BASE_SEED = 20261400

T_CRITICAL_95 = 2.262

OUTPUT_DIR = Path("results/channel")


def calculate_statistics(values):

    mean_value = statistics.mean(values)

    std_value = (
        statistics.stdev(values)
        if len(values) > 1
        else 0.0
    )

    margin = (
        T_CRITICAL_95
        * std_value
        / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
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

    print("=== PRR VALIDATION V2 ===")

    print(
        f"Pulses per experiment: "
        f"{CONFIG['pulses']:,}"
    )

    print(f"Repetitions: {REPEATS}")

    print(
        f"Dead time: {DEAD_TIME_US} us"
    )

    print(
        f"Physical afterpulse window: "
        f"{AFTERPULSE_WINDOW_US} us"
    )

    experiment = 0

    for distance in DISTANCES_KM:

        for repetition_rate in REPETITION_RATES_HZ:

            config = CONFIG.copy()

            config["pulse_repetition_rate_hz"] = (
                repetition_rate
            )

            config["dead_time_us"] = (
                DEAD_TIME_US
            )

            # Maintain a constant physical
            # afterpulse time window.
            afterpulse_gates = math.ceil(
                AFTERPULSE_WINDOW_US
                * 1e-6
                * repetition_rate
            )

            config["afterpulse_window_gates"] = (
                afterpulse_gates
            )

            group = []

            print("\n===================================")
            print(f"Distance: {distance} km")

            print(
                f"PRR: "
                f"{repetition_rate / 1e6:.2f} MHz"
            )

            print(
                f"Afterpulse window: "
                f"{afterpulse_gates} gates"
            )

            for repeat in range(REPEATS):

                experiment += 1

                seed = BASE_SEED + experiment

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

                sifted_bps = (
                    result["sifted_key_length"]
                    / duration_seconds
                )

                detection_cps = (
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
                    "afterpulse_window_us": (
                        AFTERPULSE_WINDOW_US
                    ),
                    "afterpulse_window_gates": (
                        afterpulse_gates
                    ),
                    "duration_seconds": (
                        duration_seconds
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
                        sifted_bps
                    ),
                    "detection_rate_per_second": (
                        detection_cps
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
                "detection_rate_per_second"
            ]

            summary = {
                "fiber_length_km": distance,
                "pulse_repetition_rate_hz": (
                    repetition_rate
                ),
                "dead_time_us": DEAD_TIME_US,
                "afterpulse_window_us": (
                    AFTERPULSE_WINDOW_US
                ),
                "afterpulse_window_gates": (
                    afterpulse_gates
                ),
                "repetitions": REPEATS
            }

            for metric in metrics:

                stats = calculate_statistics([
                    item[metric]
                    for item in group
                ])

                for field in [
                    "mean",
                    "std",
                    "ci_low",
                    "ci_high"
                ]:

                    summary[
                        f"{metric}_{field}"
                    ] = stats[field]

            summary_results.append(summary)

            print(
                "Mean sifted throughput: "
                f"{summary['sifted_bits_per_second_mean']:.2f} bits/s"
            )

            print(
                "95% CI throughput: "
                f"[{summary['sifted_bits_per_second_ci_low']:.2f}, "
                f"{summary['sifted_bits_per_second_ci_high']:.2f}]"
            )

            print(
                "Mean QBER: "
                f"{summary['qber_mean'] * 100:.4f}%"
            )

            print(
                "95% CI QBER: "
                f"[{summary['qber_ci_low'] * 100:.4f}%, "
                f"{summary['qber_ci_high'] * 100:.4f}%]"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_file = (
        OUTPUT_DIR / "repetition_rate_v2_raw.csv"
    )

    summary_file = (
        OUTPUT_DIR / "repetition_rate_v2_summary.csv"
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

    print("\n=== FINAL PRR V2 SUMMARY ===")

    print(
        f"{'Distance':>9} "
        f"{'PRR(MHz)':>9} "
        f"{'AP Gates':>9} "
        f"{'Sifted/s':>13} "
        f"{'QBER':>10}"
    )

    for item in summary_results:

        print(
            f"{item['fiber_length_km']:>8}km "
            f"{item['pulse_repetition_rate_hz'] / 1e6:>8.2f} "
            f"{item['afterpulse_window_gates']:>9} "
            f"{item['sifted_bits_per_second_mean']:>13.2f} "
            f"{item['qber_mean'] * 100:>9.4f}%"
        )

    print("\nSaved files:")
    print(raw_file)
    print(summary_file)

    print("\nPRR validation V2 completed.")


if __name__ == "__main__":
    main()
