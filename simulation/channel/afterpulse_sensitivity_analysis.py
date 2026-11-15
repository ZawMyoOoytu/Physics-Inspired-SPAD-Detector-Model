
import csv
import math
import statistics
from pathlib import Path

from time_gated_detector import (
    CONFIG,
    simulate_detector
)


# =========================================
# AFTERPULSE SENSITIVITY ANALYSIS
# =========================================

DISTANCES_KM = [10, 50]

AFTERPULSE_PROBABILITIES = [
    0.000,
    0.005,
    0.010,
    0.020,
    0.050
]

REPETITION_RATE_HZ = 1_000_000

DEAD_TIME_US = 10.0

AFTERPULSE_WINDOW_US = 50.0

DARK_COUNT_RATE_HZ = 1_000

REPEATS = 20

BASE_SEED = 20261500

T_CRITICAL_95 = 2.093

OUTPUT_DIR = Path("results/channel")


# =========================================
# STATISTICAL ANALYSIS
# =========================================

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


# =========================================
# MAIN EXPERIMENT
# =========================================

def main():

    raw_results = []
    summary_results = []

    pulses = CONFIG["pulses"]

    duration_seconds = (
        pulses / REPETITION_RATE_HZ
    )

    afterpulse_window_gates = math.ceil(
        AFTERPULSE_WINDOW_US
        * 1e-6
        * REPETITION_RATE_HZ
    )

    print("\n======================================")
    print(" AFTERPULSE SENSITIVITY ANALYSIS")
    print("======================================")

    print(f"Pulses per experiment: {pulses:,}")
    print(f"Repetitions: {REPEATS}")
    print(f"Total experiments: {len(DISTANCES_KM) * len(AFTERPULSE_PROBABILITIES) * REPEATS}")

    print(f"PRR: {REPETITION_RATE_HZ / 1e6:.2f} MHz")
    print(f"Dead time: {DEAD_TIME_US} us")

    print(
        f"Afterpulse window: "
        f"{AFTERPULSE_WINDOW_US} us"
    )

    print(
        f"Afterpulse window gates: "
        f"{afterpulse_window_gates}"
    )

    experiment = 0

    # -------------------------------------
    # Distance sweep
    # -------------------------------------

    for distance in DISTANCES_KM:

        # ---------------------------------
        # Afterpulse probability sweep
        # ---------------------------------

        for ap_probability in AFTERPULSE_PROBABILITIES:

            config = CONFIG.copy()

            config["pulse_repetition_rate_hz"] = (
                REPETITION_RATE_HZ
            )

            config["dead_time_us"] = (
                DEAD_TIME_US
            )

            config["afterpulse_window_gates"] = (
                afterpulse_window_gates
            )

            config["afterpulse_probability"] = (
                ap_probability
            )

            group = []

            print("\n--------------------------------------")
            print(f"Distance: {distance} km")

            print(
                f"Afterpulse probability: "
                f"{ap_probability * 100:.2f}%"
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

                sifted_bps = (
                    result["sifted_key_length"]
                    / duration_seconds
                )

                detection_cps = (
                    result["detected_events"]
                    / duration_seconds
                )

                afterpulse_cps = (
                    result["afterpulse_clicks"]
                    / duration_seconds
                )

                row = {
                    "experiment": experiment,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "fiber_length_km": distance,
                    "afterpulse_probability": (
                        ap_probability
                    ),
                    "afterpulse_probability_percent": (
                        ap_probability * 100
                    ),
                    "pulse_repetition_rate_hz": (
                        REPETITION_RATE_HZ
                    ),
                    "dead_time_us": DEAD_TIME_US,
                    "afterpulse_window_us": (
                        AFTERPULSE_WINDOW_US
                    ),
                    "afterpulse_window_gates": (
                        afterpulse_window_gates
                    ),
                    "dark_count_rate_hz": (
                        DARK_COUNT_RATE_HZ
                    ),
                    "duration_seconds": (
                        duration_seconds
                    ),
                    "signal_clicks": (
                        result["signal_clicks"]
                    ),
                    "dark_clicks": (
                        result["dark_clicks"]
                    ),
                    "afterpulse_clicks": (
                        result["afterpulse_clicks"]
                    ),
                    "dead_time_lost_signal": (
                        result["dead_time_lost_signal"]
                    ),
                    "detected_events": (
                        result["detected_events"]
                    ),
                    "double_click_events": (
                        result["double_click_events"]
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
                    ),
                    "afterpulse_clicks_per_second": (
                        afterpulse_cps
                    )
                }

                raw_results.append(row)
                group.append(row)

            # -----------------------------
            # Group statistics
            # -----------------------------

            metrics = [
                "signal_clicks",
                "dark_clicks",
                "afterpulse_clicks",
                "dead_time_lost_signal",
                "detected_events",
                "double_click_events",
                "sifted_key_length",
                "errors",
                "qber",
                "sifted_bits_per_second",
                "detection_rate_per_second",
                "afterpulse_clicks_per_second"
            ]

            summary = {
                "fiber_length_km": distance,
                "afterpulse_probability": (
                    ap_probability
                ),
                "afterpulse_probability_percent": (
                    ap_probability * 100
                ),
                "pulse_repetition_rate_hz": (
                    REPETITION_RATE_HZ
                ),
                "dead_time_us": DEAD_TIME_US,
                "afterpulse_window_us": (
                    AFTERPULSE_WINDOW_US
                ),
                "afterpulse_window_gates": (
                    afterpulse_window_gates
                ),
                "dark_count_rate_hz": (
                    DARK_COUNT_RATE_HZ
                ),
                "repetitions": REPEATS
            }

            for metric in metrics:

                values = [
                    item[metric]
                    for item in group
                ]

                stats = calculate_statistics(values)

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
                "Mean afterpulse clicks: "
                f"{summary['afterpulse_clicks_mean']:.2f}"
            )

            print(
                "Mean dead-time lost signal: "
                f"{summary['dead_time_lost_signal_mean']:.2f}"
            )

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

    # =====================================
    # SAVE RESULTS
    # =====================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_file = (
        OUTPUT_DIR
        / "afterpulse_sensitivity_raw.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "afterpulse_sensitivity_summary.csv"
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

    # =====================================
    # FINAL SUMMARY
    # =====================================

    print("\n======================================")
    print(" FINAL AFTERPULSE SENSITIVITY SUMMARY")
    print("======================================")

    print(
        f"{'Distance':>9} "
        f"{'AP %':>7} "
        f"{'AP clicks':>11} "
        f"{'Sifted/s':>13} "
        f"{'QBER':>10}"
    )

    for item in summary_results:

        print(
            f"{item['fiber_length_km']:>8}km "
            f"{item['afterpulse_probability_percent']:>6.2f}% "
            f"{item['afterpulse_clicks_mean']:>11.2f} "
            f"{item['sifted_bits_per_second_mean']:>13.2f} "
            f"{item['qber_mean'] * 100:>9.4f}%"
        )

    print("\nSaved files:")
    print(raw_file)
    print(summary_file)

    print(
        "\nAfterpulse sensitivity analysis completed."
    )


if __name__ == "__main__":
    main()
