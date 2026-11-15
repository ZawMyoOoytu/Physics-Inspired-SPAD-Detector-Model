
import csv
from pathlib import Path

from time_gated_detector import (
    CONFIG,
    simulate_detector
)


# =========================================
# DETECTOR DEAD-TIME PERFORMANCE ANALYSIS
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

BASE_SEED = 20261117


def main():

    results = []

    print("=== DETECTOR DEAD-TIME SWEEP ===")
    print(
        f"Pulses per experiment: "
        f"{CONFIG['pulses']:,}"
    )
    print(
        f"Dark count rate: "
        f"{DARK_COUNT_RATE_HZ:,} cps"
    )

    experiment = 0

    for distance in DISTANCES_KM:

        for dead_time in DEAD_TIME_VALUES_US:

            experiment += 1

            config = CONFIG.copy()

            config["dead_time_us"] = dead_time

            result = simulate_detector(
                distance_km=distance,
                dark_count_rate_hz=DARK_COUNT_RATE_HZ,
                config=config,
                seed=BASE_SEED + experiment
            )

            results.append(result)

            print("\n--------------------------------------")
            print(f"Distance: {distance} km")
            print(f"Dead time: {dead_time} us")

            print(
                f"Signal clicks: "
                f"{result['signal_clicks']}"
            )

            print(
                f"Dead-time lost signal: "
                f"{result['dead_time_lost_signal']}"
            )

            print(
                f"Detected events: "
                f"{result['detected_events']}"
            )

            print(
                f"Sifted key: "
                f"{result['sifted_key_length']}"
            )

            print(
                f"QBER: "
                f"{result['qber'] * 100:.4f}%"
            )

            print(
                f"Sifted rate: "
                f"{result['sifted_rate'] * 100:.4f}%"
            )

    # Save CSV
    output_dir = Path("results/channel")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir / "dead_time_sweep.csv"
    )

    fieldnames = [
        "fiber_length_km",
        "dark_count_rate_hz",
        "dead_time_gates",
        "signal_clicks",
        "dead_time_lost_signal",
        "detected_events",
        "sifted_key_length",
        "errors",
        "qber",
        "detection_rate",
        "sifted_rate"
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                key: result[key]
                for key in fieldnames
            })

    # Summary
    print("\n=== DEAD-TIME COMPARISON ===")

    print(
        f"{'Distance':>9} "
        f"{'Dead(us)':>9} "
        f"{'Signal':>9} "
        f"{'Lost':>8} "
        f"{'Sifted':>9} "
        f"{'QBER':>9}"
    )

    for result in results:

        print(
            f"{result['fiber_length_km']:>8}km "
            f"{result['dead_time_gates']:>9} "
            f"{result['signal_clicks']:>9} "
            f"{result['dead_time_lost_signal']:>8} "
            f"{result['sifted_key_length']:>9} "
            f"{result['qber'] * 100:>8.3f}%"
        )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
