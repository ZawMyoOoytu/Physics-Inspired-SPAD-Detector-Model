
import csv
import math
import random
from pathlib import Path


# -----------------------------------------
# QKD DETECTOR CHARACTERIZATION
# -----------------------------------------

PULSES = 200_000

DISTANCES_KM = [10, 20, 50, 100]

# Per-detector dark count probability per pulse
DARK_COUNT_LEVELS = [
    1e-6,
    1e-5,
    1e-4,
]

FIBER_ATTENUATION_DB_KM = 0.20
DETECTOR_EFFICIENCY = 0.20
MEAN_PHOTON_NUMBER = 0.50
MISALIGNMENT_PROBABILITY = 0.01

BASE_SEED = 20261115


def simulate_detector(
    distance_km,
    dark_count_probability,
    seed
):
    rng = random.Random(seed)

    # Fiber channel
    fiber_loss_db = (
        FIBER_ATTENUATION_DB_KM * distance_km
    )

    transmission = 10 ** (-fiber_loss_db / 10)

    total_efficiency = (
        transmission * DETECTOR_EFFICIENCY
    )

    signal_probability = 1 - math.exp(
        -MEAN_PHOTON_NUMBER * total_efficiency
    )

    # Counters
    detected_events = 0
    signal_clicks = 0
    dark_only_clicks = 0
    dark_involved_events = 0
    double_click_events = 0

    sifted_key_length = 0
    errors = 0

    for _ in range(PULSES):

        # Approximately 50% basis matching
        basis_match = rng.random() < 0.5

        # Signal photon detection
        signal_detected = (
            rng.random() < signal_probability
        )

        # Signal bit relative to Alice's correct bit:
        # 0 = correct detector
        # 1 = wrong detector
        signal_error = False

        if signal_detected:

            signal_clicks += 1

            if basis_match:
                signal_error = (
                    rng.random()
                    < MISALIGNMENT_PROBABILITY
                )
            else:
                # Wrong-basis measurement is random
                signal_error = (
                    rng.random() < 0.5
                )

        # Independent dark counts in two detectors
        dark_detector_0 = (
            rng.random() < dark_count_probability
        )

        dark_detector_1 = (
            rng.random() < dark_count_probability
        )

        if dark_detector_0 or dark_detector_1:
            dark_involved_events += 1

        if not signal_detected and (
            dark_detector_0 or dark_detector_1
        ):
            dark_only_clicks += 1

        # Detector 0: correct bit
        # Detector 1: error bit
        detector_0_click = (
            dark_detector_0
            or (
                signal_detected
                and not signal_error
            )
        )

        detector_1_click = (
            dark_detector_1
            or (
                signal_detected
                and signal_error
            )
        )

        number_of_clicks = (
            int(detector_0_click)
            + int(detector_1_click)
        )

        if number_of_clicks == 0:
            continue

        detected_events += 1

        # Double-click handling:
        # assign a random bit
        if number_of_clicks == 2:

            double_click_events += 1

            measured_error = (
                rng.random() < 0.5
            )

        elif detector_1_click:

            measured_error = True

        else:

            measured_error = False

        # Only matching bases contribute to sifted key
        if basis_match:

            sifted_key_length += 1

            if measured_error:
                errors += 1

    qber = (
        errors / sifted_key_length
        if sifted_key_length > 0
        else 0.0
    )

    detection_rate = detected_events / PULSES

    sifted_rate = sifted_key_length / PULSES

    signal_to_dark_ratio = (
        signal_clicks / dark_only_clicks
        if dark_only_clicks > 0
        else None
    )

    return {
        "fiber_length_km": distance_km,
        "fiber_loss_db": fiber_loss_db,
        "transmission_probability": transmission,
        "detector_efficiency": DETECTOR_EFFICIENCY,
        "total_detection_efficiency": total_efficiency,
        "signal_detection_probability": signal_probability,
        "dark_count_per_detector": dark_count_probability,
        "detected_events": detected_events,
        "signal_clicks": signal_clicks,
        "dark_only_clicks": dark_only_clicks,
        "dark_involved_events": dark_involved_events,
        "double_click_events": double_click_events,
        "sifted_key_length": sifted_key_length,
        "errors": errors,
        "qber": qber,
        "detection_rate": detection_rate,
        "sifted_rate": sifted_rate,
        "signal_to_dark_ratio": signal_to_dark_ratio,
    }


def main():

    results = []

    print("=== SINGLE-PHOTON DETECTOR CHARACTERIZATION ===")
    print(f"Pulses per experiment: {PULSES:,}")

    experiment_number = 0

    for distance in DISTANCES_KM:

        for dark_probability in DARK_COUNT_LEVELS:

            experiment_number += 1

            seed = BASE_SEED + experiment_number

            result = simulate_detector(
                distance_km=distance,
                dark_count_probability=dark_probability,
                seed=seed
            )

            results.append(result)

            print("\n-----------------------------------------")
            print(f"Fiber distance: {distance} km")
            print(
                "Dark probability per detector: "
                f"{dark_probability:.1e}"
            )
            print(
                f"Fiber loss: "
                f"{result['fiber_loss_db']:.2f} dB"
            )
            print(
                f"Signal detection probability: "
                f"{result['signal_detection_probability'] * 100:.5f}%"
            )
            print(
                f"Detected events: "
                f"{result['detected_events']}"
            )
            print(
                f"Signal clicks: "
                f"{result['signal_clicks']}"
            )
            print(
                f"Dark-only clicks: "
                f"{result['dark_only_clicks']}"
            )
            print(
                f"Double-click events: "
                f"{result['double_click_events']}"
            )
            print(
                f"Sifted key: "
                f"{result['sifted_key_length']}"
            )
            print(
                f"Errors: "
                f"{result['errors']}"
            )
            print(
                f"QBER: "
                f"{result['qber'] * 100:.4f}%"
            )

    # Save CSV
    output_dir = Path("results/channel")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "detector_characterization.csv"
    )

    fieldnames = list(results[0].keys())

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
        writer.writerows(results)

    print("\n=== EXPERIMENT SUMMARY ===")

    print(
        f"{'Distance':>9} "
        f"{'Dark rate':>12} "
        f"{'Detections':>12} "
        f"{'Dark only':>10} "
        f"{'Double':>8} "
        f"{'QBER':>10}"
    )

    for result in results:

        print(
            f"{result['fiber_length_km']:>8}km "
            f"{result['dark_count_per_detector']:>12.1e} "
            f"{result['detected_events']:>12} "
            f"{result['dark_only_clicks']:>10} "
            f"{result['double_click_events']:>8} "
            f"{result['qber'] * 100:>9.4f}%"
        )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
