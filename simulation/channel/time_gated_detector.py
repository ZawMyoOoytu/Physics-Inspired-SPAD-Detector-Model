
import csv
import math
import random
from pathlib import Path


# =========================================
# TIME-GATED SINGLE-PHOTON DETECTOR MODEL
# =========================================

CONFIG = {
    "pulses": 200_000,

    # Optical channel
    "fiber_attenuation_db_km": 0.20,
    "detector_efficiency": 0.20,
    "mean_photon_number": 0.50,

    # Laser / detector timing
    "pulse_repetition_rate_hz": 1_000_000,
    "gate_width_ns": 1.0,
    "timing_jitter_sigma_ps": 200.0,

    # Detector behavior
    "dead_time_us": 10.0,
    "afterpulse_probability": 0.01,
    "afterpulse_window_gates": 50,

    # Quantum channel
    "misalignment_probability": 0.01,

    "random_seed": 20261116,
}


DISTANCES_KM = [10, 50, 100]

# Dark count rate per detector, counts/second
DARK_COUNT_RATES = [
    100,
    1_000,
    10_000,
]


def calculate_gate_parameters(config):

    pulse_period_s = (
        1.0 / config["pulse_repetition_rate_hz"]
    )

    gate_width_s = (
        config["gate_width_ns"] * 1e-9
    )

    jitter_sigma_s = (
        config["timing_jitter_sigma_ps"] * 1e-12
    )

    # Probability that a Gaussian-timed photon
    # arrives inside the detector gate.
    if jitter_sigma_s > 0:

        gate_acceptance = math.erf(
            gate_width_s
            / (2 * math.sqrt(2) * jitter_sigma_s)
        )

    else:
        gate_acceptance = 1.0

    dead_time_gates = math.ceil(
        config["dead_time_us"]
        * 1e-6
        / pulse_period_s
    )

    return {
        "pulse_period_s": pulse_period_s,
        "gate_width_s": gate_width_s,
        "gate_acceptance": gate_acceptance,
        "dead_time_gates": dead_time_gates,
    }


def simulate_detector(
    distance_km,
    dark_count_rate_hz,
    config,
    seed
):

    rng = random.Random(seed)

    timing = calculate_gate_parameters(config)

    pulses = config["pulses"]

    # -------------------------------------
    # Fiber transmission
    # -------------------------------------

    fiber_loss_db = (
        config["fiber_attenuation_db_km"]
        * distance_km
    )

    transmission = 10 ** (
        -fiber_loss_db / 10
    )

    total_efficiency = (
        transmission
        * config["detector_efficiency"]
    )

    optical_detection_probability = (
        1 - math.exp(
            -config["mean_photon_number"]
            * total_efficiency
        )
    )

    signal_probability = (
        optical_detection_probability
        * timing["gate_acceptance"]
    )

    # -------------------------------------
    # Dark count probability per gate
    # -------------------------------------

    dark_probability_per_gate = (
        1 - math.exp(
            -dark_count_rate_hz
            * timing["gate_width_s"]
        )
    )

    # Approximate per-gate afterpulse probability.
    afterpulse_per_gate = (
        config["afterpulse_probability"]
        / config["afterpulse_window_gates"]
    )

    # Last registered click for each detector
    last_click = [-10**18, -10**18]

    # Counters
    signal_clicks = 0
    dark_clicks = 0
    afterpulse_clicks = 0

    detected_events = 0
    dark_only_events = 0
    double_click_events = 0

    dead_time_lost_signal = 0

    sifted_key_length = 0
    errors = 0

    # -------------------------------------
    # Main pulse loop
    # -------------------------------------

    for pulse_index in range(pulses):

        active = [
            (
                pulse_index - last_click[detector]
                >= timing["dead_time_gates"]
            )
            for detector in range(2)
        ]

        # Alice and Bob choose random bases.
        basis_match = rng.random() < 0.5

        # Detector click states
        clicks = [False, False]

        signal_this_pulse = False

        # ---------------------------------
        # Signal photon
        # ---------------------------------

        if rng.random() < signal_probability:

            # Determine which detector receives
            # the photon in Bob's measurement basis.
            if basis_match:

                signal_error = (
                    rng.random()
                    < config["misalignment_probability"]
                )

            else:

                # Wrong basis: random measurement
                signal_error = (
                    rng.random() < 0.5
                )

            target_detector = (
                1 if signal_error else 0
            )

            if active[target_detector]:

                clicks[target_detector] = True
                signal_this_pulse = True
                signal_clicks += 1

            else:

                dead_time_lost_signal += 1

        # ---------------------------------
        # Dark counts and afterpulsing
        # ---------------------------------

        for detector in range(2):

            if not active[detector]:
                continue

            # Independent dark event
            dark_event = (
                rng.random()
                < dark_probability_per_gate
            )

            # Afterpulse model:
            # probability distributed over the
            # configured gates after a previous click.
            gates_since_last = (
                pulse_index - last_click[detector]
            )

            afterpulse_eligible = (
                1 <= gates_since_last
                <= config["afterpulse_window_gates"]
            )

            afterpulse_event = False

            if afterpulse_eligible:

                afterpulse_event = (
                    rng.random()
                    < afterpulse_per_gate
                )

            if dark_event:

                clicks[detector] = True
                dark_clicks += 1

            if afterpulse_event:

                clicks[detector] = True
                afterpulse_clicks += 1

        # ---------------------------------
        # Count detector events
        # ---------------------------------

        number_of_clicks = (
            int(clicks[0]) + int(clicks[1])
        )

        if number_of_clicks == 0:
            continue

        detected_events += 1

        if not signal_this_pulse:
            dark_only_events += 1

        if number_of_clicks == 2:

            double_click_events += 1

            # Random assignment for double-click
            measured_error = (
                rng.random() < 0.5
            )

        else:

            # Detector 0 = correct bit
            # Detector 1 = error bit
            measured_error = clicks[1]

        # Update detector dead-time history
        for detector in range(2):

            if clicks[detector]:

                last_click[detector] = pulse_index

        # ---------------------------------
        # Basis reconciliation
        # ---------------------------------

        if basis_match:

            sifted_key_length += 1

            if measured_error:
                errors += 1

    # -------------------------------------
    # Calculate metrics
    # -------------------------------------

    qber = (
        errors / sifted_key_length
        if sifted_key_length > 0
        else 0.0
    )

    detection_rate = (
        detected_events / pulses
    )

    sifted_rate = (
        sifted_key_length / pulses
    )

    total_dark_events = (
        dark_clicks + afterpulse_clicks
    )

    signal_to_dark_ratio = (
        signal_clicks / total_dark_events
        if total_dark_events > 0
        else None
    )

    return {
        "fiber_length_km": distance_km,
        "fiber_loss_db": fiber_loss_db,
        "transmission_probability": transmission,
        "dark_count_rate_hz": dark_count_rate_hz,
        "gate_width_ns": config["gate_width_ns"],
        "gate_acceptance_probability": (
            timing["gate_acceptance"]
        ),
        "dark_probability_per_gate": (
            dark_probability_per_gate
        ),
        "dead_time_gates": (
            timing["dead_time_gates"]
        ),
        "signal_detection_probability": (
            signal_probability
        ),
        "signal_clicks": signal_clicks,
        "dark_clicks": dark_clicks,
        "afterpulse_clicks": afterpulse_clicks,
        "dark_only_events": dark_only_events,
        "double_click_events": double_click_events,
        "dead_time_lost_signal": dead_time_lost_signal,
        "detected_events": detected_events,
        "sifted_key_length": sifted_key_length,
        "errors": errors,
        "qber": qber,
        "detection_rate": detection_rate,
        "sifted_rate": sifted_rate,
        "signal_to_dark_ratio": signal_to_dark_ratio,
    }


def main():

    results = []

    print("=== TIME-GATED DETECTOR ANALYSIS ===")
    print(
        f"Pulses per experiment: "
        f"{CONFIG['pulses']:,}"
    )

    print(
        f"Gate width: "
        f"{CONFIG['gate_width_ns']} ns"
    )

    print(
        f"Timing jitter sigma: "
        f"{CONFIG['timing_jitter_sigma_ps']} ps"
    )

    print(
        f"Detector dead time: "
        f"{CONFIG['dead_time_us']} us"
    )

    experiment = 0

    for distance in DISTANCES_KM:

        for dark_rate in DARK_COUNT_RATES:

            experiment += 1

            result = simulate_detector(
                distance_km=distance,
                dark_count_rate_hz=dark_rate,
                config=CONFIG,
                seed=CONFIG["random_seed"] + experiment
            )

            results.append(result)

            print("\n--------------------------------------")
            print(f"Distance: {distance} km")
            print(f"Dark count rate: {dark_rate:,} cps")
            print(
                f"Gate acceptance: "
                f"{result['gate_acceptance_probability'] * 100:.4f}%"
            )
            print(
                f"Dark probability/gate: "
                f"{result['dark_probability_per_gate']:.3e}"
            )
            print(
                f"Signal clicks: "
                f"{result['signal_clicks']}"
            )
            print(
                f"Dark clicks: "
                f"{result['dark_clicks']}"
            )
            print(
                f"Afterpulse clicks: "
                f"{result['afterpulse_clicks']}"
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
                f"Dark-only events: "
                f"{result['dark_only_events']}"
            )
            print(
                f"Double clicks: "
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
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir / "time_gated_detector.csv"
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

    print("\n=== SUMMARY ===")

    print(
        f"{'Distance':>9} "
        f"{'DCR':>9} "
        f"{'Signal':>9} "
        f"{'Dark':>7} "
        f"{'After':>7} "
        f"{'Sifted':>8} "
        f"{'QBER':>9}"
    )

    for result in results:

        print(
            f"{result['fiber_length_km']:>8}km "
            f"{result['dark_count_rate_hz']:>8} "
            f"{result['signal_clicks']:>9} "
            f"{result['dark_clicks']:>7} "
            f"{result['afterpulse_clicks']:>7} "
            f"{result['sifted_key_length']:>8} "
            f"{result['qber'] * 100:>8.3f}%"
        )

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
