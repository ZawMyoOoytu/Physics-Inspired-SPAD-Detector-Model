
import math
import random
import json
from pathlib import Path


# ==========================================
# REALISTIC BB84 QUANTUM CHANNEL SIMULATOR
# ==========================================

CONFIG = {
    "pulses": 200_000,
    "fiber_length_km": 20.0,
    "fiber_attenuation_db_km": 0.20,
    "mean_photon_number": 0.5,
    "detector_efficiency": 0.20,
    "dark_count_probability": 1e-6,
    "misalignment_probability": 0.01,
    "eve_attack": False,
    "random_seed": 20261114
}


def prepare_photon(bit, basis):
    return bit, basis


def measure_photon(state, measurement_basis):
    bit, preparation_basis = state

    if preparation_basis == measurement_basis:
        return bit

    return random.randint(0, 1)


def simulate_realistic_bb84(config):

    random.seed(config["random_seed"])

    n = config["pulses"]

    # 1. Optical Fiber Transmission
    fiber_loss_db = (
        config["fiber_attenuation_db_km"]
        * config["fiber_length_km"]
    )

    transmission = 10 ** (-fiber_loss_db / 10)

    # 2. Total Optical Detection Efficiency
    total_efficiency = (
        transmission
        * config["detector_efficiency"]
    )

    # 3. Signal Detection Probability
    mu = config["mean_photon_number"]

    p_signal = 1 - math.exp(
        -mu * total_efficiency
    )

    # 4. Dark Count Probability (two detectors)
    p_dark_single = config["dark_count_probability"]

    p_dark = 1 - (1 - p_dark_single) ** 2

    # 5. Alice and Bob
    detected = 0
    sifted = 0
    errors = 0

    signal_clicks = 0
    dark_clicks = 0

    alice_bits = []
    alice_bases = []

    for _ in range(n):

        alice_bit = random.randint(0, 1)
        alice_basis = random.randint(0, 1)
        bob_basis = random.randint(0, 1)

        photon = prepare_photon(
            alice_bit,
            alice_basis
        )

        # 6. Optional Eve Intercept-Resend Attack
        if config["eve_attack"]:

            eve_basis = random.randint(0, 1)

            eve_bit = measure_photon(
                photon,
                eve_basis
            )

            photon = prepare_photon(
                eve_bit,
                eve_basis
            )

        # 7. Quantum Channel Detection
        signal_detected = (
            random.random() < p_signal
        )

        dark_detected = (
            random.random() < p_dark
        )

        # No detection
        if not signal_detected and not dark_detected:
            continue

        detected += 1

        # Simplified click-origin model:
        # signal clicks take priority;
        # dark-only clicks produce random outcomes.
        if signal_detected:

            signal_clicks += 1

            bob_bit = measure_photon(
                photon,
                bob_basis
            )

            # Optical misalignment
            if random.random() < config[
                "misalignment_probability"
            ]:
                bob_bit ^= 1

        else:

            dark_clicks += 1
            bob_bit = random.randint(0, 1)

        # 8. Basis Reconciliation
        if alice_basis == bob_basis:

            sifted += 1

            if alice_bit != bob_bit:
                errors += 1

    # 9. Results
    qber = errors / sifted if sifted else 0.0

    detection_rate = detected / n

    sifted_rate = sifted / n

    return {
        "pulses": n,
        "fiber_length_km": config["fiber_length_km"],
        "fiber_loss_db": fiber_loss_db,
        "transmission_probability": transmission,
        "detector_efficiency": config[
            "detector_efficiency"
        ],
        "total_detection_efficiency": total_efficiency,
        "signal_detection_probability": p_signal,
        "dark_count_probability": p_dark,
        "detected_events": detected,
        "signal_clicks": signal_clicks,
        "dark_only_clicks": dark_clicks,
        "sifted_key_length": sifted,
        "errors": errors,
        "qber": qber,
        "detection_rate": detection_rate,
        "sifted_rate": sifted_rate,
        "eve_attack": config["eve_attack"]
    }


if __name__ == "__main__":

    result = simulate_realistic_bb84(CONFIG)

    print("\n=== REALISTIC BB84 CHANNEL SIMULATION ===")

    for key, value in result.items():

        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")

    # Save experimental results
    output_dir = Path("results/channel")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "channel_result.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"\nResults saved to: {output_file}")
