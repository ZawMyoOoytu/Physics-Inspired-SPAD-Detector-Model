
import csv
import math
import random
from pathlib import Path


# ============================================================
# PRACTICAL 16
# PHYSICS-INSPIRED TIME-GATED SPAD DETECTOR MODEL
# ============================================================

CONFIG = {
    "pulses": 200_000,

    # Optical channel
    "fiber_attenuation_db_km": 0.20,
    "detector_efficiency": 0.20,
    "mean_photon_number": 0.50,

    # Timing
    "pulse_repetition_rate_hz": 1_000_000,
    "gate_width_ns": 1.0,
    "timing_jitter_sigma_ps": 200.0,

    # Detector
    "dead_time_us": 10.0,
    "dark_count_rate_hz": 1_000,

    # Effective afterpulse probability per registered avalanche.
    # This is the integrated probability of a detectable
    # afterpulse candidate within the configured observation window.
    "afterpulse_probability_per_avalanche": 0.01,

    # Effective trap lifetime populations.
    # These are illustrative values, not measured SPAD parameters.
    "trap_lifetimes_us": [
        5.0,
        50.0
    ],

    # Relative weights of trap lifetime populations.
    "trap_population_weights": [
        0.70,
        0.30
    ],

    # Maximum observation window for delayed afterpulses.
    "afterpulse_window_us": 200.0,

    # Quantum channel
    "misalignment_probability": 0.01,

    "random_seed": 20261117,
}


DISTANCES_KM = [10, 50, 100]

DARK_COUNT_RATES = [
    100,
    1_000,
    10_000,
]


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_config(config):

    if config["pulses"] <= 0:
        raise ValueError("pulses must be positive.")

    if config["pulse_repetition_rate_hz"] <= 0:
        raise ValueError(
            "Pulse repetition rate must be positive."
        )

    if config["gate_width_ns"] <= 0:
        raise ValueError(
            "Gate width must be positive."
        )

    if config["dead_time_us"] < 0:
        raise ValueError(
            "Dead time cannot be negative."
        )

    if not 0 <= config[
        "afterpulse_probability_per_avalanche"
    ] <= 1:
        raise ValueError(
            "Afterpulse probability must be between 0 and 1."
        )

    lifetimes = config["trap_lifetimes_us"]
    weights = config["trap_population_weights"]

    if len(lifetimes) == 0:
        raise ValueError(
            "At least one trap lifetime is required."
        )

    if len(lifetimes) != len(weights):
        raise ValueError(
            "Trap lifetimes and weights must have equal lengths."
        )

    if any(value <= 0 for value in lifetimes):
        raise ValueError(
            "All trap lifetimes must be positive."
        )

    if any(value < 0 for value in weights):
        raise ValueError(
            "Trap weights cannot be negative."
        )

    if sum(weights) <= 0:
        raise ValueError(
            "At least one trap weight must be positive."
        )

    if config["afterpulse_window_us"] <= 0:
        raise ValueError(
            "Afterpulse observation window must be positive."
        )

    if not 0 <= config[
        "misalignment_probability"
    ] <= 1:
        raise ValueError(
            "Misalignment probability must be between 0 and 1."
        )


# ============================================================
# TIMING PARAMETERS
# ============================================================

def calculate_timing(config):

    pulse_period_s = (
        1.0 / config["pulse_repetition_rate_hz"]
    )

    gate_width_s = (
        config["gate_width_ns"] * 1e-9
    )

    jitter_sigma_s = (
        config["timing_jitter_sigma_ps"] * 1e-12
    )

    if gate_width_s > pulse_period_s:
        raise ValueError(
            "Gate width cannot exceed pulse period "
            "in this model."
        )

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

    afterpulse_window_gates = math.floor(
        config["afterpulse_window_us"]
        * 1e-6
        / pulse_period_s
    )

    return {
        "pulse_period_s": pulse_period_s,
        "gate_width_s": gate_width_s,
        "gate_acceptance": gate_acceptance,
        "dead_time_gates": dead_time_gates,
        "afterpulse_window_gates": (
            afterpulse_window_gates
        ),
    }


# ============================================================
# EXPONENTIAL TRAP RELEASE DISTRIBUTION
# ============================================================

def build_trap_gate_distributions(config, timing):

    period = timing["pulse_period_s"]
    gate_width = timing["gate_width_s"]

    max_gates = timing[
        "afterpulse_window_gates"
    ]

    lifetimes = config["trap_lifetimes_us"]

    distributions = []

    for lifetime_us in lifetimes:

        tau = lifetime_us * 1e-6

        gate_masses = []

        for gate_index in range(1, max_gates + 1):

            # Detector gates are centered around
            # integer multiples of pulse period.

            gate_start = max(
                0.0,
                gate_index * period - gate_width / 2
            )

            gate_end = (
                gate_index * period
                + gate_width / 2
            )

            # Exponential release probability mass
            # within this detector gate.

            mass = (
                math.exp(-gate_start / tau)
                -
                math.exp(-gate_end / tau)
            )

            gate_masses.append(mass)

        total_mass = sum(gate_masses)

        if total_mass <= 0:

            raise ValueError(
                "Invalid exponential trap distribution."
            )

        normalized = [
            mass / total_mass
            for mass in gate_masses
        ]

        cumulative = []

        running = 0.0

        for probability in normalized:

            running += probability
            cumulative.append(running)

        # Protect against floating-point rounding.
        cumulative[-1] = 1.0

        distributions.append({
            "lifetime_us": lifetime_us,
            "gate_probabilities": normalized,
            "cumulative": cumulative,
        })

    return distributions


# ============================================================
# SAMPLE DELAYED AFTERPULSE GATE
# ============================================================

def sample_afterpulse_gate(
    rng,
    config,
    distributions
):

    weights = config[
        "trap_population_weights"
    ]

    total_weight = sum(weights)

    selection = (
        rng.random() * total_weight
    )

    running = 0.0

    selected_index = len(weights) - 1

    for index, weight in enumerate(weights):

        running += weight

        if selection <= running:

            selected_index = index
            break

    distribution = distributions[
        selected_index
    ]

    value = rng.random()

    for gate_index, cumulative_probability in enumerate(
        distribution["cumulative"],
        start=1
    ):

        if value <= cumulative_probability:

            return gate_index

    return len(
        distribution["cumulative"]
    )


# ============================================================
# FIBER CHANNEL
# ============================================================

def calculate_signal_probability(
    distance_km,
    config,
    timing
):

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

    return {
        "fiber_loss_db": fiber_loss_db,
        "transmission": transmission,
        "total_efficiency": total_efficiency,
        "signal_probability": signal_probability,
    }


# ============================================================
# PHYSICAL-SPAD SIMULATION
# ============================================================

def simulate_physical_spad(
    distance_km,
    dark_count_rate_hz,
    config,
    seed
):

    validate_config(config)

    rng = random.Random(seed)

    timing = calculate_timing(config)

    distributions = build_trap_gate_distributions(
        config,
        timing
    )

    channel = calculate_signal_probability(
        distance_km,
        config,
        timing
    )

    pulses = config["pulses"]

    # --------------------------------------------------------
    # Dark counts per gate
    # --------------------------------------------------------

    dark_probability_per_gate = (
        1 - math.exp(
            -dark_count_rate_hz
            * timing["gate_width_s"]
        )
    )

    # --------------------------------------------------------
    # Detector history
    # --------------------------------------------------------

    last_click = [
        -10**18,
        -10**18
    ]

    # Pending afterpulse events:
    # pending[detector][pulse_index] = number of releases

    pending = [
        {},
        {}
    ]

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    signal_clicks = 0
    dark_clicks = 0

    afterpulse_candidates_scheduled = 0
    afterpulse_release_attempts = 0
    afterpulse_registered_attempts = 0
    afterpulse_blocked_by_dead_time = 0

    detected_events = 0
    dark_only_events = 0
    double_click_events = 0

    dead_time_lost_signal = 0

    sifted_key_length = 0
    errors = 0

    # --------------------------------------------------------
    # Main pulse loop
    # --------------------------------------------------------

    for pulse_index in range(pulses):

        active = [
            (
                pulse_index - last_click[detector]
                >= timing["dead_time_gates"]
            )
            for detector in range(2)
        ]

        basis_match = (
            rng.random() < 0.5
        )

        clicks = [
            False,
            False
        ]

        signal_this_pulse = False

        # ====================================================
        # 1. SIGNAL PHOTON
        # ====================================================

        if rng.random() < channel[
            "signal_probability"
        ]:

            if basis_match:

                signal_error = (
                    rng.random()
                    < config["misalignment_probability"]
                )

            else:

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

        # ====================================================
        # 2. DARK COUNTS
        # ====================================================

        for detector in range(2):

            if not active[detector]:
                continue

            if rng.random() < dark_probability_per_gate:

                clicks[detector] = True

                dark_clicks += 1

        # ====================================================
        # 3. DELAYED AFTERPULSE RELEASES
        # ====================================================

        for detector in range(2):

            due_count = pending[
                detector
            ].pop(
                pulse_index,
                0
            )

            if due_count == 0:
                continue

            afterpulse_release_attempts += (
                due_count
            )

            if active[detector]:

                clicks[detector] = True

                afterpulse_registered_attempts += (
                    due_count
                )

            else:

                afterpulse_blocked_by_dead_time += (
                    due_count
                )

        # ====================================================
        # 4. COUNT DETECTOR EVENTS
        # ====================================================

        number_of_clicks = (
            int(clicks[0])
            +
            int(clicks[1])
        )

        if number_of_clicks == 0:
            continue

        detected_events += 1

        if not signal_this_pulse:

            dark_only_events += 1

        if number_of_clicks == 2:

            double_click_events += 1

            measured_error = (
                rng.random() < 0.5
            )

        else:

            measured_error = clicks[1]

        # ====================================================
        # 5. REGISTER AVALANCHE AND SCHEDULE TRAPS
        # ====================================================

        for detector in range(2):

            if not clicks[detector]:
                continue

            # The click is registered.
            # Start detector dead-time history.

            last_click[detector] = pulse_index

            # Each registered avalanche has a probability
            # of generating a delayed afterpulse candidate.

            if (
                rng.random()
                <
                config[
                    "afterpulse_probability_per_avalanche"
                ]
            ):

                delay_gates = sample_afterpulse_gate(
                    rng,
                    config,
                    distributions
                )

                due_pulse = (
                    pulse_index + delay_gates
                )

                pending[detector][due_pulse] = (
                    pending[detector].get(
                        due_pulse,
                        0
                    ) + 1
                )

                afterpulse_candidates_scheduled += 1

        # ====================================================
        # 6. BASIS RECONCILIATION
        # ====================================================

        if basis_match:

            sifted_key_length += 1

            if measured_error:

                errors += 1

    # ========================================================
    # AFTER SIMULATION
    # ========================================================

    # Scheduled events that lie beyond the end of the
    # simulated observation period.

    afterpulse_pending_at_end = sum(
        sum(detector_pending.values())
        for detector_pending in pending
    )

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

    duration_seconds = (
        pulses / config["pulse_repetition_rate_hz"]
    )

    sifted_bits_per_second = (
        sifted_key_length / duration_seconds
    )

    detection_rate_per_second = (
        detected_events / duration_seconds
    )

    signal_to_noise_ratio = (
        signal_clicks
        /
        (
            dark_clicks
            + afterpulse_registered_attempts
        )
        if (
            dark_clicks
            + afterpulse_registered_attempts
        ) > 0
        else None
    )

    return {
        "fiber_length_km": distance_km,

        "fiber_loss_db": channel[
            "fiber_loss_db"
        ],

        "transmission_probability": channel[
            "transmission"
        ],

        "total_efficiency": channel[
            "total_efficiency"
        ],

        "dark_count_rate_hz": dark_count_rate_hz,

        "gate_width_ns": config[
            "gate_width_ns"
        ],

        "gate_acceptance_probability": timing[
            "gate_acceptance"
        ],

        "dead_time_gates": timing[
            "dead_time_gates"
        ],

        "afterpulse_window_gates": timing[
            "afterpulse_window_gates"
        ],

        "dark_probability_per_gate": (
            dark_probability_per_gate
        ),

        "signal_detection_probability": channel[
            "signal_probability"
        ],

        "afterpulse_probability_per_avalanche": config[
            "afterpulse_probability_per_avalanche"
        ],

        "trap_lifetimes_us": ";".join(
            str(value)
            for value in config["trap_lifetimes_us"]
        ),

        "trap_population_weights": ";".join(
            str(value)
            for value in config["trap_population_weights"]
        ),

        "signal_clicks": signal_clicks,

        "dark_clicks": dark_clicks,

        "afterpulse_candidates_scheduled": (
            afterpulse_candidates_scheduled
        ),

        "afterpulse_release_attempts": (
            afterpulse_release_attempts
        ),

        "afterpulse_registered_attempts": (
            afterpulse_registered_attempts
        ),

        "afterpulse_blocked_by_dead_time": (
            afterpulse_blocked_by_dead_time
        ),

        "afterpulse_pending_at_end": (
            afterpulse_pending_at_end
        ),

        "dead_time_lost_signal": (
            dead_time_lost_signal
        ),

        "detected_events": detected_events,

        "dark_only_events": dark_only_events,

        "double_click_events": double_click_events,

        "sifted_key_length": sifted_key_length,

        "errors": errors,

        "qber": qber,

        "detection_rate": detection_rate,

        "sifted_rate": sifted_rate,

        "sifted_bits_per_second": (
            sifted_bits_per_second
        ),

        "detection_rate_per_second": (
            detection_rate_per_second
        ),

        "signal_to_noise_ratio": (
            signal_to_noise_ratio
        ),

        "duration_seconds": duration_seconds,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    validate_config(CONFIG)

    results = []

    print("\n================================================")
    print(" PRACTICAL 16")
    print(" PHYSICS-INSPIRED SPAD DETECTOR MODEL")
    print("================================================")

    print(
        f"Pulses: {CONFIG['pulses']:,}"
    )

    print(
        f"Repetition rate: "
        f"{CONFIG['pulse_repetition_rate_hz'] / 1e6:.3f} MHz"
    )

    print(
        f"Gate width: "
        f"{CONFIG['gate_width_ns']} ns"
    )

    print(
        f"Timing jitter: "
        f"{CONFIG['timing_jitter_sigma_ps']} ps"
    )

    print(
        f"Dead time: "
        f"{CONFIG['dead_time_us']} us"
    )

    print(
        f"Afterpulse probability/avalanche: "
        f"{CONFIG['afterpulse_probability_per_avalanche'] * 100:.2f}%"
    )

    print(
        f"Trap lifetimes: "
        f"{CONFIG['trap_lifetimes_us']} us"
    )

    print(
        f"Trap weights: "
        f"{CONFIG['trap_population_weights']}"
    )

    timing = calculate_timing(CONFIG)

    print(
        f"Dead-time gates: "
        f"{timing['dead_time_gates']}"
    )

    print(
        f"Afterpulse observation gates: "
        f"{timing['afterpulse_window_gates']}"
    )

    experiment = 0

    for distance in DISTANCES_KM:

        for dark_rate in DARK_COUNT_RATES:

            experiment += 1

            result = simulate_physical_spad(
                distance_km=distance,
                dark_count_rate_hz=dark_rate,
                config=CONFIG,
                seed=CONFIG["random_seed"] + experiment
            )

            results.append(result)

            print("\n--------------------------------------------")

            print(
                f"Distance: {distance} km"
            )

            print(
                f"Dark count rate: {dark_rate:,} cps"
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
                f"Afterpulse candidates: "
                f"{result['afterpulse_candidates_scheduled']}"
            )

            print(
                f"Afterpulse releases: "
                f"{result['afterpulse_release_attempts']}"
            )

            print(
                f"Afterpulse registered attempts: "
                f"{result['afterpulse_registered_attempts']}"
            )

            print(
                f"Afterpulse blocked by dead time: "
                f"{result['afterpulse_blocked_by_dead_time']}"
            )

            print(
                f"Pending at simulation end: "
                f"{result['afterpulse_pending_at_end']}"
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
                f"Double clicks: "
                f"{result['double_click_events']}"
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
                f"Sifted throughput: "
                f"{result['sifted_bits_per_second']:.2f} bit/s"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = Path("results/channel")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "physical_spad_detector_baseline.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()

        writer.writerows(results)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n================================================")
    print(" PHYSICAL SPAD BASELINE SUMMARY")
    print("================================================")

    print(
        f"{'Distance':>9} "
        f"{'DCR':>9} "
        f"{'Signal':>9} "
        f"{'Dark':>7} "
        f"{'AP':>7} "
        f"{'Sifted':>8} "
        f"{'QBER':>9}"
    )

    for result in results:

        print(
            f"{result['fiber_length_km']:>8}km "
            f"{result['dark_count_rate_hz']:>8} "
            f"{result['signal_clicks']:>9} "
            f"{result['dark_clicks']:>7} "
            f"{result['afterpulse_registered_attempts']:>7} "
            f"{result['sifted_key_length']:>8} "
            f"{result['qber'] * 100:>8.3f}%"
        )

    print("\nSaved:")
    print(output_file)

    print(
        "\nPractical 16 baseline simulation completed."
    )


if __name__ == "__main__":

    main()
