
import csv
import math
import random
from pathlib import Path

try:
    from .event_accounting import resolve_detector_gate
except ImportError:
    from event_accounting import resolve_detector_gate


# ============================================================
# PRACTICAL 16 - PHYSICAL SPAD MODEL V2
# ============================================================

MODEL_VERSION = "2.0.0"

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

    # Effective probability of a detectable afterpulse
    # candidate within the configured observation window.
    "afterpulse_probability_per_avalanche": 0.01,

    # Illustrative trap parameters.
    # These are not measured SPAD parameters.
    "trap_lifetimes_us": [
        5.0,
        50.0
    ],

    "trap_population_weights": [
        0.70,
        0.30
    ],

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

    required_keys = [
        "pulses",
        "fiber_attenuation_db_km",
        "detector_efficiency",
        "mean_photon_number",
        "pulse_repetition_rate_hz",
        "gate_width_ns",
        "timing_jitter_sigma_ps",
        "dead_time_us",
        "dark_count_rate_hz",
        "afterpulse_probability_per_avalanche",
        "trap_lifetimes_us",
        "trap_population_weights",
        "afterpulse_window_us",
        "misalignment_probability",
        "random_seed",
    ]

    for key in required_keys:

        if key not in config:
            raise KeyError(
                f"Missing configuration parameter: {key}"
            )

    if type(config["pulses"]) is not int:
        raise TypeError("pulses must be an integer.")

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

    if config["timing_jitter_sigma_ps"] < 0:
        raise ValueError(
            "Timing jitter cannot be negative."
        )

    if config["fiber_attenuation_db_km"] < 0:
        raise ValueError(
            "Fiber attenuation cannot be negative."
        )

    if not 0 <= config["detector_efficiency"] <= 1:
        raise ValueError(
            "Detector efficiency must be between 0 and 1."
        )

    if config["mean_photon_number"] < 0:
        raise ValueError(
            "Mean photon number cannot be negative."
        )

    if config["dead_time_us"] < 0:
        raise ValueError(
            "Dead time cannot be negative."
        )

    if config["dark_count_rate_hz"] < 0:
        raise ValueError(
            "Dark count rate cannot be negative."
        )

    if not 0 <= config[
        "afterpulse_probability_per_avalanche"
    ] <= 1:
        raise ValueError(
            "Afterpulse probability must be between 0 and 1."
        )

    if not 0 <= config[
        "misalignment_probability"
    ] <= 1:
        raise ValueError(
            "Misalignment probability must be between 0 and 1."
        )

    if config["afterpulse_window_us"] <= 0:
        raise ValueError(
            "Afterpulse observation window must be positive."
        )

    lifetimes = config["trap_lifetimes_us"]
    weights = config["trap_population_weights"]

    if not lifetimes:
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
            "Gate width cannot exceed pulse period."
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

    if afterpulse_window_gates < 1:

        raise ValueError(
            "Afterpulse window must cover at least one pulse gate."
        )

    return {
        "pulse_period_s": pulse_period_s,
        "gate_width_s": gate_width_s,
        "gate_acceptance": gate_acceptance,
        "dead_time_gates": dead_time_gates,
        "afterpulse_window_gates": afterpulse_window_gates,
    }


# ============================================================
# EXPONENTIAL TRAP RELEASE DISTRIBUTION
# ============================================================

def build_trap_gate_distributions(config, timing):

    period = timing["pulse_period_s"]
    gate_width = timing["gate_width_s"]

    max_gates = timing["afterpulse_window_gates"]

    distributions = []

    for lifetime_us in config["trap_lifetimes_us"]:

        tau = lifetime_us * 1e-6

        gate_masses = []

        for gate_index in range(1, max_gates + 1):

            gate_start = max(
                0.0,
                gate_index * period - gate_width / 2
            )

            gate_end = (
                gate_index * period
                + gate_width / 2
            )

            mass = (
                math.exp(-gate_start / tau)
                - math.exp(-gate_end / tau)
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

    weights = config["trap_population_weights"]

    total_weight = sum(weights)

    selection = rng.random() * total_weight

    running = 0.0

    selected_index = len(weights) - 1

    for index, weight in enumerate(weights):

        running += weight

        if selection <= running:

            selected_index = index
            break

    distribution = distributions[selected_index]

    value = rng.random()

    for gate_index, cumulative_probability in enumerate(
        distribution["cumulative"],
        start=1
    ):

        if value <= cumulative_probability:

            return gate_index

    return len(distribution["cumulative"])


# ============================================================
# FIBER CHANNEL
# ============================================================

def calculate_signal_probability(
    distance_km,
    config,
    timing
):

    if distance_km < 0:
        raise ValueError(
            "Distance cannot be negative."
        )

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
# PHYSICAL-SPAD SIMULATION V2
# ============================================================

def simulate_physical_spad(
    distance_km,
    dark_count_rate_hz,
    config,
    seed
):

    validate_config(config)

    if dark_count_rate_hz < 0:
        raise ValueError(
            "Dark count rate cannot be negative."
        )

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
    # DARK COUNTS PER GATE
    # DCR is interpreted per detector.
    # --------------------------------------------------------

    dark_probability_per_gate = (
        1 - math.exp(
            -dark_count_rate_hz
            * timing["gate_width_s"]
        )
    )

    # --------------------------------------------------------
    # DETECTOR HISTORY
    # --------------------------------------------------------

    last_click = [
        -10**18,
        -10**18
    ]

    # pending[detector][pulse_index] = release count

    pending = [
        {},
        {}
    ]

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    # Source attempts
    signal_attempts = 0
    dark_count_attempts = 0

    # Afterpulse accounting
    afterpulse_candidates_scheduled = 0
    afterpulse_release_attempts = 0
    afterpulse_blocked_by_dead_time = 0

    # Registered detector avalanches
    registered_avalanches = 0

    signal_only_avalanches = 0
    dark_only_avalanches = 0
    afterpulse_only_avalanches = 0
    mixed_source_avalanches = 0

    # Pulse-level events
    detected_events = 0
    double_click_events = 0

    # BB84 counters
    dead_time_lost_signal = 0
    sifted_key_length = 0
    errors = 0

    # ========================================================
    # MAIN PULSE LOOP
    # ========================================================

    for pulse_index in range(pulses):

        # ----------------------------------------------------
        # DETECTOR AVAILABILITY
        # ----------------------------------------------------

        active = [
            (
                pulse_index - last_click[detector]
                >= timing["dead_time_gates"]
            )
            for detector in range(2)
        ]

        basis_match = rng.random() < 0.5

        signal_attempts_by_detector = [False, False]
        dark_attempts_by_detector = [False, False]
        afterpulse_due_by_detector = [0, 0]

        clicks = [False, False]

        # ----------------------------------------------------
        # 1. SIGNAL ATTEMPT
        # ----------------------------------------------------

        if rng.random() < channel["signal_probability"]:

            if basis_match:

                signal_error = (
                    rng.random()
                    < config["misalignment_probability"]
                )

            else:

                signal_error = rng.random() < 0.5

            target_detector = (
                1 if signal_error else 0
            )

            signal_attempts += 1

            signal_attempts_by_detector[
                target_detector
            ] = True

            if not active[target_detector]:

                dead_time_lost_signal += 1

        # ----------------------------------------------------
        # 2. DARK COUNT ATTEMPTS
        # ----------------------------------------------------

        for detector in range(2):

            if not active[detector]:
                continue

            if rng.random() < dark_probability_per_gate:

                dark_count_attempts += 1

                dark_attempts_by_detector[
                    detector
                ] = True

        # ----------------------------------------------------
        # 3. AFTERPULSE RELEASE ATTEMPTS
        # ----------------------------------------------------

        for detector in range(2):

            due_count = pending[detector].pop(
                pulse_index,
                0
            )

            if due_count > 0:

                afterpulse_release_attempts += due_count

                afterpulse_due_by_detector[
                    detector
                ] = due_count

        # ----------------------------------------------------
        # 4. DETECTOR-GATE EVENT ACCOUNTING
        # ----------------------------------------------------

        for detector in range(2):

            resolution = resolve_detector_gate(
                detector_active=active[detector],
                signal_attempt=signal_attempts_by_detector[detector],
                dark_attempt=dark_attempts_by_detector[detector],
                afterpulse_release_count=afterpulse_due_by_detector[detector],
            )

            afterpulse_blocked_by_dead_time += (
                resolution[
                    "afterpulse_blocked_by_dead_time"
                ]
            )

            clicks[detector] = resolution["registered"]

            if resolution["registered"]:

                registered_avalanches += (
                    resolution["registered_avalanches"]
                )

                source_class = resolution["source_class"]

                if source_class == "signal_only":

                    signal_only_avalanches += 1

                elif source_class == "dark_only":

                    dark_only_avalanches += 1

                elif source_class == "afterpulse_only":

                    afterpulse_only_avalanches += 1

                elif source_class == "mixed_source":

                    mixed_source_avalanches += 1

        # ----------------------------------------------------
        # 5. PULSE-LEVEL DETECTION EVENTS
        # ----------------------------------------------------

        number_of_clicks = (
            int(clicks[0]) + int(clicks[1])
        )

        if number_of_clicks == 0:
            continue

        detected_events += 1

        if number_of_clicks == 2:

            double_click_events += 1

            # Preserve V1 random-assignment convention.
            measured_error = rng.random() < 0.5

        else:

            measured_error = clicks[1]

        # ----------------------------------------------------
        # 6. REGISTER AVALANCHES AND SCHEDULE AFTERPULSES
        # ----------------------------------------------------

        for detector in range(2):

            if not clicks[detector]:
                continue

            last_click[detector] = pulse_index

            # Each registered avalanche receives one
            # independent afterpulse-candidate trial.

            if (
                rng.random()
                < config["afterpulse_probability_per_avalanche"]
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

        # ----------------------------------------------------
        # 7. BB84 BASIS RECONCILIATION
        # ----------------------------------------------------

        if basis_match:

            sifted_key_length += 1

            if measured_error:
                errors += 1

    # ========================================================
    # AFTER SIMULATION
    # ========================================================

    afterpulse_pending_at_end = sum(
        sum(detector_pending.values())
        for detector_pending in pending
    )

    qber = (
        errors / sifted_key_length
        if sifted_key_length > 0
        else None
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

    # --------------------------------------------------------
    # COUNTER CONSISTENCY CHECKS
    # --------------------------------------------------------

    classified_avalanches = (
        signal_only_avalanches
        + dark_only_avalanches
        + afterpulse_only_avalanches
        + mixed_source_avalanches
    )

    if registered_avalanches != classified_avalanches:

        raise RuntimeError(
            "Avalanche classification counter mismatch."
        )

    if registered_avalanches < detected_events:

        raise RuntimeError(
            "Registered avalanches cannot be fewer "
            "than detected pulse events."
        )

    if errors > sifted_key_length:

        raise RuntimeError(
            "Error count exceeds sifted key length."
        )

    # ========================================================
    # RESULT DICTIONARY
    # ========================================================

    return {

        "model_version": MODEL_VERSION,

        "random_seed": seed,

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

        "timing_jitter_sigma_ps": config[
            "timing_jitter_sigma_ps"
        ],

        "gate_acceptance_probability": timing[
            "gate_acceptance"
        ],

        "dead_time_us": config[
            "dead_time_us"
        ],

        "dead_time_gates": timing[
            "dead_time_gates"
        ],

        "afterpulse_window_us": config[
            "afterpulse_window_us"
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

        "pulses": pulses,

        "signal_attempts": signal_attempts,

        "dark_count_attempts": dark_count_attempts,

        "afterpulse_candidates_scheduled": (
            afterpulse_candidates_scheduled
        ),

        "afterpulse_release_attempts": (
            afterpulse_release_attempts
        ),

        "afterpulse_blocked_by_dead_time": (
            afterpulse_blocked_by_dead_time
        ),

        "afterpulse_pending_at_end": (
            afterpulse_pending_at_end
        ),

        "registered_avalanches": registered_avalanches,

        "signal_only_avalanches": signal_only_avalanches,

        "dark_only_avalanches": dark_only_avalanches,

        "afterpulse_only_avalanches": afterpulse_only_avalanches,

        "mixed_source_avalanches": mixed_source_avalanches,

        "dead_time_lost_signal": dead_time_lost_signal,

        "detected_events": detected_events,

        "double_click_events": double_click_events,

        "sifted_key_length": sifted_key_length,

        "errors": errors,

        "qber": qber,

        "detection_rate": detection_rate,

        "sifted_rate": sifted_rate,

        "sifted_bits_per_second": sifted_bits_per_second,

        "detection_rate_per_second": detection_rate_per_second,

        "duration_seconds": duration_seconds,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_qber(qber, precision=4):

    if qber is None:
        return "undefined"

    return f"{qber * 100:.{precision}f}%"


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    validate_config(CONFIG)

    results = []

    print("\n================================================")
    print(" PRACTICAL 16")
    print(" PHYSICS-INSPIRED SPAD DETECTOR MODEL V2")
    print("================================================")

    print(f"Model version: {MODEL_VERSION}")

    print(f"Pulses: {CONFIG['pulses']:,}")

    print(
        f"Repetition rate: "
        f"{CONFIG['pulse_repetition_rate_hz'] / 1e6:.3f} MHz"
    )

    print(f"Gate width: {CONFIG['gate_width_ns']} ns")

    print(
        f"Timing jitter: "
        f"{CONFIG['timing_jitter_sigma_ps']} ps"
    )

    print(f"Dead time: {CONFIG['dead_time_us']} us")

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

            seed = CONFIG["random_seed"] + experiment

            result = simulate_physical_spad(
                distance_km=distance,
                dark_count_rate_hz=dark_rate,
                config=CONFIG,
                seed=seed
            )

            results.append(result)

            print("\n--------------------------------------------")

            print(f"Distance: {distance} km")

            print(f"Dark count rate: {dark_rate:,} cps/detector")

            print(
                f"Signal attempts: "
                f"{result['signal_attempts']}"
            )

            print(
                f"Dark count attempts: "
                f"{result['dark_count_attempts']}"
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
                f"Afterpulse blocked: "
                f"{result['afterpulse_blocked_by_dead_time']}"
            )

            print(
                f"Registered avalanches: "
                f"{result['registered_avalanches']}"
            )

            print(
                f"Signal-only avalanches: "
                f"{result['signal_only_avalanches']}"
            )

            print(
                f"Dark-only avalanches: "
                f"{result['dark_only_avalanches']}"
            )

            print(
                f"Afterpulse-only avalanches: "
                f"{result['afterpulse_only_avalanches']}"
            )

            print(
                f"Mixed-source avalanches: "
                f"{result['mixed_source_avalanches']}"
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
                f"{format_qber(result['qber'])}"
            )

            print(
                f"Sifted throughput: "
                f"{result['sifted_bits_per_second']:.2f} bit/s"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = Path("results/channel/v2")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "physical_spad_detector_v2_baseline.csv"
    )

    # Protect existing output from accidental overwrite.
    if output_file.exists():

        raise FileExistsError(
            f"Output already exists: {output_file}\n"
            "Rename or archive the existing file before "
            "running this experiment again."
        )

    with output_file.open(
        "x",
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
    print(" PHYSICAL SPAD V2 BASELINE SUMMARY")
    print("================================================")

    print(
        f"{'Distance':>9} "
        f"{'DCR':>9} "
        f"{'Signal':>9} "
        f"{'Dark':>9} "
        f"{'AP-only':>9} "
        f"{'Sifted':>9} "
        f"{'QBER':>10}"
    )

    for result in results:

        print(
            f"{result['fiber_length_km']:>8}km "
            f"{result['dark_count_rate_hz']:>8} "
            f"{result['signal_only_avalanches']:>9} "
            f"{result['dark_only_avalanches']:>9} "
            f"{result['afterpulse_only_avalanches']:>9} "
            f"{result['sifted_key_length']:>9} "
            f"{format_qber(result['qber'], 3):>10}"
        )

    print("\nSaved:")
    print(output_file)

    print(
        "\nPhysical SPAD V2 baseline simulation completed."
    )


if __name__ == "__main__":

    main()
