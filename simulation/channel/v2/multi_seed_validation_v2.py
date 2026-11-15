
"""
Multi-Seed Pilot Validation
Physics-Inspired SPAD Detector Model V2

Experimental design:
    3 fiber distances
    x 3 dark-count rates
    x 5 independent random seeds
    = 45 simulation runs.

Each run uses 200,000 pulses.

The original V1 reference and V2 baseline results
are preserved and never overwritten.

This script performs computational pilot validation.
It does not constitute experimental validation
of a physical SPAD device.
"""

import copy
import csv
import math
from pathlib import Path

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    simulate_physical_spad,
)


# ============================================================
# EXPERIMENT DESIGN
# ============================================================

MODEL_VERSION = "2.0.0"

DISTANCES_KM = [10, 50, 100]

DARK_COUNT_RATES_HZ = [
    100,
    1_000,
    10_000,
]

REPLICATES = 5

BASE_SEED = 20261118

PULSES_PER_RUN = 200_000

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "channel"
    / "v2"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "physical_spad_detector_v2_multiseed_pilot.csv"
)


# ============================================================
# EXPERIMENT DESIGN VALIDATION
# ============================================================

def validate_experiment_design():

    if REPLICATES < 2:
        raise ValueError(
            "At least two replicates are required."
        )

    if not DISTANCES_KM:
        raise ValueError(
            "Distance conditions cannot be empty."
        )

    if not DARK_COUNT_RATES_HZ:
        raise ValueError(
            "Dark-count conditions cannot be empty."
        )

    if PULSES_PER_RUN <= 0:
        raise ValueError(
            "PULSES_PER_RUN must be positive."
        )

    if len(set(DISTANCES_KM)) != len(DISTANCES_KM):
        raise ValueError(
            "Duplicate distance conditions detected."
        )

    if len(set(DARK_COUNT_RATES_HZ)) != len(
        DARK_COUNT_RATES_HZ
    ):
        raise ValueError(
            "Duplicate dark-count conditions detected."
        )

    if BASE_SEED < 0:
        raise ValueError(
            "BASE_SEED must be non-negative."
        )

    expected_runs = (
        len(DISTANCES_KM)
        * len(DARK_COUNT_RATES_HZ)
        * REPLICATES
    )

    print(
        "Experiment design validation passed."
    )

    print(
        f"Expected simulation runs: {expected_runs}"
    )

    return expected_runs


# ============================================================
# UNIQUE SEED GENERATION
# ============================================================

def generate_seed(
    condition_index: int,
    replicate_index: int,
) -> int:

    """
    Generate a deterministic unique seed.

    condition_index:
        0 to number_of_conditions - 1

    replicate_index:
        1 to REPLICATES

    Every condition-replicate combination receives
    a different seed.
    """

    if condition_index < 0:
        raise ValueError(
            "condition_index must be non-negative."
        )

    if not 1 <= replicate_index <= REPLICATES:
        raise ValueError(
            "Invalid replicate_index."
        )

    seed = (
        BASE_SEED
        + 1
        + condition_index * REPLICATES
        + replicate_index
    )

    return seed


# ============================================================
# RESULT VALIDATION
# ============================================================

def validate_simulation_result(result: dict) -> None:

    required_fields = [
        "model_version",
        "random_seed",
        "fiber_length_km",
        "dark_count_rate_hz",
        "pulses",
        "afterpulse_candidates_scheduled",
        "afterpulse_release_attempts",
        "afterpulse_pending_at_end",
        "afterpulse_blocked_by_dead_time",
        "registered_avalanches",
        "signal_only_avalanches",
        "dark_only_avalanches",
        "afterpulse_only_avalanches",
        "mixed_source_avalanches",
        "detected_events",
        "double_click_events",
        "sifted_key_length",
        "errors",
        "qber",
        "duration_seconds",
    ]

    for field in required_fields:

        if field not in result:
            raise KeyError(
                f"Missing simulation result field: {field}"
            )

    # --------------------------------------------------------
    # Model version
    # --------------------------------------------------------

    if result["model_version"] != MODEL_VERSION:
        raise ValueError(
            "Unexpected model version."
        )

    # --------------------------------------------------------
    # Avalanche classification
    # --------------------------------------------------------

    classified_avalanches = (
        result["signal_only_avalanches"]
        + result["dark_only_avalanches"]
        + result["afterpulse_only_avalanches"]
        + result["mixed_source_avalanches"]
    )

    if classified_avalanches != result["registered_avalanches"]:
        raise AssertionError(
            "Avalanche classification mismatch."
        )

    # --------------------------------------------------------
    # Afterpulse accounting
    # --------------------------------------------------------

    afterpulse_accounted = (
        result["afterpulse_release_attempts"]
        + result["afterpulse_pending_at_end"]
    )

    if (
        result["afterpulse_candidates_scheduled"]
        != afterpulse_accounted
    ):
        raise AssertionError(
            "Afterpulse accounting mismatch."
        )

    if (
        result["afterpulse_blocked_by_dead_time"]
        > result["afterpulse_release_attempts"]
    ):
        raise AssertionError(
            "Blocked afterpulse count exceeds release attempts."
        )

    # --------------------------------------------------------
    # Detection accounting
    # --------------------------------------------------------

    if (
        result["detected_events"]
        > result["registered_avalanches"]
    ):
        raise AssertionError(
            "Detected events exceed registered avalanches."
        )

    if (
        result["registered_avalanches"]
        > 2 * result["pulses"]
    ):
        raise AssertionError(
            "Registered avalanches exceed detector-gate limit."
        )

    if (
        result["double_click_events"]
        != (
            result["registered_avalanches"]
            - result["detected_events"]
        )
    ):
        raise AssertionError(
            "Double-click accounting mismatch."
        )

    # --------------------------------------------------------
    # Key and error accounting
    # --------------------------------------------------------

    if result["errors"] > result["sifted_key_length"]:
        raise AssertionError(
            "Errors exceed sifted key length."
        )

    if result["sifted_key_length"] > result["pulses"]:
        raise AssertionError(
            "Sifted key length exceeds pulse count."
        )

    # --------------------------------------------------------
    # QBER validation
    # --------------------------------------------------------

    if result["sifted_key_length"] == 0:

        if result["qber"] is not None:
            raise AssertionError(
                "QBER must be None when sifted key is zero."
            )

    else:

        expected_qber = (
            result["errors"]
            / result["sifted_key_length"]
        )

        if result["qber"] is None:
            raise AssertionError(
                "QBER unexpectedly undefined."
            )

        if not math.isclose(
            result["qber"],
            expected_qber,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                "QBER calculation mismatch."
            )

    # --------------------------------------------------------
    # Duration validation
    # --------------------------------------------------------

    if (
        not math.isfinite(result["duration_seconds"])
        or result["duration_seconds"] <= 0
    ):
        raise AssertionError(
            "Invalid simulation duration."
        )


# ============================================================
# MAIN MULTI-SEED VALIDATION
# ============================================================

def run_multiseed_validation():

    expected_runs = validate_experiment_design()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Protect existing results.
    if OUTPUT_FILE.exists():

        raise FileExistsError(
            f"Output already exists:\n{OUTPUT_FILE}\n"
            "Existing data will not be overwritten."
        )

    print("\n" + "=" * 65)

    print(
        "PHYSICS-INSPIRED SPAD DETECTOR MODEL V2"
    )

    print(
        "MULTI-SEED PILOT VALIDATION"
    )

    print("=" * 65)

    print(f"Model version: {MODEL_VERSION}")

    print(f"Distances: {DISTANCES_KM}")

    print(
        f"Dark-count rates: {DARK_COUNT_RATES_HZ}"
    )

    print(
        f"Replicates per condition: {REPLICATES}"
    )

    print(
        f"Pulses per run: {PULSES_PER_RUN:,}"
    )

    print(
        f"Expected runs: {expected_runs}"
    )

    results = []

    used_seeds = set()

    run_number = 0

    # --------------------------------------------------------
    # Execute experimental conditions
    # --------------------------------------------------------

    for distance_index, distance_km in enumerate(
        DISTANCES_KM
    ):

        for dcr_index, dark_count_rate_hz in enumerate(
            DARK_COUNT_RATES_HZ
        ):

            # Unique index for all nine conditions.
            condition_index = (
                distance_index
                * len(DARK_COUNT_RATES_HZ)
                + dcr_index
            )

            condition_id = (
                f"D{distance_km}_DCR{dark_count_rate_hz}"
            )

            print("\n" + "-" * 65)

            print(
                f"Condition: {condition_id}"
            )

            print("-" * 65)

            for replicate_index in range(
                1,
                REPLICATES + 1,
            ):

                run_number += 1

                seed = generate_seed(
                    condition_index,
                    replicate_index,
                )

                # Check seed uniqueness.
                if seed in used_seeds:

                    raise RuntimeError(
                        f"Duplicate seed detected: {seed}"
                    )

                used_seeds.add(seed)

                # Independent configuration copy.
                run_config = copy.deepcopy(CONFIG)

                run_config["pulses"] = PULSES_PER_RUN

                print(
                    f"\nRun {run_number}/{expected_runs}"
                )

                print(
                    f"Replicate: "
                    f"{replicate_index}/{REPLICATES}"
                )

                print(
                    f"Distance: {distance_km} km"
                )

                print(
                    f"DCR: {dark_count_rate_hz} Hz"
                )

                print(
                    f"Seed: {seed}"
                )

                # ------------------------------------------------
                # Run physical SPAD simulation
                # ------------------------------------------------

                result = simulate_physical_spad(
                    distance_km=distance_km,
                    dark_count_rate_hz=dark_count_rate_hz,
                    config=run_config,
                    seed=seed,
                )

                # ------------------------------------------------
                # Validate individual simulation result
                # ------------------------------------------------

                validate_simulation_result(result)

                # ------------------------------------------------
                # Add experiment metadata
                # ------------------------------------------------

                result["condition_id"] = condition_id

                result["condition_index"] = condition_index

                result["replicate_index"] = replicate_index

                result["run_number"] = run_number

                result["base_seed"] = BASE_SEED

                result["replicates_per_condition"] = REPLICATES

                results.append(result)

                # ------------------------------------------------
                # Progress report
                # ------------------------------------------------

                print(
                    f"Registered avalanches: "
                    f"{result['registered_avalanches']:,}"
                )

                print(
                    f"Sifted key length: "
                    f"{result['sifted_key_length']:,}"
                )

                print(
                    f"Errors: {result['errors']:,}"
                )

                print(
                    f"QBER: {result['qber']}"
                )

                print(
                    "Run validation: PASSED"
                )

    # ========================================================
    # FINAL DATA VALIDATION
    # ========================================================

    print("\n" + "=" * 65)

    print(
        "FINAL DATA VALIDATION"
    )

    print("=" * 65)

    if len(results) != expected_runs:

        raise RuntimeError(
            f"Expected {expected_runs} results, "
            f"received {len(results)}."
        )

    if len(used_seeds) != expected_runs:

        raise RuntimeError(
            "Seed uniqueness validation failed."
        )

    expected_per_condition = REPLICATES

    for distance_km in DISTANCES_KM:

        for dcr in DARK_COUNT_RATES_HZ:

            matching = [
                row for row in results
                if (
                    row["fiber_length_km"] == distance_km
                    and row["dark_count_rate_hz"] == dcr
                )
            ]

            if len(matching) != expected_per_condition:

                raise RuntimeError(
                    f"Incorrect replicate count: "
                    f"{distance_km} km, {dcr} Hz."
                )

    print(
        f"Total runs validated: {len(results)}"
    )

    print(
        f"Unique seeds validated: {len(used_seeds)}"
    )

    print(
        "All individual simulation checks passed."
    )

    # ========================================================
    # SAVE RESULTS SAFELY
    # ========================================================

    fieldnames = list(results[0].keys())

    # Exclusive creation prevents accidental overwrite.
    # If writing fails, remove the incomplete output file.
    try:

        with OUTPUT_FILE.open(
            "x",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(results)

            csv_file.flush()

    except Exception:

        if OUTPUT_FILE.exists():

            OUTPUT_FILE.unlink()

        raise

    # ========================================================
    # COMPLETION SUMMARY
    # ========================================================

    print("\n" + "=" * 65)

    print(
        "MULTI-SEED PILOT VALIDATION COMPLETED"
    )

    print("=" * 65)

    print(
        f"Model version: {MODEL_VERSION}"
    )

    print(
        f"Total simulation runs: {len(results)}"
    )

    print(
        f"Unique seeds: {len(used_seeds)}"
    )

    print(
        f"Replicates per condition: {REPLICATES}"
    )

    print(
        f"Total pulses: "
        f"{len(results) * PULSES_PER_RUN:,}"
    )

    print(
        f"CSV rows: {len(results)}"
    )

    print(
        f"CSV columns: {len(fieldnames)}"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("\nScientific scope:")

    print(
        "This is computational pilot-level evidence."
    )

    print(
        "It is not physical detector measurement "
        "or experimental validation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_multiseed_validation()
