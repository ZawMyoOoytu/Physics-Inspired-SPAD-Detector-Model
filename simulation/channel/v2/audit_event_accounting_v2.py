import csv
import math
from collections import Counter
from pathlib import Path


INPUT_FILE = Path(
    "results/channel/v2/"
    "physical_spad_detector_v2_multiseed_pilot.csv"
)

TOLERANCE = 1e-10


def close(a, b):
    return math.isclose(
        a, b,
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE,
    )


def main():

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    failures = []
    condition_counts = Counter()
    seeds = []

    for row_number, row in enumerate(rows, start=1):

        prefix = f"Row {row_number}"

        def number(field):
            return float(row[field])

        def integer(field):
            return int(row[field])

        # ------------------------------------------------
        # 1. Registered avalanche accounting
        # ------------------------------------------------

        registered_sum = sum(
            integer(field)
            for field in (
                "signal_only_avalanches",
                "dark_only_avalanches",
                "afterpulse_only_avalanches",
                "mixed_source_avalanches",
            )
        )

        if registered_sum != integer("registered_avalanches"):
            failures.append(
                f"{prefix}: registered avalanche mismatch"
            )

        # ------------------------------------------------
        # 2. Detection event accounting
        # ------------------------------------------------

        if integer("registered_avalanches") < integer("detected_events"):
            failures.append(
                f"{prefix}: registered avalanches < detected events"
            )

        if integer("detected_events") > integer("pulses"):
            failures.append(
                f"{prefix}: detected events exceed pulses"
            )

        # ------------------------------------------------
        # 3. QBER accounting
        # ------------------------------------------------

        sifted = integer("sifted_key_length")
        errors = integer("errors")

        if errors > sifted:
            failures.append(
                f"{prefix}: errors exceed sifted key length"
            )

        if sifted > 0:
            expected_qber = errors / sifted

            if not close(number("qber"), expected_qber):
                failures.append(
                    f"{prefix}: QBER mismatch"
                )

        # ------------------------------------------------
        # 4. Detection rate
        # ------------------------------------------------

        expected_detection_rate = (
            integer("detected_events") / integer("pulses")
        )

        if not close(
            number("detection_rate"),
            expected_detection_rate,
        ):
            failures.append(
                f"{prefix}: detection rate mismatch"
            )

        # ------------------------------------------------
        # 5. Sifted rate
        # ------------------------------------------------

        expected_sifted_rate = (
            sifted / integer("pulses")
        )

        if not close(
            number("sifted_rate"),
            expected_sifted_rate,
        ):
            failures.append(
                f"{prefix}: sifted rate mismatch"
            )

        # ------------------------------------------------
        # 6. Throughput
        # ------------------------------------------------

        expected_throughput = (
            sifted / number("duration_seconds")
        )

        if not close(
            number("sifted_bits_per_second"),
            expected_throughput,
        ):
            failures.append(
                f"{prefix}: sifted throughput mismatch"
            )

        # ------------------------------------------------
        # 7. Afterpulse candidate accounting
        # ------------------------------------------------

        scheduled = integer(
            "afterpulse_candidates_scheduled"
        )

        released = integer(
            "afterpulse_release_attempts"
        )

        pending = integer(
            "afterpulse_pending_at_end"
        )

        if scheduled != released + pending:
            failures.append(
                f"{prefix}: afterpulse candidate accounting mismatch"
            )

        # ------------------------------------------------
        # 8. Dead-time lost signal sanity
        # ------------------------------------------------

        if integer("dead_time_lost_signal") > integer("signal_attempts"):
            failures.append(
                f"{prefix}: dead-time losses exceed signal attempts"
            )

        # ------------------------------------------------
        # 9. Condition and seed metadata
        # ------------------------------------------------

        condition_counts[row["condition_id"]] += 1
        seeds.append(integer("random_seed"))

    # ----------------------------------------------------
    # Dataset-level checks
    # ----------------------------------------------------

    if len(rows) != 45:
        failures.append(
            f"Expected 45 rows, found {len(rows)}"
        )

    if len(set(seeds)) != len(seeds):
        failures.append(
            "Duplicate random seeds detected"
        )

    if len(condition_counts) != 9:
        failures.append(
            f"Expected 9 conditions, found {len(condition_counts)}"
        )

    for condition, count in sorted(condition_counts.items()):
        if count != 5:
            failures.append(
                f"{condition}: expected 5 replicates, found {count}"
            )

    # ----------------------------------------------------
    # Report
    # ----------------------------------------------------

    print("\n=== DATASET INTEGRITY AUDIT ===")

    print(f"Total rows       : {len(rows)}")
    print(f"Conditions       : {len(condition_counts)}")
    print(f"Unique seeds     : {len(set(seeds))}")

    print("\n=== CONDITION COUNTS ===")

    for condition, count in sorted(condition_counts.items()):
        print(f"{condition}: {count}")

    print("\n=== AUDIT RESULT ===")

    if failures:
        print(f"FAILED: {len(failures)} issue(s)")

        for failure in failures:
            print(" -", failure)

    else:
        print("ALL CHECKS PASSED")
        print("No accounting inconsistencies detected.")


if __name__ == "__main__":
    main()
