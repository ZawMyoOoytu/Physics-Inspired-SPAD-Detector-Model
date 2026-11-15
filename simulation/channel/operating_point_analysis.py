
import csv
from pathlib import Path


# =========================================
# DETECTOR OPERATING POINT ANALYSIS
# =========================================

INPUT_FILE = Path(
    "results/channel/multi_seed_summary.csv"
)

OUTPUT_FILE = Path(
    "results/channel/operating_point_analysis.csv"
)


def load_results():

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        return list(csv.DictReader(file))


def analyze(results):

    grouped = {}

    for row in results:

        distance = int(row["fiber_length_km"])

        grouped.setdefault(
            distance,
            []
        ).append(row)

    output = []

    for distance, rows in grouped.items():

        baseline = next(
            row for row in rows
            if float(row["dead_time_us"]) == 0
        )

        baseline_key = float(
            baseline["sifted_key_length_mean"]
        )

        for row in rows:

            dead_time = float(
                row["dead_time_us"]
            )

            sifted_mean = float(
                row["sifted_key_length_mean"]
            )

            qber_mean = float(
                row["qber_mean"]
            )

            qber_std = float(
                row["qber_std"]
            )

            qber_ci_low = float(
                row["qber_ci_low"]
            )

            qber_ci_high = float(
                row["qber_ci_high"]
            )

            if baseline_key > 0:

                retention = (
                    sifted_mean
                    / baseline_key
                    * 100
                )

            else:

                retention = 0.0

            loss_percent = (
                100.0 - retention
            )

            result = {
                "fiber_length_km": distance,
                "dead_time_us": dead_time,
                "sifted_key_mean": sifted_mean,
                "throughput_retention_percent": (
                    retention
                ),
                "throughput_loss_percent": (
                    loss_percent
                ),
                "qber_mean_percent": (
                    qber_mean * 100
                ),
                "qber_std_percent": (
                    qber_std * 100
                ),
                "qber_ci_low_percent": (
                    qber_ci_low * 100
                ),
                "qber_ci_high_percent": (
                    qber_ci_high * 100
                )
            }

            output.append(result)

    return output


def save_results(results):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
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


def print_report(results):

    distances = sorted(
        set(
            row["fiber_length_km"]
            for row in results
        )
    )

    for distance in distances:

        rows = [
            row for row in results
            if row["fiber_length_km"] == distance
        ]

        rows.sort(
            key=lambda x: x["dead_time_us"]
        )

        print("\n======================================")
        print(
            f"DETECTOR OPERATING POINT: "
            f"{distance} km"
        )
        print("======================================")

        print(
            f"{'Dead(us)':>9} "
            f"{'Sifted':>10} "
            f"{'Retain(%)':>11} "
            f"{'QBER(%)':>10} "
            f"{'CI Upper':>10}"
        )

        for row in rows:

            print(
                f"{row['dead_time_us']:>9.0f} "
                f"{row['sifted_key_mean']:>10.2f} "
                f"{row['throughput_retention_percent']:>10.2f}% "
                f"{row['qber_mean_percent']:>9.4f}% "
                f"{row['qber_ci_high_percent']:>9.4f}%"
            )

        # Highest measured sifted-key throughput
        best = max(
            rows,
            key=lambda x: x["sifted_key_mean"]
        )

        print("\nHighest measured sifted-key throughput:")

        print(
            f"Dead time: "
            f"{best['dead_time_us']:.0f} us"
        )

        print(
            f"Mean sifted key: "
            f"{best['sifted_key_mean']:.2f}"
        )

        print(
            f"QBER mean: "
            f"{best['qber_mean_percent']:.4f}%"
        )

        print(
            "Note: This is a model-based throughput "
            "comparison, not a security certification."
        )


def main():

    print("=== OPERATING POINT ANALYSIS ===")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    source = load_results()

    results = analyze(source)

    print_report(results)

    save_results(results)

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
