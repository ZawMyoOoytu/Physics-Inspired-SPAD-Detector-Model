
import csv
from pathlib import Path

from realistic_bb84 import CONFIG, simulate_realistic_bb84


DISTANCES = [10, 20, 50, 100]

results = []

print("=== BB84 FIBER DISTANCE ANALYSIS ===")

for index, distance in enumerate(DISTANCES):

    config = CONFIG.copy()

    config["fiber_length_km"] = distance
    config["random_seed"] = 20261114 + index

    result = simulate_realistic_bb84(config)

    results.append(result)

    print(f"\n--- Fiber Distance: {distance} km ---")

    print(f"Fiber loss: {result['fiber_loss_db']:.3f} dB")
    print(
        f"Transmission: "
        f"{result['transmission_probability'] * 100:.4f}%"
    )
    print(
        f"Detection efficiency: "
        f"{result['total_detection_efficiency'] * 100:.4f}%"
    )
    print(f"Detected events: {result['detected_events']}")
    print(f"Dark-only clicks: {result['dark_only_clicks']}")
    print(f"Sifted key: {result['sifted_key_length']}")
    print(f"Errors: {result['errors']}")
    print(f"QBER: {result['qber'] * 100:.4f}%")


# Save CSV results
output_dir = Path("results/channel")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "distance_sweep.csv"

fieldnames = [
    "fiber_length_km",
    "fiber_loss_db",
    "transmission_probability",
    "total_detection_efficiency",
    "detected_events",
    "signal_clicks",
    "dark_only_clicks",
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


print("\n=== DISTANCE ANALYSIS SUMMARY ===")

print(
    f"{'Distance':>10} "
    f"{'Detections':>12} "
    f"{'Sifted':>10} "
    f"{'QBER':>10}"
)

for result in results:

    print(
        f"{result['fiber_length_km']:>10.1f} "
        f"{result['detected_events']:>12} "
        f"{result['sifted_key_length']:>10} "
        f"{result['qber'] * 100:>9.4f}%"
    )

print(f"\nResults saved to: {output_file}")
