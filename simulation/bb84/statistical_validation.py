
import random
import statistics
from math import sqrt

from bb84_simulator import simulate_bb84

N = 500_000
REPEATS = 30
THEORETICAL_QBER = 0.25

results = []

print("=== BB84 Statistical Validation ===")

for seed in range(REPEATS):
    random.seed(seed + 20261114)

    result = simulate_bb84(n=N, eve=True)

    results.append(result["qber"])

    print(
        f"Run {seed + 1:02d}: "
        f"QBER = {result['qber'] * 100:.4f}%"
    )

mean_qber = statistics.mean(results)
std_qber = statistics.stdev(results)

standard_error = std_qber / sqrt(REPEATS)

ci_low = mean_qber - 1.96 * standard_error
ci_high = mean_qber + 1.96 * standard_error

print("\n=== Summary ===")
print(f"Experiments: {REPEATS}")
print(f"Photons per experiment: {N}")
print(f"Mean QBER: {mean_qber * 100:.4f}%")
print(f"Standard Deviation: {std_qber * 100:.4f}%")
print(f"95% CI: [{ci_low * 100:.4f}%, {ci_high * 100:.4f}%]")
print(f"Theoretical QBER: {THEORETICAL_QBER * 100:.2f}%")

if ci_low <= THEORETICAL_QBER <= ci_high:
    print("Result: Theoretical value lies within the CI.")
else:
    print("Result: Theoretical value lies outside the CI.")
