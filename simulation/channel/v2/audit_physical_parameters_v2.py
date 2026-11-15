import csv
import math
from pathlib import Path

from simulation.channel.v2.physical_spad_detector_v2 import CONFIG


INPUT_FILE = Path(
    "results/channel/v2/"
    "physical_spad_detector_v2_multiseed_pilot.csv"
)

TOLERANCE = 1e-9


def close(actual, expected):
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE,
    )


def parse_list(value):
    return [float(x) for x in value.split(";")]


def main():

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    failures = []
    checks = 0

    def verify(condition, message):
        nonlocal checks
        checks += 1

        if not condition:
            failures.append(message)

    # ----------------------------------------------------
    # CONFIG PARAMETERS
    # ----------------------------------------------------

    attenuation = CONFIG["fiber_attenuation_db_km"]
    detector_efficiency = CONFIG["detector_efficiency"]
    mean_photon_number = CONFIG["mean_photon_number"]
    repetition_rate = CONFIG["pulse_repetition_rate_hz"]

    gate_width_ns = CONFIG["gate_width_ns"]
    gate_width_s = gate_width_ns * 1e-9

    jitter_sigma_s = (
        CONFIG["timing_jitter_sigma_ps"] * 1e-12
    )

    pulse_period_s = 1.0 / repetition_rate

    expected_gate_acceptance = (
        math.erf(
            gate_width_s / (2 * math.sqrt(2) * jitter_sigma_s)
        )
        if jitter_sigma_s > 0
        else 1.0
    )

    expected_dead_time_gates = math.ceil(
        CONFIG["dead_time_us"] * 1e-6 / pulse_period_s
    )

    expected_afterpulse_window_gates = math.floor(
        CONFIG["afterpulse_window_us"] * 1e-6 / pulse_period_s
    )

    # ----------------------------------------------------
    # ROW-BY-ROW PARAMETER AUDIT
    # ----------------------------------------------------

    for index, row in enumerate(rows, start=1):

        prefix = f"Row {index} ({row['condition_id']})"

        distance = float(row["fiber_length_km"])
        dcr = float(row["dark_count_rate_hz"])
        pulses = int(row["pulses"])

        # 1. Fiber attenuation

        expected_loss = attenuation * distance

        verify(
            close(row["fiber_loss_db"], expected_loss),
            f"{prefix}: fiber loss mismatch",
        )

        # 2. Fiber transmission

        expected_transmission = 10 ** (
            -expected_loss / 10
        )

        verify(
            close(
                row["transmission_probability"],
                expected_transmission,
            ),
            f"{prefix}: transmission probability mismatch",
        )

        # 3. Total detection efficiency

        expected_efficiency = (
            expected_transmission * detector_efficiency
        )

        verify(
            close(
                row["total_efficiency"],
                expected_efficiency,
            ),
            f"{prefix}: total efficiency mismatch",
        )

        # 4. Gate acceptance

        verify(
            close(
                row["gate_acceptance_probability"],
                expected_gate_acceptance,
            ),
            f"{prefix}: gate acceptance mismatch",
        )

        # 5. Signal detection probability
        # Model convention:
        # (1 - exp(-mu * eta)) * gate_acceptance

        expected_signal_probability = (
            1 - math.exp(
                -mean_photon_number * expected_efficiency
            )
        ) * expected_gate_acceptance

        verify(
            close(
                row["signal_detection_probability"],
                expected_signal_probability,
            ),
            f"{prefix}: signal detection probability mismatch",
        )

        # 6. Dark-count probability per detector gate

        expected_dark_probability = (
            1 - math.exp(-dcr * gate_width_s)
        )

        verify(
            close(
                row["dark_probability_per_gate"],
                expected_dark_probability,
            ),
            f"{prefix}: dark probability mismatch",
        )

        # 7. Dead-time gate conversion

        verify(
            int(row["dead_time_gates"]) == expected_dead_time_gates,
            f"{prefix}: dead-time gates mismatch",
        )

        # 8. Afterpulse observation window

        verify(
            int(row["afterpulse_window_gates"])
            == expected_afterpulse_window_gates,
            f"{prefix}: afterpulse window gates mismatch",
        )

        # 9. Duration

        expected_duration = pulses / repetition_rate

        verify(
            close(
                row["duration_seconds"],
                expected_duration,
            ),
            f"{prefix}: duration mismatch",
        )

        # 10. Detection rate per second

        expected_detection_rate = (
            int(row["detected_events"]) / expected_duration
        )

        verify(
            close(
                row["detection_rate_per_second"],
                expected_detection_rate,
            ),
            f"{prefix}: detection rate per second mismatch",
        )

        # 11. Sifted throughput

        expected_throughput = (
            int(row["sifted_key_length"]) / expected_duration
        )

        verify(
            close(
                row["sifted_bits_per_second"],
                expected_throughput,
            ),
            f"{prefix}: sifted throughput mismatch",
        )

        # 12. Fixed configuration parameters

        fixed_fields = {
            "gate_width_ns": "gate_width_ns",
            "timing_jitter_sigma_ps": "timing_jitter_sigma_ps",
            "dead_time_us": "dead_time_us",
            "afterpulse_window_us": "afterpulse_window_us",
            "afterpulse_probability_per_avalanche":
                "afterpulse_probability_per_avalanche",
        }

        for csv_field, config_field in fixed_fields.items():
            verify(
                close(row[csv_field], CONFIG[config_field]),
                f"{prefix}: {csv_field} differs from CONFIG",
            )

        verify(
            pulses == CONFIG["pulses"],
            f"{prefix}: pulse count differs from CONFIG",
        )

        # 13. Trap lifetimes and population weights

        verify(
            all(
                close(a, b)
                for a, b in zip(
                    parse_list(row["trap_lifetimes_us"]),
                    CONFIG["trap_lifetimes_us"],
                )
            )
            and len(parse_list(row["trap_lifetimes_us"]))
            == len(CONFIG["trap_lifetimes_us"]),
            f"{prefix}: trap lifetimes mismatch",
        )

        verify(
            all(
                close(a, b)
                for a, b in zip(
                    parse_list(row["trap_population_weights"]),
                    CONFIG["trap_population_weights"],
                )
            )
            and len(parse_list(row["trap_population_weights"]))
            == len(CONFIG["trap_population_weights"]),
            f"{prefix}: trap population weights mismatch",
        )

    # ----------------------------------------------------
    # REPORT
    # ----------------------------------------------------

    print("\n=== PHYSICAL PARAMETER CONSISTENCY AUDIT ===")
    print(f"Dataset rows       : {len(rows)}")
    print(f"Checks performed   : {checks}")
    print(f"Attenuation        : {attenuation} dB/km")
    print(f"Detector efficiency: {detector_efficiency}")
    print(f"Mean photon number : {mean_photon_number}")
    print(f"Repetition rate    : {repetition_rate} Hz")

    print("\n=== AUDIT RESULT ===")

    if failures:
        print(f"FAILED: {len(failures)} issue(s)")

        for failure in failures:
            print(" -", failure)

    else:
        print("ALL CHECKS PASSED")
        print("CSV parameters match the implemented CONFIG equations.")

    print(
        "\nNote: This verifies numerical consistency with the "
        "implemented model, not physical experimental accuracy."
    )


if __name__ == "__main__":
    main()
