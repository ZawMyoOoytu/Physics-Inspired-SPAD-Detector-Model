import copy
import unittest

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    calculate_timing,
)


class TestDetectorTiming(unittest.TestCase):

    def setUp(self):
        self.config = copy.deepcopy(CONFIG)

    def test_default_dead_time_is_10_gates(self):
        timing = calculate_timing(self.config)

        self.assertEqual(timing["dead_time_gates"], 10)

    def test_non_integer_dead_time_rounds_up(self):
        self.config["dead_time_us"] = 10.1

        timing = calculate_timing(self.config)

        self.assertEqual(timing["dead_time_gates"], 11)

    def test_zero_dead_time_gives_zero_gates(self):
        self.config["dead_time_us"] = 0

        timing = calculate_timing(self.config)

        self.assertEqual(timing["dead_time_gates"], 0)

    def test_afterpulse_window_is_200_gates(self):
        timing = calculate_timing(self.config)

        self.assertEqual(
            timing["afterpulse_window_gates"],
            200,
        )

    def test_gate_width_cannot_exceed_pulse_period(self):
        self.config["gate_width_ns"] = 2000

        with self.assertRaises(ValueError):
            calculate_timing(self.config)

    def test_afterpulse_window_must_cover_one_gate(self):
        self.config["afterpulse_window_us"] = 0

        with self.assertRaises(ValueError):
            calculate_timing(self.config)


if __name__ == "__main__":
    unittest.main()
