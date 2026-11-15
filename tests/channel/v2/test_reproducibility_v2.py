import copy
import unittest

from simulation.channel.v2.physical_spad_detector_v2 import (
    CONFIG,
    simulate_physical_spad,
)


class TestSimulationReproducibility(unittest.TestCase):

    def setUp(self):
        self.config = copy.deepcopy(CONFIG)
        self.config["pulses"] = 2000

    def test_same_seed_produces_identical_results(self):

        result_1 = simulate_physical_spad(
            distance_km=10,
            dark_count_rate_hz=1000,
            config=copy.deepcopy(self.config),
            seed=20261120,
        )

        result_2 = simulate_physical_spad(
            distance_km=10,
            dark_count_rate_hz=1000,
            config=copy.deepcopy(self.config),
            seed=20261120,
        )

        self.assertEqual(result_1, result_2)

    def test_simulation_does_not_modify_input_config(self):

        original_config = copy.deepcopy(self.config)

        simulate_physical_spad(
            distance_km=10,
            dark_count_rate_hz=1000,
            config=self.config,
            seed=20261120,
        )

        self.assertEqual(self.config, original_config)


if __name__ == "__main__":
    unittest.main()
