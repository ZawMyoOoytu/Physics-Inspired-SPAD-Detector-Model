import unittest

from simulation.channel.v2.event_accounting import (
    resolve_detector_gate,
)


class TestDetectorGateAccounting(unittest.TestCase):

    def test_no_event(self):
        result = resolve_detector_gate(
            detector_active=True,
        )

        self.assertFalse(result["registered"])
        self.assertEqual(result["source_class"], "no_event")
        self.assertEqual(result["registered_avalanches"], 0)

    def test_signal_only(self):
        result = resolve_detector_gate(
            detector_active=True,
            signal_attempt=True,
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["source_class"], "signal_only")
        self.assertEqual(result["registered_avalanches"], 1)

    def test_dark_only(self):
        result = resolve_detector_gate(
            detector_active=True,
            dark_attempt=True,
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["source_class"], "dark_only")
        self.assertEqual(result["registered_avalanches"], 1)

    def test_afterpulse_only(self):
        result = resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=1,
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["source_class"], "afterpulse_only")
        self.assertEqual(result["afterpulse_release_attempts"], 1)

    def test_multiple_afterpulse_releases_one_avalanche(self):
        result = resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=3,
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["source_class"], "afterpulse_only")
        self.assertEqual(result["registered_avalanches"], 1)
        self.assertEqual(result["afterpulse_release_attempts"], 3)

    def test_mixed_signal_and_dark(self):
        result = resolve_detector_gate(
            detector_active=True,
            signal_attempt=True,
            dark_attempt=True,
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["source_class"], "mixed_source")
        self.assertEqual(result["registered_avalanches"], 1)

    def test_all_three_sources_mixed(self):
        result = resolve_detector_gate(
            detector_active=True,
            signal_attempt=True,
            dark_attempt=True,
            afterpulse_release_count=2,
        )

        self.assertTrue(result["registered"])
        self.assertEqual(result["source_class"], "mixed_source")
        self.assertEqual(result["registered_avalanches"], 1)
        self.assertEqual(result["afterpulse_release_attempts"], 2)

    def test_inactive_detector_blocks_event(self):
        result = resolve_detector_gate(
            detector_active=False,
            signal_attempt=True,
            dark_attempt=True,
            afterpulse_release_count=2,
        )

        self.assertFalse(result["registered"])
        self.assertEqual(result["source_class"], "blocked")
        self.assertEqual(result["registered_avalanches"], 0)
        self.assertEqual(result["afterpulse_blocked_by_dead_time"], 2)

    def test_no_event_on_inactive_detector(self):
        result = resolve_detector_gate(
            detector_active=False,
        )

        self.assertFalse(result["registered"])
        self.assertEqual(result["source_class"], "no_event")

    def test_negative_afterpulse_count_rejected(self):
        with self.assertRaises(ValueError):
            resolve_detector_gate(
                detector_active=True,
                afterpulse_release_count=-1,
            )

    def test_boolean_afterpulse_count_rejected(self):
        with self.assertRaises(TypeError):
            resolve_detector_gate(
                detector_active=True,
                afterpulse_release_count=True,
            )

    def test_invalid_detector_active_type_rejected(self):
        with self.assertRaises(TypeError):
            resolve_detector_gate(
                detector_active=1,
            )


if __name__ == "__main__":
    unittest.main()
