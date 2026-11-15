"""Detector-gate event accounting for SPAD model V2."""


def resolve_detector_gate(
    *,
    detector_active: bool,
    signal_attempt: bool = False,
    dark_attempt: bool = False,
    afterpulse_release_count: int = 0,
) -> dict:
    """
    Resolve source attempts within one detector gate.

    An active detector registers at most one avalanche per gate,
    regardless of the number of source attempts.

    Multiple afterpulse releases in the same detector gate
    represent one afterpulse source class, not multiple avalanches.
    """

    if not isinstance(detector_active, bool):
        raise TypeError("detector_active must be a boolean.")

    if not isinstance(signal_attempt, bool):
        raise TypeError("signal_attempt must be a boolean.")

    if not isinstance(dark_attempt, bool):
        raise TypeError("dark_attempt must be a boolean.")

    if type(afterpulse_release_count) is not int:
        raise TypeError("afterpulse_release_count must be an integer.")

    if afterpulse_release_count < 0:
        raise ValueError("afterpulse_release_count cannot be negative.")

    sources = []

    if signal_attempt:
        sources.append("signal")

    if dark_attempt:
        sources.append("dark")

    if afterpulse_release_count > 0:
        sources.append("afterpulse")

    if not sources:
        return {
            "registered": False,
            "source_class": "no_event",
            "registered_avalanches": 0,
            "signal_attempt": False,
            "dark_attempt": False,
            "afterpulse_release_attempts": 0,
            "afterpulse_blocked_by_dead_time": 0,
        }

    if not detector_active:
        return {
            "registered": False,
            "source_class": "blocked",
            "registered_avalanches": 0,
            "signal_attempt": signal_attempt,
            "dark_attempt": dark_attempt,
            "afterpulse_release_attempts": afterpulse_release_count,
            "afterpulse_blocked_by_dead_time": afterpulse_release_count,
        }

    if len(sources) == 1:
        source_class = f"{sources[0]}_only"
    else:
        source_class = "mixed_source"

    return {
        "registered": True,
        "source_class": source_class,
        "registered_avalanches": 1,
        "signal_attempt": signal_attempt,
        "dark_attempt": dark_attempt,
        "afterpulse_release_attempts": afterpulse_release_count,
        "afterpulse_blocked_by_dead_time": 0,
    }
