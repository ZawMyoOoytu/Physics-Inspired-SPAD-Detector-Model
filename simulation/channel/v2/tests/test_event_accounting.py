import pytest

from simulation.channel.v2.event_accounting import resolve_detector_gate


def test_signal_only_avalanche():
    result = resolve_detector_gate(
        detector_active=True,
        signal_attempt=True,
    )

    assert result["registered"] is True
    assert result["source_class"] == "signal_only"
    assert result["registered_avalanches"] == 1


def test_dark_only_avalanche():
    result = resolve_detector_gate(
        detector_active=True,
        dark_attempt=True,
    )

    assert result["source_class"] == "dark_only"
    assert result["registered_avalanches"] == 1


def test_afterpulse_only_avalanche():
    result = resolve_detector_gate(
        detector_active=True,
        afterpulse_release_count=3,
    )

    assert result["source_class"] == "afterpulse_only"
    assert result["afterpulse_release_attempts"] == 3
    assert result["registered_avalanches"] == 1


def test_multiple_sources_are_mixed():
    result = resolve_detector_gate(
        detector_active=True,
        signal_attempt=True,
        dark_attempt=True,
        afterpulse_release_count=2,
    )

    assert result["source_class"] == "mixed_source"
    assert result["registered_avalanches"] == 1


def test_inactive_detector_blocks_all_sources():
    result = resolve_detector_gate(
        detector_active=False,
        signal_attempt=True,
        dark_attempt=True,
        afterpulse_release_count=3,
    )

    assert result["registered"] is False
    assert result["registered_avalanches"] == 0
    assert result["afterpulse_blocked_by_dead_time"] == 3


def test_no_source_attempt_means_no_event():
    result = resolve_detector_gate(detector_active=True)

    assert result["registered"] is False
    assert result["source_class"] == "no_event"
    assert result["registered_avalanches"] == 0


def test_negative_afterpulse_count_is_rejected():
    with pytest.raises(ValueError):
        resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=-1,
        )


def test_boolean_afterpulse_count_is_rejected():
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=True,
        )

def test_inactive_detector_without_sources_is_no_event():
    result = resolve_detector_gate(
        detector_active=False,
    )

    assert result["registered"] is False
    assert result["source_class"] == "no_event"
    assert result["afterpulse_blocked_by_dead_time"] == 0


def test_invalid_detector_active_type_is_rejected():
    with pytest.raises(TypeError):
        resolve_detector_gate(detector_active=1)


def test_invalid_signal_attempt_type_is_rejected():
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            signal_attempt=1,
        )
def test_invalid_dark_attempt_type_is_rejected():
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            dark_attempt=1,
        )


def test_float_afterpulse_count_is_rejected():
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=1.5,
        )


def test_string_afterpulse_count_is_rejected():
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count="2",
        )