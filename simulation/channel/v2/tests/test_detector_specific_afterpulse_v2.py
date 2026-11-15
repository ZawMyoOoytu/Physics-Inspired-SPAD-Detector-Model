import pytest

from simulation.channel.v2.event_accounting import (
    resolve_detector_gate,
)


def test_no_attempts_produce_no_event():
    result = resolve_detector_gate(detector_active=True)
    assert result["registered"] is False
    assert result["source_class"] == "no_event"
    assert result["registered_avalanches"] == 0


def test_inactive_detector_with_no_attempts_produces_no_event():
    result = resolve_detector_gate(detector_active=False)
    assert result["source_class"] == "no_event"
    assert result["afterpulse_blocked_by_dead_time"] == 0


def test_single_afterpulse_release_registers_one_avalanche():
    result = resolve_detector_gate(
        detector_active=True,
        afterpulse_release_count=1,
    )
    assert result["registered"] is True
    assert result["source_class"] == "afterpulse_only"
    assert result["registered_avalanches"] == 1
    assert result["afterpulse_release_attempts"] == 1
    assert result["afterpulse_blocked_by_dead_time"] == 0


def test_multiple_afterpulse_releases_register_at_most_one_avalanche():
    result = resolve_detector_gate(
        detector_active=True,
        afterpulse_release_count=5,
    )
    assert result["registered"] is True
    assert result["source_class"] == "afterpulse_only"
    assert result["registered_avalanches"] == 1
    assert result["afterpulse_release_attempts"] == 5


def test_inactive_detector_blocks_all_afterpulse_releases():
    result = resolve_detector_gate(
        detector_active=False,
        afterpulse_release_count=4,
    )
    assert result["registered"] is False
    assert result["source_class"] == "blocked"
    assert result["registered_avalanches"] == 0
    assert result["afterpulse_release_attempts"] == 4
    assert result["afterpulse_blocked_by_dead_time"] == 4


def test_active_detector_does_not_block_afterpulse_releases():
    result = resolve_detector_gate(
        detector_active=True,
        afterpulse_release_count=3,
    )
    assert result["afterpulse_release_attempts"] == 3
    assert result["afterpulse_blocked_by_dead_time"] == 0


def test_signal_and_afterpulse_are_mixed_source():
    result = resolve_detector_gate(
        detector_active=True,
        signal_attempt=True,
        afterpulse_release_count=1,
    )
    assert result["source_class"] == "mixed_source"
    assert result["registered_avalanches"] == 1


def test_dark_and_afterpulse_are_mixed_source():
    result = resolve_detector_gate(
        detector_active=True,
        dark_attempt=True,
        afterpulse_release_count=2,
    )
    assert result["source_class"] == "mixed_source"
    assert result["afterpulse_release_attempts"] == 2


def test_all_three_sources_are_mixed_source():
    result = resolve_detector_gate(
        detector_active=True,
        signal_attempt=True,
        dark_attempt=True,
        afterpulse_release_count=3,
    )
    assert result["source_class"] == "mixed_source"
    assert result["registered_avalanches"] == 1


def test_signal_only_classification():
    result = resolve_detector_gate(
        detector_active=True,
        signal_attempt=True,
    )
    assert result["source_class"] == "signal_only"
    assert result["registered_avalanches"] == 1


def test_dark_only_classification():
    result = resolve_detector_gate(
        detector_active=True,
        dark_attempt=True,
    )
    assert result["source_class"] == "dark_only"
    assert result["registered_avalanches"] == 1


def test_negative_afterpulse_release_count_is_rejected():
    with pytest.raises(ValueError):
        resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=-1,
        )


@pytest.mark.parametrize("invalid_count", [1.5, "2", True, None])
def test_invalid_afterpulse_release_count_type_is_rejected(invalid_count):
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            afterpulse_release_count=invalid_count,
        )


@pytest.mark.parametrize("invalid_value", [1, None, "True", []])
def test_invalid_detector_active_type_is_rejected(invalid_value):
    with pytest.raises(TypeError):
        resolve_detector_gate(detector_active=invalid_value)


@pytest.mark.parametrize("invalid_value", [1, None, "True"])
def test_invalid_signal_attempt_type_is_rejected(invalid_value):
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            signal_attempt=invalid_value,
        )


@pytest.mark.parametrize("invalid_value", [1, None, "True"])
def test_invalid_dark_attempt_type_is_rejected(invalid_value):
    with pytest.raises(TypeError):
        resolve_detector_gate(
            detector_active=True,
            dark_attempt=invalid_value,
        )
