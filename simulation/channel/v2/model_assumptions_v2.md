# Physical SPAD Detector Model V2

## 1. Purpose

Develop a reproducible, event-accounting-corrected,
time-gated SPAD detector simulation for BB84 QKD
performance analysis.

## 2. Model Scope

- Discrete pulse and detector-gate simulation.
- Two-detector receiver.
- Optical signal detection.
- Poisson dark-count generation.
- Dead-time behavior.
- Delayed afterpulse release.
- BB84 basis reconciliation and QBER estimation.

The model is a gate-level approximation. It does not
claim sub-gate temporal resolution.

## 3. Event Accounting

Source attempts and registered detector avalanches
must be counted separately.

A detector can register at most one avalanche
per gate while active.

Multiple afterpulse releases in the same detector
and gate must not be counted as multiple avalanches.

If multiple source types produce attempts in the
same detector gate, record a mixed-source event.
Do not arbitrarily assign the avalanche to one source.

## 4. Required Counters

- signal_attempts
- dark_count_attempts
- afterpulse_release_attempts
- afterpulse_blocked_by_dead_time
- registered_avalanches
- signal_only_avalanches
- dark_only_avalanches
- afterpulse_only_avalanches
- mixed_source_avalanches
- double_click_events
- dead_time_lost_signal
- sifted_key_length
- errors

## 5. QBER

QBER is defined as:

QBER = errors / sifted_key_length

If sifted_key_length is zero, QBER must be undefined
(None or NaN), not zero.

Double-click handling must be explicit and configurable.
The initial comparison mode should preserve the V1
random-assignment convention.

## 6. Afterpulse Model

The configured afterpulse probability represents
the probability of scheduling a detectable afterpulse
candidate within the configured observation window.

Trap lifetimes and population weights are model
parameters, not experimentally measured detector
properties unless independently validated.

Afterpulse release attempts and registered avalanches
must remain separate metrics.

## 7. Dead-Time Convention

Dead-time behavior must be explicitly defined and
verified at the exact recovery boundary.

A detector must not register an avalanche while inactive.

## 8. Statistical Requirements

- Independent simulation replicates are the
  inferential sampling units.
- Record random seeds and full configuration.
- Report confidence intervals and effect sizes.
- Define primary outcomes before multiple-testing
  correction.
- Do not treat individual pulses as independent
  statistical replicates.

## 9. Reproducibility

The V1 reference source and original result files
must remain unchanged.

V2 outputs must be written to a separate results
directory.

Every result must record model version, seed,
configuration, and simulation duration.

## 10. Validation

Required tests include:

- Configuration validation.
- Gate acceptance probability.
- Dead-time boundary.
- Multiple releases in one gate.
- Mixed-source event accounting.
- Zero afterpulse probability.
- Zero dark-count rate.
- Zero sifted-key handling.
- Fixed-seed reproducibility.
- Counter consistency.
