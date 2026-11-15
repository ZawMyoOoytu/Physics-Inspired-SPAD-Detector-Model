# Physics-Inspired SPAD Detector Model

**A Physics-Inspired Simulation and Validation Framework for Single-Photon Avalanche Diode (SPAD) Detectors in Quantum Communication Systems**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-SPAD%20V2-purple.svg)]()
[![Tests](https://img.shields.io/badge/Tests-87%20Passed-success.svg)]()
[![Research](https://img.shields.io/badge/Project-Research%20Software-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Under%20Development-yellow.svg)]()

## 1. Overview

The **Physics-Inspired SPAD Detector Model** is a research-oriented simulation framework designed to investigate the behaviour of Single-Photon Avalanche Diode (SPAD) detectors under operating conditions relevant to quantum communication and Quantum Key Distribution (QKD) systems.

The project focuses on modelling detector-level physical and stochastic processes that influence photon detection, including:

- Optical channel attenuation.
- Photon detection probability.
- Detector efficiency.
- Timing jitter and detection gate acceptance.
- Dark count events.
- Detector dead-time behaviour.
- Afterpulse generation and delayed release.
- Multi-source detector event interactions.
- Detector-level avalanche registration.

The framework aims to provide a reproducible computational environment for studying how detector characteristics influence detection statistics and quantum communication system performance.

The current implementation includes a Version 2 (V2) simulation architecture with dedicated event-accounting logic, temporal afterpulse handling, automated tests, and simulation output datasets.

## 2. Research Objectives

The primary objectives are:

1. Develop a physics-inspired computational model of SPAD detector behaviour.
2. Investigate the influence of detector non-idealities on photon detection statistics.
3. Model delayed afterpulse events using trap-lifetime-based probability distributions.
4. Implement detector-specific event scheduling and accounting.
5. Evaluate detector behaviour under different optical channel distances and dark count conditions.
6. Establish a reproducible simulation and software-validation framework.
7. Provide a foundation for future comparison with experimentally characterized SPAD devices.

## 3. Model Architecture

The V2 implementation is organized around several interacting components.

### 3.1 Optical Channel Model

The optical channel model calculates transmission losses as a function of fibre distance and attenuation.

The transmission coefficient is represented by:

\[
T(d)=10^{-\alpha d/10}
\]

where:

- \(T(d)\) is the optical transmission.
- \(\alpha\) is the fibre attenuation in dB/km.
- \(d\) is the transmission distance in km.

The model combines channel transmission with detector efficiency and mean photon number to estimate the optical detection probability.

### 3.2 Timing and Detection Gate Model

The timing model incorporates:

- Pulse repetition rate.
- Detection gate width.
- Timing jitter.
- Detector dead time.
- Afterpulse observation window.

Timing jitter is represented through a Gaussian acceptance approximation.

The resulting gate acceptance contributes to the effective signal detection probability.

### 3.3 Dark Count Model

Dark counts are modelled as stochastic detector events.

The probability of at least one dark-count event within a detection gate is calculated using:

\[
P_{\mathrm{dark}}=1-e^{-R_{\mathrm{dark}}\Delta t}
\]

where:

- \(R_{\mathrm{dark}}\) is the dark count rate.
- \(\Delta t\) is the detection gate width.

The implementation interprets the configured dark count rate per detector.

### 3.4 Afterpulse Model

Afterpulsing is represented using delayed stochastic release events associated with illustrative trap-lifetime distributions.

The model incorporates:

- Multiple trap lifetimes.
- Trap population weights.
- Exponential release distributions.
- Discrete detection-gate delay sampling.
- Detector-specific pending afterpulse queues.
- Delayed release attempts.
- Dead-time-related blocking.

The current implementation uses a weighted mixture of exponential trap-release distributions.

This approach provides a computational framework for investigating temporal afterpulse behaviour. The configured trap parameters are illustrative and are not presented as experimentally measured device characteristics.

### 3.5 Detector Event Accounting

The V2 architecture includes a dedicated event-accounting component.

The event resolver evaluates signal, dark-count, and afterpulse attempts within an individual detector gate.

It enforces the rule that an active detector can register at most one avalanche per gate, even when multiple event sources are present.

The implementation distinguishes:

- Signal-only events.
- Dark-only events.
- Afterpulse-only events.
- Mixed-source events.
- Events blocked by detector dead time.
- Gates without registered events.

This separation improves event traceability and supports systematic validation.

## 4. V2 Event Accounting and Temporal Consistency

A central feature of the V2 architecture is explicit accounting of delayed afterpulse events.

The following accounting identity is used:

\[
N_{\mathrm{scheduled}}
=
N_{\mathrm{release\ attempts}}
+
N_{\mathrm{pending,end}}
\]

where:

- \(N_{\mathrm{scheduled}}\) represents the total number of scheduled afterpulse candidates.
- \(N_{\mathrm{release\ attempts}}\) represents the number of candidates whose release was attempted during simulation.
- \(N_{\mathrm{pending,end}}\) represents candidates still pending when the simulation terminates.

This identity provides an internal consistency check for the afterpulse event queue.

It is important to distinguish an afterpulse release attempt from a registered avalanche: a release attempt may be blocked when the detector is inactive because of dead time.

## 5. Simulation Configuration

The V2 model includes configurable parameters for optical transmission, detector characteristics, timing behaviour, and afterpulse modelling.

The current baseline configuration includes:

| Parameter | Baseline value |
|---|---:|
| Number of pulses | 200,000 |
| Fibre attenuation | 0.20 dB/km |
| Detector efficiency | 20% |
| Mean photon number | 0.50 |
| Pulse repetition rate | 1 MHz |
| Detection gate width | 1 ns |
| Timing jitter standard deviation | 200 ps |
| Detector dead time | 10 µs |
| Baseline dark count rate | 1,000 Hz |
| Afterpulse probability per avalanche | 0.01 |
| Trap lifetimes | 5 µs, 50 µs |
| Trap population weights | 0.70, 0.30 |
| Afterpulse observation window | 200 µs |
| Misalignment probability | 0.01 |

The simulation also defines parameter sweeps over:

- **Fibre distances:** 10 km, 50 km, and 100 km.
- **Dark count rates:** 100 Hz, 1,000 Hz, and 10,000 Hz.

These values describe the current simulation configuration, not universal SPAD device specifications.

## 6. Repository Structure

```text
Physics-Inspired-SPAD-Detector-Model/
│
├── simulation/
│   └── channel/
│       └── v2/
│           ├── physical_spad_detector_v2.py
│           ├── event_accounting.py
│           └── tests/
│
├── tests/
│   └── channel/
│       └── v2/
│
├── results/
│   └── channel/
│       └── v2/
│           └── *.csv
│
├── .gitignore
└── README.md
```

The repository is organized to separate simulation implementation, automated validation, and generated research datasets.

## 7. Installation and Requirements

### Prerequisites

- Python 3.x
- pip
- pytest

Clone the repository:

```bash
git clone https://github.com/ZawMyoOoytu/Physics-Inspired-SPAD-Detector-Model.git
```

Navigate to the project directory:

```bash
cd Physics-Inspired-SPAD-Detector-Model
```

Install the testing dependency:

```bash
python -m pip install pytest
```

## 8. Running the Simulation

The main V2 simulation implementation is located at:

```text
simulation/channel/v2/physical_spad_detector_v2.py
```

Run the simulation using the execution interface provided by the implementation or its associated research scripts.

The simulation configuration can be modified within the model configuration to investigate different detector and channel conditions.

For reproducible experiments, retain the random seed and record any changes to model parameters.

## 9. Automated Testing and Validation

The repository includes automated tests covering detector event accounting, afterpulse handling, temporal behaviour, and integration logic.

The complete test suite was executed using:

```bash
python -m pytest -q
```

**Latest recorded test result:**

```text
87 passed in 6.16s
```

The test suite is divided across:

- V2 simulation-specific tests.
- Repository-level channel validation tests.

The tests provide software-level verification of implemented logic and expected behaviour.

They do not independently establish the physical accuracy of the model or validate it against laboratory measurements.

## 10. Research Data and Reproducibility

Simulation outputs are organized under:

```text
results/channel/v2/
```

The repository includes CSV datasets generated for simulation and validation workflows.

These datasets support:

- Result inspection.
- Statistical analysis.
- Comparison across simulation conditions.
- Reproducibility checks.
- Future research extensions.

For reproducible analysis, users should document:

1. Model version.
2. Simulation configuration.
3. Random seed.
4. Channel distance.
5. Detector dark count rate.
6. Relevant output dataset.
7. Statistical analysis procedure.

## 11. Scientific Scope and Limitations

This project is a physics-inspired computational model under active research development.

Several limitations should be considered when interpreting its results.

**Model assumptions**

- Detector processes are represented through simplified stochastic models.
- Trap lifetimes and population weights are illustrative parameters.
- The current implementation does not establish a device-specific microscopic trap characterization.
- Simulation outcomes depend on configured parameters and modelling assumptions.

**Experimental validation**

The current automated test results demonstrate software-level consistency and implementation checks.

They should not be interpreted as:

- Experimental validation against a physical SPAD detector.
- Hardware calibration.
- A complete microscopic model of avalanche multiplication and carrier trapping.
- Proof of device-independent detector performance.

Further work is required to compare simulated behaviour with independently obtained experimental measurements and published device-characterization data.

## 12. Future Research Directions

Planned research directions include:

1. **Physical Parameter Calibration**  
   Compare model parameters with experimentally characterized SPAD detector specifications.

2. **Advanced Afterpulse Characterization**  
   Investigate multi-component trap-release distributions and detector-specific temporal correlations.

3. **Detector Performance Analysis**  
   Study the effects of dead time, dark counts, timing jitter, and afterpulsing on detection statistics.

4. **Statistical Validation**  
   Extend multi-seed simulations, uncertainty estimation, and statistical comparisons.

5. **QKD System Integration**  
   Investigate the influence of detector non-idealities on BB84 and other quantum communication protocols.

6. **Independent Experimental Comparison**  
   Establish validation procedures using external experimental datasets or laboratory measurements.

## 13. Applications

Potential research applications include:

- Quantum Key Distribution (QKD).
- BB84 quantum communication systems.
- Single-photon detection research.
- Optical communication channel modelling.
- SPAD detector performance analysis.
- Quantum communication system simulation.
- Detector non-ideality and noise analysis.

## 14. Citation and Research Use

If this software contributes to academic research, please cite the repository and the relevant version or commit used in the study.

A formal software citation and associated research publication can be added when a citable release or manuscript becomes available.

Repository:

https://github.com/ZawMyoOoytu/Physics-Inspired-SPAD-Detector-Model

## 15. Author

**Zaw Myo Oo**  
Freelance Tech Worker | Independent Researcher  
Myanmar

Research Interests:

- Quantum Communication
- Quantum Key Distribution
- Quantum Cryptography
- Single-Photon Detector Modelling
- Quantum Technology
- Electronics and Telecommunication

## 16. License

No license is currently specified in this README. Unless a license file is added to the repository, reuse and redistribution permissions should not be assumed.

---

**Project:** Physics-Inspired SPAD Detector Model  
**Model Version:** V2  
**Research Domain:** Quantum Communication and Single-Photon Detector Modelling  
**Repository:** [GitHub](https://github.com/ZawMyoOoytu/Physics-Inspired-SPAD-Detector-Model)
