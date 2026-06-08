# Monte Carlo U-235 Criticality Simulation

A Python-based Monte Carlo simulation of neutron-induced fission in Uranium-235, designed to estimate the effective multiplication factor, **k**, and explore subcritical, critical, and supercritical regimes.

## Overview

This project models neutron behavior in a simplified U-235 system using stochastic sampling. By simulating neutron paths, fission events, absorption, and leakage, the simulation estimates how the neutron population changes across generations.

The main objective is to examine how changes in material density and geometry affect the effective multiplication factor:

- **k < 1**: subcritical system
- **k ≈ 1**: critical system
- **k > 1**: supercritical system

## Methods

- Simulated neutron-induced fission using Monte Carlo random sampling
- Estimated the effective multiplication factor, **k**, across multiple trials
- Performed parameter sweeps over material density and geometry
- Quantified uncertainty using standard error of the mean (SEM)
- Validated simulation behavior through convergence analysis with increasing neutron count

## Files

- `monte-carlo-u235-criticality.ipynb`: full analysis, figures, and interpretation
- `monte-carlo-u235-criticality.py`: executable simulation logic

## Key Takeaway

This project demonstrates how stochastic simulation methods can recover large-scale nuclear behavior from probabilistic particle interactions. It also shows how uncertainty decreases as sample size increases, reinforcing the importance of convergence testing in computational physics.
