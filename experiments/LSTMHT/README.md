# LSTMHT

## Overview
This folder contains the full experiment for the **LSTMHT** model used in the solar power forecasting study.

## Model details
- **Model type:** Single LSTM
- **Time granularity:** hourly
- **Feature code:** T
- **Input features:** solar_energy_production, temperature
- **Parameter count:** 69120

## Performance (R²)
- **0–6 hours:** 93.17
- **6–12 hours:** 89.58
- **12–18 hours:** 89.35
- **18–24 hours:** 90.56
- **Mean R²:** 90.66
- **Std R²:** 2.08

## Contents
- `LSTMHT.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
