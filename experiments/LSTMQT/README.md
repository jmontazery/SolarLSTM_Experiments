# LSTMQT

## Overview
This folder contains the full experiment for the **LSTMQT** model used in the solar power forecasting study.

## Model details
- **Model type:** Single LSTM
- **Time granularity:** quarter-hourly
- **Feature code:** T
- **Input features:** solar_energy_production, temperature
- **Parameter count:** 69120

## Performance (R²)
- **0–6 hours:** 93.18
- **6–12 hours:** 86.93
- **12–18 hours:** 88.78
- **18–24 hours:** 90.34
- **Mean R²:** 89.81
- **Std R²:** 3.13

## Contents
- `LSTMQT.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
