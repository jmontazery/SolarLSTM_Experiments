# LSTMHTC

## Overview
This folder contains the full experiment for the **LSTMHTC** model used in the solar power forecasting study.

## Model details
- **Model type:** Single LSTM
- **Time granularity:** hourly
- **Feature code:** TC
- **Input features:** solar_energy_production, temperature, month, season, year
- **Parameter count:** 71168

## Performance (R²)
- **0–6 hours:** 93.34
- **6–12 hours:** 90.82
- **12–18 hours:** 91.0
- **18–24 hours:** 90.84
- **Mean R²:** 91.5
- **Std R²:** 1.65

## Contents
- `LSTMHTC.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
