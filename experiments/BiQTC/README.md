# BiQTC

## Overview
This folder contains the full experiment for the **BiQTC** model used in the solar power forecasting study.

## Model details
- **Model type:** Bidirectional LSTM
- **Time granularity:** quarter-hourly
- **Feature code:** TC
- **Input features:** solar_energy_production, temperature, month, season, year
- **Parameter count:** 142336

## Performance (R²)
- **0–6 hours:** 89.66
- **6–12 hours:** 84.56
- **12–18 hours:** 87.06
- **18–24 hours:** 90.4
- **Mean R²:** 87.92
- **Std R²:** 3.12

## Contents
- `BiQTC.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
