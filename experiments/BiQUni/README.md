# BiQUni

## Overview
This folder contains the full experiment for the **BiQUni** model used in the solar power forecasting study.

## Model details
- **Model type:** Bidirectional LSTM
- **Time granularity:** quarter-hourly
- **Feature code:** Uni
- **Input features:** solar_energy_production
- **Parameter count:** 137216

## Performance (R²)
- **0–6 hours:** 90.25
- **6–12 hours:** 84.66
- **12–18 hours:** 85.98
- **18–24 hours:** 89.7
- **Mean R²:** 87.65
- **Std R²:** 3.19

## Contents
- `BiQUni.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
