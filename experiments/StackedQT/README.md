# StackedQT

## Overview
This folder contains the full experiment for the **StackedQT** model used in the solar power forecasting study.

## Model details
- **Model type:** Stacked LSTM
- **Time granularity:** quarter-hourly
- **Feature code:** T
- **Input features:** solar_energy_production, temperature
- **Parameter count:** 131584

## Performance (R²)
- **0–6 hours:** 83.74
- **6–12 hours:** 82.11
- **12–18 hours:** 87.4
- **18–24 hours:** 89.76
- **Mean R²:** 85.75
- **Std R²:** 4.14

## Contents
- `StackedQT.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
