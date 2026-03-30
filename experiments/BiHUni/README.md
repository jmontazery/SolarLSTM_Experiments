# BiHUni

## Overview
This folder contains the full experiment for the **BiHUni** model used in the solar power forecasting study.

## Model details
- **Model type:** Bidirectional LSTM
- **Time granularity:** hourly
- **Feature code:** Uni
- **Input features:** solar_energy_production
- **Parameter count:** 137216

## Performance (R²)
- **0–6 hours:** 91.66
- **6–12 hours:** 88.54
- **12–18 hours:** 89.3
- **18–24 hours:** 90.52
- **Mean R²:** 90.0
- **Std R²:** 1.89

## Contents
- `BiHUni.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
