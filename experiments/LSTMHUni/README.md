# LSTMHUni

## Overview
This folder contains the full experiment for the **LSTMHUni** model used in the solar power forecasting study.

## Model details
- **Model type:** Single LSTM
- **Time granularity:** hourly
- **Feature code:** Uni
- **Input features:** solar_energy_production
- **Parameter count:** 68608

## Performance (R²)
- **0–6 hours:** 92.12
- **6–12 hours:** 88.45
- **12–18 hours:** 89.3
- **18–24 hours:** 90.76
- **Mean R²:** 90.16
- **Std R²:** 2.2

## Contents
- `LSTMHUni.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
