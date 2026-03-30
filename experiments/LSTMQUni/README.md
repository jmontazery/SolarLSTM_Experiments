# LSTMQUni

## Overview
This folder contains the full experiment for the **LSTMQUni** model used in the solar power forecasting study.

## Model details
- **Model type:** Single LSTM
- **Time granularity:** quarter-hourly
- **Feature code:** Uni
- **Input features:** solar_energy_production
- **Parameter count:** 68608

## Performance (R²)
- **0–6 hours:** 86.35
- **6–12 hours:** 78.77
- **12–18 hours:** 83.43
- **18–24 hours:** 89.04
- **Mean R²:** 84.4
- **Std R²:** 5.16

## Contents
- `LSTMQUni.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
