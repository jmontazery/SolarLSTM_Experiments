# BiHT

## Overview
This folder contains the full experiment for the **BiHT** model used in the solar power forecasting study.

## Model details
- **Model type:** Bidirectional LSTM
- **Time granularity:** hourly
- **Feature code:** T
- **Input features:** solar_energy_production, temperature
- **Parameter count:** 138240

## Performance (R²)
- **0–6 hours:** 93.76
- **6–12 hours:** 90.74
- **12–18 hours:** 90.15
- **18–24 hours:** 90.89
- **Mean R²:** 91.38
- **Std R²:** 1.85

## Contents
- `BiHT.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
