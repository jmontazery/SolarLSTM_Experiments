# BiQT

## Overview
This folder contains the full experiment for the **BiQT** model used in the solar power forecasting study.

## Model details
- **Model type:** Bidirectional LSTM
- **Time granularity:** quarter-hourly
- **Feature code:** T
- **Input features:** solar_energy_production, temperature
- **Parameter count:** 138240

## Performance (R²)
- **0–6 hours:** 95.07
- **6–12 hours:** 90.18
- **12–18 hours:** 89.72
- **18–24 hours:** 90.78
- **Mean R²:** 91.44
- **Std R²:** 2.56

## Contents
- `BiQT.py` — main script containing preprocessing, training, evaluation, and plotting
- `config.yaml` — experiment metadata and performance summary
- `best_model.keras` — saved best model
- `models/` — additional saved models if applicable
- `plots/` — generated plots
- `results/` — experiment outputs
- `slurm_job.sh` — SLURM job script if used for cluster execution

## Notes
This experiment is stored independently so that each model configuration remains reproducible and easy to review.
