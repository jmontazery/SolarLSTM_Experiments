# SolarLSTM Experiments

This repository contains the organized experiments from my Master's thesis on **solar power forecasting using LSTM-based deep learning models**.

## Study scope
The repository compares three neural network architectures:
- Single LSTM
- Bidirectional LSTM
- Stacked LSTM

These models are evaluated under different input settings and temporal granularities.

## Naming convention
The experiment names follow this format:

- H = hourly data
- Q = quarter-hourly data
- Uni = univariate input using only solar energy production
- T = solar energy production + temperature
- C = time-related features including month, season, and year

### Examples
- LSTMHUni = Single LSTM, hourly, univariate
- BiQT = Bidirectional LSTM, quarter-hourly, with temperature
- StackedHTC = Stacked LSTM, hourly, with temperature and time features

## Repository structure

SolarLSTM_Experiments/
├── experiments/
│   ├── LSTMHUni/
│   ├── LSTMQUni/
│   ├── LSTMHT/
│   ├── LSTMQT/
│   ├── LSTMHTC/
│   ├── LSTMQTC/
│   ├── BiHUni/
│   ├── BiQUni/
│   ├── BiHT/
│   ├── BiQT/
│   ├── BiHTC/
│   ├── BiQTC/
│   ├── StackedHUni/
│   ├── StackedQUni/
│   ├── StackedHT/
│   ├── StackedQT/
│   ├── StackedHTC/
│   └── StackedQTC/
├── summary/
├── common/
├── README.md
├── requirements.txt
└── .gitignore

## Inside each experiment folder
Each experiment folder contains:
- the main Python script named after the model
- config.yaml with metadata and performance information
- best_model.keras for the saved trained model
- models/ for additional saved models
- plots/ for generated figures
- results/ for experiment outputs
- slurm_job.sh if the experiment was run on a cluster

## Performance summary
A summary of all models is available in:
- summary/model_summary.md
- summary/parameter_table.md

## Goal of the repository
The goal of this repository is to provide a clear, reproducible, and structured presentation of the forecasting experiments, including code, trained models, plots, and summarized results.

## Tools and libraries
Main tools used in this project:
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
