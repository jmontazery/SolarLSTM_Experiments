from pathlib import Path

# ============================================
# EDIT THIS PATH
# ============================================
ROOT = Path(r"C:\Users\Jamileh\Documents\SolarLSTM_Experiments")

# ============================================
# YOUR 18 MODELS
# ============================================
MODELS = [
    "LSTMHUni",
    "LSTMQUni",
    "LSTMHT",
    "LSTMQT",
    "LSTMHTC",
    "LSTMQTC",
    "BiHUni",
    "BiQUni",
    "BiHT",
    "BiQT",
    "BiHTC",
    "BiQTC",
    "StackedHUni",
    "StackedQUni",
    "StackedHT",
    "StackedQT",
    "StackedHTC",
    "StackedQTC",
]

# ============================================
# PERFORMANCE TABLE
# ============================================
MODEL_METRICS = {
    "LSTMHUni":    {"0_6": 92.12, "6_12": 88.45, "12_18": 89.30, "18_24": 90.76, "mean_r2": 90.16, "std_r2": 2.20},
    "LSTMQUni":    {"0_6": 86.35, "6_12": 78.77, "12_18": 83.43, "18_24": 89.04, "mean_r2": 84.40, "std_r2": 5.16},
    "LSTMHT":      {"0_6": 93.17, "6_12": 89.58, "12_18": 89.35, "18_24": 90.56, "mean_r2": 90.66, "std_r2": 2.08},
    "LSTMQT":      {"0_6": 93.18, "6_12": 86.93, "12_18": 88.78, "18_24": 90.34, "mean_r2": 89.81, "std_r2": 3.13},
    "LSTMHTC":     {"0_6": 93.34, "6_12": 90.82, "12_18": 91.00, "18_24": 90.84, "mean_r2": 91.50, "std_r2": 1.65},
    "LSTMQTC":     {"0_6": 83.77, "6_12": 78.96, "12_18": 86.01, "18_24": 89.96, "mean_r2": 84.68, "std_r2": 5.43},
    "BiHUni":      {"0_6": 91.66, "6_12": 88.54, "12_18": 89.30, "18_24": 90.52, "mean_r2": 90.00, "std_r2": 1.89},
    "BiQUni":      {"0_6": 90.25, "6_12": 84.66, "12_18": 85.98, "18_24": 89.70, "mean_r2": 87.65, "std_r2": 3.19},
    "BiHT":        {"0_6": 93.76, "6_12": 90.74, "12_18": 90.15, "18_24": 90.89, "mean_r2": 91.38, "std_r2": 1.85},
    "BiQT":        {"0_6": 95.07, "6_12": 90.18, "12_18": 89.72, "18_24": 90.78, "mean_r2": 91.44, "std_r2": 2.56},
    "BiHTC":       {"0_6": 92.70, "6_12": 90.44, "12_18": 90.17, "18_24": 90.62, "mean_r2": 90.98, "std_r2": 1.50},
    "BiQTC":       {"0_6": 89.66, "6_12": 84.56, "12_18": 87.06, "18_24": 90.40, "mean_r2": 87.92, "std_r2": 3.12},
    "StackedHUni": {"0_6": 92.16, "6_12": 88.89, "12_18": 89.73, "18_24": 90.48, "mean_r2": 90.32, "std_r2": 1.81},
    "StackedQUni": {"0_6": 86.81, "6_12": 83.58, "12_18": 86.33, "18_24": 89.13, "mean_r2": 86.46, "std_r2": 3.06},
    "StackedHT":   {"0_6": 92.87, "6_12": 89.81, "12_18": 90.00, "18_24": 90.58, "mean_r2": 90.82, "std_r2": 1.78},
    "StackedQT":   {"0_6": 83.74, "6_12": 82.11, "12_18": 87.40, "18_24": 89.76, "mean_r2": 85.75, "std_r2": 4.14},
    "StackedHTC":  {"0_6": 93.00, "6_12": 91.41, "12_18": 91.83, "18_24": 91.91, "mean_r2": 92.03, "std_r2": 1.27},
    "StackedQTC":  {"0_6": 86.81, "6_12": 83.58, "12_18": 86.33, "18_24": 89.13, "mean_r2": 86.46, "std_r2": 3.06},
}

# ============================================
# PARAMETER COUNTS
# ============================================
PARAMETER_COUNTS = {
    ("single", "Uni"): 68608,
    ("single", "T"): 69120,
    ("single", "TC"): 71168,
    ("bidirectional", "Uni"): 137216,
    ("bidirectional", "T"): 138240,
    ("bidirectional", "TC"): 142336,
    ("stacked", "Uni"): 131584,
    ("stacked", "T"): 131584,
    ("stacked", "TC"): 131584,
}

def infer_model_type(model_name: str) -> str:
    if model_name.startswith("Bi"):
        return "Bidirectional LSTM"
    if model_name.startswith("Stacked"):
        return "Stacked LSTM"
    return "Single LSTM"

def infer_architecture_key(model_name: str) -> str:
    if model_name.startswith("Bi"):
        return "bidirectional"
    if model_name.startswith("Stacked"):
        return "stacked"
    return "single"

def infer_granularity(model_name: str) -> str:
    return "quarter_hourly" if "Q" in model_name else "hourly"

def infer_feature_code(model_name: str) -> str:
    if model_name.endswith("Uni"):
        return "Uni"
    if model_name.endswith("TC"):
        return "TC"
    if model_name.endswith("T"):
        return "T"
    return "unknown"

def infer_input_features(model_name: str) -> list[str]:
    code = infer_feature_code(model_name)
    if code == "Uni":
        return ["solar_energy_production"]
    if code == "T":
        return ["solar_energy_production", "temperature"]
    if code == "TC":
        return ["solar_energy_production", "temperature", "month", "season", "year"]
    return []

def infer_parameter_count(model_name: str):
    return PARAMETER_COUNTS.get((infer_architecture_key(model_name), infer_feature_code(model_name)))

def to_yaml(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"

def create_file_if_missing(path: Path, content: str = ""):
    if not path.exists():
        path.write_text(content, encoding="utf-8")

def create_root_files():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments").mkdir(exist_ok=True)
    (ROOT / "common").mkdir(exist_ok=True)
    (ROOT / "summary").mkdir(exist_ok=True)

    create_file_if_missing(
        ROOT / "README.md",
        """# SolarLSTM Experiments

This repository contains organized experiment folders for solar forecasting using:
- Single LSTM
- Bidirectional LSTM
- Stacked LSTM

## Naming convention
- H = hourly
- Q = quarter-hourly
- Uni = univariate
- T = temperature
- C = time features: month, season, year

## Structure
- experiments/
- common/
- summary/
"""
    )

    create_file_if_missing(
        ROOT / "requirements.txt",
        """numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
pyyaml
"""
    )

    create_file_if_missing(
        ROOT / ".gitignore",
        """__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
.env
.DS_Store
Thumbs.db
"""
    )

    create_file_if_missing(ROOT / "common" / "preprocessing.py", "")
    create_file_if_missing(ROOT / "common" / "metrics.py", "")
    create_file_if_missing(ROOT / "common" / "plotting.py", "")
    create_file_if_missing(ROOT / "common" / "utils.py", "")

def create_summary_files():
    summary_text = [
        "# Model Summary",
        "",
        "| Model | Mean R2 | Std R2 |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        m = MODEL_METRICS[model]
        summary_text.append(f"| {model} | {m['mean_r2']} | {m['std_r2']} |")

    create_file_if_missing(
        ROOT / "summary" / "model_summary.md",
        "\n".join(summary_text) + "\n"
    )

    parameter_text = """# Parameter Table

| Features | Single LSTM | Bidirectional | Stacked Layer |
|---|---:|---:|---:|
| Univariate (Solar energy production) | 68,608 | 137,216 | 131,584 |
| Primary + Temperature | 69,120 | 138,240 | 131,584 |
| Primary + Temperature + Month + Season + Year | 71,168 | 142,336 | 131,584 |
"""
    create_file_if_missing(ROOT / "summary" / "parameter_table.md", parameter_text)

def create_experiment_structure(model_name: str):
    exp_root = ROOT / "experiments" / model_name
    exp_root.mkdir(parents=True, exist_ok=True)

    (exp_root / "plots").mkdir(exist_ok=True)
    (exp_root / "results").mkdir(exist_ok=True)
    (exp_root / "models").mkdir(exist_ok=True)
    (exp_root / "notebooks").mkdir(exist_ok=True)

    create_file_if_missing(exp_root / "train.py", "")
    create_file_if_missing(exp_root / "evaluate.py", "")
    create_file_if_missing(exp_root / "predict.py", "")
    create_file_if_missing(exp_root / ".gitkeep", "")

    metrics = MODEL_METRICS.get(model_name, {})
    config = {
        "experiment_name": model_name,
        "model_type": infer_model_type(model_name),
        "granularity": infer_granularity(model_name),
        "feature_code": infer_feature_code(model_name),
        "input_features": infer_input_features(model_name),
        "target": "solar_energy_production",
        "parameter_count": infer_parameter_count(model_name),
        "saved_model": "best_model.keras",
        "r2_0_6_hours": metrics.get("0_6", ""),
        "r2_6_12_hours": metrics.get("6_12", ""),
        "r2_12_18_hours": metrics.get("12_18", ""),
        "r2_18_24_hours": metrics.get("18_24", ""),
        "mean_r2": metrics.get("mean_r2", ""),
        "std_r2": metrics.get("std_r2", ""),
    }
    create_file_if_missing(exp_root / "config.yaml", to_yaml(config))

    readme = f"""# {model_name}

## Model information
- Model type: {infer_model_type(model_name)}
- Granularity: {infer_granularity(model_name)}
- Feature code: {infer_feature_code(model_name)}
- Input features: {", ".join(infer_input_features(model_name))}
- Parameter count: {infer_parameter_count(model_name)}

## Performance
- 0-6 hours R²: {metrics.get("0_6", "")}
- 6-12 hours R²: {metrics.get("6_12", "")}
- 12-18 hours R²: {metrics.get("12_18", "")}
- 18-24 hours R²: {metrics.get("18_24", "")}
- Mean R²: {metrics.get("mean_r2", "")}
- Std R²: {metrics.get("std_r2", "")}

## Folder contents
- train.py
- evaluate.py
- predict.py
- config.yaml
- models/
- plots/
- results/
- notebooks/
"""
    create_file_if_missing(exp_root / "README.md", readme)

def main():
    create_root_files()
    create_summary_files()

    for model in MODELS:
        create_experiment_structure(model)

    print(f"Done. Folder structure created at:\n{ROOT}")

if __name__ == "__main__":
    main()