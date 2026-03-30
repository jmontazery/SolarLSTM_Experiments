from pathlib import Path
import shutil

# =========================================================
# EDIT THIS PATH
# =========================================================
ROOT = Path(r"C:\Users\Jamileh\Documents\SolarLSTM_Experiments")

# =========================================================
# YOUR 18 MODELS
# =========================================================
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

# =========================================================
# PERFORMANCE TABLE
# =========================================================
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

# =========================================================
# PARAMETER COUNTS
# =========================================================
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

JUNK_DIRS = {"__pycache__", ".ipynb_checkpoints", ".idea"}
JUNK_FILES = {".gitkeep", "Thumbs.db", ".DS_Store"}


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
    return "quarter-hourly" if "Q" in model_name else "hourly"


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


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def remove_junk(root: Path):
    for path in root.rglob("*"):
        if path.is_dir() and path.name in JUNK_DIRS:
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed directory: {path}")
        elif path.is_file() and path.name in JUNK_FILES:
            try:
                path.unlink()
                print(f"Removed file: {path}")
            except Exception as e:
                print(f"Could not remove file {path}: {e}")


def rename_slurm_files(experiments_root: Path):
    for exp_dir in experiments_root.iterdir():
        if not exp_dir.is_dir():
            continue

        old_path = exp_dir / "slurm_job"
        new_path = exp_dir / "slurm_job.sh"

        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            print(f"Renamed: {old_path.name} -> {new_path.name} in {exp_dir.name}")


def ensure_subfolders(exp_dir: Path):
    for folder_name in ["models", "plots", "results"]:
        (exp_dir / folder_name).mkdir(exist_ok=True)


def rewrite_experiment_config(exp_dir: Path, model_name: str):
    metrics = MODEL_METRICS.get(model_name, {})
    config = {
        "experiment_name": model_name,
        "model_type": infer_model_type(model_name),
        "granularity": infer_granularity(model_name),
        "feature_code": infer_feature_code(model_name),
        "input_features": infer_input_features(model_name),
        "target": "solar_energy_production",
        "parameter_count": infer_parameter_count(model_name),
        "main_script": f"{model_name}.py",
        "saved_model_path": "best_model.keras",
        "plots_path": "plots/",
        "results_path": "results/",
        "models_path": "models/",
        "r2_0_6_hours": metrics.get("0_6", ""),
        "r2_6_12_hours": metrics.get("6_12", ""),
        "r2_12_18_hours": metrics.get("12_18", ""),
        "r2_18_24_hours": metrics.get("18_24", ""),
        "mean_r2": metrics.get("mean_r2", ""),
        "std_r2": metrics.get("std_r2", ""),
    }
    write_text(exp_dir / "config.yaml", to_yaml(config))


def rewrite_experiment_readme(exp_dir: Path, model_name: str):
    metrics = MODEL_METRICS.get(model_name, {})
    features = infer_input_features(model_name)
    features_text = ", ".join(features) if features else "Not specified"

    readme = (
        f"# {model_name}\n\n"
        "## Overview\n"
        f"This folder contains the full experiment for the **{model_name}** model used in the solar power forecasting study.\n\n"
        "## Model details\n"
        f"- **Model type:** {infer_model_type(model_name)}\n"
        f"- **Time granularity:** {infer_granularity(model_name)}\n"
        f"- **Feature code:** {infer_feature_code(model_name)}\n"
        f"- **Input features:** {features_text}\n"
        f"- **Parameter count:** {infer_parameter_count(model_name)}\n\n"
        "## Performance (R²)\n"
        f"- **0–6 hours:** {metrics.get('0_6', '')}\n"
        f"- **6–12 hours:** {metrics.get('6_12', '')}\n"
        f"- **12–18 hours:** {metrics.get('12_18', '')}\n"
        f"- **18–24 hours:** {metrics.get('18_24', '')}\n"
        f"- **Mean R²:** {metrics.get('mean_r2', '')}\n"
        f"- **Std R²:** {metrics.get('std_r2', '')}\n\n"
        "## Contents\n"
        f"- `{model_name}.py` — main script containing preprocessing, training, evaluation, and plotting\n"
        "- `config.yaml` — experiment metadata and performance summary\n"
        "- `best_model.keras` — saved best model\n"
        "- `models/` — additional saved models if applicable\n"
        "- `plots/` — generated plots\n"
        "- `results/` — experiment outputs\n"
        "- `slurm_job.sh` — SLURM job script if used for cluster execution\n\n"
        "## Notes\n"
        "This experiment is stored independently so that each model configuration remains reproducible and easy to review.\n"
    )
    write_text(exp_dir / "README.md", readme)


def create_summary_files(root: Path):
    summary_dir = root / "summary"
    summary_dir.mkdir(exist_ok=True)

    lines = [
        "# Model Summary",
        "",
        "| Model | Model Type | Granularity | Features | Mean R² | Std R² | Parameters |",
        "|---|---|---|---|---:|---:|---:|",
    ]

    for model in MODELS:
        m = MODEL_METRICS[model]
        lines.append(
            f"| {model} | {infer_model_type(model)} | {infer_granularity(model)} | "
            f"{infer_feature_code(model)} | {m['mean_r2']} | {m['std_r2']} | {infer_parameter_count(model)} |"
        )

    write_text(summary_dir / "model_summary.md", "\n".join(lines) + "\n")

    parameter_table = (
        "# Parameter Table\n\n"
        "| Features | Single LSTM | Bidirectional | Stacked Layer |\n"
        "|---|---:|---:|---:|\n"
        "| Univariate (Solar energy production) | 68,608 | 137,216 | 131,584 |\n"
        "| Primary + Temperature | 69,120 | 138,240 | 131,584 |\n"
        "| Primary + Temperature + Month + Season + Year | 71,168 | 142,336 | 131,584 |\n"
    )
    write_text(summary_dir / "parameter_table.md", parameter_table)


def write_root_files(root: Path):
    root_readme = (
        "# SolarLSTM Experiments\n\n"
        "This repository contains the organized experiments from my Master's thesis on "
        "**solar power forecasting using LSTM-based deep learning models**.\n\n"
        "## Study scope\n"
        "The repository compares three neural network architectures:\n"
        "- Single LSTM\n"
        "- Bidirectional LSTM\n"
        "- Stacked LSTM\n\n"
        "These models are evaluated under different input settings and temporal granularities.\n\n"
        "## Naming convention\n"
        "The experiment names follow this format:\n\n"
        "- H = hourly data\n"
        "- Q = quarter-hourly data\n"
        "- Uni = univariate input using only solar energy production\n"
        "- T = solar energy production + temperature\n"
        "- C = time-related features including month, season, and year\n\n"
        "### Examples\n"
        "- LSTMHUni = Single LSTM, hourly, univariate\n"
        "- BiQT = Bidirectional LSTM, quarter-hourly, with temperature\n"
        "- StackedHTC = Stacked LSTM, hourly, with temperature and time features\n\n"
        "## Repository structure\n\n"
        "SolarLSTM_Experiments/\n"
        "├── experiments/\n"
        "│   ├── LSTMHUni/\n"
        "│   ├── LSTMQUni/\n"
        "│   ├── LSTMHT/\n"
        "│   ├── LSTMQT/\n"
        "│   ├── LSTMHTC/\n"
        "│   ├── LSTMQTC/\n"
        "│   ├── BiHUni/\n"
        "│   ├── BiQUni/\n"
        "│   ├── BiHT/\n"
        "│   ├── BiQT/\n"
        "│   ├── BiHTC/\n"
        "│   ├── BiQTC/\n"
        "│   ├── StackedHUni/\n"
        "│   ├── StackedQUni/\n"
        "│   ├── StackedHT/\n"
        "│   ├── StackedQT/\n"
        "│   ├── StackedHTC/\n"
        "│   └── StackedQTC/\n"
        "├── summary/\n"
        "├── common/\n"
        "├── README.md\n"
        "├── requirements.txt\n"
        "└── .gitignore\n\n"
        "## Inside each experiment folder\n"
        "Each experiment folder contains:\n"
        "- the main Python script named after the model\n"
        "- config.yaml with metadata and performance information\n"
        "- best_model.keras for the saved trained model\n"
        "- models/ for additional saved models\n"
        "- plots/ for generated figures\n"
        "- results/ for experiment outputs\n"
        "- slurm_job.sh if the experiment was run on a cluster\n\n"
        "## Performance summary\n"
        "A summary of all models is available in:\n"
        "- summary/model_summary.md\n"
        "- summary/parameter_table.md\n\n"
        "## Goal of the repository\n"
        "The goal of this repository is to provide a clear, reproducible, and structured "
        "presentation of the forecasting experiments, including code, trained models, plots, "
        "and summarized results.\n\n"
        "## Tools and libraries\n"
        "Main tools used in this project:\n"
        "- Python\n"
        "- TensorFlow / Keras\n"
        "- NumPy\n"
        "- Pandas\n"
        "- Matplotlib\n"
        "- Scikit-learn\n"
    )

    write_text(root / "README.md", root_readme)

    requirements = (
        "numpy\n"
        "pandas\n"
        "matplotlib\n"
        "scikit-learn\n"
        "tensorflow\n"
        "keras\n"
        "pyyaml\n"
    )
    write_text(root / "requirements.txt", requirements)

    gitignore = (
        "__pycache__/\n"
        "*.pyc\n"
        ".ipynb_checkpoints/\n"
        ".venv/\n"
        "venv/\n"
        ".env\n"
        ".DS_Store\n"
        "Thumbs.db\n"
        ".idea/\n"
    )
    write_text(root / ".gitignore", gitignore)

    common_dir = root / "common"
    common_dir.mkdir(exist_ok=True)
    for fname in ["preprocessing.py", "metrics.py", "plotting.py", "utils.py"]:
        fpath = common_dir / fname
        if not fpath.exists():
            write_text(fpath, "")


def refresh_repo():
    experiments_root = ROOT / "experiments"
    if not experiments_root.exists():
        raise FileNotFoundError(f"Experiments folder not found: {experiments_root}")

    remove_junk(ROOT)
    rename_slurm_files(experiments_root)

    for model_name in MODELS:
        exp_dir = experiments_root / model_name
        if not exp_dir.exists():
            print(f"[WARNING] Missing experiment folder: {exp_dir}")
            continue

        ensure_subfolders(exp_dir)
        rewrite_experiment_config(exp_dir, model_name)
        rewrite_experiment_readme(exp_dir, model_name)
        print(f"Updated config and README for {model_name}")

    create_summary_files(ROOT)
    write_root_files(ROOT)

    print("\nDone. Repository cleaned and refreshed.")


if __name__ == "__main__":
    refresh_repo()