import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from catboost import CatBoostRegressor


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"final_refit_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


# Tuned parameters from 10_catboost_tuning.py
BEST_PARAMS = {
    "iterations":          559,
    "learning_rate":       0.03695757989673786,
    "depth":               5,
    "subsample":           0.930573919743892,
    "colsample_bylevel":   0.7099511354894406,
    "l2_leaf_reg":         0.8983622911378986,
    "min_data_in_leaf":    9,
    "random_seed":         42,
    "verbose":             0,
}


def run_final_refit(train_path, test_path, model_dir, log_path):
    output = []
    output.append("=" * 80)
    output.append("FINAL REFIT — TRAIN + TEST — CATBOOST")
    output.append("=" * 80)

    # Merge train and test into full dataset
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    df       = pd.concat([train_df, test_df], ignore_index=True)

    output.append(f"\nTrain rows : {len(train_df):,}")
    output.append(f"Test rows  : {len(test_df):,}")
    output.append(f"Total rows : {len(df):,}")
    output.append(f"Features   : {df.shape[1] - 1}")

    y = df["fiyat_log"]
    X = df.drop("fiyat_log", axis=1)

    output.append("\nParameters:")
    for k, v in BEST_PARAMS.items():
        if k != "verbose":
            output.append(f"  {k:<22} = {v}")

    # Fit on all available data — no validation split
    model = CatBoostRegressor(**BEST_PARAMS)
    model.fit(X, y)

    # Save production model
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "catboost_production.cbm"
    model.save_model(str(model_path))

    output.append(f"\nProduction model saved → {model_path}")
    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return model


if __name__ == "__main__":
    base_path  = Path(__file__).parent.parent
    train_path = base_path / "data" / "splits" / "train.csv"
    test_path  = base_path / "data" / "splits" / "test.csv"
    model_dir  = base_path / "models"
    log_path   = setup_logging(base_path / "logs")

    run_final_refit(train_path, test_path, model_dir, log_path)
    print(f"Final refit complete. Log saved to: {log_path}")
