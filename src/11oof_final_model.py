import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"oof_final_model_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


# Best parameters from 10_catboost_tuning.py
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


def run_oof_and_final(train_path, model_dir, log_path):
    output = []
    output.append("=" * 80)
    output.append("OOF PREDICTIONS + FINAL MODEL — CATBOOST")
    output.append("=" * 80)

    df = pd.read_csv(train_path)
    output.append(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    output.append(f"Target column : fiyat_log")

    y = df["fiyat_log"]
    X = df.drop("fiyat_log", axis=1)

    output.append(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    output.append("\nParameters:")
    for k, v in BEST_PARAMS.items():
        if k != "verbose":
            output.append(f"  {k:<22} = {v}")

    # 5-fold OOF — same split as baseline for consistency
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    output.append("\n" + "-" * 60)
    output.append(f"  {'Fold':<6} {'RMSE':<14} {'MAE':<14} {'R²':<10}")
    output.append("-" * 60)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        model = CatBoostRegressor(**BEST_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        fold_preds = model.predict(X.iloc[valid_idx])
        oof_preds[valid_idx] = fold_preds

        rmse = np.sqrt(mean_squared_error(y.iloc[valid_idx], fold_preds))
        mae  = mean_absolute_error(y.iloc[valid_idx], fold_preds)
        r2   = r2_score(y.iloc[valid_idx], fold_preds)

        output.append(f"  {fold:<6} {rmse:<14.6f} {mae:<14.6f} {r2:<10.6f}")

    # OOF metrics over the full training set
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    oof_mae  = mean_absolute_error(y, oof_preds)
    oof_r2   = r2_score(y, oof_preds)

    output.append("-" * 60)
    output.append(f"\nOOF RMSE : {oof_rmse:.6f}")
    output.append(f"OOF MAE  : {oof_mae:.6f}")
    output.append(f"OOF R²   : {oof_r2:.6f}")

    # Save OOF predictions for stacking or diagnostics
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.DataFrame({"y_true": y.values, "y_pred_oof": oof_preds})
    oof_path = model_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    output.append(f"\nOOF predictions saved → {oof_path}")

    # Refit on full training set — this is the final deployable model
    output.append("\n" + "=" * 80)
    output.append("FULL TRAIN REFIT")
    output.append("=" * 80)

    final_model = CatBoostRegressor(**BEST_PARAMS)
    final_model.fit(X, y)

    model_path = model_dir / "catboost_final.cbm"
    final_model.save_model(str(model_path))

    output.append(f"\nFinal model saved → {model_path}")
    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return final_model, oof_preds


if __name__ == "__main__":
    base_path  = Path(__file__).parent.parent
    train_path = base_path / "data" / "splits" / "train.csv"
    model_dir  = base_path / "models"
    log_path   = setup_logging(base_path / "logs")

    run_oof_and_final(train_path, model_dir, log_path)
    print(f"OOF & final model complete. Log saved to: {log_path}")
