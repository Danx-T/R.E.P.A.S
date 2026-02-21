import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"boosting_baselines_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def evaluate_model(model, X, y, kf):
    """Run 5-fold CV for a single model, return per-fold metrics."""
    fold_rmse, fold_mae, fold_r2 = [], [], []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        fold_rmse.append(np.sqrt(mean_squared_error(y_valid, y_pred)))
        fold_mae.append(mean_absolute_error(y_valid, y_pred))
        fold_r2.append(r2_score(y_valid, y_pred))

    return fold_rmse, fold_mae, fold_r2


def run_boosting_baselines(train_path, log_path):
    output = []
    output.append("=" * 80)
    output.append("BOOSTING BASELINES — 5-FOLD CROSS VALIDATION")
    output.append("=" * 80)

    df = pd.read_csv(train_path)
    output.append(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    output.append(f"Target column : fiyat_log")

    y = df["fiyat_log"]
    X = df.drop("fiyat_log", axis=1)

    output.append(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # Same CV setup as RF baseline for fair comparison
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,       # no depth limit, controlled by num_leaves
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        output.append(f"\n{'─' * 60}")
        output.append(f"  {name}")
        output.append(f"{'─' * 60}")
        output.append(f"  {'Fold':<6} {'RMSE':<14} {'MAE':<14} {'R²':<10}")
        output.append(f"  {'-' * 50}")

        fold_rmse, fold_mae, fold_r2 = evaluate_model(model, X, y, kf)

        for i, (rmse, mae, r2) in enumerate(zip(fold_rmse, fold_mae, fold_r2), start=1):
            output.append(f"  {i:<6} {rmse:<14.6f} {mae:<14.6f} {r2:<10.6f}")

        mean_rmse = np.mean(fold_rmse)
        std_rmse  = np.std(fold_rmse)
        mean_mae  = np.mean(fold_mae)
        mean_r2   = np.mean(fold_r2)

        results[name] = {
            "mean_rmse": mean_rmse,
            "std_rmse":  std_rmse,
            "mean_mae":  mean_mae,
            "mean_r2":   mean_r2,
        }

        output.append(f"\n  Mean RMSE : {mean_rmse:.6f}  (± {std_rmse:.6f})")
        output.append(f"  Mean MAE  : {mean_mae:.6f}")
        output.append(f"  Mean R²   : {mean_r2:.6f}")

    # Summary comparison table
    output.append("\n" + "=" * 80)
    output.append("MODEL COMPARISON SUMMARY")
    output.append("=" * 80)
    output.append(f"\n  {'Model':<14} {'Mean RMSE':>12} {'± Std':>10} {'Mean MAE':>12} {'Mean R²':>10}")
    output.append("  " + "-" * 62)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_rmse"])
    for rank, (name, m) in enumerate(sorted_results, start=1):
        marker = "  ← best" if rank == 1 else ""
        output.append(
            f"  {name:<14} {m['mean_rmse']:>12.6f} {m['std_rmse']:>10.6f} "
            f"{m['mean_mae']:>12.6f} {m['mean_r2']:>10.6f}{marker}"
        )

    best_model, best_metrics = sorted_results[0]
    output.append(f"\nBest model   : {best_model}")
    output.append(f"Best RMSE    : {best_metrics['mean_rmse']:.6f}  (± {best_metrics['std_rmse']:.6f})")
    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return results


if __name__ == "__main__":
    base_path  = Path(__file__).parent.parent
    train_path = base_path / "data" / "splits" / "train.csv"
    log_path   = setup_logging(base_path / "logs")

    run_boosting_baselines(train_path, log_path)
    print(f"Boosting baselines complete. Log saved to: {log_path}")
