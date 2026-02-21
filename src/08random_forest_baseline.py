import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"random_forest_baseline_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def run_baseline(train_path, log_path):
    output = []
    output.append("=" * 80)
    output.append("RANDOM FOREST BASELINE — 5-FOLD CROSS VALIDATION")
    output.append("=" * 80)

    df = pd.read_csv(train_path)
    output.append(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    output.append(f"Target column : fiyat_log")

    y = df["fiyat_log"]
    X = df.drop("fiyat_log", axis=1)

    output.append(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # 5-fold CV with shuffle to avoid ordering bias
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,       # fully grown trees (baseline — no pruning)
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    fold_rmse, fold_mae, fold_r2 = [], [], []

    output.append("\n" + "-" * 60)
    output.append(f"{'Fold':<6} {'RMSE':<14} {'MAE':<14} {'R²':<10}")
    output.append("-" * 60)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        rf.fit(X_train_fold, y_train_fold)
        y_pred = rf.predict(X_valid_fold)

        rmse = np.sqrt(mean_squared_error(y_valid_fold, y_pred))
        mae  = mean_absolute_error(y_valid_fold, y_pred)
        r2   = r2_score(y_valid_fold, y_pred)

        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_r2.append(r2)

        output.append(f"  {fold:<4} {rmse:<14.6f} {mae:<14.6f} {r2:<10.6f}")

    output.append("-" * 60)

    mean_rmse = np.mean(fold_rmse)
    std_rmse  = np.std(fold_rmse)
    mean_mae  = np.mean(fold_mae)
    mean_r2   = np.mean(fold_r2)

    output.append("\nCV Summary:")
    output.append(f"  Mean RMSE : {mean_rmse:.6f}  (± {std_rmse:.6f})")
    output.append(f"  Mean MAE  : {mean_mae:.6f}")
    output.append(f"  Mean R²   : {mean_r2:.6f}")

    # Refit on full training set to get stable feature importances
    output.append("\n" + "=" * 80)
    output.append("FULL TRAIN SET REFIT — FEATURE IMPORTANCE")
    output.append("=" * 80)

    rf.fit(X, y)

    importance_df = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    output.append(f"\nTop 15 Features:")
    output.append(f"{'Rank':<6} {'Feature':<45} {'Importance':<12}")
    output.append("-" * 65)

    for rank, row in importance_df.head(15).iterrows():
        output.append(f"  {rank + 1:<4} {row['feature']:<45} {row['importance']:.6f}")

    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return mean_rmse, std_rmse, mean_mae, mean_r2


if __name__ == "__main__":
    base_path  = Path(__file__).parent.parent
    train_path = base_path / "data" / "splits" / "train.csv"
    log_path   = setup_logging(base_path / "logs")

    run_baseline(train_path, log_path)
    print(f"Random Forest baseline complete. Log saved to: {log_path}")
