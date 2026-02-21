import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"catboost_tuning_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def cv_rmse(params, X, y, kf):
    """Evaluate a parameter set via 5-fold CV, return mean RMSE."""
    fold_rmse = []

    for train_idx, valid_idx in kf.split(X):
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[valid_idx])
        fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[valid_idx], y_pred)))

    return np.mean(fold_rmse), np.std(fold_rmse)


def run_tuning(train_path, log_path, n_trials=75):
    output = []
    output.append("=" * 80)
    output.append("CATBOOST HYPERPARAMETER TUNING — OPTUNA")
    output.append("=" * 80)

    df = pd.read_csv(train_path)
    output.append(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    output.append(f"Target column : fiyat_log")
    output.append(f"Trials        : {n_trials}")

    y = df["fiyat_log"]
    X = df.drop("fiyat_log", axis=1)

    # Same CV setup as baseline scripts for fair comparison
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    trial_log = []

    def objective(trial):
        params = {
            "iterations":     trial.suggest_int("iterations", 200, 1000),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth":          trial.suggest_int("depth", 4, 10),
            "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "random_seed":    42,
        }

        mean_rmse, _ = cv_rmse(params, X, y, kf)
        trial_log.append((trial.number + 1, mean_rmse, params))
        return mean_rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Per-trial log
    output.append("\n" + "-" * 80)
    output.append(f"  {'Trial':>6} {'Mean RMSE':>12}  Best Params")
    output.append("-" * 80)

    best_so_far = float("inf")
    for trial_num, rmse, params in trial_log:
        marker = " ← best" if rmse < best_so_far else ""
        if rmse < best_so_far:
            best_so_far = rmse
        param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "random_seed")
        output.append(f"  {trial_num:>6} {rmse:>12.6f}  {param_str}{marker}")

    # Best result
    best_params = study.best_params
    best_params["random_seed"] = 42
    best_mean_rmse, best_std_rmse = cv_rmse(best_params, X, y, kf)

    output.append("\n" + "=" * 80)
    output.append("BEST RESULT")
    output.append("=" * 80)
    output.append(f"\n  Mean RMSE : {best_mean_rmse:.6f}  (± {best_std_rmse:.6f})")
    output.append(f"\n  Best Parameters:")
    for k, v in best_params.items():
        output.append(f"    {k:<22} = {v}")

    output.append("\n" + "=" * 80)
    output.append("FINAL PARAMETERS (copy-paste ready)")
    output.append("=" * 80)
    output.append("\nCatBoostRegressor(")
    for k, v in best_params.items():
        val_str = f'"{v}"' if isinstance(v, str) else str(v)
        output.append(f"    {k}={val_str},")
    output.append("    verbose=0,")
    output.append(")")
    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return best_params, best_mean_rmse


if __name__ == "__main__":
    base_path  = Path(__file__).parent.parent
    train_path = base_path / "data" / "splits" / "train.csv"
    log_path   = setup_logging(base_path / "logs")

    run_tuning(train_path, log_path, n_trials=75)
    print(f"CatBoost tuning complete. Log saved to: {log_path}")
