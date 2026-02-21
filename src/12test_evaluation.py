import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"test_evaluation_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def evaluate_test(test_path, model_path, oof_rmse, log_path):
    output = []
    output.append("=" * 80)
    output.append("TEST SET EVALUATION — CATBOOST")
    output.append("=" * 80)

    # Load model and test data
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    df = pd.read_csv(test_path)
    output.append(f"\nTest set shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

    y_true_log = df["fiyat_log"]
    X_test     = df.drop("fiyat_log", axis=1)

    # Predict in log space
    y_pred_log = model.predict(X_test)

    log_rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    log_mae  = mean_absolute_error(y_true_log, y_pred_log)
    log_r2   = r2_score(y_true_log, y_pred_log)

    output.append(f"\nLog-Space Metrics:")
    output.append(f"  RMSE : {log_rmse:.6f}")
    output.append(f"  MAE  : {log_mae:.6f}")
    output.append(f"  R²   : {log_r2:.6f}")

    # Back-transform to real price space via expm1 (inverse of log1p)
    y_true_price = np.expm1(y_true_log)
    y_pred_price = np.expm1(y_pred_log)

    price_rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    price_mae  = mean_absolute_error(y_true_price, y_pred_price)
    price_r2   = r2_score(y_true_price, y_pred_price)

    output.append(f"\nReal Price Metrics (expm1 back-transform):")
    output.append(f"  RMSE : {price_rmse:,.0f} ₺")
    output.append(f"  MAE  : {price_mae:,.0f} ₺")
    output.append(f"  R²   : {price_r2:.6f}")

    # Generalisation check: compare test RMSE against OOF RMSE
    output.append(f"\nGeneralisation Check:")
    output.append(f"  OOF RMSE  (log) : {oof_rmse:.6f}")
    output.append(f"  Test RMSE (log) : {log_rmse:.6f}")
    delta = log_rmse - oof_rmse
    output.append(f"  Delta           : {delta:+.6f}")

    if abs(delta) < 0.01:
        output.append("  → Model generalises well (delta < 0.01)")
    elif delta > 0:
        output.append("  → Test RMSE higher than OOF — possible overfit or distribution shift")
    else:
        output.append("  → Test RMSE lower than OOF — unusually favourable test set")

    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return log_rmse, price_rmse


if __name__ == "__main__":
    base_path  = Path(__file__).parent.parent
    test_path  = base_path / "data" / "splits" / "test.csv"
    model_path = base_path / "models" / "catboost_final.cbm"
    log_path   = setup_logging(base_path / "logs")

    # OOF RMSE from 11_oof_final_model.py
    OOF_RMSE = 0.184068

    evaluate_test(test_path, model_path, oof_rmse=OOF_RMSE, log_path=log_path)
    print(f"Test evaluation complete. Log saved to: {log_path}")
