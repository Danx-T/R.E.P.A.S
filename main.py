import subprocess
import sys
from pathlib import Path
from datetime import datetime


BASE_DIR  = Path(__file__).parent
SRC_DIR   = BASE_DIR / "src"
LOG_DIR   = BASE_DIR / "logs"

SCRIPTS = [
    "01null_value_analysis.py",
    "02data_cleaning.py",
    "03eda.py",
    "04feature_transformation.py",
    "05model_eda.py",
    "06feature_engineering.py",
    "07train_test_split.py",
    "08random_forest_baseline.py",
    "09boosting_baselines.py",
    "10catboost_tuning.py",
    "11oof_final_model.py",
    "12test_evaluation.py",
    "13final_refit.py",
]


def setup_pipeline_log():
    """Setup a top-level pipeline log that captures step outcomes."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"pipeline_{timestamp}.log"


def log(log_path, message):
    """Append message to pipeline log."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def run_step(script_name, log_path):
    """Run a single script as a subprocess; log success or failure."""
    step = script_name.split(".")[0]
    header = f"[{step}]"
    print(header)
    log(log_path, header)

    try:
        result = subprocess.run(
            [sys.executable, str(SRC_DIR / script_name)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            # Script exited with error — log stderr and stop pipeline
            error_msg = result.stderr.strip()
            msg = f"     FAILED (exit {result.returncode})\n{error_msg}\n"
            print(msg)
            log(log_path, msg)
            raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

        msg = f"     OK\n"

    except RuntimeError:
        raise
    except Exception as e:
        msg = f"     FAILED — {type(e).__name__}: {e}\n"
        print(msg)
        log(log_path, msg)
        raise

    print(msg)
    log(log_path, msg)


def main():
    pipeline_log = setup_pipeline_log()

    log(pipeline_log, "=" * 80)
    log(pipeline_log, "PIPELINE START")
    log(pipeline_log, "=" * 80 + "\n")

    for script in SCRIPTS:
        run_step(script, pipeline_log)

    log(pipeline_log, "=" * 80)
    log(pipeline_log, "PIPELINE COMPLETE")
    log(pipeline_log, "=" * 80)
    print(f"\nPipeline complete. Log → {pipeline_log}")


if __name__ == "__main__":
    main()
