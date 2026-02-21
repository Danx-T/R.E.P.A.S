import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"train_test_split_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def split_data(input_path, output_dir, log_path, test_size=0.2, random_state=42):
    """Split dataset into train/test with stratification on mahalle."""
    df = pd.read_csv(input_path)

    output = []
    output.append("=" * 80)
    output.append("TRAIN/TEST SPLIT REPORT")
    output.append("=" * 80)
    output.append(f"Input: {df.shape[0]:,} rows × {df.shape[1]} columns")
    output.append(f"Test size: {test_size:.0%} | Train size: {1 - test_size:.0%}")
    output.append(f"Random state: {random_state}")

    # Decode mahalle from one-hot columns back to a single label for stratification
    mahalle_cols = [col for col in df.columns if col.startswith("mahalle_")]
    if not mahalle_cols:
        raise ValueError("No mahalle_ columns found in dataset.")

    mahalle_label = df[mahalle_cols].idxmax(axis=1).str.replace("mahalle_", "", regex=False)

    output.append(f"\nStratification key: mahalle ({len(mahalle_cols)} unique values)")
    output.append(f"Mahalle distribution in full dataset:")
    for name, count in mahalle_label.value_counts().items():
        pct = count / len(df) * 100
        output.append(f"  {name:<30} {count:>5,} ({pct:>5.1f}%)")

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(df, mahalle_label))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    output.append(f"\nSplit result:")
    output.append(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df):.1%})")
    output.append(f"  Test:  {len(test_df):,} rows ({len(test_df)/len(df):.1%})")

    # Verify mahalle distribution in each split
    train_mahalle = train_df[mahalle_cols].idxmax(axis=1).str.replace("mahalle_", "", regex=False)
    test_mahalle = test_df[mahalle_cols].idxmax(axis=1).str.replace("mahalle_", "", regex=False)

    output.append(f"\nMahalle distribution comparison (Train | Test):")
    output.append(f"  {'Mahalle':<30} {'Train N':>8} {'Train %':>8} {'Test N':>8} {'Test %':>8}")
    output.append("  " + "-" * 70)

    for name in mahalle_label.value_counts().index:
        train_n = (train_mahalle == name).sum()
        test_n = (test_mahalle == name).sum()
        train_pct = train_n / len(train_df) * 100
        test_pct = test_n / len(test_df) * 100
        output.append(f"  {name:<30} {train_n:>8,} {train_pct:>7.1f}% {test_n:>8,} {test_pct:>7.1f}%")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    output.append(f"\nSaved:")
    output.append(f"  Train → {train_path}")
    output.append(f"  Test  → {test_path}")
    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return train_df, test_df


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    input_path = base_path / "data" / "processed" / "ataşehir_engineered.csv"
    output_dir = base_path / "data" / "splits"
    log_path = setup_logging(base_path / "logs")

    split_data(input_path, output_dir, log_path, test_size=0.2, random_state=42)
    print(f"Train/test split complete. Log saved to: {log_path}")
