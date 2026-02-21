import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"feature_engineering_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def engineer_features(input_path, output_path, log_path):
    """Apply feature engineering to the transformed dataset."""
    df = pd.read_csv(input_path)
    original_cols = list(df.columns)

    output = []
    output.append("=" * 80)
    output.append("FEATURE ENGINEERING REPORT")
    output.append("=" * 80)
    output.append(f"Input: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Target: log-transform price, drop original
    df["fiyat_log"] = np.log1p(df["fiyat"])
    df = df.drop(columns=["fiyat"])
    output.append("\nTarget:")
    output.append("  fiyat_log = log1p(fiyat)  [fiyat dropped]")

    # 1. M2-based features
    output.append("\n1. M2-Based Features:")

    # m2_per_room: how spacious is each room?
    df["m2_per_room"] = df["net_m2"] / df["toplam_oda"]
    output.append("  m2_per_room = net_m2 / toplam_oda")

    # room_density: rooms per m2 (inverse perspective)
    df["room_density"] = df["toplam_oda"] / df["net_m2"]
    output.append("  room_density = toplam_oda / net_m2")

    # bath_per_room: bathroom ratio
    df["bath_per_room"] = df["banyo_sayisi"] / df["toplam_oda"]
    output.append("  bath_per_room = banyo_sayisi / toplam_oda")

    # 2. Floor & location features
    output.append("\n2. Floor & Location Features:")

    # floor_ratio: relative position within building
    # Guard against division by zero (kat_sayisi == 0)
    df["floor_ratio"] = np.where(
        df["kat_sayisi"] > 0,
        df["bulundugu_kat"] / df["kat_sayisi"],
        np.nan
    )
    output.append("  floor_ratio = bulundugu_kat / kat_sayisi  (NaN if kat_sayisi == 0)")

    # is_top_floor: apartment is on the top floor
    df["is_top_floor"] = (df["bulundugu_kat"] == df["kat_sayisi"]).astype(int)
    output.append("  is_top_floor = (bulundugu_kat == kat_sayisi)")

    # is_ground_floor: ordinal 0 corresponds to Zemin Kat
    df["is_ground_floor"] = (df["bulundugu_kat"] == 0).astype(int)
    output.append("  is_ground_floor = (bulundugu_kat == 0)  [ordinal 0 = Zemin Kat]")

    # 3. Building age features
    output.append("\n3. Building Age Features:")

    # is_new: ordinal 0 = 0-5 years
    df["is_new"] = (df["bina_yasi"] == 0).astype(int)
    output.append("  is_new = (bina_yasi == 0)  [ordinal 0 = 0-5 years]")

    # is_old: ordinal >= 4 = 21+ years (21-25, 26-30, 31+)
    df["is_old"] = (df["bina_yasi"] >= 4).astype(int)
    output.append("  is_old = (bina_yasi >= 4)  [ordinal 4+ = 21+ years]")

    # Summary
    new_cols = [c for c in df.columns if c not in original_cols]
    output.append(f"\nNew columns added ({len(new_cols)}): {', '.join(new_cols)}")
    output.append(f"Final: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Basic stats for new features
    output.append("\nNew Feature Statistics:")
    output.append(f"{'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'NaN':>8}")
    output.append("-" * 80)
    for col in new_cols:
        mean = df[col].mean()
        std = df[col].std()
        mn = df[col].min()
        mx = df[col].max()
        nan_count = df[col].isnull().sum()
        output.append(f"{col:<20} {mean:>10.3f} {std:>10.3f} {mn:>10.3f} {mx:>10.3f} {nan_count:>8}")

    output.append(f"\nSaved to: {output_path}")
    output.append("=" * 80)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    log_message(log_path, "\n".join(output))
    return df


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    input_path = base_path / "data" / "processed" / "ataşehir_transformed.csv"
    output_path = base_path / "data" / "processed" / "ataşehir_engineered.csv"
    log_path = setup_logging(base_path / "logs")

    engineer_features(input_path, output_path, log_path)
    print(f"Feature engineering complete. Log saved to: {log_path}")
