import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"feature_transformation_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def transform_oda_sayisi(df):
    """Parse oda_sayisi column into oda_sayisi, salon_sayisi, toplam_oda."""
    room_mapping = {
        '1+1': (1, 1, 2),
        '2+1': (2, 1, 3),
        '3+1': (3, 1, 4),
        '4+1': (4, 1, 5),
        '5+1': (5, 1, 6),
        '6+1': (6, 1, 7),
        '4+2': (4, 2, 6),
        '5+2': (5, 2, 7),
        '3+2': (3, 2, 5),
        '2+2': (2, 2, 4),
        '2+0': (2, 0, 2),
        'Stüdyo (1+0)': (1, 0, 1),
        '1.5+1': (1.5, 1, 2.5),
        '2.5+1': (2.5, 1, 3.5),
        '3.5+1': (3.5, 1, 4.5),
    }

    oda = []
    salon = []
    toplam = []

    for val in df['oda_sayisi']:
        if val in room_mapping:
            o, s, t = room_mapping[val]
            oda.append(o)
            salon.append(s)
            toplam.append(t)
        else:
            oda.append(np.nan)
            salon.append(np.nan)
            toplam.append(np.nan)

    df['oda_sayisi'] = pd.to_numeric(oda, errors='coerce')
    df['salon_sayisi'] = pd.array(salon, dtype=pd.Int64Dtype())
    df['toplam_oda'] = pd.to_numeric(toplam, errors='coerce')

    return df


def transform_isinma(df):
    """Group heating types into categories."""
    isinma_mapping = {
        'Kombi (Doğalgaz)': 'Doğalgaz',
        'Doğalgaz Sobası': 'Doğalgaz',
        'Merkezi': 'Merkezi Sistem',
        'Merkezi (Pay Ölçer)': 'Merkezi Sistem',
        'Kat Kaloriferi': 'Merkezi Sistem',
        'Fancoil Ünitesi': 'Merkezi Sistem',
        'Yerden Isıtma': 'Yerden Isıtma',
        'Soba': 'Soba',
        'Kombi (Elektrik)': 'Elektrik / Klima',
        'Klima': 'Elektrik / Klima',
        'Yok': 'Yok',
    }

    df['isinma'] = df['isinma'].map(isinma_mapping)
    return df


def transform_bina_yasi(df):
    """Map building age strings into groups and apply ordinal encoding."""
    ordinal_order = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31+']
    ordinal_map = {label: i for i, label in enumerate(ordinal_order)}

    # Map original string values to groups
    value_mapping = {
        '0': '0-5', '1': '0-5', '2': '0-5', '3': '0-5', '4': '0-5', '5': '0-5',
        '6-10 arası': '6-10',
        '11-15 arası': '11-15',
        '16-20 arası': '16-20',
        '21-25 arası': '21-25',
        '26-30 arası': '26-30',
        '31 ve üzeri': '31+',
    }

    df['bina_yasi'] = df['bina_yasi'].astype(str).map(value_mapping)
    df['bina_yasi'] = df['bina_yasi'].map(ordinal_map)

    return df


def transform_kat_sayisi(df):
    """Map floor count strings into groups and apply ordinal encoding."""
    ordinal_order = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-29', '30+']
    ordinal_map = {label: i for i, label in enumerate(ordinal_order)}

    def map_kat_sayisi(val):
        val_str = str(val).strip()
        if val_str == '30 ve üzeri':
            return '30+'
        try:
            num = int(val_str)
            if 1 <= num <= 5:
                return '1-5'
            elif 6 <= num <= 10:
                return '6-10'
            elif 11 <= num <= 15:
                return '11-15'
            elif 16 <= num <= 20:
                return '16-20'
            elif 21 <= num <= 25:
                return '21-25'
            elif 26 <= num <= 29:
                return '26-29'
            elif num >= 30:
                return '30+'
        except ValueError:
            pass
        return np.nan

    df['kat_sayisi'] = df['kat_sayisi'].apply(map_kat_sayisi)
    df['kat_sayisi'] = df['kat_sayisi'].map(ordinal_map)

    return df


def transform_bulundugu_kat(df):
    """Map floor names/numbers into groups and apply ordinal encoding."""
    bodrum = ['Bodrum Kat', 'Giriş Altı Kot 3', 'Giriş Altı Kot 2', 'Giriş Altı Kot 1']
    zemin = ['Yüksek Giriş', 'Bahçe Katı', 'Giriş Katı', 'Zemin Kat', 'Müstakil']
    cati = ['Çatı Katı']

    ordinal_order = ['Bodrum Kat', 'Zemin Kat', '1-5', '6-10', '11-15', '16-20', '21-25', '26-29', '30+', 'Çatı Katı']
    ordinal_map = {label: i for i, label in enumerate(ordinal_order)}

    def map_floor(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()

        if val_str in bodrum:
            return 'Bodrum Kat'
        if val_str in zemin:
            return 'Zemin Kat'
        if val_str in cati:
            return 'Çatı Katı'

        try:
            num = int(val_str)
            if 1 <= num <= 5:
                return '1-5'
            elif 6 <= num <= 10:
                return '6-10'
            elif 11 <= num <= 15:
                return '11-15'
            elif 16 <= num <= 20:
                return '16-20'
            elif 21 <= num <= 25:
                return '21-25'
            elif 26 <= num <= 29:
                return '26-29'
            elif num >= 30:
                return '30+'
        except ValueError:
            pass

        return np.nan

    df['bulundugu_kat'] = df['bulundugu_kat'].apply(map_floor)
    df['bulundugu_kat'] = df['bulundugu_kat'].map(ordinal_map)

    return df


def transform_binary(df):
    """Binary encode: Evet=True, Hayır=False."""
    binary_cols = ['balkon', 'asansor', 'otopark', 'esyali', 'site_icerisinde']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Evet': True, 'Hayır': False})
    return df


def transform_features(input_path, output_path, log_path):
    """Apply all feature transformations."""
    df = pd.read_csv(input_path)
    original_shape = df.shape

    output = []
    output.append("=" * 80)
    output.append("FEATURE TRANSFORMATION REPORT")
    output.append("=" * 80)
    output.append(f"Original: {original_shape[0]:,} rows × {original_shape[1]} columns")

    # Drop unnecessary columns
    cols_to_drop = ['kullanim_durumu', 'tapu_durumu']
    df = df.drop(columns=cols_to_drop)
    output.append(f"\nDropped columns: {', '.join(cols_to_drop)}")

    # 1. oda_sayisi -> oda_sayisi, salon_sayisi, toplam_oda
    unmapped_oda = df['oda_sayisi'].value_counts()
    df = transform_oda_sayisi(df)
    oda_nulls = df['oda_sayisi'].isnull().sum()
    output.append(f"\n1. oda_sayisi parsed -> oda_sayisi, salon_sayisi, toplam_oda")
    output.append(f"   Unmapped values (NaN): {oda_nulls}")
    if oda_nulls > 0:
        original_vals = pd.read_csv(input_path)['oda_sayisi']
        missing_mask = df['oda_sayisi'].isnull()
        unmapped_vals = original_vals[missing_mask].value_counts()
        output.append(f"   Unmapped originals: {dict(unmapped_vals)}")

    # 2. Boolean encoding
    df = transform_binary(df)
    output.append(f"\n2. Boolean encoding: balkon, asansor, otopark, esyali, site_icerisinde (Evet=True, Hayır=False)")

    # 3. isinma grouping + one-hot
    df = transform_isinma(df)
    isinma_nulls = df['isinma'].isnull().sum()
    output.append(f"\n3. isinma grouped → 6 categories, then one-hot encoded")
    output.append(f"   Unmapped values (NaN): {isinma_nulls}")
    isinma_dummies = pd.get_dummies(df['isinma'], prefix='isinma')
    df = pd.concat([df.drop(columns=['isinma']), isinma_dummies], axis=1)
    output.append(f"   New columns: {list(isinma_dummies.columns)}")

    # 4. bina_yasi binning + ordinal
    df = transform_bina_yasi(df)
    output.append(f"\n4. bina_yasi binned -> 7 groups, ordinal encoded (0-6)")

    # 5. kat_sayisi binning + ordinal
    df = transform_kat_sayisi(df)
    output.append(f"\n5. kat_sayisi binned -> 7 groups, ordinal encoded (0-6)")

    # 6. bulundugu_kat mapping + ordinal
    df = transform_bulundugu_kat(df)
    output.append(f"\n6. bulundugu_kat mapped -> 10 groups, ordinal encoded (0-9)")

    # 7. mahalle one-hot encoding
    mahalle_dummies = pd.get_dummies(df['mahalle'], prefix='mahalle')
    df = pd.concat([df.drop(columns=['mahalle']), mahalle_dummies], axis=1)
    output.append(f"\n7. mahalle one-hot encoded -> {len(mahalle_dummies.columns)} new columns")

    # Save
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    output.append(f"\nFinal: {df.shape[0]:,} rows x {df.shape[1]} columns")
    output.append(f"Saved to: {output_path}")

    # Column summary
    output.append(f"\nFinal columns ({len(df.columns)}):")
    for col in df.columns:
        output.append(f"  - {col} ({df[col].dtype})")

    output.append("=" * 80)

    log_message(log_path, "\n".join(output))
    return df


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    input_path = base_path / "data" / "processed" / "ataşehir_cleaned.csv"
    output_path = base_path / "data" / "processed" / "ataşehir_transformed.csv"
    log_path = setup_logging(base_path / "logs")

    transform_features(input_path, output_path, log_path)
    print(f"Feature transformation complete. Log saved to: {log_path}")
