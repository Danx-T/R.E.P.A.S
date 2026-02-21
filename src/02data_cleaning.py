import pandas as pd
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"data_cleaning_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def clean_data(input_path, output_path, log_path):
    """Clean the dataset by removing duplicates and unnecessary columns."""
    df = pd.read_csv(input_path)
    original_rows = len(df)
    original_cols = len(df.columns)
    
    # Step 1: Remove duplicates
    df = df.drop_duplicates(subset=['ilan_no'], keep='first')
    ilan_no_removed = original_rows - len(df)
    
    df = df.drop_duplicates(subset=['url'], keep='first')
    url_removed = (original_rows - ilan_no_removed) - len(df)
    
    # Step 2: Drop unnecessary columns
    
    # Group 1: Columns that don't affect price (metadata/identifiers)
    # - baslik: Just a text description, doesn't influence price
    # - ilan_no: Unique identifier, not a feature
    # - ilan_sahibi: Owner name, not relevant for price prediction
    # - telefon_is: Business phone, contact info doesn't affect price
    # - telefon_cep: Mobile phone, contact info doesn't affect price
    # - url: Link to listing, not a feature
    # - tarih: Listing date, temporal info not needed for price modeling
    metadata_cols = ['baslik', 'ilan_no', 'ilan_sahibi', 'telefon_is', 'telefon_cep', 'url', 'tarih']
    
    # Group 2: Columns with constant values (no variance)
    # - evin_durumu: All values are 'Kiralık' (constant)
    # - il: All values are 'İstanbul' (constant)
    # - ilce: All values are 'Ataşehir' (constant)
    # - emlak_tipi: All values are constant (no variance)
    constant_cols = ['evin_durumu', 'il', 'ilce', 'emlak_tipi']
    
    # Group 3: Highly correlated columns (multicollinearity)
    # - brut_m2: High correlation with net_m2, redundant feature
    correlated_cols = ['brut_m2']
    
    columns_to_drop = metadata_cols + constant_cols + correlated_cols
    df = df.drop(columns=columns_to_drop)
    
    # Save cleaned data
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Build report
    output = []
    output.append("=" * 80)
    output.append("DATA CLEANING REPORT")
    output.append("=" * 80)
    output.append(f"Original: {original_rows:,} rows × {original_cols} columns")
    output.append("")
    output.append("Duplicates removed:")
    output.append(f"  - ilan_no: {ilan_no_removed} rows")
    output.append(f"  - url: {url_removed} rows")
    output.append("")
    output.append(f"Columns dropped ({len(columns_to_drop)}):")
    output.append(f"  - Metadata/identifiers ({len(metadata_cols)}): {', '.join(metadata_cols)}")
    output.append(f"  - Constant values ({len(constant_cols)}): {', '.join(constant_cols)}")
    output.append(f"  - Highly correlated ({len(correlated_cols)}): {', '.join(correlated_cols)}")
    output.append("")
    output.append(f"Final: {len(df):,} rows × {len(df.columns)} columns")
    output.append(f"Saved to: {output_path}")
    output.append("=" * 80)
    
    log_message(log_path, "\n".join(output))
    return df



if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    input_path = base_path / "data" / "raw" / "ataşehir.csv"
    output_path = base_path / "data" / "processed" / "ataşehir_cleaned.csv"
    log_path = setup_logging(base_path / "logs")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_data(input_path, output_path, log_path)
    print(f"Cleaning complete. Log saved to: {log_path}")
