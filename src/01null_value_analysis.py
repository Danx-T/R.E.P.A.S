import pandas as pd
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"null_analysis_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def analyze_nulls(csv_path):
    """Analyze null values in CSV file."""
    df = pd.read_csv(csv_path)
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
    row_nulls = df.isnull().sum(axis=1)
    
    return df, null_cols, row_nulls


def build_report(df, null_cols, row_nulls):
    """Build the analysis report."""
    output = []

    output.append("=" * 80)
    output.append("NULL VALUE ANALYSIS")
    output.append("=" * 80)
    output.append(f"Dataset: {len(df):,} rows × {len(df.columns)} columns\n")
    
    output.append("=" * 80)
    output.append("COLUMNS WITH NULL VALUES")
    output.append("=" * 80)
    
    if len(null_cols) > 0:
        output.append(f"{len(null_cols)} columns have null values:\n")
        output.append(f"{'Column':<30} {'Null Count':<15} {'%':<10} {'Non-Null'}")
        output.append("-" * 80)
        
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            output.append(f"{col:<30} {count:<15,} {pct:>6.2f}%    {len(df)-count:,}")
        
        total_nulls = null_cols.sum()
        output.append(f"\nTotal: {total_nulls:,} null cells")
    else:
        output.append("No null values found!")
    
    output.append("\n" + "=" * 80)
    output.append("ROWS WITH NULL VALUES")
    output.append("=" * 80)
    
    rows_with_nulls = row_nulls[row_nulls > 0]
    if len(rows_with_nulls) > 0:
        output.append(f"{len(rows_with_nulls):,} rows have nulls")
        output.append(f"Complete rows: {len(df) - len(rows_with_nulls):,}\n")
        
        output.append("Distribution:")
        for null_count, row_count in row_nulls.value_counts().sort_index().items():
            if null_count > 0:
                output.append(f"  {null_count} nulls: {row_count:,} rows")
    else:
        output.append("All rows complete!")
    
    output.append("\n" + "=" * 80)
    output.append("SUMMARY")
    output.append("=" * 80)
    total_nulls = df.isnull().sum().sum()
    output.append(f"Total nulls: {total_nulls:,} / {len(df) * len(df.columns):,} cells")
    output.append(f"Rows with nulls: {len(rows_with_nulls):,} / {len(df):,}")
    output.append(f"Columns with nulls: {len(null_cols)} / {len(df.columns)}")
    output.append("=" * 80)
    
    return "\n".join(output)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "raw" / "ataşehir.csv"
    log_path = setup_logging(base_path / "logs")
    
    df, null_cols, row_nulls = analyze_nulls(csv_path)
    report = build_report(df, null_cols, row_nulls)
    log_message(log_path, report)
    print(f"Analysis complete. Log saved to: {log_path}")
