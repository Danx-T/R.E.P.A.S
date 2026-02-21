import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"eda_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def perform_eda(csv_path):
    """Perform comprehensive exploratory data analysis."""
    df = pd.read_csv(csv_path)
    
    output = []
    
    # Dataset Overview
    output.append("=" * 80)
    output.append("DATASET OVERVIEW")
    output.append("=" * 80)
    output.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    output.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    output.append("Column Information:")
    output.append(f"{'Column':<25} {'Type':<15} {'Non-Null':<12} {'Null %'}")
    output.append("-" * 80)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        output.append(f"{col:<25} {dtype:<15} {non_null:<12,} {null_pct:>6.2f}%")
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    output.append(f"\nNumerical columns ({len(numerical_cols)}): {', '.join(numerical_cols)}")
    output.append(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")
    
    # Statistical Summary for Numerical Columns
    output.append("\n" + "=" * 80)
    output.append("NUMERICAL VARIABLES SUMMARY")
    output.append("=" * 80)
    
    if len(numerical_cols) > 0:
        stats = df[numerical_cols].describe()
        output.append("\nDescriptive Statistics:")
        output.append(stats.to_string())
        
        output.append("\n\nAdditional Statistics:")
        for col in numerical_cols:
            output.append(f"\n{col}:")
            output.append(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")
            output.append(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
            output.append(f"  Skewness: {df[col].skew():.2f}")
            output.append(f"  Kurtosis: {df[col].kurtosis():.2f}")
    else:
        output.append("No numerical columns found.")
    
    # Categorical Variables Analysis
    output.append("\n" + "=" * 80)
    output.append("CATEGORICAL VARIABLES ANALYSIS")
    output.append("=" * 80)
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            unique_count = df[col].nunique()
            output.append(f"\n{col}:")
            output.append(f"  Unique values: {unique_count}")
            
            output.append("  Value counts:")
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                output.append(f"    {val}: {count:,} ({pct:.1f}%)")
    else:
        output.append("No categorical columns found.")
    
    # Categorical Variables Impact on Price
    if len(categorical_cols) > 0 and 'fiyat' in df.columns:
        output.append("\n" + "=" * 80)
        output.append("CATEGORICAL VARIABLES IMPACT ON PRICE")
        output.append("=" * 80)
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            output.append(f"\n{col}:")
            
            grouped = df.groupby(col)['fiyat'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
            output.append(f"  {'Category':<30} {'Mean Price':<15} {'Median Price':<15} {'Count'}")
            output.append("  " + "-" * 75)
            for cat, row in grouped.iterrows():
                output.append(f"  {str(cat):<30} {row['mean']:>13,.0f} ₺  {row['median']:>13,.0f} ₺  {int(row['count']):,}")

    
    # Correlation Analysis
    if len(numerical_cols) > 1:
        output.append("\n" + "=" * 80)
        output.append("CORRELATION ANALYSIS")
        output.append("=" * 80)
        
        corr_matrix = df[numerical_cols].corr()
        output.append("\nCorrelation Matrix:")
        output.append(corr_matrix.to_string())
        
        # Find high correlations
        output.append("\n\nHigh Correlations (|r| > 0.7):")
        high_corr_found = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_found = True
                    output.append(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        if not high_corr_found:
            output.append("  No high correlations found.")
    
    # Outlier Detection (IQR method)
    if len(numerical_cols) > 0:
        output.append("\n" + "=" * 80)
        output.append("OUTLIER DETECTION (IQR Method)")
        output.append("=" * 80)
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            output.append(f"\n{col}:")
            output.append(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            output.append(f"  Outliers: {outlier_count:,} ({outlier_pct:.2f}%)")
            
            if outlier_count > 0 and outlier_count <= 10:
                output.append(f"  Outlier values: {sorted(outliers[col].unique())}")
    
    # Distribution Analysis
    if len(numerical_cols) > 0:
        output.append("\n" + "=" * 80)
        output.append("DISTRIBUTION ANALYSIS")
        output.append("=" * 80)
        
        for col in numerical_cols:
            output.append(f"\n{col}:")
            output.append(f"  Percentiles:")
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                val = df[col].quantile(p/100)
                output.append(f"    {p}%: {val:.2f}")
    
    # Summary
    output.append("\n" + "=" * 80)
    output.append("SUMMARY")
    output.append("=" * 80)
    output.append(f"Total rows: {len(df):,}")
    output.append(f"Total columns: {len(df.columns)}")
    output.append(f"Numerical columns: {len(numerical_cols)}")
    output.append(f"Categorical columns: {len(categorical_cols)}")
    output.append(f"Missing values: {df.isnull().sum().sum():,}")
    output.append(f"Duplicate rows: {df.duplicated().sum():,}")
    output.append("=" * 80)
    
    return "\n".join(output)


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "processed" / "ataşehir_cleaned.csv"
    log_path = setup_logging(base_path / "logs")
    
    report = perform_eda(csv_path)
    log_message(log_path, report)
    print(f"EDA complete. Log saved to: {log_path}")
