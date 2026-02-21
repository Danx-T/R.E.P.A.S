import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import skew, kurtosis


def setup_logging(log_dir):
    """Setup logging directory and return log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"model_eda_{timestamp}.log"


def log_message(log_path, message):
    """Append message to log file."""
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def analyze_price_distribution(df):
    """Analyze price distribution and evaluate log transformation."""
    output = []
    output.append("=" * 80)
    output.append("PRICE DISTRIBUTION ANALYSIS")
    output.append("=" * 80)
    
    # Basic stats
    output.append(f"\nPrice Statistics:")
    output.append(f"  Mean: {df['fiyat'].mean():,.0f} ₺")
    output.append(f"  Median: {df['fiyat'].median():,.0f} ₺")
    output.append(f"  Std: {df['fiyat'].std():,.0f} ₺")
    output.append(f"  Min: {df['fiyat'].min():,.0f} ₺")
    output.append(f"  Max: {df['fiyat'].max():,.0f} ₺")
    output.append(f"  Range: {df['fiyat'].max() - df['fiyat'].min():,.0f} ₺")
    
    # Distribution shape
    price_skew = skew(df['fiyat'])
    price_kurt = kurtosis(df['fiyat'])
    output.append(f"\nDistribution Shape:")
    output.append(f"  Skewness: {price_skew:.3f} {'(right-skewed)' if price_skew > 0 else '(left-skewed)'}")
    output.append(f"  Kurtosis: {price_kurt:.3f} {'(heavy-tailed)' if price_kurt > 0 else '(light-tailed)'}")
    
    # Log transformation evaluation
    log_price = np.log1p(df['fiyat'])
    log_skew = skew(log_price)
    log_kurt = kurtosis(log_price)
    output.append(f"\nLog-Transformed Price:")
    output.append(f"  Skewness: {log_skew:.3f} (improvement: {abs(price_skew) - abs(log_skew):.3f})")
    output.append(f"  Kurtosis: {log_kurt:.3f} (improvement: {abs(price_kurt) - abs(log_kurt):.3f})")
    
    if abs(log_skew) < abs(price_skew):
        output.append(f"\nLog transformation RECOMMENDED - reduces skewness")
    else:
        output.append(f"\nLog transformation NOT recommended - increases skewness")
    
    # Percentiles
    output.append(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = df['fiyat'].quantile(p/100)
        output.append(f"  {p:2d}%: {val:>10,.0f} ₺")
    
    return "\n".join(output)


def analyze_numeric_correlations(df):
    """Analyze correlations between price and all numeric features."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("PRICE CORRELATION ANALYSIS")
    output.append("=" * 80)
    
    # Get numeric columns (excluding boolean columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'fiyat' and not pd.api.types.is_bool_dtype(df[col])]
    
    # Calculate correlations with price
    correlations = df[numeric_cols + ['fiyat']].corr()['fiyat'].drop('fiyat').sort_values(ascending=False)
    
    output.append(f"\nNumeric Features Correlation with Price (sorted):")
    output.append(f"{'Feature':<25} {'Correlation':>12} {'Strength'}")
    output.append("-" * 80)
    
    for feat, corr in correlations.items():
        strength = ""
        if abs(corr) > 0.7:
            strength = "STRONG"
        elif abs(corr) > 0.4:
            strength = "MODERATE"
        elif abs(corr) > 0.2:
            strength = "WEAK"
        else:
            strength = "VERY WEAK"
        
        output.append(f"{feat:<25} {corr:>12.3f}  {strength}")
    
    # Key features analysis
    output.append(f"\nKey Features Analysis:")
    key_features = ['net_m2', 'toplam_oda', 'bina_yasi', 'kat_sayisi', 'bulundugu_kat']
    for feat in key_features:
        if feat in correlations:
            output.append(f"  {feat}: r = {correlations[feat]:.3f}")
    
    return "\n".join(output)


def analyze_dummy_variables(df):
    """Analyze price differences across dummy variable groups."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("DUMMY VARIABLES IMPACT ON PRICE")
    output.append("=" * 80)
    
    # Get mahalle dummies
    mahalle_cols = [col for col in df.columns if col.startswith('mahalle_')]
    isinma_cols = [col for col in df.columns if col.startswith('isinma_')]
    
    # Mahalle analysis
    if mahalle_cols:
        output.append(f"\nMahalle Impact (n={len(mahalle_cols)} categories):")
        mahalle_stats = []
        for col in mahalle_cols:
            mask = df[col] == True
            if mask.sum() > 0:
                mean_price = df[mask]['fiyat'].mean()
                median_price = df[mask]['fiyat'].median()
                count = mask.sum()
                mahalle_name = col.replace('mahalle_', '')
                mahalle_stats.append((mahalle_name, mean_price, median_price, count))
        
        mahalle_stats.sort(key=lambda x: x[1], reverse=True)
        output.append(f"  {'Mahalle':<30} {'Mean Price':>15} {'Median Price':>15} {'Count':>8}")
        output.append("  " + "-" * 75)
        for name, mean, median, count in mahalle_stats:
            output.append(f"  {name:<30} {mean:>13,.0f} ₺  {median:>13,.0f} ₺  {count:>8,}")
        
        # Price variance across mahalle
        prices_by_mahalle = [df[df[col] == True]['fiyat'].values for col in mahalle_cols if (df[col] == True).sum() > 0]
        overall_mean = df['fiyat'].mean()
        mahalle_means = [df[df[col] == True]['fiyat'].mean() for col in mahalle_cols if (df[col] == True).sum() > 0]
        price_range = max(mahalle_means) - min(mahalle_means)
        output.append(f"\n  Price range across mahalle: {price_range:,.0f} ₺")
        output.append(f"  Highest/Lowest ratio: {max(mahalle_means)/min(mahalle_means):.2f}x")
    
    # Isinma analysis
    if isinma_cols:
        output.append(f"\n\nIsinma Impact (n={len(isinma_cols)} categories):")
        isinma_stats = []
        for col in isinma_cols:
            mask = df[col] == True
            if mask.sum() > 0:
                mean_price = df[mask]['fiyat'].mean()
                median_price = df[mask]['fiyat'].median()
                count = mask.sum()
                isinma_name = col.replace('isinma_', '')
                isinma_stats.append((isinma_name, mean_price, median_price, count))
        
        isinma_stats.sort(key=lambda x: x[1], reverse=True)
        output.append(f"  {'Heating Type':<30} {'Mean Price':>15} {'Median Price':>15} {'Count':>8}")
        output.append("  " + "-" * 75)
        for name, mean, median, count in isinma_stats:
            output.append(f"  {name:<30} {mean:>13,.0f} ₺  {median:>13,.0f} ₺  {count:>8,}")
        
        # Price variance across isinma
        isinma_means = [df[df[col] == True]['fiyat'].mean() for col in isinma_cols if (df[col] == True).sum() > 0]
        price_range = max(isinma_means) - min(isinma_means)
        output.append(f"\n  Price range across heating types: {price_range:,.0f} ₺")
        output.append(f"  Highest/Lowest ratio: {max(isinma_means)/min(isinma_means):.2f}x")
    
    return "\n".join(output)


def analyze_outlier_impact(df):
    """Analyze outlier impact on price prediction."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("OUTLIER IMPACT ANALYSIS")
    output.append("=" * 80)
    
    # IQR method for price
    Q1 = df['fiyat'].quantile(0.25)
    Q3 = df['fiyat'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['fiyat'] < lower_bound) | (df['fiyat'] > upper_bound)]
    outlier_pct = (len(outliers) / len(df)) * 100
    
    output.append(f"\nPrice Outliers (IQR method):")
    output.append(f"  IQR: {IQR:,.0f} ₺")
    output.append(f"  Bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}] ₺")
    output.append(f"  Outliers: {len(outliers):,} ({outlier_pct:.1f}%)")
    
    if len(outliers) > 0:
        output.append(f"\n  Outlier price range: {outliers['fiyat'].min():,.0f} - {outliers['fiyat'].max():,.0f} ₺")
        output.append(f"  Mean price (with outliers): {df['fiyat'].mean():,.0f} ₺")
        output.append(f"  Mean price (without outliers): {df[~df.index.isin(outliers.index)]['fiyat'].mean():,.0f} ₺")
        
        # Check if outliers have extreme feature values
        numeric_cols = ['net_m2', 'toplam_oda', 'banyo_sayisi']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        
        output.append(f"\n  Outlier characteristics:")
        for col in numeric_cols:
            outlier_mean = outliers[col].mean()
            normal_mean = df[~df.index.isin(outliers.index)][col].mean()
            diff_pct = ((outlier_mean - normal_mean) / normal_mean) * 100
            output.append(f"    {col}: outlier mean = {outlier_mean:.1f}, normal mean = {normal_mean:.1f} (diff: {diff_pct:+.1f}%)")
    
    # Recommendation
    output.append(f"\nRecommendation:")
    if outlier_pct > 10:
        output.append(f"High outlier percentage ({outlier_pct:.1f}%) - consider robust scaling or outlier removal")
    elif outlier_pct > 5:
        output.append(f"Moderate outlier percentage ({outlier_pct:.1f}%) - monitor model performance")
    else:
        output.append(f"Low outlier percentage ({outlier_pct:.1f}%) - acceptable")
    
    return "\n".join(output)


def feature_signal_evaluation(df):
    """Evaluate which features carry real signal vs noise."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("FEATURE SIGNAL vs NOISE EVALUATION")
    output.append("=" * 80)
    
    # Get all numeric features (excluding boolean columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'fiyat' and not pd.api.types.is_bool_dtype(df[col])]
    
    # Calculate signal metrics
    correlations = df[numeric_cols + ['fiyat']].corr()['fiyat'].drop('fiyat').abs().sort_values(ascending=False)
    
    # Categorize features
    strong_signal = correlations[correlations > 0.4].index.tolist()
    moderate_signal = correlations[(correlations > 0.2) & (correlations <= 0.4)].index.tolist()
    weak_signal = correlations[(correlations > 0.1) & (correlations <= 0.2)].index.tolist()
    noise = correlations[correlations <= 0.1].index.tolist()
    
    output.append(f"\nFeature Categorization by Price Correlation:")
    output.append(f"\nSTRONG SIGNAL (|r| > 0.4) - {len(strong_signal)} features:")
    for feat in strong_signal:
        output.append(f"    {feat}: r = {correlations[feat]:.3f}")
    
    output.append(f"\nMODERATE SIGNAL (0.2 < |r| ≤ 0.4) - {len(moderate_signal)} features:")
    for feat in moderate_signal:
        output.append(f"    {feat}: r = {correlations[feat]:.3f}")
    
    output.append(f"\nWEAK SIGNAL (0.1 < |r| ≤ 0.2) - {len(weak_signal)} features:")
    for feat in weak_signal:
        output.append(f"    {feat}: r = {correlations[feat]:.3f}")
    
    output.append(f"\nNOISE (|r| ≤ 0.1) - {len(noise)} features:")
    for feat in noise:
        output.append(f"    {feat}: r = {correlations[feat]:.3f}")
    
    # Recommendations
    output.append(f"\nRecommendations:")
    output.append(f"Focus on STRONG + MODERATE features for initial modeling")
    output.append(f"Consider feature engineering for WEAK features")
    output.append(f"Consider removing NOISE features to reduce overfitting")
    
    return "\n".join(output)


def perform_model_eda(csv_path, log_path):
    """Perform model-focused EDA."""
    df = pd.read_csv(csv_path)
    
    output = []
    output.append("=" * 80)
    output.append("MODEL-FOCUSED EDA")
    output.append("=" * 80)
    output.append(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")
    output.append(f"Target: fiyat")
    
    # Run all analyses
    output.append(analyze_price_distribution(df))
    output.append(analyze_numeric_correlations(df))
    output.append(analyze_dummy_variables(df))
    output.append(analyze_outlier_impact(df))
    output.append(feature_signal_evaluation(df))
    
    output.append("\n" + "=" * 80)
    output.append("END OF REPORT")
    output.append("=" * 80)
    
    report = "\n".join(output)
    log_message(log_path, report)
    return report


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    csv_path = base_path / "data" / "processed" / "ataşehir_transformed.csv"
    log_path = setup_logging(base_path / "logs")
    
    perform_model_eda(csv_path, log_path)
    print(f"Model-focused EDA complete. Log saved to: {log_path}")
