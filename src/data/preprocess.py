import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_pipeline(raw_file_path: Path, output_file_path: Path):
    logging.info(f"Loading raw dataset from: {raw_file_path}")
    if not raw_file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_file_path}")

    df_raw = pd.read_csv(raw_file_path, index_col='Date', parse_dates=True)

    # 1. Alignment: Drop days when gold market is closed, forward-fill exogenous indicators
    df_aligned = df_raw.dropna(subset=['Gold_Price']).copy()
    df_aligned = df_aligned.ffill()

    # Truncate at 2008 where Gold_VIX becomes available
    df_aligned = df_aligned.dropna()
    logging.info(f"Aligned date range: {df_aligned.index.min().date()} to {df_aligned.index.max().date()} | Shape: {df_aligned.shape}")

    # 2. Transformations to ensure Stationarity (avoiding spurious regression)
    df_transformed = pd.DataFrame(index=df_aligned.index)

    # 2.1 Returns for Price series (I(1))
    price_cols = [
        'Gold_Price', 
        'Silver_Futures', 
        'Copper_Futures', 
        'SP_500', 
        'USD_Index'
    ]
    for col in price_cols:
        if col in df_aligned.columns:
            df_transformed[f'{col}_Return'] = np.log(df_aligned[col] / df_aligned[col].shift(1))

    if 'Crude_Oil' in df_aligned.columns:
        df_transformed['Crude_Oil_Return'] = df_aligned['Crude_Oil'].pct_change()

    # 2.2 First-Difference for interest rate yields (I(1))
    # Excluded Treasury_Yield_10Y to eliminate collinearity with Real_Interest_Rate_10Y
    diff_cols = ['Treasury_Yield_2Y', 'Real_Interest_Rate_10Y']
    for col in diff_cols:
        if col in df_aligned.columns:
            df_transformed[f'{col}_Diff'] = df_aligned[col].diff()

    # 2.3 Stationary variables at level (I(0))
    # Log-transformed volatility indicators to normalize right-skewness
    if 'VIX' in df_aligned.columns:
        df_transformed['VIX_Level'] = np.log(df_aligned['VIX'])
    if 'Gold_VIX' in df_aligned.columns:
        df_transformed['Gold_VIX_Level'] = np.log(df_aligned['Gold_VIX'])
    if 'Breakeven_Inflation_10Y' in df_aligned.columns:
        df_transformed['Breakeven_Inflation_10Y_Level'] = df_aligned['Breakeven_Inflation_10Y']

    # 2.4 Retain component GPRs; Exclude composite GPRD to remove r=0.90 multicollinearity
    if 'GPRD_ACT' in df_aligned.columns:
        df_transformed['GPRD_ACT_Level'] = df_aligned['GPRD_ACT']
    if 'GPRD_THREAT' in df_aligned.columns:
        df_transformed['GPRD_THREAT_Level'] = df_aligned['GPRD_THREAT']

    # 3. Drop initial row produced by differencing/shifting
    df_processed = df_transformed.dropna()

    # 4. Export processed dataset
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_file_path)
    logging.info(f"Processed dataset successfully saved to: {output_file_path}")
    logging.info(f"Final training-ready shape: {df_processed.shape}")
    logging.info(f"Features: {list(df_processed.columns)}")

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    default_input = project_root / "data" / "raw" / "daily_raw_features.csv"
    default_output = project_root / "data" / "processed" / "daily_processed_features.csv"

    parser = argparse.ArgumentParser(description="Preprocess daily features for model training")
    parser.add_argument('--input', type=str, default=str(default_input), help='Path to raw CSV')
    parser.add_argument('--output', type=str, default=str(default_output), help='Path to output processed CSV')
    args = parser.parse_args()

    preprocess_pipeline(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()