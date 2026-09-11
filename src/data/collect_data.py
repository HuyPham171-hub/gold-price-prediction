import os
import argparse
import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path

# ==============================================================================
# 1. SETUP LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# 2. CONFIGURATION & TICKERS
# ==============================================================================
YF_TICKERS = {
    'GC=F': 'Gold_Price',
    'SI=F': 'Silver_Futures',
    'HG=F': 'Copper_Futures',    # Added Copper
    'CL=F': 'Crude_Oil',
    '^GSPC': 'SP_500',
    '^VIX': 'VIX',
    '^GVZ': 'Gold_VIX',          
    'DX-Y.NYB': 'USD_Index'
}

FRED_TICKERS = {
    'T10YIE': 'Breakeven_Inflation_10Y', 
    'DGS2': 'Treasury_Yield_2Y',         
    'DGS10': 'Treasury_Yield_10Y',
    'DFII10': 'Real_Interest_Rate_10Y'   
}

# ==============================================================================
# 3. DATA EXTRACTION FUNCTIONS
# ==============================================================================
def fetch_yfinance_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily market data from Yahoo Finance."""
    logging.info("Fetching data from Yahoo Finance...")
    try:
        tickers_list = list(YF_TICKERS.keys())
        df_raw = yf.download(tickers_list, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        df_close = df_raw['Close'].copy()
        
        if isinstance(df_close.columns, pd.MultiIndex):
            df_close.columns = df_close.columns.get_level_values(0)
            
        df_close.rename(columns=YF_TICKERS, inplace=True)
        df_close.index = pd.to_datetime(df_close.index).tz_localize(None).normalize()
        df_close.index.name = 'Date'
        
        logging.info(f"Successfully fetched YFinance data. Shape: {df_close.shape}")
        return df_close
    except Exception as e:
        logging.error(f"Error fetching YFinance data: {e}")
        return pd.DataFrame()

def fetch_fred_data(api_key: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Fetch daily macroeconomic indicators from FRED."""
    logging.info("Fetching data from FRED API...")
    try:
        fred = Fred(api_key=api_key)
        df_fred = pd.DataFrame()
        
        for code, name in FRED_TICKERS.items():
            series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
            df_fred[name] = series
            
        df_fred.index = pd.to_datetime(df_fred.index).tz_localize(None).normalize()
        df_fred.index.name = 'Date'
        
        logging.info(f"Successfully fetched FRED data. Shape: {df_fred.shape}")
        return df_fred
    except Exception as e:
        logging.error(f"Error fetching FRED data. Details: {e}")
        return pd.DataFrame()

def load_local_gpr_data(file_path: str) -> pd.DataFrame:
    """Load local GPR daily data from an Excel file."""
    logging.info(f"Loading local GPR data from {file_path}...")
    try:
        df_gpr = pd.read_excel(file_path)
        
        # Parse the 'date' column to datetime and set as index
        df_gpr.index = pd.to_datetime(df_gpr['date']).dt.tz_localize(None).dt.normalize()
        df_gpr.index.name = 'Date'
        
        # Select only the relevant GPR columns
        cols_to_keep = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT']
        df_gpr = df_gpr[cols_to_keep]
        
        logging.info(f"Successfully loaded GPR data. Shape: {df_gpr.shape}")
        return df_gpr
    except Exception as e:
        logging.error(f"Error loading GPR data. Ensure the path is correct and openpyxl is installed. Details: {e}")
        return pd.DataFrame()

# ==============================================================================
# 4. DATA INTEGRATION WITH FULL CALENDAR REINDEXING
# ==============================================================================
def merge_raw_data_full_calendar(df_market: pd.DataFrame, df_macro: pd.DataFrame, df_gpr: pd.DataFrame) -> pd.DataFrame:
    """Merge market, macro, and GPR datasets and reindex to a complete daily calendar."""
    logging.info("Merging datasets (Raw Outer Join)...")
    
    # Pandas join can accept a list of DataFrames for multiple outer joins
    dfs_to_join = [df for df in [df_macro, df_gpr] if not df.empty]
    df_merged = df_market.join(dfs_to_join, how='outer')
    
    start_dt = df_merged.index.min()
    end_dt = df_merged.index.max()
    full_calendar = pd.date_range(start=start_dt, end=end_dt, freq='D', name='Date')
    df_merged = df_merged.reindex(full_calendar)
    
    logging.info(f"Full calendar reindexing complete. Date range: {start_dt.date()} to {end_dt.date()}")
    logging.info(f"Raw Dataset Shape: {df_merged.shape}")
        
    return df_merged

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    # Anchor to the project root: collect_data.py -> src/data/ -> src/ -> Project/
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DEFAULT_GPR_PATH = PROJECT_ROOT / "data" / "raw" / "data_gpr_daily_recent.xls"
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

    parser = argparse.ArgumentParser(description="Collect Raw Daily Financial & Macro Data")
    parser.add_argument('--start', type=str, default='1990-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR), help='Output directory path')
    parser.add_argument('--gpr_path', type=str, default=str(DEFAULT_GPR_PATH), 
                        help='Path to the local GPR daily Excel file')
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        logging.error("FRED_API_KEY not found. Please set it in .env file.")
        return

    df_market = fetch_yfinance_data(start_date=args.start, end_date=args.end)
    df_macro = fetch_fred_data(api_key=fred_api_key, start_date=args.start, end_date=args.end)
    
    # Load the local GPR dataset
    df_gpr = pd.DataFrame()
    gpr_file = Path(args.gpr_path).resolve()
    if gpr_file.exists():
        df_gpr = load_local_gpr_data(file_path=str(gpr_file))
    else:
        logging.warning(f"GPR file not found at {gpr_file}. Proceeding without GPR data.")
    
    if not df_market.empty and not df_macro.empty:
        df_final = merge_raw_data_full_calendar(df_market, df_macro, df_gpr)
        
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "daily_raw_features.csv"
        
        df_final.to_csv(file_path)
        logging.info(f"Pipeline executed successfully. Raw data saved to: {file_path}")
    else:
        logging.error("Pipeline failed. Market or Macro sources returned empty DataFrames.")

if __name__ == "__main__":
    main()