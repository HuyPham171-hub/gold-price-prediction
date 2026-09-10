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
    'CL=F': 'Crude_Oil',
    '^GSPC': 'SP_500',
    '^VIX': 'VIX',
    '^GVZ': 'Gold_VIX',          # Gold Volatility Index (Proxy for Geopolitical/Tail-Risk)
    'DX-Y.NYB': 'USD_Index'
}

FRED_TICKERS = {
    'T10YIE': 'Breakeven_Inflation_10Y', # Proxy for CPI
    'DGS2': 'Treasury_Yield_2Y',         # Proxy for Fed Funds Rate
    'DGS10': 'Treasury_Yield_10Y',
    'DFII10': 'Real_Interest_Rate_10Y'   # Available from 2003, establishes the latest common start date
}

# ==============================================================================
# 3. DATA EXTRACTION FUNCTIONS
# ==============================================================================
def fetch_yfinance_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily market data from Yahoo Finance."""
    logging.info("Fetching data from Yahoo Finance...")
    try:
        tickers_list = list(YF_TICKERS.keys())
        # auto_adjust=False is explicitly set to comply with the latest yfinance API
        df_raw = yf.download(tickers_list, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        # Extract only the 'Close' prices
        df_close = df_raw['Close'].copy()
        
        # Flatten MultiIndex columns if returned by yfinance
        if isinstance(df_close.columns, pd.MultiIndex):
            df_close.columns = df_close.columns.get_level_values(0)
            
        df_close.rename(columns=YF_TICKERS, inplace=True)
        
        # Normalize index to timezone-naive datetime normalized to midnight
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
            
        # Normalize index to timezone-naive datetime normalized to midnight
        df_fred.index = pd.to_datetime(df_fred.index).tz_localize(None).normalize()
        df_fred.index.name = 'Date'
        
        logging.info(f"Successfully fetched FRED data. Shape: {df_fred.shape}")
        return df_fred
    except Exception as e:
        logging.error(f"Error fetching FRED data. Please check API Key. Details: {e}")
        return pd.DataFrame()

# ==============================================================================
# 4. DATA INTEGRATION WITH FULL CALENDAR REINDEXING
# ==============================================================================
def merge_raw_data_full_calendar(df_market: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market and macro datasets and reindex to a complete daily calendar.
    Every single date (including weekends and holidays) is preserved with explicit NaNs.
    """
    logging.info("Merging datasets (Raw Outer Join)...")
    df_merged = df_market.join(df_macro, how='outer')
    
    # Determine boundary dates across all collected data
    start_dt = df_merged.index.min()
    end_dt = df_merged.index.max()
    
    # Generate full unbroken calendar range (freq='D')
    full_calendar = pd.date_range(start=start_dt, end=end_dt, freq='D', name='Date')
    
    # Reindex to insert missing weekend/holiday rows as explicit NaNs
    df_merged = df_merged.reindex(full_calendar)
    
    logging.info(f"Full calendar reindexing complete. Date range: {start_dt.date()} to {end_dt.date()}")
    logging.info(f"Raw Dataset Shape (with complete calendar NaNs): {df_merged.shape}")
        
    return df_merged

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Collect Raw Daily Financial & Macro Data with Full Calendar Dates")
    parser.add_argument('--start', type=str, default='1990-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD) - Default is Today')
    parser.add_argument('--output', type=str, default='../data/raw', help='Output directory path')
    args = parser.parse_args()

    load_dotenv()
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        logging.error("FRED_API_KEY not found. Please set it in .env file.")
        return

    df_market = fetch_yfinance_data(start_date=args.start, end_date=args.end)
    df_macro = fetch_fred_data(api_key=fred_api_key, start_date=args.start, end_date=args.end)
    
    if not df_market.empty and not df_macro.empty:
        df_final = merge_raw_data_full_calendar(df_market, df_macro)
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "daily_raw_features.csv"
        
        df_final.to_csv(file_path)
        logging.info(f"Pipeline executed successfully. Raw data saved to: {file_path}")
    else:
        logging.error("Pipeline failed. One or both data sources returned empty DataFrames.")

if __name__ == "__main__":
    main()