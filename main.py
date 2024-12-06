# main.py
import pandas as pd
from data_fetcher import fetch_ticker_data, save_data_to_csv
from signal_generator import TopBottomStrategy
from backtest import run_top_bottom_strategy

# Example usage
tickers = pd.read_excel(r"Input\Complete-List-of-Biotech-Stocks-Listed-on-NASDAQ-Jan-1-24.xlsx")
tickers = tickers[tickers.Country == "United States"]
tickers = tickers.Ticker.unique().tolist()

# USING MINI for Testing
tickers_mini = tickers[:10]
benchmark_tickers = ["XBI","SPY"]
tickers_full = tickers_mini + benchmark_tickers

# Folder setup
folder_name = 'Input'

# Fetch daily data for the year-to-date (YTD)
df_daily_ticker = fetch_ticker_data(tickers_full, period='max', interval='1d', filename=None)
save_data_to_csv(df_daily_ticker, folder_name, 'daily_data.csv')

# Fetch today's minute-level data
df_today_minute = fetch_ticker_data(tickers_full, period='1d', interval='1m', filename=None)
save_data_to_csv(df_today_minute, folder_name, 'minute_data.csv')

# Load the data
df_ticker_daily = pd.read_csv("INPUT/daily_data.csv")

# Ensure date column is converted to datetime
df_ticker_daily['date'] = pd.to_datetime(df_ticker_daily['date'])
df_today_minute['date'] = pd.to_datetime(df_today_minute['date'])

# Run the backtest strategy
run_top_bottom_strategy(df_ticker_daily, TopBottomStrategy)
