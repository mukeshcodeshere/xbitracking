# data_fetcher.py
from yahooquery import Ticker
import os
import pandas as pd

def create_folder_if_not_exists(folder_name):
    """Create folder if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def fetch_ticker_data(tickers, period='ytd', interval='1d', filename=None):
    """Fetch historical data for the given tickers and save it as a CSV file."""
    try:
        tickers_obj = Ticker(tickers, asynchronous=True)
        df = tickers_obj.history(period=period, interval=interval)
        df.reset_index(inplace=True)
        
        # Save to CSV if filename is provided
        if filename:
            df.to_csv(filename, index=False)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {tickers}: {e}")
        return None

def save_data_to_csv(df, folder_name, file_name):
    """Save the DataFrame to a CSV file in the specified folder."""
    create_folder_if_not_exists(folder_name)
    file_path = os.path.join(folder_name, file_name)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
