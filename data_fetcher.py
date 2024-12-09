# data_fetcher.py
from yahooquery import Ticker
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_folder_if_not_exists(folder_name):
    """Create folder if it doesn't already exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def fetch_ticker_data(tickers, period='ytd', interval='1d', filename=None):
    """Fetch historical data for the given tickers and save it as a CSV file."""
    try:
        # Initialize Ticker object for asynchronous fetching
        tickers_obj = Ticker(tickers, asynchronous=True)
        
        # Fetch historical data for the specified period and interval
        df = tickers_obj.history(period=period, interval=interval)
        
        # If data is empty, raise an exception
        if df.empty:
            raise ValueError(f"No data returned for tickers: {tickers}")
        
        df.reset_index(inplace=True)
        
        # Save to CSV if filename is provided
        if filename:
            df.to_csv(filename, index=False)
            logging.info(f"Data for {tickers} saved to {filename}")
        
        return df
    except ValueError as ve:
        # Handle cases where no data is returned
        logging.error(f"ValueError: {ve}")
    except Exception as e:
        # Log any other exception
        logging.error(f"Error fetching data for {tickers}: {e}")
    
    return pd.DataFrame()  # Return an empty DataFrame instead of None

def save_data_to_csv(df, folder_name, file_name):
    """Save the DataFrame to a CSV file in the specified folder."""
    if df.empty:
        logging.warning("No data to save. DataFrame is empty.")
        return
    
    create_folder_if_not_exists(folder_name)
    file_path = os.path.join(folder_name, file_name)
    df.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")
