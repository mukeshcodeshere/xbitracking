import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from yahooquery import Ticker
from data_fetcher import fetch_ticker_data

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to add download link for images
def get_image_download_link(plt_fig, filename):
    """Generate a download link for matplotlib figures"""
    buf = BytesIO()
    plt_fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    href = f'<a href="data:image/png;base64,{image_base64}" download="{filename}">Download {filename}</a>'
    return href

# Replicate all helper functions from the original script
def compute_rsi(series, window=14):
    """Compute the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    """Compute the MACD and signal line."""
    short_ema = series.ewm(span=short_window, min_periods=1).mean()
    long_ema = series.ewm(span=long_window, min_periods=1).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1).mean()
    return macd, signal

def compute_market_breadth_indicators(df_ticker_data, tickers_mini):
    """Compute additional market breadth and sentiment indicators."""
    df_filtered = df_ticker_data[df_ticker_data['symbol'].isin(tickers_mini)]
    
    def is_above_200d_ma(stock_data):
        try:
            return stock_data['adjclose'].iloc[-1] > stock_data['adjclose'].rolling(window=200).mean().iloc[-1]
        except:
            return False
    
    above_200d_ma = df_filtered.groupby('symbol').apply(is_above_200d_ma).sum()
    pct_above_200d_ma = (above_200d_ma / len(tickers_mini)) * 100
    
    def compute_return_std(group):
        try:
            return group['adjclose'].pct_change().std()
        except:
            return 0
    
    market_return_std = df_filtered.groupby('symbol').apply(compute_return_std).mean()
    
    def is_52_week_high(stock_data):
        try:
            return stock_data['adjclose'].iloc[-1] == stock_data['adjclose'].rolling(window=252).max().iloc[-1]
        except:
            return False
    
    def is_52_week_low(stock_data):
        try:
            return stock_data['adjclose'].iloc[-1] == stock_data['adjclose'].rolling(window=252).min().iloc[-1]
        except:
            return False
    
    new_highs = df_filtered.groupby('symbol').apply(is_52_week_high).sum()
    new_lows = df_filtered.groupby('symbol').apply(is_52_week_low).sum()
    new_highs_ratio = (new_highs / len(tickers_mini)) * 100
    new_lows_ratio = (new_lows / len(tickers_mini)) * 100
    
    def compute_correlation(group):
        try:
            returns = group['adjclose'].pct_change()
            return returns
        except:
            return pd.Series()
    
    try:
        daily_returns = df_filtered.groupby('symbol').apply(compute_correlation)
        correlations = daily_returns.corr().mean().mean()
    except:
        correlations = 0
    
    def compute_momentum(stock_data, momentum_window=20):
        try:
            return stock_data['adjclose'].iloc[-1] / stock_data['adjclose'].iloc[-momentum_window] - 1
        except:
            return 0
    
    sector_momentum = df_filtered.groupby('symbol').apply(
        lambda stock: compute_momentum(stock)
    ).mean() * 100
    
    return {
        'pct_above_200d_ma': pct_above_200d_ma,
        'market_return_std': market_return_std,
        'new_highs_ratio': new_highs_ratio,
        'new_lows_ratio': new_lows_ratio,
        'avg_stock_correlation': correlations,
        'sector_momentum': sector_momentum
    }

def load_tickers(file_path):
    """Load tickers from an Excel file."""
    tickers = pd.read_excel(file_path)
    return tickers[tickers.Country == "United States"].Ticker.unique().tolist()

def fetch_data(tickers, period='max', interval='1d'):
    """Fetch data for the tickers."""
    return fetch_ticker_data(tickers, period=period, interval=interval, filename=None)

def prepare_features_and_targets(df_ticker_data, tickers_mini, period=5):
    """Prepare features and targets for modeling."""
    # Filter data for XBI and SPY
    df_xbi = df_ticker_data[df_ticker_data['symbol'] == 'XBI'].copy()
    df_spy = df_ticker_data[df_ticker_data['symbol'] == 'SPY'].copy()
    
    # Calculate indicators for XBI
    df_xbi['7d_ma_XBI'] = df_xbi['adjclose'].rolling(window=7).mean()
    df_xbi['14d_rsi_XBI'] = compute_rsi(df_xbi['adjclose'])
    df_xbi['30d_ma_XBI'] = df_xbi['adjclose'].rolling(window=30).mean()
    df_xbi['macd_XBI'], df_xbi['macd_signal_XBI'] = compute_macd(df_xbi['adjclose'])
    df_xbi['volatility_XBI'] = df_xbi['adjclose'].rolling(window=14).std()
    
    # Compute market breadth indicators
    market_breadth = compute_market_breadth_indicators(df_ticker_data, tickers_mini)
    
    # Add market breadth indicators to the XBI dataframe
    df_xbi['pct_above_200d_ma'] = market_breadth['pct_above_200d_ma']
    df_xbi['market_return_std'] = market_breadth['market_return_std']
    df_xbi['new_highs_ratio'] = market_breadth['new_highs_ratio']
    df_xbi['new_lows_ratio'] = market_breadth['new_lows_ratio']
    df_xbi['avg_stock_correlation'] = market_breadth['avg_stock_correlation']
    df_xbi['sector_momentum'] = market_breadth['sector_momentum']
    
    # Define target for XBI: Local maxima = Top, Local minima = Bottom
    df_xbi['target_XBI'] = 0
    df_xbi.loc[df_xbi['adjclose'] == df_xbi['adjclose'].rolling(window=30).max(), 'target_XBI'] = 1
    df_xbi.loc[df_xbi['adjclose'] == df_xbi['adjclose'].rolling(window=30).min(), 'target_XBI'] = -1
    # OUTPUT THIS

    # Calculate indicators for SPY
    df_spy['7d_ma_SPY'] = df_spy['adjclose'].rolling(window=7).mean()
    df_spy['14d_rsi_SPY'] = compute_rsi(df_spy['adjclose'])
    df_spy['30d_ma_SPY'] = df_spy['adjclose'].rolling(window=30).mean()
    df_spy['macd_SPY'], df_spy['macd_signal_SPY'] = compute_macd(df_spy['adjclose'])
    df_spy['volatility_SPY'] = df_spy['adjclose'].rolling(window=14).std()
    
    # Merge SPY features into XBI
    df_combined = pd.merge(
        df_xbi, 
        df_spy[['date', 'adjclose', '7d_ma_SPY', '14d_rsi_SPY', '30d_ma_SPY', 'macd_SPY', 'macd_signal_SPY', 'volatility_SPY']], 
        on='date', 
        suffixes=('_XBI', '_SPY')
    )
    
    # Filter columns for features and target
    feature_columns = ['7d_ma_XBI', '14d_rsi_XBI', '30d_ma_XBI', 'macd_XBI', 'macd_signal_XBI', 
                       'volatility_XBI', 
                       '7d_ma_SPY', '14d_rsi_SPY', '30d_ma_SPY', 'macd_SPY', 'macd_signal_SPY', 'volatility_SPY']

    #feature_columns = ['new_highs_ratio', 'new_lows_ratio',  '30d_ma_SPY']
    X_combined = df_combined[feature_columns].fillna(0)  # Handle any missing values in features
    y_combined = df_combined['target_XBI']  # The target variable
    
    return X_combined, y_combined

def initialize_models():
    """Initialize and return a dictionary of models."""
    return {
        'DecisionTree': DecisionTreeClassifier(
            max_depth=3, 
            min_samples_split=10, 
            min_samples_leaf=5,
            random_state=42
        )  
    }

def main():
    st.title("Biotech Market Analysis Dashboard")
    
    # Load tickers and fetch data
    st.write("Loading Biotech Stock Tickers...")
    tickers = load_tickers("Input/Complete-List-of-Biotech-Stocks-Listed-on-NASDAQ-Jan-1-24.xlsx")
    tickers_mini = tickers[:50]  # Limit to first 50 for demonstration
    benchmark_tickers = ["XBI", "SPY"]
    tickers_full = tickers_mini + benchmark_tickers
    
    # Fetch data
    st.write("Fetching Stock Data...")
    df_ticker_daily = fetch_data(tickers_full)
    
    # Prepare features and targets
    st.write("Preparing Model Features...")
    X_combined, y_combined = prepare_features_and_targets(df_ticker_daily, tickers_full, period=5)
    
    # Train model
    st.write("Training Machine Learning Model...")
    models = initialize_models()
    model = models['DecisionTree']
    model.fit(X_combined, y_combined)
    
    # Model Evaluation Section
    st.header("Model Evaluation")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, shuffle=False)
    y_pred = model.predict(X_test)
    
    # Display model metrics
    st.subheader("Classification Metrics")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, y_pred))
    
    st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    
    # Plot Decision Tree
    st.subheader("Decision Tree Visualization")
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=X_combined.columns, 
              class_names=['Bottom', 'Neutral', 'Top'], 
              filled=True, 
              rounded=True)
    st.pyplot(plt.gcf())
    plt.close()

if __name__ == "__main__":
    main()