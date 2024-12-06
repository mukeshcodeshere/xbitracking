import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from yahooquery import Ticker
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from data_fetcher import fetch_ticker_data
import pickle, os
import warnings
warnings.filterwarnings("ignore")

# %% Helper functions for technical indicators
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

def compute_trending_down_percentage(df_ticker_data, period=5):
    """Compute the percentage of stocks in the dataset trending down over the specified period."""
    trending_down = sum(
        df_ticker_data.groupby('symbol').apply(
            lambda stock: stock['adjclose'].iloc[-1] / stock['adjclose'].iloc[-(period + 1)] - 1 < 0
        )
    )
    total_stocks = len(df_ticker_data['symbol'].unique())
    return (trending_down / total_stocks) * 100

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

# %% Data fetching and preprocessing
def load_tickers(file_path):
    """Load tickers from an Excel file."""
    tickers = pd.read_excel(file_path)
    return tickers[tickers.Country == "United States"].Ticker.unique().tolist()

def fetch_data(tickers, period='max', interval='1d'):
    """Fetch data for the tickers."""
    return fetch_ticker_data(tickers, period=period, interval=interval, filename=None)

def prepare_features_and_targets(df_ticker_data, tickers_mini, period=5):
    """Prepare features and targets for modeling."""
    df_xbi = df_ticker_data[df_ticker_data['symbol'] == 'XBI'].copy()
    df_spy = df_ticker_data[df_ticker_data['symbol'] == 'SPY'].copy()

    df_xbi['7d_ma_XBI'] = df_xbi['adjclose'].rolling(window=7).mean()
    df_xbi['14d_rsi_XBI'] = compute_rsi(df_xbi['adjclose'])
    df_xbi['30d_ma_XBI'] = df_xbi['adjclose'].rolling(window=30).mean()
    df_xbi['macd_XBI'], df_xbi['macd_signal_XBI'] = compute_macd(df_xbi['adjclose'])
    df_xbi['volatility_XBI'] = df_xbi['adjclose'].rolling(window=14).std()

    market_breadth = compute_market_breadth_indicators(df_ticker_data, tickers_mini)

    df_xbi['pct_above_200d_ma'] = market_breadth['pct_above_200d_ma']
    df_xbi['market_return_std'] = market_breadth['market_return_std']
    df_xbi['new_highs_ratio'] = market_breadth['new_highs_ratio']
    df_xbi['new_lows_ratio'] = market_breadth['new_lows_ratio']
    df_xbi['avg_stock_correlation'] = market_breadth['avg_stock_correlation']
    df_xbi['sector_momentum'] = market_breadth['sector_momentum']

    df_xbi['target_XBI'] = 0
    df_xbi.loc[df_xbi['adjclose'] == df_xbi['adjclose'].rolling(window=30).max(), 'target_XBI'] = 1
    df_xbi.loc[df_xbi['adjclose'] == df_xbi['adjclose'].rolling(window=30).min(), 'target_XBI'] = -1

    df_spy['7d_ma_SPY'] = df_spy['adjclose'].rolling(window=7).mean()
    df_spy['14d_rsi_SPY'] = compute_rsi(df_spy['adjclose'])
    df_spy['30d_ma_SPY'] = df_spy['adjclose'].rolling(window=30).mean()
    df_spy['macd_SPY'], df_spy['macd_signal_SPY'] = compute_macd(df_spy['adjclose'])
    df_spy['volatility_SPY'] = df_spy['adjclose'].rolling(window=14).std()

    df_combined = pd.merge(
        df_xbi, 
        df_spy[['date', 'adjclose', '7d_ma_SPY', '14d_rsi_SPY', '30d_ma_SPY', 'macd_SPY', 'macd_signal_SPY', 'volatility_SPY']], 
        on='date', 
        suffixes=('_XBI', '_SPY')
    )

    feature_columns = ['7d_ma_XBI', '14d_rsi_XBI', '30d_ma_XBI', 'macd_XBI', 'macd_signal_XBI', 
                       'volatility_XBI', '7d_ma_SPY', '14d_rsi_SPY', '30d_ma_SPY', 'macd_SPY', 'macd_signal_SPY', 'volatility_SPY']
    
    X_combined = df_combined[feature_columns].fillna(0)
    y_combined = df_combined['target_XBI']

    return X_combined, y_combined

# %% Model training and evaluation
def initialize_models():
    """Initialize and return a dictionary of models."""
    return {
        'DecisionTree': DecisionTreeClassifier(
            max_depth=3,  
            min_samples_split=10, 
            random_state=42
        )
    }

def train_and_evaluate_model(X, y, model):
    """Train the model and print evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write('Accuracy:', accuracy_score(y_test, y_pred))
    st.write('Classification Report:', classification_report(y_test, y_pred))
    st.write('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['Down', 'Neutral', 'Up'])
    ax.set_yticklabels(['Down', 'Neutral', 'Up'])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    st.pyplot(fig)

def main():
    # Streamlit title and description
    st.title("Stock Prediction and Analysis App")
    st.write("This app uses stock data, technical indicators, and machine learning to predict market trends.")

    # Load tickers and fetch data
    tickers = load_tickers("Input/Complete-List-of-Biotech-Stocks-Listed-on-NASDAQ-Jan-1-24.xlsx")
    df_ticker_data = fetch_data(tickers)
    
    # Prepare features and targets
    X_combined, y_combined = prepare_features_and_targets(df_ticker_data, tickers)

    # Train the model and display evaluation metrics
    models = initialize_models()
    selected_model = models['DecisionTree']
    train_and_evaluate_model(X_combined, y_combined, selected_model)

if __name__ == "__main__":
    main()
