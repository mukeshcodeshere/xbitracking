import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from yahooquery import Ticker

# Pre-defined weights for features
feature_weights = {
    '14d_rsi_XBI': 0.9347673756567654,
    'macd_XBI': 0.04290946317958141,
    'macd_signal_XBI': 0.022323161163653328,
    # All other features have weight 0
    '7d_ma_XBI': 0,
    '30d_ma_XBI': 0,
    'volatility_XBI': 0,
    '7d_ma_SPY': 0,
    '14d_rsi_SPY': 0,
    '30d_ma_SPY': 0,
    'macd_SPY': 0,
    'macd_signal_SPY': 0,
    'volatility_SPY': 0
}

# Fetch the live XBI data from Yahoo Finance
def fetch_live_data(ticker='XBI'):
    """Fetch the latest data for the given ticker from Yahoo Finance."""
    xbi = Ticker(ticker)
    xbi_history = xbi.history(period='1d', interval='1m')  # Fetch daily data
    return xbi_history

# Calculate the necessary technical indicators
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

def fetch_spy_data(ticker='SPY'):
    """Fetch the latest data for SPY from Yahoo Finance."""
    spy = Ticker(ticker)
    spy_history = spy.history(period='1d', interval='1m')  # Fetch daily data for SPY
    return spy_history

# Prepare the missing features (as placeholders for now)
def prepare_live_features(xbi_data, spy_data):
    """Prepare features for the live XBI data, including SPY-related features."""
    # Calculate indicators for XBI
    xbi_data['7d_ma_XBI'] = xbi_data['close'].rolling(window=7).mean()
    xbi_data['14d_rsi_XBI'] = compute_rsi(xbi_data['close'])
    xbi_data['30d_ma_XBI'] = xbi_data['close'].rolling(window=30).mean()
    xbi_data['macd_XBI'], xbi_data['macd_signal_XBI'] = compute_macd(xbi_data['close'])
    xbi_data['volatility_XBI'] = xbi_data['close'].rolling(window=14).std()

    # Calculate indicators for SPY (or other related benchmark data)
    spy_data['7d_ma_SPY'] = spy_data['close'].rolling(window=7).mean()
    spy_data['14d_rsi_SPY'] = compute_rsi(spy_data['close'])
    spy_data['30d_ma_SPY'] = spy_data['close'].rolling(window=30).mean()
    spy_data['macd_SPY'], spy_data['macd_signal_SPY'] = compute_macd(spy_data['close'])
    spy_data['volatility_SPY'] = spy_data['close'].rolling(window=14).std()

    # Use the most recent row for prediction
    latest_xbi_data = xbi_data.iloc[-1:]
    latest_spy_data = spy_data.iloc[-1:]
    
    features = {
        '7d_ma_XBI': latest_xbi_data['7d_ma_XBI'].values[0],
        '14d_rsi_XBI': latest_xbi_data['14d_rsi_XBI'].values[0],
        '30d_ma_XBI': latest_xbi_data['30d_ma_XBI'].values[0],
        'macd_XBI': latest_xbi_data['macd_XBI'].values[0],
        'macd_signal_XBI': latest_xbi_data['macd_signal_XBI'].values[0],
        'volatility_XBI': latest_xbi_data['volatility_XBI'].values[0],
        '7d_ma_SPY': latest_spy_data['7d_ma_SPY'].values[0],
        '14d_rsi_SPY': latest_spy_data['14d_rsi_SPY'].values[0],
        '30d_ma_SPY': latest_spy_data['30d_ma_SPY'].values[0],
        'macd_SPY': latest_spy_data['macd_SPY'].values[0],
        'macd_signal_SPY': latest_spy_data['macd_signal_SPY'].values[0],
        'volatility_SPY': latest_spy_data['volatility_SPY'].values[0],
    }
    return pd.DataFrame([features])

def predict_xbi_conditions(features):
    """Predict market conditions based on feature weights."""
    weighted_sum = sum(features[feature] * feature_weights[feature] for feature in feature_weights)
    
    # Determine the market condition based on weighted sum
    if weighted_sum > 0.5:
        return 'Top'  # Strong positive signal, likely at a peak
    elif weighted_sum < -0.5:
        return 'Bottom'  # Strong negative signal, likely at a bottom
    else:
        return 'Neutral'  # Neither strongly positive nor negative

def main():
    st.title('XBI Market Condition Predictor')

    # Sidebar for model information
    st.sidebar.header('Model Details')
    
    # Display feature weights (importance)
    st.sidebar.subheader('Feature Weights')
    for feature, weight in feature_weights.items():
        st.sidebar.write(f"{feature}: {weight:.4f}")
    
    # Main prediction interface
    st.header('XBI Market Condition Prediction')

    # Fetch live data for XBI and SPY
    st.subheader('Live Data')
    xbi_data = fetch_live_data('XBI')
    spy_data = fetch_spy_data('SPY')

    if xbi_data.empty or spy_data.empty:
        st.error("Failed to fetch live data for XBI or SPY.")
        return

    st.write(f"Latest Data for XBI: {xbi_data.tail(1)}")
    st.write(f"Latest Data for SPY: {spy_data.tail(1)}")

    # Prepare features from live data
    features = prepare_live_features(xbi_data, spy_data)

    # Make prediction based on the latest data and weights
    prediction = predict_xbi_conditions(features.iloc[0])

    # Display prediction
    st.metric('Predicted Market Condition', prediction)
    
    # Prediction explanation
    if prediction == 'Top':
        st.success('The model suggests the XBI is at a potential peak.')
    elif prediction == 'Bottom':
        st.warning('The model suggests the XBI is at a potential bottom.')
    else:
        st.info('The model suggests a neutral market condition.')

if __name__ == '__main__':
    main()
