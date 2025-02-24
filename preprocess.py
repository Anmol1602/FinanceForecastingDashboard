import pandas as pd
import numpy as np

# Load data from CSV
stock_df = pd.read_csv("stock_data.csv", parse_dates=["Date"], index_col="Date")
crypto_df = pd.read_csv("crypto_data.csv", parse_dates=["timestamp"], index_col="timestamp")
forex_df = pd.read_csv("forex_data.csv", parse_dates=["timestamp"], index_col="timestamp") 

# Drop Unnecessary Columns 
stock_df.drop(columns=["Dividends", "Stock Splits"], inplace=True)
crypto_df.drop(columns=["volume"], inplace=True)

# Convert all headers to uppercase
# For regular columns
stock_df.columns = stock_df.columns.str.upper()
crypto_df.columns = crypto_df.columns.str.upper()
forex_df.columns = forex_df.columns.str.upper()

# For index names
stock_df.index.name = stock_df.index.name.upper()
crypto_df.index.name = crypto_df.index.name.upper()
forex_df.index.name = forex_df.index.name.upper()

# Function to handle missing values
def handle_missing_values(df):
    df = df.ffill().bfill()  # Forward fill first, then backward fill
    return df

# Normalize prices (Min-Max Scaling)
def normalize_prices(df, columns=["OPEN", "HIGH", "LOW", "CLOSE"]):
    for col in columns:
        if col in df.columns:  # Ensure the column exists
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

# Compute technical indicators
def add_technical_indicators(df):
    if "CLOSE" in df.columns:
        df["SMA_10"] = df["CLOSE"].rolling(window=10).mean()  # 10-day SMA
        df["SMA_50"] = df["CLOSE"].rolling(window=50).mean()  # 50-day SMA
        df["RSI"] = compute_rsi(df["CLOSE"])
        df["Upper_Band"], df["Lower_Band"] = compute_bollinger_bands(df["CLOSE"])
    return df

# Relative Strength Index (RSI)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Bollinger Bands
def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, lower_band

# Apply preprocessing
stock_df = handle_missing_values(stock_df)
crypto_df = handle_missing_values(crypto_df)
forex_df = handle_missing_values(forex_df)

stock_df = normalize_prices(stock_df)
crypto_df = normalize_prices(crypto_df)
forex_df = normalize_prices(forex_df)

stock_df = add_technical_indicators(stock_df)
crypto_df = add_technical_indicators(crypto_df)
forex_df = add_technical_indicators(forex_df)

# Save processed data
stock_df.to_csv("processed_stock_data.csv")
crypto_df.to_csv("processed_crypto_data.csv")
forex_df.to_csv("processed_forex_data.csv")

print("Preprocessing complete! Processed data saved.")
