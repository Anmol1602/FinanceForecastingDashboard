import yfinance as yf
import pandas as pd
import ccxt  # For crypto data
from alpha_vantage.foreignexchange import ForeignExchange
import time

# Function to fetch stock data
def fetch_stock_data(ticker='AAPL', period='1y', interval='1d'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

# Function to fetch crypto data (BTC/ETH from Binance)
def fetch_crypto_data(symbol='BTC/USDT', exchange='binance', timeframe='1d', limit=365):
    exchange = getattr(ccxt, exchange)()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to fetch Forex data (EUR/USD from Alpha Vantage)
def fetch_forex_data(pair='EUR/USD', api_key='YOUR_ALPHA_VANTAGE_KEY'):
    fx = ForeignExchange(key=api_key)
    data, _ = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')
    
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Reset index and rename columns to match crypto format
    df = df.reset_index()
    
    # Rename columns to match crypto format (without volume)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close']
    
    return df


# Example usage
stock_data = fetch_stock_data()
crypto_data = fetch_crypto_data()
forex_data = fetch_forex_data(api_key='IY4CC6VEMI9P95FY')

# Save to CSV
stock_data.to_csv('stock_data.csv', index=False)
crypto_data.to_csv('crypto_data.csv', index=False)
forex_data.to_csv('forex_data.csv', index=False)

print("Data fetched and saved!")
