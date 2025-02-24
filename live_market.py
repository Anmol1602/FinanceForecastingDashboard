import streamlit as st
import yfinance as yf
import requests
import time

ALPHA_VANTAGE_API_KEY = "IY4CC6VEMI9P95FY"
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return price
    except:
        return None

def get_forex_price(pair):
    try:
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={pair[:3]}&to_currency={pair[3:]}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url).json()
        return float(response["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
    except:
        return None

def get_crypto_price(symbol):
    try:
        response = requests.get(f"{BINANCE_API_URL}?symbol={symbol.upper()}USDT")
        return float(response.json()["price"])
    except:
        return None

def live_market():
    st.title("📈 Live Market Data")
    st.markdown("---")

    asset_type = st.sidebar.radio("Select Asset Type", ["Stock", "Forex", "Crypto"])
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, EURUSD, BTC):", "").upper()

    if ticker:
        with st.spinner("Fetching real-time data..."):
            if asset_type == "Stock":
                price = get_stock_price(ticker)
            elif asset_type == "Forex":
                price = get_forex_price(ticker)
            elif asset_type == "Crypto":
                price = get_crypto_price(ticker)

        if price:
            st.success(f"✅ {ticker} Price: **${price:.2f}**")
        else:
            st.error(f"❌ Unable to fetch data for {ticker}")

    st.write("🔄 Auto-refreshing every 10 seconds...")
    time.sleep(10)
    st.rerun()
