import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import talib

# Load historical data
# Load historical data with correct date column
def load_historical_data(asset):
    try:
        df = pd.read_csv(f"processed_{asset}_data.csv", parse_dates=True)

        # Rename date columns to a standard name
        if "DATE" in df.columns:
            df.rename(columns={"DATE": "Date"}, inplace=True)
        elif "TIMESTAMP" in df.columns:
            df.rename(columns={"TIMESTAMP": "Date"}, inplace=True)
        
        # Set Date as the index
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error loading historical data: {e}")
        return None


def calculate_indicators(df):
    df["SMA_50"] = talib.SMA(df["CLOSE"], timeperiod=50)
    df["SMA_200"] = talib.SMA(df["CLOSE"], timeperiod=200)
    df["RSI"] = talib.RSI(df["CLOSE"], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df["CLOSE"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal
    return df

def create_price_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["CLOSE"], name="CLOSE Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="50-Day SMA", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="200-Day SMA", line=dict(color="red", dash="dot")))
    fig.update_layout(title=f"{asset} Historical Prices & Moving Averages", xaxis_title="Date", yaxis_title="Price")
    return fig

def create_rsi_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")))
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
    fig.update_layout(title=f"{asset} RSI Indicator", xaxis_title="Date", yaxis_title="RSI")
    return fig

def create_macd_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="MACD Signal", line=dict(color="orange")))
    fig.update_layout(title=f"{asset} MACD Indicator", xaxis_title="Date", yaxis_title="MACD")
    return fig

def main():
    st.title("📊 Historical Data Analysis")
    st.markdown("---")

    asset = st.selectbox("Select Asset", ["Stock", "Crypto", "Forex"])
    data = load_historical_data(asset.lower())

    if data is not None:
        data = calculate_indicators(data)

        # Display price chart
        st.plotly_chart(create_price_chart(data, asset), use_container_width=True)
        
        # Display RSI chart
        st.plotly_chart(create_rsi_chart(data, asset), use_container_width=True)
        
        # Display MACD chart
        st.plotly_chart(create_macd_chart(data, asset), use_container_width=True)

        # Display raw data
        st.write("### Raw Data")
        st.dataframe(data.tail())

if __name__ == "__main__":
    main()
