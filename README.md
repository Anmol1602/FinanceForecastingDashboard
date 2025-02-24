# Financial Market Analysis and Forecasting

A comprehensive Python-based project for analyzing and forecasting multiple financial markets including stocks, cryptocurrency, and forex data using technical indicators and machine learning approaches.

## 📋 Project Overview

This project processes and analyzes financial data from multiple markets:

- Stock Market Data
- Cryptocurrency Data
- Forex (Foreign Exchange) Data

The system implements various technical indicators and normalization techniques to prepare data for financial forecasting.

## 🔧 Prerequisites

- Python 3.x
- pandas
- numpy

## 📊 Data Processing Features

- Data Normalization using Min-Max Scaling
- Technical Indicators:
  - Simple Moving Averages (10-day and 50-day)
  - Relative Strength Index (RSI)
  - Bollinger Bands
- Missing Value Handling
- Automated Data Preprocessing Pipeline

🚀 Usage
Prepare your input data:

Place your stock data in stock_data.csv

Place your cryptocurrency data in crypto_data.csv

Place your forex data in forex_data.csv

Run the preprocessing script:

python preprocess.py

Copy

Insert at cursor
bash
Access processed data in the output files:

processed_stock_data.csv

processed_crypto_data.csv

processed_forex_data.csv

## Input Data Format

- Stock Data
Required columns: Date, Open, High, Low, Close

Optional columns: Dividends, Stock Splits (will be removed during preprocessing)

Date should be in datetime format

Cryptocurrency Data
Required columns: timestamp, Open, High, Low, Close

Optional columns: volume (will be removed during preprocessing)

timestamp should be in datetime format

Forex Data
Required columns: timestamp, Open, High, Low, Close

timestamp should be in datetime format

🔍 Technical Indicators
The project calculates the following technical indicators:

Simple Moving Averages (SMA)

10-day SMA

50-day SMA

Relative Strength Index (RSI)

14-period RSI

Helps identify overbought/oversold conditions

Bollinger Bands

20-day moving average

Upper and lower bands (2 standard deviations)

📈 Data Processing Steps
Data Loading

Unnecessary Column Removal

Column Name Standardization

Missing Value Handling

Forward fill followed by backward fill

Price Normalization

Min-Max scaling for Open, High, Low, Close prices

Technical Indicator Calculation

Processed Data Export

⚠️ Important Notes
All column names are converted to uppercase during processing

Missing values are handled using forward fill followed by backward fill

Prices are normalized to the range 0,1

Technical indicators are automatically computed for all datasets
