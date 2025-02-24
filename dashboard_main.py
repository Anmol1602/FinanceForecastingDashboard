import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import h5py
from datetime import datetime, timedelta



def load_models():
    """Load all three LSTM models"""
    try:
        stock_model = load_model('stock_lstm_model.h5')
        crypto_model = load_model('crypto_lstm_model.h5')
        forex_model = load_model('forex_lstm_model.h5')
        return stock_model, crypto_model, forex_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

def load_scalers():
    """Load and initialize all three scalers"""
    try:
        stock_scaler = MinMaxScaler()
        crypto_scaler = MinMaxScaler()
        forex_scaler = MinMaxScaler()

        # Load scaler parameters
        stock_params = np.load('stock_lstm_model.h5_scaler.npy', allow_pickle=True).item()
        crypto_params = np.load('crypto_lstm_model.h5_scaler.npy', allow_pickle=True).item()
        forex_params = np.load('forex_lstm_model.h5_scaler.npy', allow_pickle=True).item()
        
        # Initialize scalers with saved parameters
        for scaler, params in [(stock_scaler, stock_params),
                             (crypto_scaler, crypto_params),
                             (forex_scaler, forex_params)]:
            scaler.min_ = params['min_']
            scaler.scale_ = params['scale_']
            scaler.data_min_ = params['data_min_']
            scaler.data_max_ = params['data_max_']
            scaler.data_range_ = params['data_range_']
            scaler.feature_range = (0, 1)
            scaler.n_features_in_ = 1
            scaler.n_samples_seen_ = params.get('n_samples_seen_', None)
            
        return stock_scaler, crypto_scaler, forex_scaler
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        return None, None, None

def load_predictions(asset_type):
    """Load predictions from HDF5 file"""
    try:
        with h5py.File(f'{asset_type.lower()}_predictions.h5', 'r') as f:
            predictions = np.array(f['predictions'])
            dates = pd.to_datetime([date.decode('ascii') for date in f['dates']])
            actual = np.array(f['actual'])
            
        pred_df = pd.DataFrame({
            'actual': actual,
            'predicted': predictions.flatten()
        }, index=dates)
        
        return pred_df
    except Exception as e:
        st.error(f"❌ Error loading predictions for {asset_type}: {e}")
        return None

def predict_future(model, last_sequence, scaler):
    """Generate future predictions"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(30):  # Predict next 30 days
        current_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        future_predictions.append(scaler.inverse_transform(current_pred)[0,0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = current_pred

    return future_predictions

def create_forecast_plot(historical_data, future_data, asset_type):
    """Create interactive plot with historical and future data"""
    fig = go.Figure()

    # Historical actual values
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['actual'],
        name='Actual',
        line=dict(color='blue')
    ))

    # Historical predictions
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['predicted'],
        name='Historical Predictions',
        line=dict(color='green', dash='dash')
    ))

    # Future predictions
    future_dates = pd.date_range(
        start=historical_data.index[-1] + timedelta(days=1),
        periods=len(future_data)
    )
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_data,
        name='Future Forecast',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f'{asset_type} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

def main():
    st.title("📈 Financial Forecasting Dashboard")
    st.markdown("---")

    # Load models and scalers
    stock_model, crypto_model, forex_model = load_models()
    stock_scaler, crypto_scaler, forex_scaler = load_scalers()

    if None in (stock_model, crypto_model, forex_model, stock_scaler, crypto_scaler, forex_scaler):
        st.error("Failed to load models or scalers. Please check the error messages above.")
        return

    # Create tabs for different asset types
    tabs = st.tabs(["Stocks", "Cryptocurrency", "Forex"])

    # Load historical predictions
    stock_pred = load_predictions("stock")
    crypto_pred = load_predictions("crypto")
    forex_pred = load_predictions("forex")

    # Generate future predictions
    with st.spinner("Generating future predictions..."):
        # Stocks
        stock_last_sequence = stock_scaler.transform(
            stock_pred['actual'].values[-50:].reshape(-1, 1)
        )
        stock_future = predict_future(stock_model, stock_last_sequence, stock_scaler)

        # Crypto
        crypto_last_sequence = crypto_scaler.transform(
            crypto_pred['actual'].values[-50:].reshape(-1, 1)
        )
        crypto_future = predict_future(crypto_model, crypto_last_sequence, crypto_scaler)

        # Forex
        forex_last_sequence = forex_scaler.transform(
            forex_pred['actual'].values[-50:].reshape(-1, 1)
        )
        forex_future = predict_future(forex_model, forex_last_sequence, forex_scaler)

    # Stock Tab
    with tabs[0]:
        st.header("Stock Market Forecast")
        st.plotly_chart(
            create_forecast_plot(stock_pred, stock_future, "Stock"),
            use_container_width=True
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            last_actual = stock_pred['actual'].iloc[-1]
            st.metric("Latest Actual Price", f"${last_actual:.2f}")
        with col2:
            next_pred = stock_future[0]
            change = ((next_pred - last_actual) / last_actual) * 100
            st.metric("Next Day Forecast", f"${next_pred:.2f}", f"{change:+.2f}%")
        with col3:
            forecast_30d = stock_future[-1]
            change_30d = ((forecast_30d - last_actual) / last_actual) * 100
            st.metric("30-Day Forecast", f"${forecast_30d:.2f}", f"{change_30d:+.2f}%")

    # Crypto Tab
    with tabs[1]:
        st.header("Cryptocurrency Forecast")
        st.plotly_chart(
            create_forecast_plot(crypto_pred, crypto_future, "Cryptocurrency"),
            use_container_width=True
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            last_actual = crypto_pred['actual'].iloc[-1]
            st.metric("Latest Actual Price", f"${last_actual:.2f}")
        with col2:
            next_pred = crypto_future[0]
            change = ((next_pred - last_actual) / last_actual) * 100
            st.metric("Next Day Forecast", f"${next_pred:.2f}", f"{change:+.2f}%")
        with col3:
            forecast_30d = crypto_future[-1]
            change_30d = ((forecast_30d - last_actual) / last_actual) * 100
            st.metric("30-Day Forecast", f"${forecast_30d:.2f}", f"{change_30d:+.2f}%")

    # Forex Tab
    with tabs[2]:
        st.header("Forex Forecast")
        st.plotly_chart(
            create_forecast_plot(forex_pred, forex_future, "Forex"),
            use_container_width=True
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            last_actual = forex_pred['actual'].iloc[-1]
            st.metric("Latest Actual Rate", f"{last_actual:.4f}")
        with col2:
            next_pred = forex_future[0]
            change = ((next_pred - last_actual) / last_actual) * 100
            st.metric("Next Day Forecast", f"{next_pred:.4f}", f"{change:+.2f}%")
        with col3:
            forecast_30d = forex_future[-1]
            change_30d = ((forecast_30d - last_actual) / last_actual) * 100
            st.metric("30-Day Forecast", f"{forecast_30d:.4f}", f"{change_30d:+.2f}%")

if __name__ == "__main__":
    main()
