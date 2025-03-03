import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import h5py
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
            scaler.n_features_in_ = params['n_features_in_']
            scaler.n_samples_seen_ = params.get('n_samples_seen_', None)
            
        return stock_scaler, crypto_scaler, forex_scaler
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        return None, None, None

def load_predictions(asset_type):
    """Load predictions from HDF5 file"""
    try:
        features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        with h5py.File(f'{asset_type.lower()}_predictions.h5', 'r') as f:
            dates = pd.to_datetime([date.decode('ascii') for date in f['dates']])
            
            pred_dict = {}
            for feature in features:
                if f'actual_{feature}' in f:
                    pred_dict[f'actual_{feature}'] = np.array(f[f'actual_{feature}'])
                if f'predictions_{feature}' in f:
                    pred_dict[f'predicted_{feature}'] = np.array(f[f'predictions_{feature}'])
            
        pred_df = pd.DataFrame(pred_dict, index=dates)
        return pred_df
    except Exception as e:
        st.error(f"❌ Error loading predictions for {asset_type}: {e}")
        return None

def predict_future(model, last_sequence, scaler):
    """Generate future predictions for multiple features"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(30):  # Predict next 30 days
        scaled_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        pred = scaler.inverse_transform(scaled_pred)[0]
        future_predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = scaled_pred
    
    return np.array(future_predictions)

def get_available_features(pred_df):
    """Get list of available features from the prediction dataframe"""
    features = set()
    for column in pred_df.columns:
        if column.startswith('actual_'):
            features.add(column.replace('actual_', ''))
    return sorted(list(features))

def create_multi_feature_plot(historical_data, future_data, asset_type, selected_features):
    """Create interactive plot with multiple features"""
    fig = go.Figure()
    
    # Add historical actual values for each selected feature
    for feature in selected_features:
        if f'actual_{feature}' in historical_data.columns:
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data[f'actual_{feature}'],
                name=f'Actual {feature}',
                line=dict(width=2)
            ))
            
            # Add historical predictions if available
            if f'predicted_{feature}' in historical_data.columns:
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data[f'predicted_{feature}'],
                    name=f'Predicted {feature}',
                    line=dict(dash='dash', width=2)
                ))
            
            # Add future predictions
            feature_idx = selected_features.index(feature)
            if feature_idx < future_data.shape[1]:
                future_dates = pd.date_range(
                    start=historical_data.index[-1] + timedelta(days=1),
                    periods=len(future_data)
                )
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_data[:, feature_idx],
                    name=f'Future {feature}',
                    line=dict(dash='dot', width=2)
                ))

    fig.update_layout(
        title=f'{asset_type} Multi-Feature Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_feature_correlation_plot(historical_data, features):
    """Create correlation heatmap for features"""
    correlation_data = {}
    for feature in features:
        correlation_data[feature] = historical_data[f'actual_{feature}']
    
    corr_matrix = pd.DataFrame(correlation_data).corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        template='plotly_white'
    )
    
    return fig

def create_prediction_accuracy_plot(historical_data, features):
    """Create prediction accuracy comparison plot"""
    fig = go.Figure()
    
    for feature in features:
        mse = mean_squared_error(
            historical_data[f'actual_{feature}'],
            historical_data[f'predicted_{feature}']
        )
        mae = mean_absolute_error(
            historical_data[f'actual_{feature}'],
            historical_data[f'predicted_{feature}']
        )
        
        fig.add_trace(go.Bar(
            name=feature,
            x=['MSE', 'MAE'],
            y=[mse, mae],
            text=[f'{mse:.4f}', f'{mae:.4f}'],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Prediction Accuracy Metrics by Feature',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def display_asset_analysis(tab, pred_df, predictions, asset_type, features):
    """Display analysis for a specific asset type"""
    # Get available features for this asset
    available_features = get_available_features(pred_df)
    
    if not available_features:
        st.error(f"No features found for {asset_type}")
        return
        
    selected_features = st.multiselect(
        "Select features to display",
        available_features,
        default=['CLOSE'] if available_features else [],
        key=f"{asset_type.lower()}_features"
    )
    
    if selected_features:
        # Main prediction plot
        st.plotly_chart(
            create_multi_feature_plot(
                pred_df, predictions,
                asset_type, selected_features
            ),
            use_container_width=True
        )
        
        # Only show correlation and accuracy if we have more than one feature
        if len(selected_features) > 1:
            # Correlation and accuracy analysis
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_feature_correlation_plot(pred_df, selected_features),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    create_prediction_accuracy_plot(pred_df, selected_features),
                    use_container_width=True
                )
        
        # Feature metrics
        st.subheader("Feature Metrics")
        cols = st.columns(len(selected_features))
        for i, feature in enumerate(selected_features):
            with cols[i]:
                if f'actual_{feature}' in pred_df.columns:
                    last_actual = pred_df[f'actual_{feature}'].iloc[-1]
                    next_pred = predictions[0][i] if i < predictions.shape[1] else None
                    
                    if next_pred is not None:
                        change = ((next_pred - last_actual) / last_actual) * 100
                        
                        if asset_type == "Forex":
                            st.metric(
                                f"{feature}",
                                f"{next_pred:.4f}",
                                f"{change:+.2f}%"
                            )
                        else:
                            st.metric(
                                f"{feature}",
                                f"${next_pred:.2f}",
                                f"{change:+.2f}%"
                            )

def main():
    st.title("📈 Advanced Financial Forecasting Dashboard")
    st.markdown("---")

    # Load models and scalers
    stock_model, crypto_model, forex_model = load_models()
    stock_scaler, crypto_scaler, forex_scaler = load_scalers()

    if None in (stock_model, crypto_model, forex_model, stock_scaler, crypto_scaler, forex_scaler):
        st.error("Failed to load models or scalers. Please check the error messages above.")
        return

    # Load historical predictions
    stock_pred = load_predictions("stock")
    crypto_pred = load_predictions("crypto")
    forex_pred = load_predictions("forex")

    # Create tabs for different asset types
    tabs = st.tabs(["Stocks", "Cryptocurrency", "Forex"])

    # Generate future predictions
    with st.spinner("Generating future predictions..."):
        predictions = {}
        for asset_type, pred_df, model, scaler in [
            ("stock", stock_pred, stock_model, stock_scaler),
            ("crypto", crypto_pred, crypto_model, crypto_scaler),
            ("forex", forex_pred, forex_model, forex_scaler)
        ]:
            if pred_df is not None:
                available_features = get_available_features(pred_df)
                if available_features:
                    last_sequence = np.column_stack([
                        pred_df[f'actual_{feature}'].values[-50:] 
                        for feature in available_features
                    ])
                    scaled_sequence = scaler.transform(last_sequence)
                    predictions[asset_type] = predict_future(model, scaled_sequence, scaler)
                else:
                    predictions[asset_type] = None
            else:
                predictions[asset_type] = None

    # Display analysis for each asset type
    with tabs[0]:
        st.header("Stock Market Analysis")
        if stock_pred is not None and predictions.get('stock') is not None:
            display_asset_analysis(tabs[0], stock_pred, predictions['stock'], "Stock", 
                                 get_available_features(stock_pred))
        else:
            st.error("Stock data not available")

    with tabs[1]:
        st.header("Cryptocurrency Analysis")
        if crypto_pred is not None and predictions.get('crypto') is not None:
            display_asset_analysis(tabs[1], crypto_pred, predictions['crypto'], "Cryptocurrency", 
                                 get_available_features(crypto_pred))
        else:
            st.error("Cryptocurrency data not available")

    with tabs[2]:
        st.header("Forex Analysis")
        if forex_pred is not None and predictions.get('forex') is not None:
            display_asset_analysis(tabs[2], forex_pred, predictions['forex'], "Forex", 
                                 get_available_features(forex_pred))
        else:
            st.error("Forex data not available")

if __name__ == "__main__":
    main()