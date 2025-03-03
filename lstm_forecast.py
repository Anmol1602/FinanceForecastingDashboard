import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import h5py

def load_stock_data():
    """Load stock data from CSV"""
    try:
        stock_df = pd.read_csv('processed_stock_data.csv', parse_dates=["DATE"], index_col="DATE")
        return stock_df
    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def load_crypto_data():
    """Load cryptocurrency data from CSV"""
    try:
        crypto_df = pd.read_csv('processed_crypto_data.csv', parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
        return crypto_df
    except Exception as e:
        print(f"Error loading crypto data: {e}")
        return None

def load_forex_data():
    """Load forex data from CSV"""
    try:
        forex_df = pd.read_csv('processed_forex_data.csv', parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
        return forex_df
    except Exception as e:
        print(f"Error loading forex data: {e}")
        return None

def create_sequences(data, seq_length):
    """Create sequences for LSTM model with multiple features"""
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        # For labels, we'll predict all features for the next timestamp
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

def train_lstm(df, asset_name, model_filename, seq_length=50, epochs=20):
    """Train LSTM model with multiple features"""
    # Select features for prediction
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    
    # Prepare data - scale each feature independently
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model - adjust input shape for multiple features
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(len(features))  # Output layer predicts all features
    ])
    
    # Compile and train
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model
    model.save(model_filename)
    
    # Save scaler parameters
    scaler_params = {
        'min_': scaler.min_,
        'scale_': scaler.scale_,
        'data_min_': scaler.data_min_,
        'data_max_': scaler.data_max_,
        'data_range_': scaler.data_range_,
        'n_samples_seen_': scaler.n_samples_seen_,
        'feature_range': scaler.feature_range,
        'n_features_in_': scaler.n_features_in_
    }
    np.save(f"{model_filename}_scaler.npy", scaler_params)
    
    # Generate and save predictions
    generate_predictions(model, df, scaler, asset_name, seq_length, features)
    
    return model, history

def generate_predictions(model, df, scaler, asset_name, seq_length, features):
    """Generate and save predictions for multiple features"""
    # Prepare data
    scaled_data = scaler.transform(df[features])
    
    # Create sequences
    X, _ = create_sequences(scaled_data, seq_length)
    
    # Generate predictions
    scaled_predictions = model.predict(X)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(scaled_predictions)
    
    # Get dates
    pred_dates = df.index[seq_length:].map(lambda x: x.strftime('%Y-%m-%d')).values
    
    # Save predictions to HDF5
    with h5py.File(f'{asset_name.lower()}_predictions.h5', 'w') as f:
        # Save predictions for each feature
        for i, feature in enumerate(features):
            f.create_dataset(f'predictions_{feature}', data=predictions[:, i])
        
        # Save dates as ASCII strings
        dt_string = h5py.string_dtype(encoding='ascii')
        f.create_dataset('dates', data=pred_dates, dtype=dt_string)
        
        # Save actual values for each feature
        for i, feature in enumerate(features):
            f.create_dataset(f'actual_{feature}', data=df[feature].values[seq_length:])

def predict_future(model, last_sequence, scaler, n_future=30):
    """Generate future predictions for multiple features"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        scaled_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        pred = scaler.inverse_transform(scaled_pred)[0]
        future_predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = scaled_pred
    
    return np.array(future_predictions)

def load_predictions(asset_name):
    """Load predictions for multiple features"""
    try:
        features = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        with h5py.File(f'{asset_name.lower()}_predictions.h5', 'r') as f:
            dates = pd.to_datetime([date.decode('ascii') for date in f['dates']])
            
            pred_dict = {'actual_' + feat: np.array(f[f'actual_{feat}']) for feat in features}
            pred_dict.update({'predicted_' + feat: np.array(f[f'predictions_{feat}']) for feat in features})
            
        pred_df = pd.DataFrame(pred_dict, index=dates)
        return pred_df
    except Exception as e:
        print(f"Error loading predictions for {asset_name}: {e}")
        return None


def load_scaler(model_filename):
    """Load and initialize scaler from saved parameters"""
    try:
        scaler = MinMaxScaler()
        scaler_params = np.load(f'{model_filename}_scaler.npy', allow_pickle=True).item()
        
        # Set all required attributes
        scaler.min_ = scaler_params['min_']
        scaler.scale_ = scaler_params['scale_']
        scaler.data_min_ = scaler_params['data_min_']
        scaler.data_max_ = scaler_params['data_max_']
        scaler.data_range_ = scaler_params['data_range_']
        scaler.feature_range = scaler_params.get('feature_range', (0, 1))
        scaler.n_features_in_ = scaler_params.get('n_features_in_', 1)
        scaler.n_samples_seen_ = scaler_params.get('n_samples_seen_', None)
        
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def main():
    # Load all data
    stock_df = load_stock_data()
    crypto_df = load_crypto_data()
    forex_df = load_forex_data()

    if stock_df is None or crypto_df is None or forex_df is None:
        print("Error: Could not load one or more datasets")
        return

    # Train models and generate predictions
    print("\nTraining Stock Model...")
    stock_model, stock_history = train_lstm(stock_df, "Stock", "stock_lstm_model.h5")
    
    print("\nTraining Crypto Model...")
    crypto_model, crypto_history = train_lstm(crypto_df, "Crypto", "crypto_lstm_model.h5")
    
    print("\nTraining Forex Model...")
    forex_model, forex_history = train_lstm(forex_df, "Forex", "forex_lstm_model.h5")
    
    # Load and display predictions
    print("\nLoading predictions...")
    stock_predictions = load_predictions("stock")
    crypto_predictions = load_predictions("crypto")
    forex_predictions = load_predictions("forex")
    print("Stock Predictions:", stock_predictions)
    print("Crypto Predictions:", crypto_predictions)
    print("Forex Predictions:", forex_predictions)
    # Define features
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    
    # Generate future predictions for all assets
    for asset_name, df, model, model_file in [
        ("Stock", stock_df, stock_model, "stock_lstm_model.h5"),
        ("Crypto", crypto_df, crypto_model, "crypto_lstm_model.h5"),
        ("Forex", forex_df, forex_model, "forex_lstm_model.h5")
    ]:
        # Load the saved scaler
        scaler = load_scaler(model_file)
        if scaler is None:
            print(f"Error loading scaler for {asset_name}")
            continue
            
        # Prepare the last sequence using all features
        last_data = df[features].values[-50:]  # Get last 50 rows for all features
        scaled_data = scaler.transform(last_data)
        
        # Generate future predictions
        future_preds = predict_future(model, scaled_data, scaler)
        
        print(f"\n{asset_name} Future Predictions (next 30 days):")
        for i, feature in enumerate(features):
            print(f"\n{feature}:")
            print(future_preds[:, i])

        # Print model performance metrics
        print(f"\n{asset_name} Model Training History:")
        history = stock_history if asset_name == "Stock" else (crypto_history if asset_name == "Crypto" else forex_history)
        print(f"Final loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
