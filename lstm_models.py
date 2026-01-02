"""
LSTM (Long Short-Term Memory) models for time series forecasting
Requires TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. LSTM models will not work.")
    print("   Install with: pip install tensorflow")

import config

class LSTMForecaster:
    """LSTM model for time series forecasting"""
    
    def __init__(self, lookback=60, units=50, dropout=0.2, bidirectional=False):
        """
        Initialize LSTM forecaster
        
        Args:
            lookback: Number of past time steps to use for prediction
            units: Number of LSTM units
            dropout: Dropout rate for regularization
            bidirectional: Whether to use Bidirectional LSTM
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def create_model(self, input_shape):
        """Create LSTM model architecture"""
        model = Sequential()
        
        if self.bidirectional:
            # Bidirectional LSTM
            model.add(Bidirectional(
                LSTM(units=self.units, return_sequences=True, input_shape=input_shape)
            ))
            model.add(Dropout(self.dropout))
            
            model.add(Bidirectional(LSTM(units=self.units // 2, return_sequences=False)))
            model.add(Dropout(self.dropout))
        else:
            # Standard LSTM
            model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(self.dropout))
            
            model.add(LSTM(units=self.units // 2, return_sequences=False))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        return model
    
    def prepare_data(self, data: pd.Series):
        """
        Prepare time series data for LSTM
        
        Args:
            data: Time series data
            
        Returns:
            X, y arrays ready for LSTM
        """
        # Scale data
        data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        X, y = [], []
        
        for i in range(self.lookback, len(data_scaled)):
            X.append(data_scaled[i - self.lookback:i, 0])
            y.append(data_scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def fit(self, data: pd.Series, validation_split=0.2, epochs=50, batch_size=32, verbose=0):
        """
        Train LSTM model
        
        Args:
            data: Training data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Create model
        self.model = self.create_model(input_shape=(X.shape[1], 1))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        return self
    
    def predict(self, data: pd.Series, steps: int = 1):
        """
        Make predictions
        
        Args:
            data: Historical data for prediction
            steps: Number of steps to predict
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Scale data
        data_scaled = self.scaler.transform(data.values.reshape(-1, 1))
        
        predictions = []
        
        # Get last lookback values
        current_batch = data_scaled[-self.lookback:].reshape(1, self.lookback, 1)
        
        for i in range(steps):
            # Predict next value
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def get_training_history(self):
        """Get training history"""
        if self.history is None:
            return None
        
        return pd.DataFrame({
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'mae': self.history.history['mae'],
            'val_mae': self.history.history['val_mae']
        })


class AdvancedLSTMForecaster(LSTMForecaster):
    """Advanced LSTM with additional features"""
    
    def __init__(self, lookback=60, units=100, dropout=0.3, layers=3):
        """
        Initialize advanced LSTM forecaster
        
        Args:
            lookback: Number of past time steps
            units: Number of LSTM units
            dropout: Dropout rate
            layers: Number of LSTM layers
        """
        super().__init__(lookback, units, dropout)
        self.layers = layers
    
    def create_model(self, input_shape):
        """Create advanced multi-layer LSTM architecture"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.units,
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout))
        
        # Middle layers
        for i in range(self.layers - 2):
            model.add(LSTM(
                units=self.units // (2 ** (i + 1)),
                return_sequences=True
            ))
            model.add(Dropout(self.dropout))
        
        # Last LSTM layer
        model.add(LSTM(units=self.units // (2 ** (self.layers - 1))))
        model.add(Dropout(self.dropout))
        
        # Dense layers
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1))
        
        # Compile with custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model


class LSTMPipeline:
    """Pipeline to manage multiple LSTM models"""
    
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.models = {}
        self.predictions = {}
        self.training_history = {}
    
    def fit_models(self, data: pd.Series, asset_name: str, 
                   model_types=['standard', 'bidirectional', 'advanced']):
        """
        Fit multiple LSTM variants
        
        Args:
            data: Time series data
            asset_name: Name of the asset
            model_types: List of model types to train
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available, skipping LSTM training")
            return self
        
        # Standard LSTM
        if 'standard' in model_types:
            try:
                print(f"  Training Standard LSTM for {asset_name}...")
                standard_lstm = LSTMForecaster(
                    lookback=self.lookback,
                    units=50,
                    dropout=0.2
                )
                standard_lstm.fit(data, epochs=50, verbose=0)
                self.models[f'{asset_name}_LSTM_Standard'] = standard_lstm
                self.training_history[f'{asset_name}_LSTM_Standard'] = standard_lstm.get_training_history()
            except Exception as e:
                print(f"    ⚠️  Standard LSTM failed: {e}")
        
        # Bidirectional LSTM
        if 'bidirectional' in model_types:
            try:
                print(f"  Training Bidirectional LSTM for {asset_name}...")
                bi_lstm = LSTMForecaster(
                    lookback=self.lookback,
                    units=50,
                    dropout=0.2,
                    bidirectional=True
                )
                bi_lstm.fit(data, epochs=50, verbose=0)
                self.models[f'{asset_name}_LSTM_Bidirectional'] = bi_lstm
                self.training_history[f'{asset_name}_LSTM_Bidirectional'] = bi_lstm.get_training_history()
            except Exception as e:
                print(f"    ⚠️  Bidirectional LSTM failed: {e}")
        
        # Advanced LSTM
        if 'advanced' in model_types:
            try:
                print(f"  Training Advanced LSTM for {asset_name}...")
                advanced_lstm = AdvancedLSTMForecaster(
                    lookback=self.lookback,
                    units=100,
                    dropout=0.3,
                    layers=3
                )
                advanced_lstm.fit(data, epochs=50, verbose=0)
                self.models[f'{asset_name}_LSTM_Advanced'] = advanced_lstm
                self.training_history[f'{asset_name}_LSTM_Advanced'] = advanced_lstm.get_training_history()
            except Exception as e:
                print(f"    ⚠️  Advanced LSTM failed: {e}")
        
        return self
    
    def predict_all(self, data: pd.Series, steps: int = 1):
        """Generate predictions from all fitted LSTM models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(data, steps=steps)
                predictions[model_name] = pred
            except Exception as e:
                print(f"  ⚠️  Error predicting with {model_name}: {e}")
                predictions[model_name] = None
        
        self.predictions = predictions
        return predictions
    
    def get_model_summary(self):
        """Get summary of all LSTM models"""
        summary = []
        
        for model_name, model in self.models.items():
            summary.append({
                'Model': model_name,
                'Type': 'LSTM',
                'Lookback': model.lookback,
                'Units': model.units,
                'Bidirectional': getattr(model, 'bidirectional', False)
            })
        
        return pd.DataFrame(summary)
    
    def get_training_history_summary(self):
        """Get training history for all models"""
        return self.training_history


def create_lstm_forecasts(data: pd.DataFrame, asset_names: list, 
                          lookback: int = 60, horizon: int = 1):
    """
    Create LSTM forecasts for multiple assets
    
    Args:
        data: Price data DataFrame
        asset_names: List of asset names
        lookback: Lookback window
        horizon: Prediction horizon
        
    Returns:
        Dictionary with LSTM predictions for each asset
    """
    if not TENSORFLOW_AVAILABLE:
        print("⚠️  TensorFlow not available. Install with: pip install tensorflow")
        return {}, {}
    
    all_predictions = {}
    all_models = {}
    
    for asset in asset_names:
        print(f"Training LSTM models for {asset}...")
        
        pipeline = LSTMPipeline(lookback=lookback)
        pipeline.fit_models(
            data[asset],
            asset,
            model_types=['standard', 'bidirectional']
        )
        
        # Make predictions
        predictions = pipeline.predict_all(data[asset], steps=horizon)
        
        all_predictions[asset] = predictions
        all_models[asset] = pipeline
    
    return all_models, all_predictions
