"""
Configuration file for STOKASPORT project
"""

# Data Configuration
DATA_PATH = "data/"
NUM_ASSETS = 35
PREDICTION_HORIZON = 1  # days
SECONDARY_HORIZON = 5  # days

# Time Series Configuration
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
WALK_FORWARD_WINDOW = 252  # trading days (1 year)
REBALANCING_FREQ = 5  # days (weekly)

# Model Configuration
ARIMA_MAX_P = 5
ARIMA_MAX_D = 2
ARIMA_MAX_Q = 5
SARIMA_SEASONAL_PERIOD = 5  # weekly seasonality

# Machine Learning Configuration
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

# LSTM Configuration
LSTM_LOOKBACK = 60  # Number of past time steps to use
LSTM_UNITS = 50  # Number of LSTM units
LSTM_DROPOUT = 0.2  # Dropout rate
LSTM_EPOCHS = 50  # Training epochs
LSTM_BATCH_SIZE = 32  # Batch size
LSTM_VALIDATION_SPLIT = 0.2  # Validation split

# Feature Engineering
LAG_FEATURES = [1, 2, 3, 5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20, 60]
TECHNICAL_INDICATORS = True

# Monte Carlo Configuration
MC_SIMULATIONS = 3000
MC_TIME_HORIZON = 20  # trading days

# Portfolio Optimization
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.15  # maximum 15% per asset
RISK_FREE_RATE = 0.02  # annual risk-free rate
TRADING_DAYS_PER_YEAR = 252

# Risk Tolerance Mapping
RISK_TOLERANCE = {
    'Low': {'target_volatility': 0.10, 'max_weight': 0.10},
    'Medium': {'target_volatility': 0.15, 'max_weight': 0.15},
    'High': {'target_volatility': 0.25, 'max_weight': 0.20}
}

# Evaluation Metrics
METRICS = ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']

# Streamlit Configuration
PAGE_TITLE = "STOKASPORT - AI Portfolio Optimizer"
PAGE_ICON = "📈"
LAYOUT = "wide"