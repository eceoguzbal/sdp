"""
STOKASPORT - AI-Powered Portfolio Optimization System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append('.')
import config
from modules.data_loader import DataLoader
from modules.statistical_models import StatisticalForecaster
from modules.ml_models import MLPipeline, create_ml_forecasts
from modules.lstm_models import LSTMPipeline, create_lstm_forecasts
from modules.ensemble import EnsembleForecaster, create_ensemble_forecast
from modules.stochastic_simulation import MonteCarloSimulator
from modules.portfolio_optimizer import PortfolioOptimizer
from modules.backtesting import Backtester
from modules.evaluation import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main title
st.markdown('<div class="main-header">📈 STOKASPORT</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">AI-Powered Dynamic Portfolio Optimization System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# User inputs
investment_amount = st.sidebar.number_input(
    "💰 Investment Amount ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000
)

risk_tolerance = st.sidebar.selectbox(
    "🎯 Risk Tolerance",
    options=['Low', 'Medium', 'High'],
    index=1
)

prediction_horizon = st.sidebar.selectbox(
    "📅 Prediction Horizon",
    options=[1, 5],
    format_func=lambda x: f"{x} day(s)",
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Options")

run_monte_carlo = st.sidebar.checkbox("Run Monte Carlo Simulation", value=True)
generate_efficient_frontier = st.sidebar.checkbox("Generate Efficient Frontier", value=True)

st.sidebar.markdown("---")

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Data Overview",
    "🤖 Model Training",
    "🧠 LSTM Deep Learning",
    "📈 Predictions",
    "💼 Portfolio Optimization",
    "📊 Performance Analysis"
])

# TAB 1: Data Overview
with tab1:
    st.markdown('<div class="sub-header">📁 Data Overview</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Load Data", type="primary", key="load_data_btn"):
        with st.spinner("Loading data from JSON files..."):
            try:
                loader = DataLoader()
                st.session_state.loader = loader
                st.session_state.asset_data = loader.load_all_assets()
                st.session_state.combined_df = loader.get_combined_dataframe()
                st.session_state.returns_df = loader.calculate_returns(st.session_state.combined_df)
                st.session_state.data_loaded = True
                st.success(f"✅ Successfully loaded {len(st.session_state.asset_data)} assets!")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    if st.session_state.data_loaded:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assets", len(st.session_state.asset_data))
        with col2:
            st.metric("Date Range", 
                     f"{st.session_state.combined_df.index[0].date()} to {st.session_state.combined_df.index[-1].date()}")
        with col3:
            st.metric("Total Observations", len(st.session_state.combined_df))
        
        # Asset summary
        st.markdown("### 📋 Asset Summary")
        summary = st.session_state.loader.get_data_summary()
        st.dataframe(summary, use_container_width=True)
        
        # Price visualization
        st.markdown("### 📊 Price History")
        selected_assets = st.multiselect(
            "Select assets to visualize:",
            options=st.session_state.combined_df.columns.tolist(),
            default=st.session_state.combined_df.columns.tolist()[:5]
        )
        
        if selected_assets:
            fig = go.Figure()
            for asset in selected_assets:
                fig.add_trace(go.Scatter(
                    x=st.session_state.combined_df.index,
                    y=st.session_state.combined_df[asset],
                    mode='lines',
                    name=asset
                ))
            fig.update_layout(
                title="Asset Price History",
                xaxis_title="Date",
                yaxis_title="Price",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        st.markdown("### 📈 Returns Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Returns Statistics**")
            returns_stats = st.session_state.returns_df.describe().T
            st.dataframe(returns_stats, use_container_width=True)
        
        with col2:
            st.write("**Correlation Matrix**")
            corr_matrix = st.session_state.returns_df.corr()
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

# TAB 2: Model Training
with tab2:
    st.markdown('<div class="sub-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Overview tab.")
    else:
        st.markdown("""
        This system trains the following models:
        - **Statistical Models**: ARIMA, SARIMA, VAR
        - **Machine Learning Models**: XGBoost, LightGBM
        - **Deep Learning Models**: LSTM (Standard, Bidirectional, Advanced)
        - **Ensemble**: Performance-weighted combination
        """)
        
        if st.button("🚀 Train All Models", type="primary", key="train_models_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Select subset of assets for faster training (for demo)
                n_assets_to_train = min(10, len(st.session_state.combined_df.columns))
                selected_train_assets = st.session_state.combined_df.columns[:n_assets_to_train].tolist()
                
                st.info(f"Training models on {n_assets_to_train} assets for demonstration...")
                
                # Train statistical models
                status_text.text("Training statistical models (ARIMA, SARIMA, VAR)...")
                progress_bar.progress(0.15)
                
                stat_forecaster = StatisticalForecaster()
                
                # Train ARIMA/SARIMA for each asset
                for i, asset in enumerate(selected_train_assets):
                    stat_forecaster.fit_all_models(st.session_state.combined_df, asset)
                
                # Train VAR model
                stat_forecaster.fit_var_model(st.session_state.combined_df[selected_train_assets])
                
                st.session_state.stat_forecaster = stat_forecaster
                progress_bar.progress(0.35)
                
                # Train ML models
                status_text.text("Training machine learning models (XGBoost, LightGBM)...")
                ml_models, ml_predictions = create_ml_forecasts(
                    st.session_state.combined_df[selected_train_assets],
                    selected_train_assets,
                    horizon=prediction_horizon
                )
                
                st.session_state.ml_models = ml_models
                st.session_state.ml_predictions = ml_predictions
                progress_bar.progress(0.95)
                st.session_state.trained_assets = selected_train_assets
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training complete!")
                
                st.session_state.models_trained = True
                st.success(f"✅ Successfully trained statistical and ML models on {n_assets_to_train} assets!")
                
                # Display model summary
                st.markdown("### 📋 Model Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Statistical Models**")
                    model_summary = stat_forecaster.get_model_summary()
                    st.dataframe(model_summary, use_container_width=True)
                
                with col2:
                    st.write("**Machine Learning Models**")
                    ml_summary = pd.DataFrame({
                        'Model': ['XGBoost', 'LightGBM'],
                        'Type': ['Gradient Boosting', 'Gradient Boosting'],
                        'Assets Trained': [n_assets_to_train, n_assets_to_train]
                    })
                    st.dataframe(ml_summary, use_container_width=True)
                
                st.info("💡 To train LSTM deep learning models, go to the 'LSTM Deep Learning' tab →")
                
            except Exception as e:
                st.error(f"❌ Error training models: {e}")
                import traceback
                st.code(traceback.format_exc())

# TAB 3: LSTM Deep Learning
with tab3:
    st.markdown('<div class="sub-header">🧠 LSTM Deep Learning Models</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Overview tab.")
    else:
        st.markdown("""
        ### Long Short-Term Memory (LSTM) Networks
        
        LSTM is a type of recurrent neural network (RNN) designed to learn long-term dependencies in sequential data.
        Perfect for financial time series forecasting where past patterns influence future prices.
        
        **Available LSTM Architectures:**
        - **Standard LSTM**: Basic 2-layer architecture (fast, efficient)
        - **Bidirectional LSTM**: Processes sequences in both directions (better accuracy)
        - **Advanced LSTM**: Deep 3-layer architecture (captures complex patterns)
        """)
        
        # LSTM Configuration
        st.markdown("### ⚙️ LSTM Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lstm_lookback = st.number_input(
                "Lookback Window (days)",
                min_value=20,
                max_value=120,
                value=config.LSTM_LOOKBACK,
                step=10,
                help="Number of past days to use for prediction"
            )
        
        with col2:
            lstm_epochs = st.number_input(
                "Training Epochs",
                min_value=10,
                max_value=100,
                value=config.LSTM_EPOCHS,
                step=10,
                help="Number of training iterations"
            )
        
        with col3:
            lstm_units = st.number_input(
                "LSTM Units",
                min_value=25,
                max_value=200,
                value=config.LSTM_UNITS,
                step=25,
                help="Number of LSTM memory cells"
            )
        
        # Model selection
        st.markdown("### 🎯 Select LSTM Architectures to Train")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_standard = st.checkbox("Standard LSTM", value=True)
        with col2:
            train_bidirectional = st.checkbox("Bidirectional LSTM", value=True)
        with col3:
            train_advanced = st.checkbox("Advanced LSTM", value=False)
        
        # Asset selection
        if st.session_state.data_loaded:
            n_assets_for_lstm = st.slider(
                "Number of assets to train LSTM on",
                min_value=1,
                max_value=min(10, len(st.session_state.combined_df.columns)),
                value=min(5, len(st.session_state.combined_df.columns)),
                help="LSTM training is time-intensive. Start with fewer assets."
            )
            
            st.info(f"⏱️ Estimated training time: {n_assets_for_lstm * 3}-{n_assets_for_lstm * 5} minutes")
        
        # Training button
        if st.button("🚀 Train LSTM Models", type="primary", key="train_lstm_btn"):
            
            # Check if TensorFlow is available
            try:
                import tensorflow as tf
                tf_available = True
            except ImportError:
                tf_available = False
            
            if not tf_available:
                st.error("❌ TensorFlow is not installed!")
                st.markdown("""
                Please install TensorFlow to use LSTM models:
                
                ```bash
                pip install tensorflow==2.15.0
                ```
                
                Or update all requirements:
                ```bash
                pip install -r requirements.txt
                ```
                """)
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Select assets
                    selected_lstm_assets = st.session_state.combined_df.columns[:n_assets_for_lstm].tolist()
                    
                    status_text.text(f"Training LSTM models on {n_assets_for_lstm} assets...")
                    
                    # Determine which models to train
                    model_types = []
                    if train_standard:
                        model_types.append('standard')
                    if train_bidirectional:
                        model_types.append('bidirectional')
                    if train_advanced:
                        model_types.append('advanced')
                    
                    if not model_types:
                        st.warning("⚠️ Please select at least one LSTM architecture to train.")
                    else:
                        all_lstm_models = {}
                        all_lstm_predictions = {}
                        training_logs = []
                        
                        for idx, asset in enumerate(selected_lstm_assets):
                            status_text.text(f"Training LSTM for {asset} ({idx+1}/{n_assets_for_lstm})...")
                            progress = (idx / n_assets_for_lstm) * 0.9
                            progress_bar.progress(progress)
                            
                            try:
                                # Create LSTM pipeline
                                lstm_pipeline = LSTMPipeline(lookback=lstm_lookback)
                                
                                # Get asset data
                                asset_data = st.session_state.combined_df[asset]
                                
                                # Train models
                                lstm_pipeline.fit_models(
                                    asset_data,
                                    asset,
                                    model_types=model_types
                                )
                                
                                # Generate predictions
                                predictions = lstm_pipeline.predict_all(
                                    asset_data,
                                    steps=prediction_horizon
                                )
                                
                                all_lstm_models[asset] = lstm_pipeline
                                all_lstm_predictions[asset] = predictions
                                
                                # Log success
                                training_logs.append({
                                    'Asset': asset,
                                    'Status': '✅ Success',
                                    'Models Trained': len(model_types)
                                })
                                
                            except Exception as e:
                                training_logs.append({
                                    'Asset': asset,
                                    'Status': f'❌ Failed: {str(e)[:50]}',
                                    'Models Trained': 0
                                })
                                continue
                        
                        # Save to session state
                        st.session_state.lstm_models = all_lstm_models
                        st.session_state.lstm_predictions = all_lstm_predictions
                        st.session_state.lstm_trained_assets = selected_lstm_assets
                        st.session_state.lstm_available = True
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ LSTM training complete!")
                        
                        # Display training summary
                        st.success(f"✅ Successfully trained LSTM models on {len(all_lstm_models)} assets!")
                        
                        st.markdown("### 📋 Training Summary")
                        training_df = pd.DataFrame(training_logs)
                        st.dataframe(training_df, use_container_width=True)
                        
                        # Show model details for first asset
                        if len(all_lstm_models) > 0:
                            st.markdown("### 🔍 Model Details (First Asset)")
                            first_asset = selected_lstm_assets[0]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Model Architecture**")
                                model_summary = all_lstm_models[first_asset].get_model_summary()
                                st.dataframe(model_summary, use_container_width=True)
                            
                            with col2:
                                st.write("**Training History**")
                                history_dict = all_lstm_models[first_asset].get_training_history_summary()
                                
                                if history_dict:
                                    # Show final metrics
                                    metrics_data = []
                                    for model_name, history_df in history_dict.items():
                                        final_loss = history_df['val_loss'].iloc[-1]
                                        final_mae = history_df['val_mae'].iloc[-1]
                                        metrics_data.append({
                                            'Model': model_name.split('_')[-1],
                                            'Final Val Loss': f"{final_loss:.4f}",
                                            'Final Val MAE': f"{final_mae:.4f}"
                                        })
                                    
                                    metrics_df = pd.DataFrame(metrics_data)
                                    st.dataframe(metrics_df, use_container_width=True)
                        
                        # Visualization
                        st.markdown("### 📊 Training History Visualization")
                        
                        if len(all_lstm_models) > 0:
                            selected_viz_asset = st.selectbox(
                                "Select asset to visualize training history:",
                                options=selected_lstm_assets,
                                key="lstm_viz_asset"
                            )
                            
                            history_dict = all_lstm_models[selected_viz_asset].get_training_history_summary()
                            
                            if history_dict:
                                for model_name, history_df in history_dict.items():
                                    st.write(f"**{model_name}**")
                                    
                                    fig = make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Loss Over Epochs', 'MAE Over Epochs')
                                    )
                                    
                                    # Loss plot
                                    fig.add_trace(
                                        go.Scatter(
                                            y=history_df['loss'],
                                            name='Train Loss',
                                            line=dict(color='blue')
                                        ),
                                        row=1, col=1
                                    )
                                    fig.add_trace(
                                        go.Scatter(
                                            y=history_df['val_loss'],
                                            name='Val Loss',
                                            line=dict(color='red')
                                        ),
                                        row=1, col=1
                                    )
                                    
                                    # MAE plot
                                    fig.add_trace(
                                        go.Scatter(
                                            y=history_df['mae'],
                                            name='Train MAE',
                                            line=dict(color='green')
                                        ),
                                        row=1, col=2
                                    )
                                    fig.add_trace(
                                        go.Scatter(
                                            y=history_df['val_mae'],
                                            name='Val MAE',
                                            line=dict(color='orange')
                                        ),
                                        row=1, col=2
                                    )
                                    
                                    fig.update_xaxes(title_text="Epoch", row=1, col=1)
                                    fig.update_xaxes(title_text="Epoch", row=1, col=2)
                                    fig.update_yaxes(title_text="Loss", row=1, col=1)
                                    fig.update_yaxes(title_text="MAE", row=1, col=2)
                                    
                                    fig.update_layout(height=400, showlegend=True)
                                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error training LSTM models: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Show existing LSTM models if available
        if st.session_state.get('lstm_available', False):
            st.markdown("---")
            st.markdown("### ✅ Trained LSTM Models")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Assets with LSTM",
                    len(st.session_state.get('lstm_trained_assets', []))
                )
            
            with col2:
                total_models = sum(
                    len(pipeline.models) 
                    for pipeline in st.session_state.lstm_models.values()
                )
                st.metric("Total LSTM Models", total_models)
            
            with col3:
                st.metric("Status", "Ready ✅")

# TAB 4: Predictions
with tab4:
    st.markdown('<div class="sub-header">📈 Price Predictions</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training tab.")
    else:
        selected_asset_pred = st.selectbox(
            "Select Asset for Prediction Visualization:",
            options=st.session_state.trained_assets
        )
        
        if st.button("🔮 Generate Predictions", key="generate_predictions_btn"):
            with st.spinner("Generating predictions..."):
                try:
                    # Get predictions from all models
                    asset_idx = st.session_state.trained_assets.index(selected_asset_pred)
                    
                    # Statistical predictions
                    stat_preds = st.session_state.stat_forecaster.predict_all(steps=prediction_horizon)
                    
                    # ML predictions (already stored)
                    ml_preds = st.session_state.ml_predictions[selected_asset_pred]
                    
                    # LSTM predictions
                    lstm_preds = {}
                    if st.session_state.get('lstm_available', False) and selected_asset_pred in st.session_state.lstm_predictions:
                        lstm_preds = st.session_state.lstm_predictions[selected_asset_pred]
                    
                    # Combine predictions
                    all_predictions = {}
                    
                    # Add statistical model predictions
                    for key, value in stat_preds.items():
                        if selected_asset_pred in key and value is not None:
                            all_predictions[key] = value if isinstance(value, np.ndarray) else np.array([value])
                    
                    # Add ML model predictions
                    for key, value in ml_preds.items():
                        if value is not None:
                            all_predictions[key] = value[:prediction_horizon]
                    
                    # Add LSTM predictions
                    for key, value in lstm_preds.items():
                        if value is not None:
                            all_predictions[key] = value[:prediction_horizon]
                    
                    st.session_state.all_predictions = all_predictions
                    st.session_state.selected_asset_pred = selected_asset_pred
                    
                    # Create visualization
                    st.markdown("### 📊 Prediction Results")
                    
                    # Tabs for different model types
                    pred_tab1, pred_tab2, pred_tab3 = st.tabs(["All Models", "Model Comparison", "LSTM Details"])
                    
                    with pred_tab1:
                        # Show predictions table
                        pred_df = pd.DataFrame(all_predictions, index=[f"Day {i+1}" for i in range(prediction_horizon)])
                        st.dataframe(pred_df.T, use_container_width=True)
                        
                        # Visualization
                        fig = go.Figure()
                        
                        # Historical prices
                        historical = st.session_state.combined_df[selected_asset_pred].tail(60)
                        fig.add_trace(go.Scatter(
                            x=historical.index,
                            y=historical.values,
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Predictions
                        last_date = historical.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=prediction_horizon+1, freq='D')[1:]
                        
                        for model_name, pred in all_predictions.items():
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=pred,
                                mode='lines+markers',
                                name=model_name
                            ))
                        
                        fig.update_layout(
                            title=f'Price Predictions for {selected_asset_pred}',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=500,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with pred_tab2:
                        # Compare model predictions
                        st.markdown("**Prediction Variance Analysis**")
                        
                        pred_values = [pred[0] if len(pred) > 0 else np.nan for pred in all_predictions.values()]
                        comparison_df = pd.DataFrame({
                            'Model': list(all_predictions.keys()),
                            'Day 1 Prediction': pred_values
                        })
                        comparison_df['Deviation from Mean'] = comparison_df['Day 1 Prediction'] - comparison_df['Day 1 Prediction'].mean()
                        comparison_df = comparison_df.sort_values('Day 1 Prediction', ascending=False)
                        
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Box plot
                        fig_box = go.Figure()
                        fig_box.add_trace(go.Box(
                            y=pred_values,
                            name='Predictions',
                            boxmean='sd'
                        ))
                        fig_box.update_layout(
                            title='Prediction Distribution Across Models',
                            yaxis_title='Predicted Price',
                            height=400
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    with pred_tab3:
                        if st.session_state.get('lstm_available', False) and selected_asset_pred in st.session_state.lstm_models:
                            st.markdown("**LSTM Training History**")
                            
                            lstm_pipeline = st.session_state.lstm_models[selected_asset_pred]
                            history_dict = lstm_pipeline.get_training_history_summary()
                            
                            if history_dict:
                                # Show training history for each LSTM model
                                for model_name, history_df in history_dict.items():
                                    st.write(f"**{model_name}**")
                                    
                                    fig_history = make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Loss', 'MAE')
                                    )
                                    
                                    fig_history.add_trace(
                                        go.Scatter(y=history_df['loss'], name='Train Loss', line=dict(color='blue')),
                                        row=1, col=1
                                    )
                                    fig_history.add_trace(
                                        go.Scatter(y=history_df['val_loss'], name='Val Loss', line=dict(color='red')),
                                        row=1, col=1
                                    )
                                    
                                    fig_history.add_trace(
                                        go.Scatter(y=history_df['mae'], name='Train MAE', line=dict(color='green')),
                                        row=1, col=2
                                    )
                                    fig_history.add_trace(
                                        go.Scatter(y=history_df['val_mae'], name='Val MAE', line=dict(color='orange')),
                                        row=1, col=2
                                    )
                                    
                                    fig_history.update_layout(height=300, showlegend=True)
                                    st.plotly_chart(fig_history, use_container_width=True)
                        else:
                            st.info("LSTM models not available for this asset")
                    
                    st.success("✅ Predictions generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# TAB 5: Portfolio Optimization
with tab5:
    st.markdown('<div class="sub-header">💼 Portfolio Optimization</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first.")
    else:
        st.markdown("""
        ### Optimization Strategy
        - **Objective**: Maximize Sharpe Ratio
        - **Constraints**: Long-only portfolio with position limits
        - **Risk Adjustment**: Based on selected risk tolerance
        """)
        
        if st.button("⚡ Optimize Portfolio", type="primary", key="optimize_portfolio_btn"):
            with st.spinner("Optimizing portfolio..."):
                try:
                    # Calculate expected returns and covariance
                    returns_df = st.session_state.returns_df
                    expected_returns = returns_df.mean().values
                    cov_matrix = returns_df.cov().values
                    
                    # Initialize optimizer
                    optimizer = PortfolioOptimizer()
                    
                    # Get risk parameters
                    risk_params = config.RISK_TOLERANCE[risk_tolerance]
                    max_weight = risk_params['max_weight']
                    
                    # Optimize for maximum Sharpe ratio
                    optimal_weights = optimizer.optimize_sharpe_ratio(
                        expected_returns,
                        cov_matrix,
                        min_weight=config.MIN_WEIGHT,
                        max_weight=max_weight
                    )
                    
                    # Apply risk tolerance adjustment
                    adjusted_weights = optimizer.apply_risk_tolerance(
                        optimal_weights,
                        risk_level=risk_tolerance
                    )
                    
                    # Calculate portfolio metrics
                    p_return, p_std, sharpe = optimizer.calculate_portfolio_metrics(
                        adjusted_weights,
                        expected_returns,
                        cov_matrix
                    )
                    
                    # Store results
                    st.session_state.optimal_weights = adjusted_weights
                    st.session_state.optimizer = optimizer
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Annual Return", 
                                f"{p_return * config.TRADING_DAYS_PER_YEAR * 100:.2f}%")
                    with col2:
                        st.metric("Annual Volatility",
                                f"{p_std * np.sqrt(config.TRADING_DAYS_PER_YEAR) * 100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                    
                    # Portfolio allocation
                    st.markdown("### 📊 Portfolio Allocation")
                    
                    # Create portfolio DataFrame
                    portfolio_df = pd.DataFrame({
                        'Asset': returns_df.columns,
                        'Weight': adjusted_weights,
                        'Allocation ($)': adjusted_weights * investment_amount
                    })
                    portfolio_df = portfolio_df[portfolio_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(portfolio_df, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=portfolio_df['Asset'],
                            values=portfolio_df['Weight'],
                            hole=0.3
                        )])
                        fig_pie.update_layout(title="Portfolio Weights", height=400)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Efficient Frontier
                    if generate_efficient_frontier:
                        st.markdown("### 📈 Efficient Frontier")
                        with st.spinner("Generating efficient frontier..."):
                            frontier_df = optimizer.generate_efficient_frontier(
                                expected_returns,
                                cov_matrix,
                                n_points=50
                            )
                            
                            fig_ef = go.Figure()
                            fig_ef.add_trace(go.Scatter(
                                x=frontier_df['Volatility'],
                                y=frontier_df['Return'],
                                mode='lines',
                                name='Efficient Frontier',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Mark optimal portfolio
                            fig_ef.add_trace(go.Scatter(
                                x=[p_std * np.sqrt(config.TRADING_DAYS_PER_YEAR)],
                                y=[p_return * config.TRADING_DAYS_PER_YEAR],
                                mode='markers',
                                name='Optimal Portfolio',
                                marker=dict(color='red', size=12, symbol='star')
                            ))
                            
                            fig_ef.update_layout(
                                title='Efficient Frontier',
                                xaxis_title='Volatility (Annual)',
                                yaxis_title='Return (Annual)',
                                height=500,
                                hovermode='closest'
                            )
                            st.plotly_chart(fig_ef, use_container_width=True)
                    
                    # Monte Carlo Simulation
                    if run_monte_carlo:
                        st.markdown("### 🎲 Monte Carlo Simulation")
                        with st.spinner("Running Monte Carlo simulation..."):
                            simulator = MonteCarloSimulator(
                                n_simulations=config.MC_SIMULATIONS,
                                time_horizon=config.MC_TIME_HORIZON
                            )
                            
                            # Simulate portfolio
                            portfolio_paths = simulator.simulate_portfolio(
                                st.session_state.combined_df,
                                adjusted_weights,
                                returns_df
                            )
                            
                            # Scale by investment amount
                            portfolio_paths_scaled = portfolio_paths * investment_amount
                            
                            # Get statistics
                            sim_stats = simulator.get_simulation_statistics(portfolio_paths_scaled)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Final Value", f"${sim_stats['Mean_Final_Value']:,.0f}")
                            with col2:
                                st.metric("Median Final Value", f"${sim_stats['Median_Final_Value']:,.0f}")
                            with col3:
                                st.metric("95% VaR", f"${sim_stats['VaR_95'] * investment_amount:,.0f}")
                            with col4:
                                st.metric("95% CVaR", f"${sim_stats['CVaR_95'] * investment_amount:,.0f}")
                            
                            # Plot simulation paths
                            lower, median, upper = simulator.get_confidence_intervals(
                                portfolio_paths_scaled,
                                confidence_level=0.95
                            )
                            
                            fig_mc = go.Figure()
                            
                            # Add sample paths
                            for i in range(min(100, config.MC_SIMULATIONS)):
                                fig_mc.add_trace(go.Scatter(
                                    y=portfolio_paths_scaled[i],
                                    mode='lines',
                                    line=dict(color='lightgray', width=0.5),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                            
                            # Add confidence intervals
                            fig_mc.add_trace(go.Scatter(
                                y=upper,
                                mode='lines',
                                name='95% Upper Bound',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            fig_mc.add_trace(go.Scatter(
                                y=median,
                                mode='lines',
                                name='Median',
                                line=dict(color='blue', width=3)
                            ))
                            fig_mc.add_trace(go.Scatter(
                                y=lower,
                                mode='lines',
                                name='95% Lower Bound',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig_mc.update_layout(
                                title=f'Monte Carlo Simulation ({config.MC_SIMULATIONS} paths, {config.MC_TIME_HORIZON} days)',
                                xaxis_title='Trading Days',
                                yaxis_title='Portfolio Value ($)',
                                height=500,
                                showlegend=True
                            )
                            st.plotly_chart(fig_mc, use_container_width=True)
                    
                    st.success("✅ Portfolio optimization complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error optimizing portfolio: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# TAB 6: Performance Analysis
with tab6:
    st.markdown('<div class="sub-header">📊 Performance Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Model Performance & Portfolio Analysis
    
    This section provides comprehensive performance metrics and analysis for:
    - Prediction accuracy across all models
    - Portfolio optimization results
    - Risk-return analysis
    - Monte Carlo simulation outcomes
    """)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Overview tab.")
    elif not st.session_state.models_trained:
        st.warning("⚠️ Please train models first to see performance analysis.")
    else:
        st.markdown("### 🎯 Model Performance Comparison")
        
        # If predictions have been generated, show comparison
        if 'all_predictions' in st.session_state:
            predictions = st.session_state.all_predictions
            selected_asset = st.session_state.selected_asset_pred
            
            # Get actual recent prices for comparison
            actual_prices = st.session_state.combined_df[selected_asset].tail(20)
            
            st.markdown(f"#### Performance for {selected_asset}")
            
            # Create comparison metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Models Evaluated", len(predictions))
            
            with col2:
                pred_values = [pred[0] if len(pred) > 0 else np.nan for pred in predictions.values()]
                pred_std = np.std([p for p in pred_values if not np.isnan(p)])
                st.metric("Prediction Std Dev", f"${pred_std:.2f}")
            
            with col3:
                pred_mean = np.mean([p for p in pred_values if not np.isnan(p)])
                st.metric("Mean Prediction", f"${pred_mean:.2f}")
            
            # Model type breakdown
            st.markdown("#### 📊 Predictions by Model Type")
            
            model_types = {
                'Statistical': [k for k in predictions.keys() if 'ARIMA' in k or 'SARIMA' in k or 'VAR' in k],
                'Machine Learning': [k for k in predictions.keys() if 'XGBoost' in k or 'LightGBM' in k],
                'Deep Learning': [k for k in predictions.keys() if 'LSTM' in k]
            }
            
            type_summary = []
            for model_type, models in model_types.items():
                if models:
                    type_preds = [predictions[m][0] for m in models if m in predictions and len(predictions[m]) > 0]
                    if type_preds:
                        type_summary.append({
                            'Model Type': model_type,
                            'Count': len(models),
                            'Mean Prediction': np.mean(type_preds),
                            'Std Dev': np.std(type_preds)
                        })
            
            if type_summary:
                summary_df = pd.DataFrame(type_summary)
                st.dataframe(summary_df, use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                
                for model_type in model_types.keys():
                    type_models = [m for m in model_types[model_type] if m in predictions]
                    if type_models:
                        type_preds = [predictions[m][0] for m in type_models if len(predictions[m]) > 0]
                        fig.add_trace(go.Box(
                            y=type_preds,
                            name=model_type,
                            boxmean='sd'
                        ))
                
                fig.update_layout(
                    title='Prediction Distribution by Model Type',
                    yaxis_title='Predicted Price',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio Performance (if optimization was run)
        if 'optimal_weights' in st.session_state:
            st.markdown("---")
            st.markdown("### 💼 Portfolio Performance Analysis")
            
            weights = st.session_state.optimal_weights
            returns_df = st.session_state.returns_df
            
            # Calculate portfolio statistics
            expected_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values
            
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Annualize
            annual_return = portfolio_return * config.TRADING_DAYS_PER_YEAR
            annual_std = portfolio_std * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            sharpe_ratio = (annual_return - config.RISK_FREE_RATE) / annual_std
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Annual Return", f"{annual_return*100:.2f}%")
            with col2:
                st.metric("Annual Volatility", f"{annual_std*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
            with col4:
                diversification = 1 / np.sum(weights ** 2)
                st.metric("Diversification", f"{diversification:.1f}")
            
            # Asset allocation analysis
            st.markdown("#### 📊 Asset Allocation Analysis")
            
            # Top holdings
            portfolio_df = pd.DataFrame({
                'Asset': returns_df.columns,
                'Weight': weights
            })
            portfolio_df = portfolio_df[portfolio_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart of top holdings
                fig_bar = go.Figure()
                top_10 = portfolio_df.head(10)
                fig_bar.add_trace(go.Bar(
                    x=top_10['Asset'],
                    y=top_10['Weight'] * 100,
                    marker_color='skyblue'
                ))
                fig_bar.update_layout(
                    title='Top 10 Holdings',
                    xaxis_title='Asset',
                    yaxis_title='Weight (%)',
                    height=400
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Concentration metrics
                st.markdown("**Concentration Metrics**")
                
                top_5_weight = portfolio_df.head(5)['Weight'].sum()
                top_10_weight = portfolio_df.head(10)['Weight'].sum()
                herfindahl = np.sum(weights ** 2)
                
                concentration_df = pd.DataFrame({
                    'Metric': ['Top 5 Weight', 'Top 10 Weight', 'Herfindahl Index', 'Effective N'],
                    'Value': [
                        f"{top_5_weight*100:.1f}%",
                        f"{top_10_weight*100:.1f}%",
                        f"{herfindahl:.4f}",
                        f"{1/herfindahl:.1f}"
                    ]
                })
                st.dataframe(concentration_df, use_container_width=True)
        
        # Historical performance simulation
        st.markdown("---")
        st.markdown("### 📈 Historical Performance Simulation")
        
        st.info("This section shows how the portfolio would have performed on historical data.")
        
        # Simple historical simulation
        returns = st.session_state.returns_df
        
        # Equal weight benchmark
        equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
        equal_weight_returns = (returns.values @ equal_weights)
        
        cumulative_equal = (1 + equal_weight_returns).cumprod()
        
        # Plot
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Scatter(
            x=returns.index,
            y=cumulative_equal,
            mode='lines',
            name='Equal Weight Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        if 'optimal_weights' in st.session_state:
            optimized_returns = (returns.values @ st.session_state.optimal_weights)
            cumulative_optimized = (1 + optimized_returns).cumprod()
            
            fig_hist.add_trace(go.Scatter(
                x=returns.index,
                y=cumulative_optimized,
                mode='lines',
                name='Optimized Portfolio',
                line=dict(color='green', width=2)
            ))
        
        fig_hist.update_layout(
            title='Historical Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Performance metrics comparison
        if 'optimal_weights' in st.session_state:
            st.markdown("#### 📊 Performance Comparison")
            
            # Calculate metrics for both portfolios
            equal_return = equal_weight_returns.mean() * config.TRADING_DAYS_PER_YEAR
            equal_std = equal_weight_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            equal_sharpe = (equal_return - config.RISK_FREE_RATE) / equal_std
            
            opt_return = optimized_returns.mean() * config.TRADING_DAYS_PER_YEAR
            opt_std = optimized_returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            opt_sharpe = (opt_return - config.RISK_FREE_RATE) / opt_std
            
            comparison_df = pd.DataFrame({
                'Portfolio': ['Equal Weight', 'Optimized'],
                'Annual Return': [f"{equal_return*100:.2f}%", f"{opt_return*100:.2f}%"],
                'Annual Volatility': [f"{equal_std*100:.2f}%", f"{opt_std*100:.2f}%"],
                'Sharpe Ratio': [f"{equal_sharpe:.3f}", f"{opt_sharpe:.3f}"]
            })
            
            st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>STOKASPORT</strong> - AI-Powered Portfolio Optimization System</p>
    <p>Developed for Graduation Project | 2025</p>
</div>
""", unsafe_allow_html=True)