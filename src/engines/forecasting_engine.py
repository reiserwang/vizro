#!/usr/bin/env python3
"""
Forecasting Engine Module
Handles all forecasting models and time series analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import gradio as gr
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core import dashboard_config

def prepare_time_series_data(df, target_col, additional_cols=None):
    """Prepare data for time series forecasting"""
    try:
        # Create a simple time index if none exists
        if not any(df.dtypes == 'datetime64[ns]'):
            df = df.copy()
            
            # Limit the number of periods to prevent timestamp overflow
            max_periods = min(len(df), 10000)  # Limit to 10,000 periods max
            
            # Use daily frequency for smaller datasets, monthly for larger ones
            if max_periods <= 1000:
                freq = 'D'  # Daily
                start_date = '2020-01-01'
            elif max_periods <= 5000:
                freq = 'W'  # Weekly
                start_date = '2020-01-01'
            else:
                freq = 'M'  # Monthly
                start_date = '2020-01-01'
            
            try:
                # Create date range with safe limits
                df['time_index'] = pd.date_range(
                    start=start_date, 
                    periods=max_periods, 
                    freq=freq
                )[:len(df)]  # Truncate to actual data length
                df = df.set_index('time_index')
            except (pd.errors.OutOfBoundsDatetime, OverflowError):
                # Fallback: use simple integer index if date creation fails
                df['time_index'] = range(len(df))
                df = df.set_index('time_index')
        
        # Select target and additional columns
        if additional_cols:
            cols = [target_col] + [col for col in additional_cols if col in df.columns and col != target_col]
            ts_data = df[cols].dropna()
        else:
            ts_data = df[[target_col]].dropna()
        
        # Ensure we have enough data for forecasting
        if len(ts_data) < 3:
            raise ValueError("Insufficient data for forecasting (minimum 3 data points required)")
        
        return ts_data
    except Exception as e:
        raise ValueError(f"Error preparing time series data: {str(e)}")

def linear_regression_forecast(data, target_col, periods, confidence_level=0.95):
    """Simple linear regression forecasting"""
    try:
        y = data[target_col].values
        X = np.arange(len(y)).reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecasts
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Simple confidence intervals (assuming normal residuals)
        residuals = y - model.predict(X)
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        
        from scipy.stats import t
        alpha = 1 - confidence_level
        t_val = t.ppf(1 - alpha/2, len(y) - 2)
        margin = t_val * std_error
        
        lower_bound = forecast - margin
        upper_bound = forecast + margin
        
        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fitted_values': model.predict(X),
            'model_info': {
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r_squared': model.score(X, y),
                'mse': mse
            }
        }
    except Exception as e:
        raise ValueError(f"Linear regression forecast failed: {str(e)}")

def arima_forecast(data, target_col, periods, confidence_level=0.95):
    """ARIMA forecasting"""
    try:
        # Try to import statsmodels
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            raise ValueError("ARIMA requires statsmodels. Install with: pip install statsmodels")
        
        y = data[target_col].dropna()
        
        # Simple auto-ARIMA (try common configurations)
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                            best_order = (p, d, q)
                    except:
                        continue
        
        if best_model is None:
            raise ValueError("Could not fit ARIMA model. Try a simpler model.")
        
        # Generate forecasts
        forecast_result = best_model.forecast(steps=periods, alpha=1-confidence_level)
        forecast = forecast_result
        
        # Get confidence intervals
        forecast_ci = best_model.get_forecast(steps=periods, alpha=1-confidence_level).conf_int()
        
        return {
            'forecast': forecast,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values,
            'fitted_values': best_model.fittedvalues,
            'model_info': {
                'order': best_order,
                'aic': best_aic,
                'bic': best_model.bic,
                'log_likelihood': best_model.llf
            }
        }
    except Exception as e:
        raise ValueError(f"ARIMA forecast failed: {str(e)}")

def sarima_forecast(data, target_col, periods, seasonal_period, confidence_level=0.95):
    """SARIMA forecasting with seasonality"""
    try:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ValueError("SARIMA requires statsmodels. Install with: pip install statsmodels")
        
        y = data[target_col].dropna()
        
        # Simple SARIMA configuration
        try:
            model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
            fitted_model = model.fit(disp=False)
        except:
            # Fallback to simpler model
            model = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 1, seasonal_period))
            fitted_model = model.fit(disp=False)
        
        # Generate forecasts
        forecast_result = fitted_model.get_forecast(steps=periods, alpha=1-confidence_level)
        forecast = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        return {
            'forecast': forecast.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values,
            'fitted_values': fitted_model.fittedvalues,
            'model_info': {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'seasonal_period': seasonal_period,
                'log_likelihood': fitted_model.llf
            }
        }
    except Exception as e:
        raise ValueError(f"SARIMA forecast failed: {str(e)}")

def var_forecast(data, target_col, additional_cols, periods, confidence_level=0.95):
    """Vector Autoregression forecasting"""
    try:
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
        except ImportError:
            raise ValueError("VAR requires statsmodels. Install with: pip install statsmodels")
        
        # Prepare multivariate data
        cols = [target_col] + [col for col in additional_cols if col in data.columns and col != target_col]
        if len(cols) < 2:
            raise ValueError("VAR requires at least 2 variables. Please select additional variables.")
        
        var_data = data[cols].dropna()
        
        # Fit VAR model
        model = VAR(var_data)
        
        # Select optimal lag order (max 5 for performance)
        max_lags = min(5, len(var_data) // 4)
        lag_order = model.select_order(maxlags=max_lags)
        optimal_lags = lag_order.aic
        
        fitted_model = model.fit(optimal_lags)
        
        # Generate forecasts
        forecast_result = fitted_model.forecast_interval(var_data.values, steps=periods, alpha=1-confidence_level)
        
        # Extract results for target variable
        target_idx = cols.index(target_col)
        forecast = forecast_result[0][:, target_idx]
        lower_bound = forecast_result[1][:, target_idx]
        upper_bound = forecast_result[2][:, target_idx]
        
        return {
            'forecast': forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'fitted_values': fitted_model.fittedvalues[target_col],
            'model_info': {
                'lag_order': optimal_lags,
                'variables': cols,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'n_variables': len(cols)
            }
        }
    except Exception as e:
        raise ValueError(f"VAR forecast failed: {str(e)}")

def dynamic_factor_forecast(data, target_col, additional_cols, periods, confidence_level=0.95):
    """Dynamic Factor Model forecasting"""
    try:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ValueError("Dynamic Factor Model requires scikit-learn")
        
        # Prepare multivariate data
        cols = [target_col] + [col for col in additional_cols if col in data.columns and col != target_col]
        if len(cols) < 3:
            raise ValueError("Dynamic Factor Model requires at least 3 variables for meaningful factors.")
        
        factor_data = data[cols].dropna()
        
        # Extract factors using PCA
        n_factors = min(3, len(cols) - 1)
        pca = PCA(n_components=n_factors)
        factors = pca.fit_transform(factor_data.values)
        
        # Forecast factors using simple AR(1) model
        factor_forecasts = []
        for i in range(n_factors):
            factor_series = factors[:, i]
            # Simple AR(1) forecast
            if len(factor_series) > 1:
                ar_coef = np.corrcoef(factor_series[:-1], factor_series[1:])[0, 1]
                last_value = factor_series[-1]
                factor_forecast = [last_value * (ar_coef ** (j+1)) for j in range(periods)]
            else:
                factor_forecast = [0] * periods
            factor_forecasts.append(factor_forecast)
        
        factor_forecasts = np.array(factor_forecasts).T
        
        # Transform back to original space
        forecast = pca.inverse_transform(factor_forecasts)
        target_idx = cols.index(target_col)
        target_forecast = forecast[:, target_idx]
        
        # Simple confidence intervals based on factor variance
        factor_var = np.var(factors, axis=0)
        total_var = np.sum(factor_var)
        std_error = np.sqrt(total_var)
        margin = 1.96 * std_error  # Approximate 95% CI
        
        return {
            'forecast': target_forecast,
            'lower_bound': target_forecast - margin,
            'upper_bound': target_forecast + margin,
            'fitted_values': pca.inverse_transform(factors)[:, target_idx],
            'model_info': {
                'n_factors': n_factors,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': np.sum(pca.explained_variance_ratio_),
                'variables': cols
            }
        }
    except Exception as e:
        raise ValueError(f"Dynamic Factor Model forecast failed: {str(e)}")

def state_space_forecast(data, target_col, periods, confidence_level=0.95):
    """State-Space Model (Unobserved Components) forecasting"""
    try:
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError:
            raise ValueError("State-Space Model requires statsmodels. Install with: pip install statsmodels")
        
        y = data[target_col].dropna()
        
        # Fit Unobserved Components model
        try:
            model = UnobservedComponents(y, 'local linear trend', seasonal=12 if len(y) > 24 else None)
            fitted_model = model.fit(disp=False)
        except:
            # Fallback to simpler model
            model = UnobservedComponents(y, 'local level')
            fitted_model = model.fit(disp=False)
        
        # Generate forecasts
        forecast_result = fitted_model.get_forecast(steps=periods, alpha=1-confidence_level)
        forecast = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        return {
            'forecast': forecast.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values,
            'fitted_values': fitted_model.fittedvalues,
            'model_info': {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'log_likelihood': fitted_model.llf,
                'components': 'trend + seasonal' if hasattr(fitted_model, 'seasonal') else 'trend only'
            }
        }
    except Exception as e:
        raise ValueError(f"State-Space Model forecast failed: {str(e)}")

def nowcasting_forecast(data, target_col, periods, confidence_level=0.95):
    """Nowcasting - simple short-term forecasting"""
    try:
        y = data[target_col].dropna()
        
        if len(y) < 3:
            raise ValueError("Nowcasting requires at least 3 data points")
        
        # Simple exponential smoothing for nowcasting
        alpha = 0.3  # Smoothing parameter
        
        # Calculate smoothed values
        smoothed = [y.iloc[0]]
        for i in range(1, len(y)):
            smoothed.append(alpha * y.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast using last smoothed value with slight trend
        if len(y) >= 2:
            trend = (y.iloc[-1] - y.iloc[-2]) * 0.5  # Dampened trend
        else:
            trend = 0
        
        last_smoothed = smoothed[-1]
        forecast = [last_smoothed + trend * (i + 1) for i in range(periods)]
        
        # Simple confidence intervals
        residuals = y.values - np.array(smoothed)
        std_error = np.std(residuals)
        margin = 1.96 * std_error
        
        return {
            'forecast': np.array(forecast),
            'lower_bound': np.array(forecast) - margin,
            'upper_bound': np.array(forecast) + margin,
            'fitted_values': np.array(smoothed),
            'model_info': {
                'smoothing_parameter': alpha,
                'trend_component': trend,
                'mse': np.mean(residuals**2),
                'method': 'Exponential Smoothing'
            }
        }
    except Exception as e:
        raise ValueError(f"Nowcasting failed: {str(e)}")

def perform_forecasting(target_var, additional_vars, model_type, periods, seasonal_period, confidence_level, progress=gr.Progress()):
    """Main forecasting function"""
    
    if dashboard_config.current_data is None:
        return None, "⚠️ Please upload data first", "No data available for forecasting"
    
    if not target_var:
        return None, "⚠️ Please select a target variable", "Target variable not selected"
    
    try:
        progress(0.1, desc="📊 Preparing time series data...")
        
        # Prepare data with enhanced error handling
        try:
            ts_data = prepare_time_series_data(dashboard_config.current_data, target_var, additional_vars)
        except ValueError as ve:
            error_msg = str(ve)
            if "Out of bounds nanosecond timestamp" in error_msg:
                return None, "⚠️ Dataset too large for time series analysis. Please use a smaller subset (< 10,000 rows)", "Timestamp overflow error"
            elif "Insufficient data" in error_msg:
                return None, "⚠️ Not enough data for forecasting (minimum 3 data points required)", "Insufficient data"
            else:
                return None, f"⚠️ Data preparation failed: {error_msg}", "Data preparation error"
        
        if len(ts_data) < 10:
            return None, "⚠️ Insufficient data for reliable forecasting (minimum 10 points recommended)", "Not enough data"
        
        progress(0.3, desc=f"🤖 Fitting {model_type} model...")
        
        # Select and run forecasting model
        if model_type == "Linear Regression":
            result = linear_regression_forecast(ts_data, target_var, periods, confidence_level)
        elif model_type == "ARIMA":
            result = arima_forecast(ts_data, target_var, periods, confidence_level)
        elif model_type == "SARIMA":
            result = sarima_forecast(ts_data, target_var, periods, seasonal_period, confidence_level)
        elif model_type == "VAR (Vector Autoregression)":
            if not additional_vars:
                return None, "⚠️ VAR model requires additional variables", "Additional variables needed for VAR"
            result = var_forecast(ts_data, target_var, additional_vars, periods, confidence_level)
        elif model_type == "Dynamic Factor Model":
            if not additional_vars or len(additional_vars) < 2:
                return None, "⚠️ Dynamic Factor Model requires at least 2 additional variables", "More variables needed"
            result = dynamic_factor_forecast(ts_data, target_var, additional_vars, periods, confidence_level)
        elif model_type == "State-Space Model":
            result = state_space_forecast(ts_data, target_var, periods, confidence_level)
        elif model_type == "Nowcasting":
            result = nowcasting_forecast(ts_data, target_var, periods, confidence_level)
        else:
            return None, f"⚠️ Unknown model type: {model_type}", "Invalid model selection"
        
        progress(0.7, desc="📈 Creating forecast visualization...")
        
        # Create forecast plot
        fig = create_forecast_plot(ts_data, target_var, result, model_type, periods)
        
        progress(0.9, desc="📋 Generating forecast summary...")
        
        # Create summary and metrics
        summary, metrics = create_forecast_summary(result, model_type, target_var, periods, confidence_level)
        
        progress(1.0, desc="✅ Forecast complete!")
        
        return fig, summary, metrics
        
    except Exception as e:
        return None, f"❌ Forecasting failed: {str(e)}", f"Error: {str(e)}"

def create_forecast_plot(data, target_var, result, model_type, periods):
    """Create forecast visualization"""
    try:
        # Prepare historical data
        historical_values = data[target_var].values
        historical_dates = data.index
        
        # Handle different index types
        if pd.api.types.is_datetime64_any_dtype(historical_dates):
            # DateTime index - try to infer frequency
            last_date = historical_dates[-1]
            if hasattr(last_date, 'freq') and last_date.freq:
                freq = last_date.freq
            else:
                # Infer frequency from data
                if len(historical_dates) > 1:
                    try:
                        freq = pd.infer_freq(historical_dates) or 'M'
                    except:
                        freq = 'M'
                else:
                    freq = 'M'
            
            # Create future dates
            try:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), periods=periods, freq=freq)
            except:
                # Fallback: create simple date range
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        else:
            # Numeric index - create simple sequential dates
            if len(historical_dates) > 0:
                # Use the last index value as starting point
                start_idx = historical_dates[-1] + 1
                future_dates = pd.RangeIndex(start=start_idx, stop=start_idx + periods, step=1)
                
                # Convert to simple date range for plotting
                base_date = pd.Timestamp('2024-01-01')  # Use a base date
                historical_dates = pd.date_range(start=base_date, periods=len(historical_dates), freq='D')
                future_dates = pd.date_range(start=historical_dates[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
            else:
                # Fallback for empty data
                base_date = pd.Timestamp('2024-01-01')
                historical_dates = pd.date_range(start=base_date, periods=1, freq='D')
                future_dates = pd.date_range(start=base_date + pd.Timedelta(days=1), periods=periods, freq='D')
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Fitted values (if available)
        if 'fitted_values' in result and len(result['fitted_values']) == len(historical_values):
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=result['fitted_values'],
                mode='lines',
                name='Fitted Values',
                line=dict(color='green', width=1, dash='dot'),
                opacity=0.7
            ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=result['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence intervals
        if 'lower_bound' in result and 'upper_bound' in result:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=result['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=result['lower_bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name=f'{int((result.get("confidence_level", 0.95))*100)}% Confidence Interval',
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{model_type} Forecast for {target_var}',
            xaxis_title='Time',
            yaxis_title=target_var,
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        # Return error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating forecast plot: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red')
        )
        return fig

def create_forecast_summary(result, model_type, target_var, periods, confidence_level):
    """Create forecast summary and metrics"""
    try:
        # Basic forecast statistics
        forecast_values = result['forecast']
        forecast_mean = np.mean(forecast_values)
        forecast_std = np.std(forecast_values)
        forecast_min = np.min(forecast_values)
        forecast_max = np.max(forecast_values)
        
        # Model-specific information
        model_info = result.get('model_info', {})
        
        # Create summary
        summary = f"""
        ## 📈 {model_type} Forecast Results
        
        **Target Variable:** {target_var}  
        **Forecast Periods:** {periods}  
        **Confidence Level:** {confidence_level*100:.0f}%  
        
        ### 📊 Forecast Statistics
        - **Mean Forecast:** {forecast_mean:.4f}
        - **Forecast Range:** {forecast_min:.4f} to {forecast_max:.4f}
        - **Standard Deviation:** {forecast_std:.4f}
        
        ### 🔍 Model Information
        """
        
        # Add model-specific details
        if model_type == "Linear Regression":
            summary += f"""
        - **Slope:** {model_info.get('slope', 'N/A'):.4f}
        - **R-squared:** {model_info.get('r_squared', 'N/A'):.4f}
        - **MSE:** {model_info.get('mse', 'N/A'):.4f}
        """
        elif model_type in ["ARIMA", "SARIMA"]:
            summary += f"""
        - **Model Order:** {model_info.get('order', 'N/A')}
        - **AIC:** {model_info.get('aic', 'N/A'):.2f}
        - **BIC:** {model_info.get('bic', 'N/A'):.2f}
        """
        elif model_type == "VAR (Vector Autoregression)":
            summary += f"""
        - **Variables:** {', '.join(model_info.get('variables', []))}
        - **Lag Order:** {model_info.get('lag_order', 'N/A')}
        - **AIC:** {model_info.get('aic', 'N/A'):.2f}
        """
        elif model_type == "Dynamic Factor Model":
            summary += f"""
        - **Number of Factors:** {model_info.get('n_factors', 'N/A')}
        - **Total Explained Variance:** {model_info.get('total_explained_variance', 0)*100:.1f}%
        - **Variables Used:** {', '.join(model_info.get('variables', []))}
        """
        elif model_type == "State-Space Model":
            summary += f"""
        - **Components:** {model_info.get('components', 'N/A')}
        - **AIC:** {model_info.get('aic', 'N/A'):.2f}
        - **BIC:** {model_info.get('bic', 'N/A'):.2f}
        """
        elif model_type == "Nowcasting":
            summary += f"""
        - **Method:** {model_info.get('method', 'N/A')}
        - **Smoothing Parameter:** {model_info.get('smoothing_parameter', 'N/A')}
        - **MSE:** {model_info.get('mse', 'N/A'):.4f}
        """
        
        # Create metrics table
        metrics_data = []
        for i, value in enumerate(forecast_values):
            period = i + 1
            lower = result.get('lower_bound', [None]*len(forecast_values))[i]
            upper = result.get('upper_bound', [None]*len(forecast_values))[i]
            
            metrics_data.append({
                'Period': f'T+{period}',
                'Forecast': f'{value:.4f}',
                'Lower Bound': f'{lower:.4f}' if lower is not None else 'N/A',
                'Upper Bound': f'{upper:.4f}' if upper is not None else 'N/A'
            })
        
        # Create HTML table
        metrics_df = pd.DataFrame(metrics_data)
        metrics_html = f"""
        <div class="table-container">
            <h4>📊 Detailed Forecast Values</h4>
            {metrics_df.to_html(classes='table table-striped table-hover', index=False, escape=False)}
        </div>
        """
        
        return summary, metrics_html
        
    except Exception as e:
        return f"Error creating summary: {str(e)}", f"Error creating metrics: {str(e)}"