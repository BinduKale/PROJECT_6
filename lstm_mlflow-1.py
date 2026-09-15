"""
lstm_mlflow.py

Task 2.6 - LSTM deep learning model (TensorFlow/Keras)
Task 2.7 - MLflow tracking and served inference

Run this in Google Colab (tensorflow, statsmodels, and mlflow are preinstalled there).
Requires store_train.csv (from Notebook 2) and, for the MLflow section, either the
`pipeline` object from Notebook 5 still in memory, or a saved rf_sales_model_*.pkl to
reload.

NOT executed in this environment -- written directly from the already-tested logic
in 06_lstm_deep_learning_mlflow.ipynb (the pandas/numpy/sklearn portions of that
notebook were verified against real data; this script follows the same structure).
"""

import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logger setup -- same pattern used throughout the project
# ---------------------------------------------------------------------------
LOG_FILE = 'lstm_mlflow.log'
logger = logging.getLogger('lstm_mlflow')
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

DATA_DIR = ''
TARGET_STORE = 1
WINDOW_SIZE = 7


# ===========================================================================
# Task 2.6, Step 1 -- Isolate the time series data
# ===========================================================================
def isolate_time_series(store_id: int) -> pd.DataFrame:
    """One store's daily Sales on open-trading days only, sorted chronologically."""
    store_train = pd.read_csv(DATA_DIR + 'store_train.csv', low_memory=False, parse_dates=['Date'])

    ts = store_train[(store_train['Store'] == store_id) & (store_train['Open'] == 1)]
    ts = ts.sort_values('Date').reset_index(drop=True)[['Date', 'Sales']]

    logger.info(f"Store {store_id} time series: {len(ts)} open-trading days, "
                f"{ts['Date'].min().date()} to {ts['Date'].max().date()}")
    return ts


# ===========================================================================
# Task 2.6, Step 2 -- Check stationarity
# ===========================================================================
def check_stationarity(ts: pd.DataFrame) -> bool:
    """Augmented Dickey-Fuller test. Null hypothesis: series is non-stationary."""
    from statsmodels.tsa.stattools import adfuller

    adf_result = adfuller(ts['Sales'])
    logger.info(f"ADF Statistic: {adf_result[0]:.4f}")
    logger.info(f"p-value: {adf_result[1]:.4f}")
    logger.info(f"Critical values: {adf_result[4]}")

    is_stationary = adf_result[1] < 0.05
    logger.info(f"Series is {'STATIONARY' if is_stationary else 'NOT stationary'} "
                f"(p-value {'<' if is_stationary else '>='} 0.05)")

    # Supplementary visual check, no statsmodels required
    rolling_mean = ts['Sales'].rolling(window=30).mean()
    rolling_std = ts['Sales'].rolling(window=30).std()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts['Date'], ts['Sales'], alpha=0.4, label='Sales')
    ax.plot(ts['Date'], rolling_mean, color='red', label='30-day rolling mean')
    ax.plot(ts['Date'], rolling_std, color='green', label='30-day rolling std')
    ax.legend()
    ax.set_title('Rolling Mean & Std (30-day window)')
    plt.show()

    return is_stationary


# ===========================================================================
# Task 2.6, Step 3 -- Difference the series if needed
# ===========================================================================
def difference_if_needed(ts: pd.DataFrame, is_stationary: bool) -> np.ndarray:
    """Only differences when the ADF test actually found non-stationarity."""
    from statsmodels.tsa.stattools import adfuller

    if not is_stationary:
        ts['Sales_diff'] = ts['Sales'].diff()
        modeling_series = ts['Sales_diff'].dropna().values
        logger.info("Series was non-stationary -> using the first-differenced series.")

        adf_diff = adfuller(ts['Sales_diff'].dropna())
        logger.info(f"ADF p-value AFTER differencing: {adf_diff[1]:.4f} "
                    f"({'now stationary' if adf_diff[1] < 0.05 else 'still not stationary'})")
    else:
        modeling_series = ts['Sales'].values
        logger.info("Series was already stationary -> using the raw series, no differencing.")

    return modeling_series


# ===========================================================================
# Task 2.6, Step 4 -- Autocorrelation and partial autocorrelation
# ===========================================================================
def plot_acf_pacf(modeling_series: np.ndarray, ts: pd.DataFrame):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(modeling_series, lags=30, ax=axes[0])
    axes[0].set_title('Autocorrelation (ACF)')
    plot_pacf(modeling_series, lags=30, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation (PACF)')
    plt.tight_layout()
    plt.show()

    # statsmodels-free numeric fallback / cross-check
    for lag in [1, 7, 14, 30]:
        logger.info(f"Autocorrelation at lag={lag}: {ts['Sales'].autocorr(lag=lag):.3f}")


# ===========================================================================
# Task 2.6, Step 5 -- Sliding window: transform into supervised learning data
# ===========================================================================
def create_windows(series: np.ndarray, window_size: int = WINDOW_SIZE):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


# ===========================================================================
# Task 2.6, Step 6 -- Scale to (-1, 1)
# ===========================================================================
def scale_and_window(modeling_series: np.ndarray):
    from sklearn.preprocessing import MinMaxScaler

    # Same 6-week holdout convention used for the Random Forest in Notebook 5
    train_values = modeling_series[:-42].reshape(-1, 1)
    test_values = modeling_series[-42:].reshape(-1, 1)
    logger.info(f"Train series length: {len(train_values)}, Test series length: {len(test_values)}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_values)   # fit on TRAIN ONLY -- avoids leaking test scale
    test_scaled = scaler.transform(test_values)

    X_train_w, y_train_w = create_windows(train_scaled[:, 0])

    # Bridge the train/test boundary with train's tail so the first test predictions
    # aren't thrown away for lack of a full window of history
    bridge = np.concatenate([train_scaled[-WINDOW_SIZE:], test_scaled])
    X_test_w, y_test_w = create_windows(bridge[:, 0])

    X_train_lstm = X_train_w.reshape((X_train_w.shape[0], WINDOW_SIZE, 1))
    X_test_lstm = X_test_w.reshape((X_test_w.shape[0], WINDOW_SIZE, 1))
    logger.info(f"X_train_lstm shape: {X_train_lstm.shape}, X_test_lstm shape: {X_test_lstm.shape}")

    return scaler, X_train_lstm, y_train_w, X_test_lstm, y_test_w


# ===========================================================================
# Task 2.6, Step 7 -- Build and train a 2-layer LSTM
# ===========================================================================
def build_and_train_lstm(X_train_lstm, y_train_w):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # Two stacked LSTM layers, per the brief's "not very deep" requirement.
    # return_sequences=True on the first layer is required whenever a second
    # LSTM layer sits on top -- it passes the full hidden-state sequence forward
    # instead of only the final one.
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.summary()

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = lstm_model.fit(
        X_train_lstm, y_train_w,
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    logger.info(f"LSTM training finished after {len(history.history['loss'])} epochs "
                f"(final loss: {history.history['loss'][-1]:.4f})")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'])
    ax.set_title('LSTM Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (scaled space)')
    plt.show()

    return lstm_model


# ===========================================================================
# Evaluate on the test period
# ===========================================================================
def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def evaluate_lstm(lstm_model, scaler, X_test_lstm, y_test_w):
    lstm_preds_scaled = lstm_model.predict(X_test_lstm)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test_w.reshape(-1, 1)).flatten()

    lstm_rmspe = rmspe(actual, lstm_preds)
    logger.info(f"LSTM test RMSPE: {lstm_rmspe*100:.2f}%")
    print(f"LSTM RMSPE: {lstm_rmspe*100:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label='Actual', marker='o')
    ax.plot(lstm_preds, label='LSTM Predicted', marker='x')
    ax.legend()
    ax.set_title(f'Store {TARGET_STORE} — LSTM Predictions vs Actual (test period)')
    plt.show()

    return lstm_preds, actual


# ===========================================================================
# Task 2.7 -- MLflow: track training and serve predictions
#
# Applied to the Random Forest pipeline from Notebook 5, not the LSTM above --
# the RF pipeline is the model that generates the finance team's real,
# feature-rich forecasts on store_test.csv. The LSTM here is a single-store
# proof of concept.
#
# Assumes `pipeline`, `val_rmspe`, `val_mae`, `val_rmse`, and `ALL_FEATURES`
# are available from Notebook 5 (same kernel session, or reloaded via joblib).
# ===========================================================================
def log_and_serve_with_mlflow(pipeline, val_rmspe, val_mae, val_rmse, ALL_FEATURES):
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment('rossmann_sales_forecasting')

    with mlflow.start_run(run_name='random_forest_baseline'):
        mlflow.log_param('n_estimators', pipeline.named_steps['model'].regressor_.n_estimators)
        mlflow.log_param('max_depth', pipeline.named_steps['model'].regressor_.max_depth)
        mlflow.log_param('features', len(ALL_FEATURES))

        mlflow.log_metric('val_rmspe', val_rmspe)
        mlflow.log_metric('val_mae', val_mae)
        mlflow.log_metric('val_rmse', val_rmse)

        mlflow.sklearn.log_model(pipeline, 'rf_sales_pipeline')
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run logged. run_id={run_id}")

    print(f"Logged to MLflow. Run ID: {run_id}")

    # --- Serve predictions on store_test.csv through the logged model ---
    loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/rf_sales_pipeline')

    store_test = pd.read_csv(DATA_DIR + 'store_test.csv', low_memory=False, parse_dates=['Date'])
    store_test['StateHoliday'] = store_test['StateHoliday'].astype(str)

    # Re-apply the SAME feature engineering used in training (see preprocessing.py
    # from the Task 3 deployment files -- import engineer_features from there instead
    # of duplicating it here, if this script lives alongside that module).
    from preprocessing import engineer_features, ALL_FEATURES as FEATURES

    test_df = engineer_features(store_test)
    test_df['DaysToNextHoliday'] = 90    # test.csv's short window has few/no holidays --
    test_df['DaysSinceLastHoliday'] = 90  # documented simplification, same as app.py

    X_test_serve = test_df[FEATURES]

    predictions = np.where(
        test_df['Open'] == 1,
        loaded_model.predict(X_test_serve),
        0   # closed stores: skip the model, same rule used throughout training
    )

    submission = pd.DataFrame({'Id': test_df['Id'], 'Sales': predictions})
    logger.info(f"Generated {len(submission)} predictions via the MLflow-served model.")
    return submission


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    ts = isolate_time_series(TARGET_STORE)
    is_stationary = check_stationarity(ts)
    modeling_series = difference_if_needed(ts, is_stationary)
    plot_acf_pacf(modeling_series, ts)

    scaler, X_train_lstm, y_train_w, X_test_lstm, y_test_w = scale_and_window(modeling_series)
    lstm_model = build_and_train_lstm(X_train_lstm, y_train_w)
    lstm_preds, actual = evaluate_lstm(lstm_model, scaler, X_test_lstm, y_test_w)

    # Task 2.7 requires `pipeline` etc. from Notebook 5 -- uncomment once available:
    # submission = log_and_serve_with_mlflow(pipeline, val_rmspe, val_mae, val_rmse, ALL_FEATURES)
