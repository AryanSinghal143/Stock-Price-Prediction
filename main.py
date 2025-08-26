import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from twelvedata import TDClient
from math import sqrt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found! Please set it in your .env file.")

# Set plot style
plt.style.use('fivethirtyeight')

# Initialize Twelve Data API
td = TDClient(apikey=API_KEY)

# Logging function
def log_output(message, file_path="results/results_log.txt"):
    print(message)
    with open(file_path, "a") as f:
        f.write(message + "\n")

# Fetch intraday data
def fetch_training_intraday(symbol, start_date, end_date):
    try:
        ts = td.time_series(
            symbol=symbol,
            interval="1min",
            start_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
            outputsize=5000
        ).as_pandas()
    except Exception as e:
        log_output(f"❌ Error fetching data for {symbol}: {e}")
        return None

    if ts is not None and not ts.empty:
        ts = ts.sort_index()
        ts = ts[['close']]
        return ts
    else:
        log_output(f"⚠ Could not fetch training data for {symbol}")
        return None

# Prepare data
def prepare_training_data(data):
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i - 60:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_intraday_model(symbols, start_date, end_date):
    all_data = {}
    excel_records = []

    # Clear log file
    if os.path.exists("results/results_log.txt"):
        os.remove("results/results_log.txt")

    for symbol in symbols:
        log_output(f"\n📥 Fetching data for {symbol}...")
        data = fetch_training_intraday(symbol, start_date, end_date)
        if data is None or len(data) < 100:
            log_output(f"❌ Not enough data to train for {symbol}.")
            continue
        all_data[symbol] = data

    if not all_data:
        log_output("❌ No valid data retrieved for any symbol.")
        return

    for symbol, data in all_data.items():
        log_output(f"\n🚀 Training model for {symbol}...")
        x_train, y_train, scaler = prepare_training_data(data)
        model = build_lstm_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)
        model.save(f'results/stock_price_model_intraday_{symbol}.keras')
        log_output(f"✅ Model for {symbol} trained and saved successfully!")

        # Predictions & metrics
        train_pred = model.predict(x_train, verbose=0)
        train_pred_inv = scaler.inverse_transform(train_pred)
        actual_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

        test_data = data[-(60+30):]
        scaled_test = scaler.transform(test_data.values)
        x_test = [scaled_test[i-60:i, 0] for i in range(60, len(scaled_test))]
        x_test = np.array(x_test).reshape(-1, 60, 1)
        test_pred = model.predict(x_test, verbose=0)
        test_pred_inv = scaler.inverse_transform(test_pred)
        actual_test_inv = scaler.inverse_transform(scaled_test[60:])

        train_mse = mean_squared_error(actual_train_inv, train_pred_inv)
        train_mae = mean_absolute_error(actual_train_inv, train_pred_inv)
        train_rmse = sqrt(train_mse)
        test_mse = mean_squared_error(actual_test_inv, test_pred_inv)
        test_mae = mean_absolute_error(actual_test_inv, test_pred_inv)
        test_rmse = sqrt(test_mse)

        log_output(f"📊 Performance for {symbol}:")
        log_output(f"   - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        # Future prediction
        last_60 = scaled_test[-60:].reshape(1, 60, 1)
        future_price = model.predict(last_60, verbose=0)
        future_price = scaler.inverse_transform(future_price)
        actual_latest_price = data.values[-1][0]

        log_output(f"📉 Last Actual Price: ${actual_latest_price:.2f}")
        log_output(f"🔮 Predicted Next Price: ${future_price[0][0]:.2f}")

        # Excel record
        excel_records.append({
            "Symbol": symbol,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Train MAE": train_mae,
            "Test MAE": test_mae,
            "Last Actual Price": actual_latest_price,
            "Predicted Next Price": future_price[0][0]
        })

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(data.index[60:], actual_train_inv, label='Actual Price', color='blue')
        plt.plot(data.index[60:], train_pred_inv, label='Predicted Price', color='red')
        plt.title(f"{symbol} - Intraday Prediction vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"results/{symbol}_chart.png")
        plt.close()

    # Export summary to Excel
    if excel_records:
        df_results = pd.DataFrame(excel_records)
        df_results.to_excel("results/results_summary.xlsx", index=False)
        log_output("📄 Summary saved to 'results/results_summary.xlsx'.")

# Main loop
def main():
    while True:
        stock_symbols = input("Enter stock symbols (comma-separated): ").split(',')
        stock_symbols = [s.strip().upper() for s in stock_symbols]

        start_input = input("Enter START datetime (YYYY-MM-DD HH:MM:SS): ")
        end_input = input("Enter END datetime (YYYY-MM-DD HH:MM:SS): ")

        try:
            start_date = datetime.strptime(start_input, "%Y-%m-%d %H:%M:%S")
            end_date = datetime.strptime(end_input, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            log_output("❌ Invalid date format. Use YYYY-MM-DD HH:MM:SS")
            continue

        if start_date >= end_date:
            log_output("❌ Start date must be before end date.")
            continue

        train_intraday_model(stock_symbols, start_date, end_date)
        another = input("Do you want to train another model? (y/n): ").lower()
        if another != 'y':
            log_output("👋 Exiting program.")
            break

if __name__ == "__main__":
    main()
