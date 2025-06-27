import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Title and description
st.title("Tata Steel Stock Price Prediction")
st.markdown("""
This app predicts future stock prices for Tata Steel based on historical data.
You can view the closing prices and predict for a custom number of days into the future.
""")

# Stock symbol
stock_symbol = 'TATASTEEL.NS'

# Input for number of days to predict
num_days = st.number_input("Enter the number of future days to predict:", min_value=1, max_value=30, value=10, step=1)

# Fetch data and predict
if st.button("Predict"):
    with st.spinner("Fetching data and making predictions..."):
        try:
            # Fetch historical data
            data = yf.download(stock_symbol, start="2015-01-01")
            st.success("Data fetched successfully!")
            
            # Display line graph of historical closing prices
            st.subheader("Historical Closing Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Close'])
            ax.set_title("Closing Price GRAPH", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price", fontsize=12)
            ax.legend()
            st.pyplot(fig)

             # Load pre-trained model and scaler
            model = load_model('tata_steel_model.h5')
            scaler = joblib.load('scaler.pkl')

            # Fetch stock data
            data['RSI'] = calculate_rsi(data['Close'])
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data = data.dropna()

            # Scale features
            features = data[['Close', 'RSI', 'MA_20', 'Volume']]
            scaled_data = scaler.transform(features)

            # Prepare sequences for prediction
            sequence_length = 60
            last_sequence = scaled_data[-sequence_length:]
            current_sequence = last_sequence
            future_predictions = []

            for _ in range(num_days):
                prediction = model.predict(current_sequence.reshape(1, sequence_length, 4))
                future_predictions.append(prediction[0, 0])
                current_sequence = np.append(current_sequence[1:], prediction, axis=0)

            # Ensure future_predictions is reshaped to (num_days, 1)
            future_predictions = np.array(future_predictions).reshape(-1, 1)

            # Ensure dummy features have the correct shape: (num_days, 3)
            dummy_features = np.zeros((future_predictions.shape[0], 3))

            # Concatenate predictions with dummy features
            full_features = np.concatenate((future_predictions, dummy_features), axis=1)

            # Perform inverse transformation
            future_predictions_rescaled = scaler.inverse_transform(full_features)[:, 0]  # Extract the 'Close' column



            # Generate future dates
            last_date = data.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]

            # Display predictions
            st.subheader("Predicted Prices")
            predicted_data = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_predictions
            })
            st.write(predicted_data)

            # Visualize predictions
            st.subheader("Actual and Predicted Prices")
            plt.figure(figsize=(14, 7))
            plt.plot(data.index[-sequence_length:], data['Close'][-sequence_length:], label="Actual Prices", color="blue")
            plt.plot(future_dates, future_predictions, label="Predicted Prices", color="orange")
            plt.title(f"{stock_symbol} Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("Developed with ❤️ using Streamlit and TensorFlow")