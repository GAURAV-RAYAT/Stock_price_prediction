import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

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
if st.button("Fetch Data and Predict"):
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

            # Load the saved scaler
            scaler = joblib.load('scaler.pkl')

            # Preprocess data using the loaded scaler
            scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))

            # Load the trained model
            model = load_model('tata_steel_model.h5')

            # Prepare the last 60 days of data from the entire dataset for prediction
            sequence_length = 60
            last_sequence = scaled_data[-sequence_length:]
            current_sequence = last_sequence
            future_predictions = []

            # Predict for the specified number of future days
            for _ in range(num_days):
                prediction = model.predict(current_sequence.reshape(1, sequence_length, 1))
                future_predictions.append(prediction[0, 0])
                current_sequence = np.append(current_sequence[1:], prediction, axis=0)

            # Transform predictions back to original scale
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Generate future dates
            last_date = data.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days + 1)]

            # Display predictions
            st.subheader(f"Predicted Prices for the Next {num_days} Days")
            predicted_data = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_predictions.flatten()
            })
            st.write(predicted_data)

            # Combine historical data with predictions for visualization
            combined_dates = list(data.index) + future_dates
            combined_prices = list(data['Close'].values) + list(future_predictions.flatten())

            # Plot combined graph
            st.subheader("Historical and Predicted Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(combined_dates[:len(data)], combined_prices[:len(data)], label='Historical Prices', color='blue')
            ax.plot(combined_dates[len(data):], combined_prices[len(data):], label='Predicted Prices', color='red')
            ax.set_title("Historical and Predicted Prices", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Stock Price", fontsize=12)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("Developed with ❤️ using Streamlit and TensorFlow")