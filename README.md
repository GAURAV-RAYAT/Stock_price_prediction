# Tata Steel Stock Price Prediction

This project is a Streamlit-based web application that uses an LSTM (Long Short-Term Memory) model to predict the stock prices of Tata Steel for the next 30 days.

## Features
- Fetches historical Tata Steel stock price data from Yahoo Finance.
- Visualizes the fetched data with interactive charts.
- Predicts future stock prices using a trained LSTM model.
- Displays and visualizes the predicted prices for the next 30 days.

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- Pip (Python package manager)

### Installation
1. Clone this repository or download the project files.
2. Navigate to the project directory:
   ```bash
   cd tata-steel-stock-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. Place the trained LSTM model (`tata_steel_model.h5`) in the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the URL provided by Streamlit (e.g., `http://localhost:8501`) in your browser.

## Files in the Repository
- `app.py`: Main Streamlit application script.
- `tata_steel_model.h5`: Pre-trained LSTM model (place this file in the project directory).
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: Project documentation.

## How It Works
1. **Data Fetching**:
   - The app fetches historical stock price data for Tata Steel from Yahoo Finance based on user-specified dates.

2. **Prediction**:
   - The LSTM model predicts stock prices for the next 30 days using the last 60 days of data as input.

3. **Visualization**:
   - The app visualizes the actual stock prices and the predicted future prices using interactive charts.

## Example
1. Fetch Tata Steel stock data from 2015-01-01 to 2025-01-01.
2. Predict stock prices for the next 30 days starting from 2025-01-01.
3. View the results as a table and an interactive chart.

## Requirements
- Streamlit
- TensorFlow
- Yahoo Finance API (via `yfinance`)
- scikit-learn
- Matplotlib
- Pandas
- NumPy