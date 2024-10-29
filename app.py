import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Crypto Price Prediction", page_icon="💰", layout="wide")

# Apply custom CSS for sidebar styling
st.markdown(
    """
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1F1B24;
        padding-top: 20px;
        color: #FFFFFF;
    }
    /* Sidebar title styling */
    [data-testid="stSidebar"] h2 {
        color: #FFD700;
        font-size: 24px;
        margin-bottom: 20px;
    }
    /* Radio button styling */
    [data-testid="stSidebar"] .stRadio {
        color: #FFFFFF;
        font-size: 18px;
    }
    /* Custom style for the radio options */
    [data-testid="stSidebar"] .css-1vq4p4l.e1fqkh3o2 {
        padding: 8px;
        font-size: 16px;
        color: #FFFFFF;
        border: 1px solid #FFD700;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #2B2B35;
    }
    /* Hover effect for sidebar options */
    .css-1vq4p4l.e1fqkh3o2:hover {
        background-color: #FFD700;
        color: #1F1B24;
    }
    /* Make navigation active option distinct */
    .css-1vq4p4l.e1fqkh3o2 [aria-selected="true"] {
        background-color: #FFD700;
        color: #1F1B24;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("🔹 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Features"])

# Function to fetch data from Yahoo Finance with error handling
def fetch_data(ticker):
    try:
        data = yf.download(ticker, start="2021-01-01", end="2023-01-01")
        if data.empty:
            st.error(f"Failed to retrieve data for {ticker}. The ticker might be unavailable. ")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to preprocess the data
def preprocess_data(data):
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    return X, y

# Load or train model
def train_model(X, y, coin):
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, f'models/trained_model_{coin}.pkl')
    return model

def load_model(coin):
    try:
        model = joblib.load(f'models/trained_model_{coin}.pkl')
    except:
        model = None
    return model

# Home page function
def home():
    st.title('Live Cryptocurrency Price Prediction')
    st.markdown(
        """
        **Developer:** Sai Kumar Thota - [GitHub Profile](https://github.com/SAIKUMAR039/)
        """
    )

    coins = ['BTC-USD', 'ETH-USD', 'LTC-USD']
    coin = st.selectbox('Select Cryptocurrency', coins)
    data = fetch_data(coin)

    if data is not None and not data.empty:
        X, y = preprocess_data(data)
        
        # Load or train model
        model = load_model(coin)
        if model is None:
            model = train_model(X, y, coin)

        st.write(f"Data for {coin}")
        st.write(data.tail())

        # Show historical closing price
        st.subheader('Historical Closing Prices')
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], label='Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{coin} Historical Closing Prices')
        plt.legend()
        st.pyplot(plt)

        # Live Prediction
        st.header('Live Cryptocurrency Price Prediction')
        open_price = st.number_input('Enter Open Price')
        high_price = st.number_input('Enter High Price')
        low_price = st.number_input('Enter Low Price')
        volume = st.number_input('Enter Volume')

        if st.button('Predict'):
            live_data = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=['Open', 'High', 'Low', 'Volume'])
            prediction = model.predict(live_data)
            st.write(f"Predicted {coin} Closing Price: ${float(prediction[0]):.2f}")

            # Show prediction features
            st.subheader('Prediction Input Features')
            pred_df = pd.DataFrame({'Feature': ['Open', 'High', 'Low', 'Volume'], 'Value': [open_price, high_price, low_price, volume]})
            fig, ax = plt.subplots()
            sns.barplot(x='Feature', y='Value', data=pred_df, ax=ax)
            ax.set_title('Prediction Input Features')
            st.pyplot(fig)

# About page function
def about():
    st.title('About')
    st.write(
        """
        This application is developed to provide live cryptocurrency price predictions based on historical data. 
        Utilizing a trained machine learning model, it helps users predict closing prices for selected cryptocurrencies.
        """
    )

# Features page function
def features():
    st.title('Features')
    st.write(
        """
        - **Live Price Prediction**: Predict the closing prices of cryptocurrencies in real-time.
        - **Historical Data Visualization**: View historical closing prices for better understanding of trends.
        - **Interactive Data Input**: Easily input various metrics to get instant predictions.
        - **Machine Learning Model**: Uses a linear regression model for predicting prices based on historical data.
        """
    )

# Run the appropriate page function
if page == "Home":
    home()
elif page == "About":
    about()
elif page == "Features":
    features()
