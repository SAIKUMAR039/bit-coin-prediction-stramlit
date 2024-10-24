import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Function to fetch data from Yahoo Finance
def fetch_data(ticker):
    data = yf.download(ticker, start="2021-01-01", end="2023-01-01")
    return data

# Function to preprocess the data
def preprocess_data(data):
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    return X, y

# Load model or train if model does not exist
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

# Main function to run the Streamlit app
def main():
    st.title('Live Cryptocurrency Price Prediction')

    # Developer details
    st.markdown(
        """
        **Developer:** Sai Kumar Thota
        - [GitHub Profile](https://github.com/SAIKUMAR039/)
        """
    )

    coins = ['BTC-USD', 'ETH-USD', 'LTC-USD']
    coin = st.selectbox('Select Cryptocurrency', coins)
    
    data = fetch_data(coin)
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

        # Show prediction
        st.subheader('Prediction')
        pred_df = pd.DataFrame({'Feature': ['Open', 'High', 'Low', 'Volume'], 'Value': [open_price, high_price, low_price, volume]})
        fig, ax = plt.subplots()
        sns.barplot(x='Feature', y='Value', data=pred_df, ax=ax)
        ax.set_title('Prediction Input Features')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
