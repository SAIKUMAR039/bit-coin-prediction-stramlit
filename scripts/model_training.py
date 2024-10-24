import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def load_data():
    data = pd.read_csv("../data/raw/bitcoin_data.csv")
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    joblib.dump(model, "../models/trained_model.pkl")
    return mse

if __name__ == "__main__":
    X, y = load_data()
    mse = train_model(X, y)
    print(f"Model Mean Squared Error: {mse}")
