import pandas as pd
import joblib

def load_model():
    model = joblib.load("../models/trained_model.pkl")
    return model

def predict_live(open_price, high_price, low_price, volume):
    model = load_model()
    live_data = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=['Open', 'High', 'Low', 'Volume'])
    prediction = model.predict(live_data)
    return prediction

if __name__ == "__main__":
    open_price = float(input("Enter Open Price: "))
    high_price = float(input("Enter High Price: "))
    low_price = float(input("Enter Low Price: "))
    volume = float(input("Enter Volume: "))
    prediction = predict_live(open_price, high_price, low_price, volume)
    print(f"Predicted Bitcoin Closing Price: ${prediction[0]:.2f}")
