from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form["ticker"].strip().upper()

    # Download stock data for the last 60 days
    stock_data = yf.download(ticker, period="60d")

    # Check if data is available
    if stock_data.empty:
        return render_template("result.html", error="Invalid stock ticker or no data available.")

    # Prepare data for Linear Regression
    stock_data["Day"] = np.arange(len(stock_data))  # Add Day number
    X = stock_data[["Day"]]  # Feature (Day Number)
    y = stock_data["Close"]  # Target (Closing Price)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next day's price
    next_day = np.array([[len(stock_data)]])
    predicted_price = model.predict(next_day).item()

    # Render result.html with predicted price
    return render_template("result.html", ticker=ticker, predicted_price=f"{predicted_price:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
