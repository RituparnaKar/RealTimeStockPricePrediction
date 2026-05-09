# 📊 India Market Prediction Terminal

A machine learning-based stock market prediction dashboard built with **Python**, **Streamlit**, and **yFinance**.
This application fetches the latest market data, trains a **Decision Tree Regression** model, and predicts the **next trading day's market movement and expected closing price** for major Indian indices and stocks. 

---

## 🚀 Features

* 📈 Predicts next trading day price movement
* 📊 Interactive Streamlit dashboard
* 🔄 Auto-downloads latest 12 months market data using yFinance
* 🧠 Machine Learning model using Decision Tree Regressor
* 📉 Displays expected closing price and trend direction
* 📌 Supports multiple Indian indices and stocks
* 📋 Analytics mode with feature inspection and raw data view
* ⚡ Fast and lightweight implementation

---

## 🛠️ Technology Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **yFinance**

---

## 📦 Supported Markets

* NIFTY 50
* SENSEX
* BANK NIFTY
* NIFTY IT
* NIFTY AUTO
* RELIANCE
* TCS
* INFOSYS
* HDFC BANK
* ICICI BANK

---

## 🧠 Machine Learning Logic

The project uses a **Decision Tree Regressor** trained on engineered market features such as:

* Lag Returns (`Lag1`, `Lag2`, `Lag3`)
* Market Volatility
* Short-term Trend Indicators

The model predicts:

* Next day percentage movement
* Expected closing price
* Market trend direction (UP/DOWN)

---

## 📷 Dashboard Highlights

* Market selection dropdown
* Prediction metrics
* Confidence meter
* Line chart visualization
* 52-week High/Low stats
* Analytics view with latest features and raw data

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/RituparnaKar/RealTimeStockPricePrediction.git
cd RealTimeStockPricePrediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run RealTimeIndexPredictorMultipleStocks.py
```

---

## 📁 Project Structure

```text
├── RealTimeIndexPredictorMultipleStocks.py
├── Nifty50StockPredictorWithExplanation.ipynb
├── requirements.txt
└── README.md
```

---

## 📌 Future Improvements

* Add LSTM / Deep Learning models
* Real-time market streaming
* Candlestick charts
* News sentiment analysis
* Model accuracy comparison dashboard
* Portfolio prediction support

---

## ⚠️ Disclaimer

This project is created for **educational and learning purposes only**.
It should **not** be considered financial or investment advice.

---

## 👩‍💻 Author
Rituparna Kar
Developed using Python, Machine Learning, and Streamlit to explore real-time financial prediction systems.
