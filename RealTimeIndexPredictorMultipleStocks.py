import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

st.set_page_config(page_title='NIFTY Predictor', layout='wide')

st.title('📊 India Market Prediction Terminal')
st.caption('Select an index or stock. App auto-downloads latest 12 months data, retrains model, and predicts next trading day move.')

indices = {
    'NIFTY 50': '^NSEI',
    'SENSEX': '^BSESN',
    'BANK NIFTY': '^NSEBANK',
    'NIFTY IT': '^CNXIT',
    'NIFTY AUTO': '^CNXAUTO',
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'INFOSYS': 'INFY.NS',
    'HDFC BANK': 'HDFCBANK.NS',
    'ICICI BANK': 'ICICIBANK.NS'
}

colA, colB = st.columns([2,1])
with colA:
    selected_name = st.selectbox('Select Index / Stock', list(indices.keys()))
with colB:
    theme = st.selectbox('View', ['Dashboard','Analytics'])
ticker = indices[selected_name]

@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period='12mo', interval='1d')
    return df


def prepare(df):
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Lag1'] = df['Returns'].shift(1)
    df['Lag2'] = df['Returns'].shift(2)
    df['Lag3'] = df['Returns'].shift(3)
    df['Volatility'] = df['Returns'].rolling(5).std()
    df['Trend'] = np.where(df['Close'] > df['Close'].rolling(5).mean(), 1, 0)
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    df.dropna(inplace=True)
    return df


def next_trading_day(last_date):
    d = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.date()

if st.button('🚀 Run Prediction', type='primary'):
    with st.spinner('Fetching data and training model...'):
        raw = load_data(ticker)
        df = prepare(raw)
        X = df[['Lag1','Lag2','Lag3','Volatility','Trend']]
        y = df['Target']
        split = int(len(df)*0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Regression target: next day % move
        y = ((df['Close'].shift(-1) - df['Close']) / df['Close']) * 100
        y = y.loc[X.index]
        y = y.dropna()
        X = X.loc[y.index]
        split = int(len(X)*0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_pred)

        latest = X.tail(1)
        pred = float(model.predict(latest)[0])
        conf = max(0, round(100 - abs(pred)*10, 2))
        pdate = next_trading_day(df.index[-1])

    c1,c2,c3 = st.columns(3)
    close_data = raw['Close'].dropna()
    if isinstance(close_data, pd.DataFrame):
        last_close = float(close_data.iloc[-1, 0])
        prev_close = float(close_data.iloc[-2, 0])
        today_date = close_data.index[-1].date()
    else:
        last_close = float(close_data.iloc[-1])
        prev_close = float(close_data.iloc[-2])
        today_date = close_data.index[-1].date()

    today_change_pct = round(((last_close - prev_close) / prev_close) * 100, 2)
    expected_close = round(last_close * (1 + pred/100), 2)

    c1.metric('Avg Error (MAE)', f'{mae:.2f}%')
    c2.metric('Today Close', f'{last_close:.2f}')
    c3.metric('Today Change', f'{today_change_pct}%')

    c4, c5 = st.columns(2)
    c4.metric('Today Date', str(today_date))
    c5.metric('Prediction Date', str(pdate))

    st.metric('Predicted Move', f'{pred:.2f}%')

    if pred >= 0:
        st.success(f'Predicted Trend for {pdate}: UP 📈')
    else:
        st.error(f'Predicted Trend for {pdate}: DOWN 📉')

    st.metric('Expected Closing Price', expected_close)
    st.write(f'Today ({today_date}) Close: {last_close:.2f}')
    st.write(f'Today Move: {today_change_pct}%')
    st.metric('Confidence Meter', f'{conf}%')

    st.subheader(f'Recent {selected_name} Close')
    st.line_chart(raw['Close'])

    st.subheader('Quick Stats')
    s1, s2, s3 = st.columns(3)

    high_data = raw["High"]
    low_data = raw["Low"]

    # Convert DataFrame to Series if needed
    if isinstance(high_data, pd.DataFrame):
       high_data = high_data.iloc[:, 0]

    if isinstance(low_data, pd.DataFrame):
       low_data = low_data.iloc[:, 0]

    high_52 = float(high_data.dropna().max())
    low_52 = float(low_data.dropna().min())

    s1.metric("52W High", f"{high_52:.2f}")
    s2.metric("52W Low", f"{low_52:.2f}")
    
    s3.metric('Data Rows', len(raw))

    if theme == 'Analytics':
        st.subheader('Latest Features Used')
        st.dataframe(latest)
        st.subheader('Recent Raw Data')
        st.dataframe(raw.tail(10))
else:
    st.info('Choose a market from dropdown and click Run Prediction.')

st.markdown('---')
st.markdown('---')
st.caption('Premium demo dashboard • Educational project • Not financial advice.')
