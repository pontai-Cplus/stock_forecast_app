import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# --- Streamlit 設定 ---
st.set_page_config(page_title="📈 Stock Forecast with Prophet model", layout="centered")
st.title("📊 Stock Price Forecast")
st.write(
    "Enter any **stock ticker symbol** (e.g., `AAPL`, `TSLA`, `PLTR`) to predict its future price."
)

# --- 入力欄 ---
ticker = st.text_input("Enter stock ticker:", value="AAPL").upper()
forecast_days = st.slider("Forecast Days", 30, 365, 100)

# --- 3年前からの開始日を定義 ---
three_years_ago = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()


# --- 予測関数 ---
def forecast_stock_price(ticker, forecast_days=100):
    try:
        st.info(f"📥 Downloading data for {ticker}...")
        df_raw = yf.download(
            ticker,
            start=three_years_ago,
            end=pd.Timestamp.today(),
            auto_adjust=True
        )

        if df_raw.empty or "Close" not in df_raw.columns:
            st.error(f"No data found for {ticker}.")
            return None, None

        df = df_raw.reset_index()[["Date", "Close"]]
        df.columns = ["ds", "y"]
        df = df.dropna(subset=["y"])

        model = Prophet(daily_seasonality=True)
        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # --- グラフ作成 ---
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["ds"], df["y"], label="Actual Price")
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Price", linestyle="--")
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="skyblue",
            alpha=0.3,
            label="Confidence Interval",
        )
        ax.set_title(f"{ticker} Stock Price Forecast", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)

        # --- メモリに画像を保存 ---
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        return fig, buf

    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None, None


# --- 実行 ---
if st.button("Run Forecast"):
    if ticker:
        fig, buf = forecast_stock_price(ticker, forecast_days)
        if fig and buf:
            st.pyplot(fig)
            st.download_button(
                label="📥 画像をダウンロード",
                data=buf,
                file_name=f"{ticker}_stock_forecast.png",
                mime="image/png"
            )
