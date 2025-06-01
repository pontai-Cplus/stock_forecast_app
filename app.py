import streamlit as st
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from datetime import datetime
import time

# --- Streamlit Page Config ---
st.set_page_config(page_title="📈 Stock Forecast App", layout="centered")
st.title("📊 Stock Price Forecast with Prophet")

st.write(
    "Enter a **stock ticker** (e.g., AAPL, TSLA, GOOGL) and generate a future price forecast using the Prophet model."
)

# --- User Inputs ---
ticker = st.text_input("Enter stock ticker:", value="AAPL").upper()
forecast_days = st.slider("Forecast period (days):", 30, 365, 100)
theme = st.toggle("🌙 Dark theme", value=False)

# --- Theme styling ---
if theme:
    plt.style.use("dark_background")
else:
    plt.style.use("default")

# --- Date setting ---
three_years_ago = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()

# --- Forecasting function ---
def forecast_stock_price(ticker, forecast_days=100):
    try:
        st.info(f"📥 Downloading data for {ticker}...")

        df_raw = yf.download(
            ticker,
            start=three_years_ago,
            end=pd.Timestamp.today(),
            auto_adjust=False
        )

        if df_raw.empty or "Close" not in df_raw.columns:
            st.error(f"No data found for {ticker}.")
            return None, None, None, None, None

        df = df_raw.reset_index()[["Date", "Close"]]
        df.columns = ["ds", "y"]
        df = df.dropna(subset=["y"])

        start_time = time.time()

        model = Prophet(daily_seasonality=True)
        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        elapsed_time = time.time() - start_time

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["ds"], df["y"], label="Actual Price")
        ax.plot(
            forecast["ds"], forecast["yhat"], label="Predicted Price", linestyle="--"
        )
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

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        return df_raw, forecast, fig, buf, elapsed_time

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None, None, None


# --- Main ---
if st.button("Run Forecast"):
    if ticker:
        df_raw, forecast, fig, buf, elapsed_time = forecast_stock_price(ticker, forecast_days)
        if df_raw is not None:
            # --- Show forecast summary ---
            last_day = forecast.iloc[-1]
            st.success(
                f"📅 Forecast for {last_day['ds'].date()}: "
                f"**${last_day['yhat']:.2f}** "
                f"(Range: ${last_day['yhat_lower']:.2f} ~ ${last_day['yhat_upper']:.2f})"
            )

            # --- Show elapsed time ---
            st.caption(f"⏱️ Model trained in {elapsed_time:.2f} seconds")

            # --- Show image chart by default ---
            st.image(buf, caption=f"{ticker} Forecast", use_column_width=True)

            # --- Download buttons ---
            st.download_button(
                label="📥 Download Forecast Image (PNG)",
                data=buf,
                file_name=f"{ticker}_stock_forecast.png",
                mime="image/png",
            )

            st.download_button(
                label="📄 Download Historical Data (CSV)",
                data=df_raw.to_csv(index=True).encode("utf-8"),
                file_name=f"{ticker}_historical_data.csv",
                mime="text/csv",
            )
