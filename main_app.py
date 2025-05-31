import nltk
import pandas as pd
import requests
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from Feature_Engineering import add_features
from anomaly_detection import detect_anomalies
import plotly.express as px
import os
from datetime import datetime, timedelta
import time

# Setup
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "0SAQZUUG1WOLO1ST")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "c81cfb9a44494861856886a980e7b387")

# Streamlit UI Configuration
st.set_page_config(page_title="Stock + News Analyzer", layout="wide")
st.title("📊 Stock & News Sentiment Analyzer")

# Sidebar Controls
with st.sidebar:
    st.header("Analysis Parameters")
    symbol = st.text_input("Stock Symbol (e.g., AAPL, TSLA):", "AAPL").upper()
    days_back = st.slider("Days to analyze", 1, 30, 7)
    anomaly_threshold = st.slider("Anomaly Threshold", 0.5, 1.0, 0.7, 0.01)

# Data Fetching Functions with Retry Logic
@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def fetch_stock_data(symbol, days=7, max_retries=3):
    for attempt in range(max_retries):
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url, timeout=15)  # Increased timeout
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" not in data:
                if "Note" in data:  # API limit message
                    st.warning(f"API Limit: {data['Note']}")
                return None

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df = df.rename(columns={
                "1. open": "open_price",
                "2. high": "high_price",
                "3. low": "low_price",
                "4. close": "close_price",
                "5. volume": "volume"
            })
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)

            cutoff_date = datetime.now() - timedelta(days=days)
            return df[df.index >= cutoff_date].sort_index()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
                continue
            st.error("Stock data request timed out. Please try again later.")
            return None
        except Exception as e:
            st.error(f"Stock data error: {str(e)}")
            return None

@st.cache_data(ttl=3600, show_spinner="Fetching news articles...")
def fetch_news(symbol, days=7):
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get("articles", []))
    except Exception as e:
        st.error(f"News fetch error: {str(e)}")
        return pd.DataFrame()


# Analysis Functions
def analyze_sentiment(news_df):
    if news_df.empty:
        return news_df
    news_df["sentiment"] = news_df["content"].fillna("").apply(
        lambda x: sia.polarity_scores(str(x))["compound"])
    return news_df


def combine_data(stock_df, news_df):
    """Combine stock data with news sentiment"""
    if news_df.empty:
        stock_df["sentiment"] = 0
        return stock_df

    # Convert news timestamps to timezone-naive datetime
    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"])
    if news_df["publishedAt"].dt.tz is not None:
        news_df["publishedAt"] = news_df["publishedAt"].dt.tz_convert(None)

    # Create date column (without time component)
    news_df["date"] = news_df["publishedAt"].dt.normalize()

    # Calculate average sentiment by date
    sentiment = news_df.groupby("date")["sentiment"].mean().reset_index()

    # Prepare stock data
    stock_df = stock_df.reset_index()
    stock_df["date"] = pd.to_datetime(stock_df["index"]).dt.normalize()

    # Ensure both date columns are identical types
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    stock_df["date"] = pd.to_datetime(stock_df["date"])

    # Merge the data
    merged = pd.merge(
        stock_df,
        sentiment,
        on="date",
        how="left"
    )

    # Fill missing sentiment values with 0
    merged["sentiment"] = merged["sentiment"].fillna(0)

    return merged.drop(columns=["date"])
# Main Execution Flow
if st.sidebar.button("🔍 Run Analysis"):
    with st.spinner(f"Analyzing {symbol} data..."):
        # Data Collection
        stock_df = fetch_stock_data(symbol, days_back)
        news_df = fetch_news(symbol, days_back)

        if stock_df is None:
            st.error("Failed to fetch stock data. Please try again later.")
            st.stop()

        # Data Processing
        news_df = analyze_sentiment(news_df)
        combined_df = combine_data(stock_df, news_df)
        feature_df = add_features(combined_df)
        final_df = detect_anomalies(feature_df, threshold=anomaly_threshold)
        # Display Results
        st.success("Analysis completed successfully!")

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(final_df))
            st.metric("Avg Sentiment", f"{final_df['sentiment'].mean():.2f}")
        with col2:
            anomalies = final_df[final_df["suspicious"] == "🔴 Yes"]
            st.metric("Anomalies Detected", len(anomalies))
        with col3:
            st.metric("Max Price", f"${final_df['close_price'].max():.2f}")
            st.metric("Volume (Avg)", f"{final_df['volume'].mean():,.0f}")

        # Visualization Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📰 News Sentiment", "🚨 Anomalies"])

        with tab1:
            fig1 = px.line(final_df, x="index", y="close_price",
                           title=f"{symbol} Closing Prices")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.bar(final_df, x="index", y="volume",
                          title="Trading Volume")
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            if not news_df.empty:
                fig3 = px.bar(news_df, x="publishedAt", y="sentiment",
                              color="sentiment", title="News Sentiment Over Time")
                st.plotly_chart(fig3, use_container_width=True)

                st.dataframe(news_df[["publishedAt", "title", "source", "sentiment"]]
                             .sort_values("publishedAt", ascending=False),
                             use_container_width=True)
            else:
                st.warning("No news articles found for this period")

        with tab3:
            if not anomalies.empty:
                st.error(f"⚠️ {len(anomalies)} suspicious trades detected!")
                fig4 = px.scatter(anomalies, x="index", y="anomaly_score",
                                  size="volume", color="close_price",
                                  title="Anomaly Detection Results")
                st.plotly_chart(fig4, use_container_width=True)

                st.dataframe(anomalies[["index", "close_price", "volume",
                                        "anomaly_score", "suspicious"]]
                             .sort_values("anomaly_score", ascending=False),
                             use_container_width=True)
            else:
                st.success("✅ No anomalies detected in this time period")
