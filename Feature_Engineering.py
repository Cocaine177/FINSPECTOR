def add_features(df):
    df = df.copy()

    # 1. Price change percentage (feature_0)
    df['feature_0'] = df['close_price'].pct_change().fillna(0)

    # 2. Volume change percentage (feature_1)
    df['feature_1'] = df['volume'].pct_change().fillna(0)

    # 3. 5-day moving average (feature_2)
    df['feature_2'] = df['close_price'].rolling(5).mean().bfill()

    # 4. 5-day price volatility (feature_3)
    df['feature_3'] = df['close_price'].rolling(5).std().fillna(0)

    # 5. Sentiment change (feature_4)
    if 'sentiment' in df.columns:
        df['feature_4'] = df['sentiment'].diff().fillna(0)
    else:
        df['feature_4'] = 0

    # 6. Price-sentiment divergence (feature_5)
    df['feature_5'] = df['feature_0'] - df['feature_4']

    # 7. Price z-score (feature_6)
    df['feature_6'] = (df['close_price'] - df['close_price'].mean()) / df['close_price'].std()

    # 8. Volume spike flag (feature_7)
    df['feature_7'] = (df['volume'] > df['volume'].mean() * 1.5).astype(int)

    # 9. Price spike flag (feature_8)
    df['feature_8'] = (abs(df['feature_0']) > 0.03).astype(int)

    # 10. 3-day sentiment moving average (feature_9)
    if 'sentiment' in df.columns:
        df['feature_9'] = df['sentiment'].rolling(3).mean().bfill()
    else:
        df['feature_9'] = 0

    # Keep original features if needed for display
    df["price_change"] = df['feature_0']
    df["volume_change"] = df['feature_1']
    df["ma_5"] = df['feature_2']
    df["volatility"] = df['feature_3']

    return df
