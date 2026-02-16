import yfinance as yf
import pandas as pd
import os

TICKERS = ["AAPL", "MSFT", "TSLA"]
START = "2023-01-01"
END = "2024-12-31"


def download_data():
    all_rows = []

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")
        df = yf.download(
            ticker,
            start=START,
            end=END,
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            print(f"Warning: No data for {ticker}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        df["ticker"] = ticker

        df = df[
            ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        ]

        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No data downloaded")

    final_df = pd.concat(all_rows, ignore_index=True)

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in numeric_cols:
        final_df[c] = pd.to_numeric(final_df[c], errors="coerce")

    return final_df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = download_data()

    df.to_csv("data/prices.csv", index=False)

    print("Saved data/prices.csv")
    print(df.head())
