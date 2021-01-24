import datetime
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from fbprophet import Prophet

BASE_DIR = Path(__file__).resolve(strict=True).parent
TODAY = datetime.date.today()


def train(ticker="MSFT"):
    data = yf.download(ticker, "2020-01-01", TODAY.strftime("%Y-%m-%d"))
    data.head()
    data["Adj Close"].plot(title=f"{ticker} Stock Adjusted Closing Price")

    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Adj Close"]
    df_forecast = df_forecast[["ds", "y"]]

    model = Prophet()
    model.fit(df_forecast)
    joblib.dump(model, Path(BASE_DIR).joinpath(f"{ticker}.joblib"))
