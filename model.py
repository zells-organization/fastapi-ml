import datetime
from pathlib import Path
import os
import joblib
import pandas as pd
import yfinance as yf
from fbprophet import Prophet

BASE_DIR = Path(__file__).resolve(strict=True).parent
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
TODAY = datetime.date.today()
START_PREDICTION_DATE = "2020-01-01"
STRF_FORMAT = "%m/%d/%Y"


def train(ticker="MSFT"):
    '''
    downloads historical stock data with yfinance, creates a new Prophet model, fits the model to the stock data,
    and then serializes and saves the model as a Joblib file
    :param ticker: Company's ticker of stocks you want to predict. By default Microsoft Company
    :return: None
    '''
    data = yf.download(ticker, START_PREDICTION_DATE, TODAY.strftime("%Y-%m-%d"))
    data.head()
    data["Adj Close"].plot(title=f"{ticker} Stock Adjusted Closing Price")

    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Adj Close"]
    df_forecast = df_forecast[["ds", "y"]]

    model = Prophet()
    model.fit(df_forecast)
    joblib.dump(model, Path(MODELS_DIR).joinpath(f"{ticker}.joblib"))


def predict(ticker="MSFT", days=7):
    '''
    loads and deserializes the saved model, generates a new forecast, creates images of the forecast plot and
    forecast components, and returns the days included in the forecast as a list of dicts.
    :param ticker: Company's ticker of stocks you want to predict. By default Microsoft Company
    :param days: period of prediction in days. By default 7 days
    :return:
    '''
    model_file = Path(MODELS_DIR).joinpath(f"{ticker}.joblib")
    if not model_file.exists():
        return False

    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)
    dates = pd.date_range(start=START_PREDICTION_DATE, end=future.strftime(STRF_FORMAT), )
    df = pd.DataFrame({"ds": dates})

    forecast = model.predict(df)

    model.plot(forecast).savefig(os.path.join(PREDICTIONS_DIR, f"{ticker}_plot.png"))
    model.plot_components(forecast).savefig(os.path.join(PREDICTIONS_DIR, f"{ticker}_plot_components.png"))

    return forecast.tail(days).to_dict("records")


def convert(predictions_list):
    '''
    takes the list of dicts from predict and outputs a dict of dates and
    forecasted values (i.e., {"07/02/2020": 200}).
    :param predictions_list:
    :return:
    '''
    output = {}
    for data in predictions_list:
        date = data["ds"].strftime(STRF_FORMAT)
        output[date] = data["trend"]
    return output
