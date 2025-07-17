import pandas as pd 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

def prepare_data(df, target):
    df = df[['Date', target]].dropna()
    df = df.sort_values(by='Date')
    df = df.set_index('Date').resample('D').mean() 
    df = df.fillna(method='ffill') 
    df = df.reset_index()
    df = df.rename(columns={'Date': 'ds', target: 'y'})
    return df

def arima(df, target, forecast_days=10):
    ts = df[[target]].set_index(df['Date']).dropna()
    model = ARIMA(ts, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq = 'D')
    forecast = pd.Series(forecast.values, index=forecast_index)
    return model_fit, forecast

def exponential_smoothing_forecast(df, target, forecast_days=10):
    ts = df[[target]].set_index(df['Date']).dropna()
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq = 'D')
    forecast = pd.Series(forecast.values, index=forecast_index)
    return model_fit, forecast

def prophet(df, target, forecast_days=10):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].set_index('ds')
    return model, result.tail(forecast_days)
