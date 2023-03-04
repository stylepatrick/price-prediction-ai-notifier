import os
import yfinance as yf
import numpy as np
import requests
import schedule

from datetime import datetime, timedelta
from keras.models import Sequential, load_model
from keras.layers import LSTM

# Telegram
TOKEN = "TOKEN"
chat_id = "CHAT_ID"

# Configs
model_name = 'price-prediction-ai.h5'
ticker = 'ETH-USD'
train_days = 30
retrain_days = train_days

model = Sequential()


def init_model():
    start = datetime.today() - timedelta(days=365)
    df = load_history_data(start)
    # print(df)

    y_train, x_train = build_train_data(df)
    # print(y_train[0])
    # print(x_train[0])

    model.add(LSTM(1, input_shape=(train_days, 1)))
    model.compile(optimizer="rmsprop", loss="mse")
    model.fit(x_train, y_train, batch_size=32, epochs=30)
    model.save(model_name)
    predict_next_day()


def load_history_data(start):
    df = yf.download(ticker, start)
    df["Open_last_day"] = df["Open"].shift(1)
    df = df.dropna()
    df["Open_changes"] = ((df["Open"] / df["Open_last_day"]) - 1) * 100
    return df


def build_train_data(df):
    y_train = []
    x_train = []
    for i in range(0, len(df) - train_days):
        y_train.append(df["Open_changes"][i])
        x_train.append(np.array(df["Open_changes"][i + 1:i + train_days + 1]))
    y_train = np.array(y_train)
    x_train = np.array(x_train).reshape(-1, train_days)
    return y_train, x_train


def predict_next_day():
    start = datetime.today() - timedelta(days=train_days + 1)
    df_predict = load_history_data(start)
    next_day = df_predict["Open_changes"][-train_days:]
    next_day = np.array(next_day)
    next_day = next_day.reshape(-1, train_days)
    pred = model.predict(next_day)
    predict_price = (df_predict["Open"][-1:][0] / 100 * pred[0][0]) + df_predict["Open"][-1:][0]
    message = str(round(pred[0][0], 2)) + '% ' + str(round(predict_price, 2)) + '$'
    print(message)
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()


def retrain_model():
    start = datetime.today() - timedelta(days=retrain_days + 1)
    df = load_history_data(start)
    y_train, x_train = build_train_data(df)
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    model.save('price-prediction-ai.h5')


def predict():
    predict_next_day()
    retrain_model()


if __name__ == '__main__':
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        init_model()

schedule.every().day.at("09:30").do(predict)
schedule.every().day.at("15:30").do(predict)
# schedule.every(10).seconds.do(predict)
while True:
    schedule.run_pending()
