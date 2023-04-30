import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score
from sklearn.model_selection import train_test_split,cross_val_score
import copy

import seaborn as sns

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

def montecarlo(search_value):
    try:
        df = yf.download("{}".format(search_value),startDate=startDate,endDate=endDate)
        if df.empty:
            raise Exception("Dataframe is empty. No data found for the given stock symbol.")
        else:
            returns = np.log(1 + df['Adj Close'].pct_change())
            mu, sigma = returns.mean(), returns.std()
            num_simulations = 100
            num_trading_days = 252
            results = np.zeros((num_simulations, num_trading_days + 1))
            initial = df['Adj Close'].iloc[-1]
            results[:, 0] = initial
            for i in range(num_simulations):
                for j in range(1, num_trading_days + 1):
                    results[i, j] = results[i, j-1] * np.exp(np.random.normal(mu, sigma))
            plt.switch_backend('AGG')
            plt.figure(figsize=(10, 5))
            for i in range(num_simulations):
                plt.plot(results[i])
            plt.axhline(initial, c='k', label='Starting Value')
            plt.title(f"{num_simulations} Monte Carlo Simulations for {search_value}")
            plt.xlabel('Trading Days')
            plt.ylabel('Stock Price')
            graph = get_graph()
            
            
            mean_results = np.mean(results, axis=0)
            mean_price = mean_results[-1]
            upper_price = np.percentile(results[:, -1], q=95)
            lower_price = np.percentile(results[:, -1], q=5)
            print(f"Average Price for the next 252 Days: {mean_price:.2f}")
            print(f"Upper Price: {upper_price:.2f}")
            print(f"Lower Price: {lower_price:.2f}")

            return graph
            
    except Exception as e:
        print("An error occurred: ", e)

def get_graph():
  buffer = BytesIO()
  plt.savefig(buffer , format = 'png')
  buffer.seek(0)
  image_png = buffer.getvalue()
  graph = base64.b64encode(image_png)
  graph = graph.decode('utf-8')
  buffer.close()
  return graph


def randomforest(search_value):
    try:
        df = yf.download("{}".format(search_value),startDate=startDate,endDate=endDate)
        if df.empty:
            raise Exception("Dataframe is empty. No data found for the given stock symbol.")
        else:
            print(df)
            df = df.loc['2007-01-01':].copy()            
            days_out = 30

            #Pre-process Data            
            df[['Close', 'Low', 'High', 'Open']] = df[['Close', 'Low', 'High', 'Open']].transform(lambda x: x.ewm(span=days_out).mean())
            df['Signal_Flag'] = df['Close'].transform(lambda x : np.sign(x.diff(days_out)))

            #Target Column
            df["Tomorrow"] = df["Close"].shift(-1)
            df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
          
            # Calculate the Stochastic Oscillator.
            n = 14
            # Make a copy of the high and low column.
            low_14, high_14 = df['Low'].copy(), df['High'].copy()

            # Apply the rolling function and grab the Min and Max.
            low_14 = low_14.transform(lambda x: x.rolling(window = n).min())
            high_14 = high_14.transform(lambda x: x.rolling(window = n).max())
            k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['low_14'] = low_14
            df['high_14'] = high_14
            df['k_percent'] = k_percent

            # Calculate the MACD
            ema_26 = df['Close'].transform(lambda x: x.ewm(span = 26).mean())
            ema_12 = df['Close'].transform(lambda x: x.ewm(span = 12).mean())
            macd = ema_12 - ema_26
            ema_9_macd = macd.ewm(span = 9).mean()

            # Store the data in the data frame.
            df['MACD'] = macd
            df['MACD_EMA'] = ema_9_macd

            # Calculate the Price Rate of Change
            m = 9
            df['Price_Rate_Of_Change'] = df['Close'].transform(lambda x: x.pct_change(periods = m))

            # calculate the On Balance Volume
            volume = df['Volume']
            change = df['Close'].diff()

            # intialize the previous OBV
            prev_obv = 0
            obv_values = []
          
            for i, j in zip(change, volume):

                if i > 0:
                    current_obv = prev_obv + j
                elif i < 0:
                    current_obv = prev_obv - j
                else:
                    current_obv = prev_obv

                prev_obv = current_obv
                obv_values.append(current_obv)

            # Add the values to the data frame.
            df['On Balance Volume'] = obv_values

            # Calculate RSI
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
            loss = (-delta).where(delta < 0, 0).ewm(span=14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
                                                                           
            # Calculate ADX
            up_move = df["High"] - df["High"].shift(1)
            down_move = df["Low"].shift(1) - df["Low"]
            up_move[up_move < 0] = 0
            down_move[down_move < 0] = 0
            pos_direction = up_move.rolling(window=14).mean()
            neg_direction = down_move.rolling(window=14).mean()
            direction_index = pos_direction / (pos_direction + neg_direction)
            df["ADX"] = 100 * direction_index.rolling(window=14).mean()

            predictors = ["RSI", "Price_Rate_Of_Change", "MACD", "ADX","On Balance Volume"]
            df = df.dropna()            
            X_train, X_test, y_train, y_test = train_test_split(df[predictors], df["Target"], test_size=0.2)        
            model = RandomForestClassifier(n_estimators=100,oob_score = True,criterion = "gini", random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print('Correct Prediction (%): ', accuracy_score(y_test, y_pred))
            scores = cross_val_score(model, X_train, y_train, cv=5)
            print("Cross-validation scores:", scores)

            # Calculate precision, recall, and F1-score
            y_pred = model.predict(X_test)
            print("Precision:", precision_score(y_test, y_pred))
            print("Recall:", recall_score(y_test, y_pred))
            print("F1-score:", f1_score(y_test, y_pred))

            last_date = df.index[-1]
            last_close = df.iloc[-1]

            #creating an empty df with future dates       
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
            future_df = pd.DataFrame(index=future_dates, columns=predictors)
            future_df.fillna(df.mean(), inplace=True)
            future_preds = model.predict(future_df)
           
            # Get the last known closing price
            last_close = df['Close'].iloc[-1]

            # Calculate the future prices based on the predicted direction
            future_prices = [last_close]
            last_30_days = df.tail(30)
            mean_price_change = last_30_days['Close'].pct_change().mean()
            std_price_change = last_30_days['Close'].pct_change().std()
            
            for pred in future_preds:
                if pred == 1:  # If the prediction is for an increase in price
                    future_prices.append(future_prices[-1] * (1 + np.random.normal(mean_price_change, std_price_change)))
                else:  # If the prediction is for a decrease in price
                    future_prices.append(future_prices[-1] * (1 - np.random.normal(mean_price_change, std_price_change)))
              
            plt.switch_backend('AGG')                       
            # Plot the future prices
            sns.lineplot(x=future_dates, y=future_prices[1:])
            plt.xticks(rotation=45)
            plt.xlabel('Date')
            plt.ylabel('Price')            
            plt.title('Predicted Future Prices')
          
            #plt.savefig('png')
            graphic = get_graph()
            return graphic
                        
    except Exception as e:
        print("An error occurred: ", e)   
    






