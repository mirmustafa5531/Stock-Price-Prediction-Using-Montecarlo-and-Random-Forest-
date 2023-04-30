import json
import requests
import time
import pyotp
import os
import requests
from urllib.parse import parse_qs,urlparse
import sys
from fyers_api import fyersModel
from fyers_api import accessToken
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import yfinance as yf
from io import BytesIO
import base64

#FY_ID, APP_ID_TYPE, TOTP_KEY, PIN,client_id,SECRET_KEY,REDIRECT_URI

APP_ID =  "1HJC4Q80HO"
APP_TYPE = "100"
SECRET_KEY = 'H5LNBNZFF7'
client_id= f'{APP_ID}-{APP_TYPE}'

FY_ID = "xm10751"  # fyers ID
APP_ID_TYPE = "2"  
TOTP_KEY = "GEXMPY776EARHFVKWZVKFKAIZNJY47XS"  # TOTP secret 
PIN = "8501"  #  pin for fyers account

REDIRECT_URI = "https://www.google.co.uk/"  # Redirect url from the app.


# API endpoints

BASE_URL = "https://api-t2.fyers.in/vagator/v2"
BASE_URL_2 = "https://api.fyers.in/api/v2"
URL_SEND_LOGIN_OTP = BASE_URL + "/send_login_otp"   #/send_login_otp_v2
URL_VERIFY_TOTP = BASE_URL + "/verify_otp"
URL_VERIFY_PIN = BASE_URL + "/verify_pin"
URL_TOKEN = BASE_URL_2 + "/token"
URL_VALIDATE_AUTH_CODE = BASE_URL_2 + "/validate-authcode"
SUCCESS = 1
ERROR = -1



def send_login_otp(fy_id, app_id):
    try:
        result_string = requests.post(url=URL_SEND_LOGIN_OTP, json= {"fy_id": fy_id, "app_id": app_id })
        if result_string.status_code != 200:
            return [ERROR, result_string.text]
        result = json.loads(result_string.text)
        request_key = result["request_key"]
        return [SUCCESS, request_key]
    except Exception as e:
        return [ERROR, e]

def verify_totp(request_key, totp):
    try:
        result_string = requests.post(url=URL_VERIFY_TOTP, json={"request_key": request_key,"otp": totp})
        if result_string.status_code != 200:
            return [ERROR, result_string.text]
        result = json.loads(result_string.text)
        request_key = result["request_key"]
        return [SUCCESS, request_key]
    except Exception as e:
        return [ERROR, e]


def accesstoken(client_id,SECRET_KEY,REDIRECT_URI):
    session = accessToken.SessionModel(client_id=client_id, secret_key=SECRET_KEY, redirect_uri=REDIRECT_URI,response_type='code', grant_type='authorization_code')
    urlToActivate = session.generate_authcode()
    return urlToActivate



def send_otp_and_verify_totp(FY_ID, APP_ID_TYPE, TOTP_KEY, PIN,client_id,SECRET_KEY,REDIRECT_URI):
    success = False
    for i in range(1, 3):
        send_otp_result = send_login_otp(fy_id=FY_ID, app_id=APP_ID_TYPE)
        if send_otp_result[0] != SUCCESS:
            print(f"send_login_otp failure - {send_otp_result[1]}")
            time.sleep(1)
        else:
            print("send_login_otp success")
            success = True
            break

    if not success:
        print("Failed to send OTP after multiple attempts.")
        sys.exit()

    request_key = send_otp_result[1]

    success = False
    for i in range(1, 3):
        verify_totp_result = verify_totp(request_key=request_key, totp=pyotp.TOTP(TOTP_KEY).now())
        if verify_totp_result[0] != SUCCESS:
            print(f"verify_totp_result failure - {verify_totp_result[1]}")
            time.sleep(1)
        else:
            print(f"verify_totp_result success {verify_totp_result}")
            success = True
            break

    if not success:
        print("Failed to verify OTP after multiple attempts.")
        sys.exit()
    request_key_2 = verify_totp_result[1]

    # Verify pin and send back access token
    ses = requests.Session()
    payload_pin = {"request_key":f"{request_key_2}","identity_type":"pin","identifier":f"{PIN}","recaptcha_token":""}
    res_pin = ses.post('https://api-t2.fyers.in/vagator/v2/verify_pin', json=payload_pin).json()
   
    ses.headers.update({
        'authorization': f"Bearer {res_pin['data']['access_token']}"
    })

    authParam = {"fyers_id":FY_ID,"app_id":APP_ID,"redirect_uri":REDIRECT_URI,"appType":APP_TYPE,"code_challenge":"","state":"None","scope":"","nonce":"","response_type":"code","create_cookie":True}
    authres = ses.post('https://api.fyers.in/api/v2/token', json=authParam).json()

    url = authres['Url']
   
    parsed = urlparse(url)
    auth_code = parse_qs(parsed.query)['auth_code'][0]
    
    session = accessToken.SessionModel(client_id=client_id, secret_key=SECRET_KEY, redirect_uri=REDIRECT_URI, response_type='code', grant_type='authorization_code')
    session.set_token(auth_code)
    response = session.generate_token()
    access_token= response["access_token"]

    fyers = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path=os.getcwd())
    weights = sharperatio(fyers)
    portfolio_volatility(fyers,weights)

def sharperatio(fyers):
    holdings = fyers.holdings()
    print(holdings)
    total_cost = 0                                #  Total price at which the user bought security 
    total_current_value = 0
    total_quantity = 0
    quantity_1 = []                               # total Quantity of stock in our portfolio
    quantity_2 = []                               # Calculating individual Portfolio weights based on quantity
    quantity_3 = []                               # Calculating Portfolio weights based on price
    daily_returns = []
    holding_volatilities = []
    symbols = [] 

    for holding in holdings['holdings']:
        cost_price = holding['costPrice']
        quantity = holding['quantity']
        total_quantity += quantity
        quantity_1.append(quantity)                      # total Quantity
        current_price = holding['ltp']
        holding_value = quantity * current_price
        total_cost += cost_price * quantity          
        total_current_value += holding_value             #  Current value of the holdings 

        daily_return = (current_price - cost_price) / cost_price
        daily_returns.append(daily_return)
        holding_volatility = daily_return ** 2
        holding_volatilities.append(holding_volatility)
        symbols.append(holding['symbol'])     

        a = holdings['overall']['total_current_value']
        quantity_3.append(cost_price*quantity/a)

    
    for i in quantity_1 :
        quantity_2.append(i/total_quantity * 100)                  # Calculating individual Portfolio weights
    
    
    portfolio_return = math.log(total_current_value / total_cost) 
    annual_return = portfolio_return * 252                          # Assuming 252 trading days in a year
    portfolio_individual_volatility = math.sqrt(sum([h * w for h, w in zip(holding_volatilities, quantity_2)]))
    holding_contributions = [h * w / portfolio_individual_volatility for h, w in zip(holding_volatilities, quantity_3)]
    plt.switch_backend('AGG')
    # Plot a pie chart
    sns.set_style('darkgrid')
    plt.figure(figsize=(8,6))
    plt.pie(quantity_3, labels=symbols, autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Weights')
    graphic = get_graph()

    # create a pandas dataframe with symbol and holding contributions
    df = pd.DataFrame({'Symbol': symbols, 'Holding Contributions': holding_contributions})
    # create the barplot using seaborn
    sns.set_style('darkgrid')
    ax = sns.barplot(x='Symbol', y='Holding Contributions', data=df)
    ax.set_title('Holding Contributions to Portfolio Volatility')
    ax.set_ylabel('Holding Contributions (%)')
    ax.set_xlabel('Symbol')
    ax.tick_params(axis='x', labelrotation=45)
    graphic1 = get_graph()

    return quantity_3,annual_return,holding_contributions,quantity_2,graphic,graphic1

def portfolio_volatility(fyers, weights):
    holdings = fyers.holdings()
    symbols = [h['symbol'] for h in holdings['holdings']]
    
    # Convert symbols to desired format
    tickers = [s.split(':')[1] + '.' + s.split(':')[0] for s in symbols]
    tickers = [t.replace('-EQ.NSE', '.NS') for t in tickers]
    
    # Download historical data for the tickers
    start_date = '2018-03-24'  # Replace with desired start date
    end_date = '2021-03-24'  # Replace with desired end date
    df = yf.download(tickers=tickers, start=start_date, end=end_date)       
    
    returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # Calculate the weights of each asset
    weight = np.array(weights)
    volatility = returns.std() * np.sqrt(252) 
    sharpe_ratio = ((returns.mean()*252) - 0.05) / volatility
    # Calculate the portfolio volatility using np.std()
    portfolio_volatility = np.std(returns.dot(weight.T)) * np.sqrt(252)
    return portfolio_volatility,sharpe_ratio
   
def get_graph():
  buffer = BytesIO()
  plt.savefig(buffer , format = 'png')
  buffer.seek(0)
  image_png = buffer.getvalue()
  graph = base64.b64encode(image_png)
  graph = graph.decode('utf-8')
  buffer.close()
  return graph

 
#a = send_otp_and_verify_totp(FY_ID, APP_ID_TYPE, TOTP_KEY, PIN,client_id,SECRET_KEY,REDIRECT_URI)
   
    






