from django.shortcuts import render ,HttpResponse
from django.http import JsonResponse
import yfinance as yf
import requests
import json
from stocksearch.models import Stocksearch
from django.db.models import Q
import matplotlib.pyplot as plt
from .utils import montecarlo,randomforest
from stocksearch import portfolio

apikey = '0dac9ce89e53e21cbbf756d1db458c8e'  #financial Modelling prep
apikey1 = 'Z8GFU8EAMTIOWOFZ'                 #alphavantage

def show(request):
    return render(request,'base.html')

# Give suggestions to search bar 
def search(request):
    if 'term' in request.GET:
        qs = Stocksearch.objects.filter(Q(stockname__startswith=request.GET.get('term')) | Q(company_name__startswith=request.GET.get('term')))  
        titles = list()
        for i in qs:
            titles.append(i.stockname + " - " + i.company_name)
        print(titles)
        return JsonResponse(titles, safe=False)
    return render(request, 'base.html')

# Handle post request
def search_post(request):
  if request.method == 'POST':  
    search_ticker = request.POST.get('search_value')
    payload = suggestionsearchbar(search_ticker)
    chart = montecarlo(search_ticker)
    news(request,search_ticker)
    payload1=randomforest(search_ticker)
    companyinfo = companyoverview(search_ticker)
  return render(request,'base.html',{'charts':chart,'name':companyinfo,'MLDT':payload1})

# Saving data from Api into database 
def suggestionsearchbar(search_ticker):
    response = requests.get("https://financialmodelingprep.com/api/v3/search?query={}&limit=10&exchange=NASDAQ&apikey={}".format(search_ticker, apikey)).json()
    details = []        
    if not response:
        return "No results found"
    for i in response:
        details.append(i)
        if Stocksearch.objects.filter(stockname=i['symbol']).exists():
            continue
        stocksearch = Stocksearch(stockname=i['symbol'],company_name=i['name'])
        stocksearch.save()
    return details

def news(request,category='latest'):
    m = yf.Ticker(category)
    info = m.news    
    return render(request, 'news.html',{'articles': info})

def companyoverview(search_ticker):
    response = requests.get('https://www.alphavantage.co/query?function=OVERVIEW&symbol={}&apikey={}'.format(search_ticker, apikey1)).json()

    return response
   


APP_ID =  "1HJC4Q80HO"
APP_TYPE = "100"
SECRET_KEY = 'H5LNBNZFF7'
client_id= f'{APP_ID}-{APP_TYPE}'

FY_ID = "xm10751"  # fyers ID
APP_ID_TYPE = "2"  
TOTP_KEY = "GEXMPY776EARHFVKWZVKFKAIZNJY47XS"  # TOTP secret 
PIN = "8501"  #  pin for fyers account

REDIRECT_URI = "https://www.google.co.uk/"  # Redirect url from the app.


def portfolioanalyser(request):
    viewportfolio = portfolio.send_otp_and_verify_totp(FY_ID, APP_ID_TYPE, TOTP_KEY, PIN,client_id,SECRET_KEY,REDIRECT_URI)
    portfolio_weight = viewportfolio[0][0]
    individual_weights = viewportfolio[0][1]
    individual_volatility = viewportfolio[0][2]
    annual_return = viewportfolio[0][3]
    total_volatility = viewportfolio[1][0]
    sharpe_ratio = viewportfolio[1][1]
    graph1  = viewportfolio[0][4]
    graph2 = viewportfolio[2]
  
     
    return render(request, 'portfolio.html', {'portfolio_weight': portfolio_weight,
                                               'individual_weights': individual_weights,
                                               'individual_volatility': individual_volatility,
                                               'annual_return': annual_return,
                                               'total_volatility': total_volatility,
                                               'sharpe_ratio': sharpe_ratio,
                                               'holding_contribution_fig': graph1,
                                               'portfolio_weight_distribution':graph2                                             
                                               })
    
    
   




    


