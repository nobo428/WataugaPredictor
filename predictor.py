import pandas as pd
import urllib.request
import urllib.error
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#define lookback and forecast windows
lookback = 60
forecast = 10

#getting the correct dates
today = datetime.date.today()
startdate = today - datetime.timedelta(days=lookback)
enddate = today + datetime.timedelta(days=forecast)

BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'

ApiKey='YFPMR6XA4CTYQ26UNFK2C9Q2Y'

#UnitGroup sets the units of the output - us or metric
UnitGroup='us'

#Location for the weather data
Location1 = 'Banner%20Elk,NC'
Location2 = 'Valle%20Crucis,NC'
Location3 = 'Bethel,NC'

#set start and end dates
StartDate = str(startdate)
EndDate = str(enddate)

ContentType="csv"

#specify elements to get
Elements='datetime%2Cprecip'

#include sections
Include="days"

#basic query including location
ApiQueryBE=BaseURL + Location1
ApiQueryVC=BaseURL + Location2
ApiQueryBTH=BaseURL + Location3

#append the start and end date if present
if (len(StartDate)):
    ApiQueryBE += "/" + StartDate
    ApiQueryVC += "/" + StartDate
    ApiQueryBTH += "/" + StartDate
    if (len(EndDate)):
        ApiQueryBE += "/" + EndDate
        ApiQueryVC += "/" + EndDate
        ApiQueryBTH += "/" + EndDate

#Url is completed. Now add query parameters (could be passed as GET or POST)
ApiQueryBE += "?"
ApiQueryVC += "?"
ApiQueryBTH += "?"

#append each parameter as necessary
if (len(UnitGroup)):
    ApiQueryBE += "&unitGroup=" + UnitGroup
    ApiQueryVC += "&unitGroup=" + UnitGroup
    ApiQueryBTH += "&unitGroup=" + UnitGroup

if (len(Elements)):
    ApiQueryBE += "&elements=" + Elements
    ApiQueryVC += "&elements=" + Elements
    ApiQueryBTH += "&elements=" + Elements

if (len(ContentType)):
    ApiQueryBE += "&contentType=" + ContentType
    ApiQueryVC += "&contentType=" + ContentType
    ApiQueryBTH += "&contentType=" + ContentType

if (len(Include)):
    ApiQueryBE += "&include=" + Include
    ApiQueryVC += "&include=" + Include
    ApiQueryBTH += "&include=" + Include

ApiQueryBE += "&key=" + ApiKey
ApiQueryVC += "&key=" + ApiKey
ApiQueryBTH += "&key=" + ApiKey

#query for Banner Elk
print(' - Running query URL: ', ApiQueryBE)
print()

try:
    CSVBytesBE = urllib.request.urlopen(ApiQueryBE)
    be = pd.read_table(CSVBytesBE, sep=",")
    #be.to_pickle('betest')
except urllib.error.HTTPError  as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code, ErrorInfo)
    sys.exit()
except  urllib.error.URLError as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code,ErrorInfo)
    sys.exit()

#query for Valle Crucis
print(' - Running query URL: ', ApiQueryVC)
print()

try:
    CSVBytesVC = urllib.request.urlopen(ApiQueryVC)
    vc = pd.read_table(CSVBytesVC, sep=",")
except urllib.error.HTTPError  as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code, ErrorInfo)
    sys.exit()
except  urllib.error.URLError as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code,ErrorInfo)
    sys.exit()

#query for Bethel
print(' - Running query URL: ', ApiQueryBTH)
print()

try:
    CSVBytesBTH = urllib.request.urlopen(ApiQueryBTH)
    bth = pd.read_table(CSVBytesBTH, sep=",")
except urllib.error.HTTPError  as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code, ErrorInfo)
    sys.exit()
except  urllib.error.URLError as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code,ErrorInfo)
    sys.exit()

#create date list for index
dates = pd.to_datetime(be["datetime"], format="%Y-%m-%d")

# combine multiple dataframes into one with readable columns
og = pd.concat([be["precip"], vc["precip"], bth["precip"]], axis=1)
og.index = dates
og.columns = ["be_precip", "vc_precip", "bth_precip"]
datenames = be["datetime"]

# helper for manipulating data into model readable format
def convert2matrix(data, look_back):
    X = []
    for i in range(len(data) - look_back):
        d = i + look_back
        X.append(data.iloc[i:d+1, 0:3])
    return np.array(X)

testvals = convert2matrix(og, lookback)

#load saved model
daModel = keras.models.load_model('taugModel')

prediction = daModel.predict(testvals)

# Basic prediction plot for testing
def prediction_plot(test_predict):
    fig, ax = plt.subplots(2, figsize=(8, 8))
    ax[0].plot(og.index[lookback:], test_predict, 'r', label="prediction")
    ax[1].plot(og.index[lookback:], og.iloc[lookback:, 0], 'b', label="be_precip", alpha=0.3)
    ax[1].plot(og.index[lookback:], og.iloc[lookback:, 1], 'b', label="vc_precip", alpha=0.3)
    ax[1].plot(og.index[lookback:], og.iloc[lookback:, 2], 'b', label="bth_precip", alpha=0.3)
    ax[0].set_ylabel('Level (cfs)', size=15)
    ax[1].set_ylabel('Expected Rainfall Amount', size=15)
    ax[0].set_xlabel('Date', size=15)
    ax[1].set_xlabel('Date', size=15)
    ax[0].legend(fontsize=15)
    ax[1].legend(fontsize=15)
    plt.tight_layout()
    plt.savefig('forecast.png')

# Final prediction plot for data output
def generate_predictions(test_predict):
    fig, ax = plt.subplots(2, figsize=(8, 8))
    ax[0].plot(og.index[lookback:], test_predict, 'r', label="Predicted Level")
    ax[0].set_ylabel('Level (cfs)', size=15)
    ax[0].set_xlabel('Date', size=15)
    ax[0].legend(fontsize=15)
    ax[0].set_xticks(og.index[lookback:])
    ax[0].tick_params(axis='x', labelrotation=30)
    ax[0].grid(axis='y')

    dadates = datenames[lookback:]
    x = np.arange(len(dadates))

    width = .25

    be_precip = og.iloc[lookback:, 0]
    vc_precip = og.iloc[lookback:, 1]
    bth_precip = og.iloc[lookback:, 2]

    ax[1].bar(x, be_precip, width, label='Banner Elk')
    ax[1].bar(x + width, vc_precip, width, label='Valle Crucis')
    ax[1].bar(x + 2 * width, bth_precip, width, label='Bethel')

    ax[1].set_xticks(x + (width * 2) / 2, dadates, rotation=30)

    ax[1].set_ylabel('Expected Rainfall Amount (in)', size=15)
    ax[1].set_xlabel('Date', size=15)
    ax[1].legend(fontsize=15)
    ax[1].grid(axis='y')
    plt.tight_layout()
    plt.savefig('TaugOnline/static/graphs/forecast.png')

# Generate the final prediction plot for site
generate_predictions(prediction)