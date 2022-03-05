# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:01:42 2021

@author: Woo Young Hwang
"""

# Source: https://leedakyeong.tistory.com/entry/Python-%EB%82%A0%EC%94%A8-%EC%8B%9C%EA%B3%84%EC%97%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0Kaggle%EB%A1%9C-ARIMA-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0

import timeit
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pandas import datetime
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from statsmodels.tsa.arima_model import ARIMA

# =============================================================================
# Read Data
# =============================================================================
#cnc-milling dataset
cnc_df = pd.read_csv(r'C:\Users\WooYoungHwang\Desktop\SPS\데이터\cnc mill tool wear\experiment_01.csv')
cnc_df = cnc_df.drop(columns = ['Machining_Process'])
cnc_df = cnc_df.iloc[:, [1, 4, 6, 9, 15, 20]]
data = np.array(cnc_df)

#turbofan dataset z-norm data
turbofan_df = pd.read_csv(r'C:\Users\WooYoungHwang\Desktop\SPS\연구실 project\비스텔\코드\비스텔_터보팬_RUL_구현\데이터\train_FD002_RUL.csv')
turbofan_df = turbofan_df[turbofan_df.unit == 2.0]
turbofan_df = turbofan_df.drop(['sensor1', 'sensor5', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19'], axis = 1)
turbofan_df = turbofan_df.iloc[:, 2:-1]
data = np.array(turbofan_df)

#alltrainmescla5D dataset
train_data = pd.read_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\데이터\predictive useful life based into telemetry\ALLtrainMescla5D.csv')
#input 4개 센서 데이터로 코드 검증 중...
data = train_data[train_data.machineID == 2.0]
data = data.iloc[75:,[3, 4, 5, 6, 11, 12, 13, 14]]


# =============================================================================
#  Basic EDA
# =============================================================================
# Plot the data
plt.figure(figsize = (20, 15))
plt.plot(data.iloc[:, 0])
plt.show()

# Create univaraite time series from original data
timeSeries = data.iloc[:, 0]
timeSeries.index = range(len(timeSeries))

# Seasonal decomposition
sd_result = seasonal_decompose(timeSeries, model = 'additive', period = 2)  # seasonal: must look to the original data for the right seasonality 
fig = plt.figure()
fig = sd_result.plot()
fig.set_size_inches(20, 15)

# AutoCorrelation FUcntion (ACF) Graph --> Lag에 따른 관측치들 사이의 관련성을 측정하는 함수
fig = plt.figure(figsize = (20, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(timeSeries, lags = 20, ax = ax1)

# Augmented Dickey-Fuller Test (ADF) --> Check time series stationarity (if p-value is over 0.05 not stationary)
adf_result = adfuller(timeSeries)
print('ADF Statistics: %f' %adf_result[0])
print('p-value: %f' %adf_result[1])
print('Critical Values: ')
for key, values in adf_result[4].items():
    print('\t%s: %.3f' %(key, values))

# 1st order difference (for non-stationary data)
ts_diff = timeSeries - timeSeries.shift()
plt.figure(figsize = (22, 8))
plt.plot(ts_diff)
plt.title('Differencing Method')
plt.xlabel('time')
plt.ylabel('Difference')
plt.show()


# =============================================================================
# Auto-regressive Integrated Moving Average (ARIMA)
# =============================================================================
# 정상성을 만족하는 데이터로 ACF와 PACF 그래프를 그려 ARIMA 모형의 p와 q를 결정한다
 #  p와 d, q는 어떻게 정해야 할까? Rules of thumb이긴 하지만 ACF plot와 PACF plot을 통해 AR 및 MA의 모수를 추정
# ACF: Lag에 따른 관측치들 사이의 관련성을 측정하는 함수
# PACF: k 이외의 모든 다른 시점 관측치의 영향력을 배제하고 y(t)와 y(t-k)두 관측치의 관련성을 측정하는 함수
fig = plt.figure(figsize = (20, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(timeSeries, lags = 20, ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(timeSeries, lags = 20, ax = ax2)

#fit model
p = 0
d = 1
q = 1
model = ARIMA(timeSeries, order = (p,d,q))
model_fit = model.fit(disp = 0)

#predict
start_index = 1     #datetime(1944, 6, 25)
end_index = 250      #datetime(1945, 5, 31)
forecast = model_fit.predict(start = start_index, end = end_index, typ = 'linear')

#visualization
plt.figure(figsize = (22, 8))
plt.plot(timeSeries, label = 'Origianl')
plt.plot(forecast, label = 'predicted')
plt.title('Time Series Forecast')
plt.xlabel('time')
plt.ylabel('y')
plt.legend()
plt.show()




################undone
# create train & test dataset
train_df = data[:200]
test_df = data[200:]

# scaling (Minmax & Standard Scaler)
ss = StandardScaler()
x_train_ss = ss.fit_transform(data)

mm = MinMaxScaler()
x_train_ss = mm.fit_transform(cnc_df)

#outlier removal
z_scores = zscore(train_df)                          #compute the z-score 
abs_z_score = np.abs(z_scores)                          
filtered_entries = (abs_z_score < 3).all(axis = 1)      # log entries that go over 3-sigma
train_df = train_df[filtered_entries]               # filter out entries that go over 3-sigma



#create cae train batch
test_data = []
for i in range(20):
    test_data.append(x_train_ss[i:234 + i, [5]])

true = x_train_ss[234:254, [5]]

from statsmodels.tsa.arima_model import ARIMA

start = timeit.default_timer()
result = []
for i, data in enumerate(test_data):
    model = ARIMA(data, order=(1,1,0))
    model_fit = model.fit(trend='c',full_output=True, disp=1)
    #print(model_fit.summary())
    fore = model_fit.forecast(steps=1)
    result.append(fore[0])

plt.plot(result)
plt.plot(true)
stop = timeit.default_timer()
print('Time: ', stop - start)  

dt = np.concatenate((result, true), axis = 1)
result_dt = pd.DataFrame(dt)
result_dt.to_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\연구\논문\석사논문경진대회\코드\result\telemetry_arima\mm_arima7비교.csv')


