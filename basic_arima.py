# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:01:42 2021

@author: Woo Young Hwang
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
import timeit
import seaborn as sns

cnc_df = pd.read_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\데이터\cnc mill tool wear\experiment_01.csv')
cnc_df = cnc_df.drop(columns = ['Machining_Process'])
cnc_df = cnc_df.iloc[:, [1, 4, 6, 9, 15, 20]]


#turbofan dataset z-norm data
train_data = pd.read_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\연구실 project\비스텔\코드\비스텔_터보팬_RUL_구현\데이터\train_FD002_RUL.csv')
#input 4개 센서 데이터로 코드 검증 중...
data = train_data[train_data.unit == 2.0]
data = data.drop(['sensor1', 'sensor5', 'sensor6', 'sensor10', 'sensor16', 'sensor18', 'sensor19'], axis = 1)
data = data.iloc[:, 2:-1]
x_train_ss = np.array(data)
x_train_ss.shape


train_data = pd.read_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\데이터\predictive useful life based into telemetry\ALLtrainMescla5D.csv')
#input 4개 센서 데이터로 코드 검증 중...
data = train_data[train_data.machineID == 2.0]
data = data.iloc[75:,[3, 4, 5, 6, 11, 12, 13, 14]]
# =============================================================================
# heat = pd.DataFrame((x_train_ss.transpose()))
# 
# plt.pcolor(heat)
# plt.xticks(np.arange(0.5, len(heat.columns), 1), heat.columns)
# #plt.yticks(np.arange(0.5, len(heat.index), 1), heat.index)
# plt.title('Data Correleation by Heatmap', fontsize = 20)
# plt.xlabel('time', fontsize = 14)
# plt.ylabel('variables', fontsize = 14)
# plt.colorbar()
# plt.show()
# =============================================================================
data.shape


#create train data
ss = StandardScaler()
x_train_ss = ss.fit_transform(data)

mm = MinMaxScaler()
x_train_ss = mm.fit_transform(cnc_df)

#outlier removal
z_scores = zscore(x_train_ss)
abs_z_score = np.abs(z_scores)
filtered_entries = (abs_z_score < 3).all(axis = 1)
x_train_ss = x_train_ss[filtered_entries]
x_train_ss = x_train_ss[:1000]

#create cae 석사 train batch
test_data = []
for i in range(20):
    test_data.append(x_train_ss[i:234 + i, [7]])

true = x_train_ss[234:254, [7]]

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


