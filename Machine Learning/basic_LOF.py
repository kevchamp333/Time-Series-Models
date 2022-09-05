# -*- coding: utf-8 -*-
"""
Created on Thu May 19 13:59:50 2022

@author: WooYoungHwang
"""
import sys
sys.path.append(r'C:\Users\WooYoungHwang\Desktop\SPS\연구\code')

import torch
import pyreadr
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Read Data
# =============================================================================
# read 해외N사 data file (file is R)
result = pyreadr.read_r(r'C:\Users\WooYoungHwang\Desktop\SPS\데이터\비스텔\2차 데이터 전달\해외 N사\raw_data.RData') 

data = result['nis']
data = data.drop('serialnum', 1)        # serialnum is not needed
data = data.sort_values('time_stamp')   # sort values by 'time-stamp'

# sensorID & replacement data
sensorid = 'N119'
sensor_N = data[data['SensorID'] == sensorid]
sensor_N = sensor_N.drop(['spt_cnt'], axis = 1)

sensor_data = sensor_N.iloc[:, 3:]
sensor_N['time_stamp'] = pd.to_datetime(sensor_N['time_stamp'])
sensor_data.set_index(sensor_N['time_stamp'].dt.date, inplace = True)

sensor_data= sensor_data.fillna(method = 'ffill')


replacement_date = '2017-11-29'
def find_replace_index(data, rd_date):
    
    replacement_date = rd_date          #replacement date in string 'y-m-d'
    replacement_date_ = datetime.strptime(replacement_date, '%Y-%m-%d')

    for i in range(len(data)):
        if data.index[i] == replacement_date_.date():
            return data[:i], i

_, fault_index = find_replace_index(sensor_data, replacement_date)         #find replacement date's index
fault_index += 5000                                                         # fault index 뒤에 buffer
# fault_index = 40473     #104
# subdata
sub_data = sensor_data.iloc[: int(fault_index)].fillna(method = 'ffill')

# scaling
sub_scaler = StandardScaler()                                                   # scaling for PCA
data_rescaled = sub_scaler.fit_transform(sub_data)                       
scaled_pca = pd.DataFrame(data_rescaled,  columns = sub_data.columns).fillna(method='bfill')
scaled_pca = scaled_pca[:].reset_index(drop = True)


# =============================================================================
# Local Outlier Factor
# =============================================================================

from sklearn.neighbors import LocalOutlierFactor

clf=LocalOutlierFactor(contamination=0.1)
y_pred=clf.fit_predict(scaled_pca)



#f1_score
import math
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#n104
y_true = [1 for x in range(0,33950)] + [-1 for x in range(33950, 40473)]

#n110
y_true = [1 for x in range(0,137377)] + [-1 for x in range(137377, 142377)]

#n119
y_true = [1 for x in range(0,48723)] + [-1 for x in range(48723, 53723)]

print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))

