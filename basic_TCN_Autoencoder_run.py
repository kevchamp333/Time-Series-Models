# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:36:24 2022

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
from basic_TCN_Autoencoder import TCN_AE


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
# Train Data & Test Data
# =============================================================================
def sliding_window(df, time_window, variable_num, create_target):
    x = []
    y = []

    for i in range(len(df)-time_window-1):
        _x = df[i:(i+time_window)]
        x.append(_x)

        if create_target == True:
            _y = df[i+time_window-1]
            y.append(_y)        
        
    return np.array(x), np.array(y)

train_x_o = scaled_pca.iloc[int(fault_index * 0.7) :int(fault_index * 0.9)]  # 전체 데이터의 30% train data로 지정
test_x_o  = scaled_pca.iloc[: int(fault_index)]
 
#train data
window_size = 64
variable_num = None
total_train_data = train_x_o
create_target = False
train_x, train_y = sliding_window(total_train_data, window_size, variable_num, create_target)      # x --> window sized train data
train_dataloader = DataLoader(train_x, batch_size = 50, shuffle = False)

#test data
window_size = 64
variable_num = None
total_test_data = test_x_o
create_target = False
test_x, test_y= sliding_window(total_test_data, window_size, variable_num, create_target)      # x --> window sized train data
test_dataloader = DataLoader(test_x, batch_size = 50, shuffle = False)

# =============================================================================
# Initialize Model
# =============================================================================
model = TCN_AE(input_channel= len(train_x_o.columns), hidden_dims = 1024, output_channel = len(train_x_o.columns), seq_length = window_size,  num_channels= [20, 20, 20, 1])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.0002)
loss_function = torch.nn.MSELoss()          

epoch = 200

# =============================================================================
# Train Model
# =============================================================================

model.train()
progress = tqdm(range(epoch))
recon_record = []

for i in progress:
    batchloss = 0.0

    for batch_idx , batch_x  in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch_x_device = batch_x.float().to(device).permute(0, 2, 1)

        # train the model
        result = model(batch_x_device)
        loss = loss_function(result, batch_x_device)
        loss.backward()
        optimizer.step()

        batchloss += loss

    recon_record.append(batchloss.cpu().item() / len(train_dataloader))
    progress.set_description("{:0.5f}".format(loss.cpu().detach()))
    
# plot loss
plt.plot(recon_record[10:])

# =============================================================================
# Test
# =============================================================================
model.eval()
test_loss_record = []
test_results = []

with torch.no_grad():
    for batch_idx , batch_x  in enumerate(test_dataloader):
        batch_x_device = batch_x.float().to(device).permute(0, 2, 1)
        result = model(batch_x_device)
        test_results.append(result[-1,-1,:].detach().cpu())
        recon_loss = loss_function(result, batch_x_device).cpu().detach()
        test_loss_record.append(recon_loss.cpu().detach())


# =============================================================================
# Dynamic Threshold
# =============================================================================
test_score_df = pd.DataFrame(test_loss_record[:])
test_score_df['loss'] = pd.DataFrame(test_loss_record[:])

# Dynamic Threshold
window = 32
std_coef = 0.8
test_pred_errors_windowed = test_score_df['loss'] 
# test_pred_errors_windowed = pd.Series(test_score_df['loss']).rolling(window=window, min_periods=5)
test_pred_errors_windowed = test_score_df['loss'].ewm(alpha = 0.4, min_periods = 3, adjust = True).mean()
upper_test_dynamic_threshold = test_pred_errors_windowed + std_coef * test_pred_errors_windowed.std()
lower_test_dynamic_threshold = test_pred_errors_windowed - std_coef * test_pred_errors_windowed.std()

plt.figure(figsize = (15, 10))
plt.plot(test_score_df['loss'], color = 'r', linewidth = 3, label ='Test loss')
plt.plot(upper_test_dynamic_threshold, color = 'b', label ='Upper_Threshold')
plt.plot(lower_test_dynamic_threshold, color = 'y', label ='Lower_Threshold')
plt.axvline(x = int((fault_index-5000)/50), color = 'r', linestyle = '--', linewidth = 2, label = 'J3')
plt.axvspan(0, int((fault_index* 0.3)/50), alpha=0.2, color='orange', label = 'train_ratio')
plt.title('%s_(J3)Test loss vs. Threshold & fault timestep' % sensorid , fontsize = 15)
plt.legend(loc = 'best')
plt.show()

# Dynamic Threshold 결과 zoom in (normal)
plt.figure(figsize = (15, 10))
plt.plot(test_score_df['loss'][400: 700], color = 'r', linewidth = 3, label ='Test loss')
plt.plot(upper_test_dynamic_threshold[400: 700], color = 'b', label ='Upper_Threshold')
plt.plot(lower_test_dynamic_threshold[400: 700], color = 'y', label ='Lower_Threshold')
plt.legend(loc = 'best')
plt.title('Machinery3_Normal Data Test Loss' , fontsize = 15)
plt.show()

# Dynamic Threshold 결과 zoom in (abnormal)
plt.figure(figsize = (15, 10))
plt.plot(test_score_df['loss'][int((fault_index-5000)/50) - 300: int((fault_index-5000)/50)  + 500], color = 'r', linewidth = 3, label ='Test loss')
plt.plot(upper_test_dynamic_threshold[int((fault_index-5000)/50) - 300: int((fault_index-5000)/50)  + 500], color = 'b', label ='Upper_Threshold')
plt.plot(lower_test_dynamic_threshold[int((fault_index-5000)/50) - 300: int((fault_index-5000)/50)  + 500], color = 'y', label ='Lower_Threshold')
plt.axvline(x = int((fault_index-5000)/50) - 30, color = 'r', linestyle = '--', linewidth = 2, label = 'Type 1 anomaly')
plt.axvspan(int((fault_index-5000)/50) - 30, 1008, alpha=0.2, color='orange', label = 'anomaly')
plt.legend(loc = 'best')
plt.title('Machinery3_Abnormal Data Test Loss' , fontsize = 15)
plt.show()


# extra
# threshold
final_loss = []
for i in range(len(test_score_df['loss'])):
    if test_score_df['loss'][i] >= upper_test_dynamic_threshold[i]:
        final_loss.append(abs(test_score_df['loss'][i] - upper_test_dynamic_threshold[i]))
    elif test_score_df['loss'][i] <= lower_test_dynamic_threshold[i]:
        final_loss.append(abs(test_score_df['loss'][i] - lower_test_dynamic_threshold[i]))
    else:
        final_loss.append(0)

# zoom in
plt.figure(figsize = (15, 10))
plt.plot(final_loss, color = 'r', linewidth = 3, label ='Test loss')
plt.axvline(x = int((fault_index-5000)/50) -30 , color = 'b', linestyle = '--', linewidth = 2, label = 'PIN')
plt.axvspan(0, int((fault_index* 0.3)/50), alpha=0.2, color='orange', label = 'train_ratio')
plt.legend(loc = 'best')
plt.show()


#f1_score
import math
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#n104
y_pred = np.ceil(final_loss)
y_true = [0 for x in range(int((fault_index-5000)/50) - 30)] + [1 for x in range(0, len(test_score_df['loss']) - (int((fault_index-5000)/50) - 30))]

f1_score(y_true, y_pred, average='micro')

print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))

#n110
y_pred = np.ceil(final_loss)
y_true = [0 for x in range(int((fault_index-5000)/50))] + [1 for x in range(0, len(test_score_df['loss']) - (int((fault_index-5000)/50)))]

f1_score(y_true, y_pred, average='micro')

print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))

#n119
y_pred = np.ceil(final_loss)
y_true = [0 for x in range(int((fault_index-5000)/50)- 30)] + [1 for x in range(0, len(test_score_df['loss']) - (int((fault_index-5000)/50) - 30))]

f1_score(y_true, y_pred, average='micro')

print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))
