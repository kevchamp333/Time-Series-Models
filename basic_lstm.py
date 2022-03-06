# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:09:36 2021

@author: Woo Young Hwang
"""
import torch
import timeit
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from scipy.stats import zscore
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#####GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))


# =============================================================================
# 정규화 과정
# =============================================================================
# StandardScaler 각 특징의 평균을 0, 분산을 1이 되도록 변경
# MinMaxScaler 최대/최소값이 각각 1, 0이 되도록 변경


# =============================================================================
# Read Data (total data)
# =============================================================================
cnc_df = pd.read_csv(r'C:\Users\WooYoungHwang\Desktop\SPS\데이터\cnc mill tool wear\experiment_14.csv')
cnc_df = cnc_df.drop(columns = ['Machining_Process'])
data = cnc_df.iloc[:, [1, 4, 6, 9, 15, 20]]
label = cnc_df.iloc[:, [3]].shift()

#train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size = 0.3, shuffle = False)

#scaling
ss = StandardScaler()
x_train_S = ss.fit_transform(X_train)
x_test_S = ss.transform(X_test)

#outlier removal (optinal)
z_scores = zscore(x_train_S)
abs_z_score = np.abs(z_scores)
filtered_entries = (abs_z_score < 3).all(axis = 1)
x_train_S_outlier, y_train_outlier = x_train_S[filtered_entries], Y_train[filtered_entries]

#multivariable data에서 sliding window 
def sliding_window(x_df, y_df, time_window, variable_num, create_target):
    x = []
    y = []

    for i in range(len(x_df)-time_window-1):
        _x = x_df[i:(i+time_window)]
        x.append(_x)

        if create_target == True:
            #_y = data[i+seq_length, [variable_num]]        # specific target variable
            _y = y_df.iloc[i+time_window]
            y.append(_y)        
        
    return np.array(x), np.array(y)

seq_length = 7
variable_num = 0
create_target = True

x_train, y_train = sliding_window(x_train_S_outlier, y_train_outlier, seq_length, variable_num, create_target)   
x_test, y_test = sliding_window(x_test_S, Y_test, seq_length, variable_num, create_target)   

# convert to torch.tensor
x_train_tensor = Variable(torch.Tensor(x_train))
x_test_tensor = Variable(torch.Tensor(x_test))
y_train_tensor = Variable(torch.Tensor(y_train))
y_test_tensor = Variable(torch.Tensor(y_test))

print('Training Shape', x_train_tensor.shape, y_train_tensor.shape)   
print('Testing Shape', x_test_tensor.shape, y_test_tensor.shape)

# create train & test dataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
test_dataloader = DataLoader(test_dataset, shuffle = False)

# =============================================================================
# Long-Short Term Memory (LSTM)
# =============================================================================
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes      #number of classes
        self.num_layers = num_layers        #number of layers
        self.input_size = input_size        #input size
        self.hidden_size = hidden_size      #hidden state
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)        #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)        #internal state
        
        #propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))     #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)       #reshaping the data for Dense layer next
        #hn = hn[-1]
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        
        return out
    
    
# =============================================================================
# LSTM parameters
# =============================================================================
epochs = 3000
learing_rate = 0.01

input_size = 6     #number of features
hidden_size = 2     #number of features in hidden state
num_layers = 1      #number of stacked lstm layers

num_classes = 1     #number of output classes

lstm= LSTM(num_classes, input_size, hidden_size, num_layers).to(device)

criterion = torch.nn.MSELoss()          #mean squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr = learing_rate)


# =============================================================================
# Train
# =============================================================================
# ver1
start = timeit.default_timer()
train_losses = []
model = lstm.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx , (batch_x, target)  in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch_x_tensor = batch_x.float().to(device)
        seq_pred = model(batch_x_tensor)
        loss = criterion(seq_pred, target.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss/len(train_dataloader))
    print("[Epoch %d/%d] [train loss: %f]" % (epoch, epochs, epoch_loss/len(train_dataloader)))
stop = timeit.default_timer()
print('Time: ', stop - start)  


# ver2
start = timeit.default_timer()
for epoch in range(epochs):
    lstm.train()
    iter_loss = 0
    optimizer.zero_grad() 
    outputs = lstm.forward(x_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.float().to(device))
    loss.backward()                                                 #calculate the loss of the loss function
    optimizer.step()   #calculate the gradient, manually setting to 
    print("[Epoch %d/%d] [train loss: %f]" % (epoch, epochs, loss.item()))
stop = timeit.default_timer()
print('Time: ', stop - start)  

plt.plot(np.array(outputs.detach().cpu()))
plt.plot(y_train)


# =============================================================================
# Test
# =============================================================================
train_predict = lstm(x_test_tensor.to(device))#forward pass
data_predict = train_predict.detach().cpu().numpy() #numpy conversion


plt.figure(figsize=(10,6)) #plotting
plt.plot(y_test_tensor, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 

dt = np.concatenate((result, true), axis = 1)
result_dt = pd.DataFrame(dt)
result_dt.to_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\연구\code\multi levvel convolutional autoencoder network\lstm\lstm5비교.csv')


