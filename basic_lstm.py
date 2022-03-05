# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:09:36 2021

@author: Woo Young Hwang
"""
import timeit
import numpy as np
import pandas as pd
import pandas_datareader.data as pdf
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
import timeit

#####정규화 과정
'''
StandardScaler 각 특징의 평균을 0, 분산을 1이 되도록 변경
MinMaxScaler 최대/최소값이 각각 1, 0이 되도록 변경
'''

# =============================================================================
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# mm = MinMaxScaler()
# ss = StandardScaler()
# 
# =============================================================================
cnc_df = pd.read_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\데이터\cnc mill tool wear\experiment_14.csv')
cnc_df = cnc_df.drop(columns = ['Machining_Process'])
cnc_df = cnc_df.iloc[:, [1, 4, 6, 9, 15, 20]]


#create train data
ss = StandardScaler()
x_train_ss = ss.fit_transform(cnc_df)

#outlier removal
z_scores = zscore(x_train_ss)
abs_z_score = np.abs(z_scores)
filtered_entries = (abs_z_score < 3).all(axis = 1)
x_train_ss = x_train_ss[filtered_entries]
x_train_ss = x_train_ss[:1000]


X_train = x_train_ss[:2100]
y_train = x_train_ss[1:2101, [1]]



test_data = []
for i in range(100):
    test_data.append(x_train_ss[i:2100 + i])

true = x_train_ss[2100:2200, [0]]


    
print('Training Shape', X_train.shape, y_train.shape)   #이 shape으론 lstm학습 불가능
#print('Testing Shape', X_test.shape, y_test.shape)



#####학습할 수 있는 형태로 변환하기 위해 Torch 변환
'''
torch Variable에는 3개의 형태가 있다.
data, grad, grad_fn
'''
X_train_tensors = Variable(torch.Tensor(X_train))       #torch.Size([4500, 5])
#X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
#y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors_final = torch.reshape(X_train_tensors, (2100, 1, 6))  #torch.Size([4500, 1, 5])
#X_test_tensors_final = torch.reshape(X_test_tensors,  (200, 1, 1))      #torch.Size([795, 1, 5])

#####GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(0))



##### LSTM 네트워크 구성하기
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes      #number of classes
        self.num_layers = num_layers        #number of layers
        self.input_size = input_size        #input size
        self.hidden_size = hidden_size      #hidden state
        self.seq_length = seq_length        #sequence length
        
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
    
    
#####네트워크 파아미터 구성하기
num_epochs = 3000
learing_rate = 0.01

input_size = 6     #number of features
hidden_size = 2     #number of features in hidden state
num_layers = 1      #number of stacked lstm layers

num_classes = 1     #number of output classes

lstm1= LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[2]).to(device)

loss_function = torch.nn.MSELoss()          #mean squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr = learing_rate)

#####학습하기
start = timeit.default_timer()
for epoch in range(num_epochs):
    lstm1.train()
    iter_loss = 0
    optimizer.zero_grad() 
    outputs = lstm1.forward(X_train_tensors_final.to(device))
    loss = loss_function(outputs, torch.tensor(y_train).float().to(device))
    loss.backward()                                                 #calculate the loss of the loss function
    optimizer.step()   #calculate the gradient, manually setting to 
    print("[Epoch %d/%d] [train loss: %f]" % (epoch, num_epochs, loss.item()))
stop = timeit.default_timer()
print('Time: ', stop - start)  

plt.plot(np.array(outputs.detach().cpu()))
plt.plot(y_train)


####예측하기
#reshaping the dataset
result = []
for i, data in enumerate(test_data):
    train_predict = lstm1(torch.tensor(data).unsqueeze(0).permute(1, 0, 2).float().to(device))#forward pass
    data_predict = train_predict[-1].detach().cpu().numpy() #numpy conversion
    result.append(data_predict)

dataY_plot = true

plt.figure(figsize=(10,6)) #plotting
plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(result[:], label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 

dt = np.concatenate((result, true), axis = 1)
result_dt = pd.DataFrame(dt)
result_dt.to_csv(r'C:\Users\Woo Young Hwang\Desktop\SPS\연구\code\multi levvel convolutional autoencoder network\lstm\lstm5비교.csv')


#plt.axvline(x=4500, c='r', linestyle='--') #size of the training set
