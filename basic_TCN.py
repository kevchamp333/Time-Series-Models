# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:06:26 2022

@author: WooYoungHwang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import timeit


# =============================================================================
# ##### CUDA
# =============================================================================
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# #  Temporal Convolutional Network (intra relationship)
# =============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size                            #제로 패딩된 패딩 크기

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Equivalent to a Residual block

        :param n_inputs: int, number of input channels
        :param n_outputs: int, number of output channels
        :param kernel_size: int, convolution kernel size
        :param stride: int, stride, generally 1
        :param dilation: int, expansion coefficient
        :param padding: int, padding factor
        :param dropout: float, dropout ratio
        """
        super(TemporalBlock, self).__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))          #weight_norm: dailation convolution 이후에 weight normalization을 적용
        # After conv1, the output size is actually (Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)   # Cut out the extra padding part and maintain the output time step as seq_len
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None      # residal function의 출력과 입력의 channel 너비가 다를 경우 1x1 convolution을 정의해 줌
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
                 TCN, the TCN structure given in the paper currently supports the situation where each time is a number, that is, the sequence structure.
                 For the one-dimensional structure where each time is a vector, the vector can barely be split into several input channels at that time.
                 It is not easy to handle the situation where each moment is a matrix or higher-dimensional image.

                 :param num_inputs: int, the number of input channels
                 :param num_channels: list, the number of hidden_channels in each layer, for example [25,25,25,25] means there are 4 hidden layers, and the number of hidden_channels in each layer is 25
                 :param kernel_size: int, convolution kernel size
                 :param dropout: float, drop_out ratio
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # Expansion coefficient: 1, 2, 4, 8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]   # Determine the number of input channels for each layer, the input layer channel is 1, and the hidden layer is 25.
            out_channels = num_channels[i]                              # Determine the number of output channels for each layer
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # *The role is to split the input iterator into elements

    def forward(self, x):
        """
                 The structure of input x is different from RNN. Generally, the size of RNN is (Batch, seq_len, channels) or (seq_len, Batch, channels),
                 Here put seq_len behind the channels, put the data of all time steps together, as the input size of Conv1d, realize the operation of convolution across time steps,
                 Very clever design.

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        forecast = self.linear(y1[:, :, -1])# Add a dimension, 1D·FCN
        #forecast = self.softmax(o)
        return forecast