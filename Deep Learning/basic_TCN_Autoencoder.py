# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:37:26 2022

@author: WooYoungHwang
"""

import sys
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.nn import functional as F
from basic_TCN import TemporalConvNet


class TCN_AE(nn.Module):
    def __init__(self, input_channel: int, hidden_dims: int, output_channel: int, seq_length: int, num_channels: list):
        super(TCN_AE, self).__init__()
        
        self.input_channel = input_channel
        self.hidden_dims = hidden_dims
        self.output_channel = output_channel
        self.seq_length = seq_length
        self.num_channels = num_channels
        kernel_size = 3
        dropout = 0.1
        
        # =============================================================================
        # Build Encoder
        # =============================================================================
        """
        :param num_inputs: int, the number of input channels
        :param num_channels: list, the number of hidden_channels in each layer, for example [25,25,25,25] means there are 4 hidden layers, and the number of hidden_channels in each layer is 25
        :param kernel_size: int, convolution kernel size
        :param dropout: float, drop_out ratio
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        encoder_modules = []
        
        self.encoder = nn.Sequential(
                            TemporalConvNet(num_inputs = self.input_channel, num_channels = self.num_channels, kernel_size=kernel_size, dropout=dropout),
                            # nn.AvgPool1d(3, stride = 2)
                        )

        # =============================================================================
        # Build Encoder
        # =============================================================================
        self.decoder = nn.Sequential(
                            nn.Linear(1, self.input_channel // 2),
                            nn.Linear(self.input_channel // 2, self.input_channel)
                        )
        
        
    def forward(self, input: torch.tensor):
        latent = self.encoder(input)
        latent = latent.permute(0, 2, 1)
        recon = self.decoder(latent)
        
        return  recon.permute(0, 2, 1)
