from __future__ import absolute_import

from typing import Tuple
import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, n_ts: int, n_features: int, n_channels: int=1,
                 out_dim: int=1, n_filters: Tuple[int, int, int]=(8, 8, 8), 
                 dropout_p: float=0.2) -> None:
        
        # Initialise module class
        super(ConvNet, self).__init__()

        # Define Layers
        self.conv_1 = nn.Conv2d(in_channels=n_channels,
                                out_channels=n_filters[0],
                                kernel_size=(2, 1),
                                bias=True)
        
        self.conv_2 = nn.Conv2d(in_channels=n_filters[0],
                                 out_channels=n_filters[1],
                                 kernel_size=(1, 2),
                                 bias=True)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_3 = nn.Conv2d(in_channels=n_filters[1],
                                 out_channels=n_filters[2],
                                 kernel_size=(2, 2),
                                 bias=True)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(in_features=n_filters[-1]*(n_features-7)*((n_ts-1)//2), 
                                out_features=out_dim)
        
    
    def forward(self, x):
        out = self.conv_1(x).relu()
        out = self.conv_2(out).relu()
        out = self.max_pool(out)
        out = self.conv_3(out).relu()
        out = self.dropout(self.flatten(out))
        out = self.linear(out)
        return out
    

class ConvNetAE(nn.Module):
    
    def __init__(self, ts_in, in_dim, ts_out, out_dim=1, hidden_dim=64, kernel_size=(3,3), stride=1, padding=1):
        super(ConvNetAE, self).__init__()
        
        # Define parameters
        self.ts_in = ts_in
        self.in_dim = in_dim
        self.ts_out = ts_out
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, 
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, 
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )
        
        # Bottleneck: Flattening temporal dimension before feeding into decoder
        self.fc_enc = nn.Linear(hidden_dim * ts_in * in_dim, 
                                hidden_dim * ts_out * in_dim)
        self.fc_dec = nn.Linear(hidden_dim * ts_out * in_dim, 
                                hidden_dim * ts_out * in_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, 
                               kernel_size=kernel_size, stride=stride, 
                               padding=padding),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=1, 
                               kernel_size=kernel_size, stride=stride, 
                               padding=padding)
        )
        
        # Final output projection
        self.fc_out = nn.Linear(ts_out * in_dim, ts_out * out_dim)
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_enc(x)
        x = self.fc_dec(x)
        x = x.view(x.size(0), self.hidden_dim, self.ts_out, self.in_dim)  # Reshape back for decoder
        
        # Decode
        x = self.decoder(x)
        x = x.view(x.size(0), -1)  # Flatten for final projection
        x = self.fc_out(x)
        x = x.view(x.size(0), self.ts_out, self.out_dim)  # Reshape to desired output shape
        return x