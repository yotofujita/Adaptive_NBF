#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import speechbrain as sb


class LSTM_CSS(nn.Module):
    """
        LSTM Module for 
        Continuous Speech Separation (CSS) and Voice Activity Detection (VAD)
        with Direction Attractor Network (DAN)
    """

    def __init__(self, input_lstm=1024, hidden_lstm=512, n_layer=3, n_freq=513, n_mic=5, **kwargs):
        super().__init__()

        input_dim = n_freq * (2 * n_mic)
    
        self.preprocess = nn.Sequential(
            sb.nnet.normalization.GroupNorm(num_groups=1, input_size=input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_lstm * 2),
            nn.SiLU(),
            nn.Linear(input_lstm * 2, input_lstm),
            nn.SiLU(),
        )

        self.dan = nn.Sequential(
            nn.Linear(2, input_lstm // 4),
            nn.SiLU(),  
            nn.Linear(input_lstm // 4, input_lstm // 2),
            nn.SiLU(),
            nn.Linear(input_lstm // 2, input_lstm),
            nn.SiLU(),
        )

        self.lstm = nn.LSTM(
            input_size=input_lstm,
            hidden_size=hidden_lstm,
            num_layers=n_layer,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        self.output_mask = nn.Linear(hidden_lstm * 2, n_freq)

    def forward(self, batch):
        """
        args
            x: Batch x Time x Feature
            doa: Batch x 2 (cos(theta), sin(theta))
        return
            output: Batch x Time x Frequency
        """
        x, doa = batch
        
        x = self.preprocess(x)

        directional_feature = torch.sigmoid(self.dan(doa))
        
        x, _ = self.lstm(x * directional_feature.unsqueeze(1))

        mask = torch.sigmoid(self.output_mask(x))

        return mask


if __name__ == "__main__":
    input_dim = 5
    hidden_lstm = 3
    batch_size = 2
    model = LSTM_CSS(input_dim=input_dim, hidden_lstm=hidden_lstm)

    x = torch.arange(batch_size * 189 * input_dim).to(torch.float32).reshape(batch_size, 189, input_dim)
    doa = torch.rand([batch_size, 2])
    model.forward(x, doa)
