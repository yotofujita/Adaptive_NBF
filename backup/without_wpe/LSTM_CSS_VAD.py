#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import nn
import speechbrain as sb
import pytorch_lightning as pl


class LSTM_CSS_VAD(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LSTM")
        parser.add_argument("--input_lstm", type=int, default=1024)
        parser.add_argument("--hidden_lstm", type=int, default=512)
        parser.add_argument("--n_layer", type=int, default=3)
        return parent_parser

    def __init__(self, input_dim=5130, input_lstm=1024, hidden_lstm=512, n_layer=3, n_freq=513, **kwargs):
        super().__init__()

        self.preprocess = nn.Sequential(
            sb.nnet.normalization.GroupNorm(num_groups=1, input_size=input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_lstm),
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
        self.output_vad = nn.Sequential(
            nn.Conv1d(in_channels=hidden_lstm * 2, out_channels=hidden_lstm, kernel_size=3, stride=3),
            nn.SiLU(),
            nn.Conv1d(in_channels=hidden_lstm, out_channels=128, kernel_size=3, stride=3),
            nn.SiLU(),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, stride=2),
        )

    def forward(self, x):
        """
        args
            x: Batch x Time x Feature
        return
            output: Batch x Time x Frequency
        """
        x = self.preprocess(x)
        x, _ = self.lstm(x)
        mask = torch.sigmoid(self.output_mask(x))
        vad = torch.sigmoid(self.output_vad(x.permute(0, 2, 1)).squeeze())
        # vad = self.expand_vad(vad)

        return mask, vad

    def expand_vad(self, vad):
        """
        args
            vad: Batch x Time
        return
            output: Batch x (Time * 18 + 9)
        """
        n_batch, _ = vad.shape
        vad = nn.ReplicationPad1d(2)(torch.tile(vad, (4, 1, 1)).permute(1, 2, 0).reshape(n_batch, -1))
        vad = torch.nn.MaxPool1d(kernel_size=3, stride=2)(vad)
        vad = torch.tile(vad, (9, 1, 1)).permute(1, 2, 0).reshape(n_batch, -1)

        return vad


if __name__ == "__main__":
    input_dim = 5
    hidden_lstm = 3
    batch_size = 2
    model = LSTM_CSS_VAD(input_dim=input_dim, hidden_lstm=hidden_lstm)

    x = torch.arange(batch_size * 189 * input_dim).to(torch.float32).reshape(batch_size, 189, input_dim)
    model.forward(x)
