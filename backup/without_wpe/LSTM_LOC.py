#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import nn
import speechbrain as sb
import pytorch_lightning as pl


class LSTM_LOC(pl.LightningModule):
    """
    LSTM Module for
    Continuous Speech Separation (CSS) with localization cost
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LSTM")
        parser.add_argument("--input_lstm", type=int, default=1024)
        parser.add_argument("--hidden_lstm", type=int, default=512)
        parser.add_argument("--n_layer", type=int, default=3)
        return parent_parser

    def __init__(
        self, use_BF, use_DAN, use_SV, input_lstm=1024, hidden_lstm=512, n_layer=3, n_freq=513, n_mic=5, **kwargs
    ):
        super().__init__()

        self.use_BF = use_BF
        self.use_DAN = use_DAN
        self.use_SV = use_SV

        if self.use_BF and self.use_SV:
            input_dim = n_freq * (4 * n_mic - 2)
        elif self.use_BF:
            input_dim = n_freq * (2 * n_mic)
        elif self.use_SV:
            input_dim = n_freq * (4 * n_mic - 3)
        else:
            input_dim = n_freq * (2 * n_mic - 1)

        if self.use_DAN:
            self.dan = nn.Sequential(
                nn.Linear(2, input_lstm // 4),
                nn.SiLU(),
                nn.Linear(input_lstm // 4, input_lstm // 2),
                nn.SiLU(),
                nn.Linear(input_lstm // 2, input_lstm),
                nn.SiLU(),
            )

        self.preprocess = nn.Sequential(
            sb.nnet.normalization.GroupNorm(num_groups=1, input_size=input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_lstm * 2),
            nn.SiLU(),
            nn.Linear(input_lstm * 2, input_lstm),
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

        self.output_mask_speech = nn.Sequential(
            nn.Linear(hidden_lstm * 2, hidden_lstm), nn.SiLU(), nn.Linear(hidden_lstm, 1)
        )
        # self.output_mask_noise = nn.Sequential(
        #     nn.Linear(hidden_lstm * 2, hidden_lstm), nn.SiLU(), nn.Linear(hidden_lstm, 1)
        # )

    def forward(self, batch):
        """
        args
            x: Batch x Time x Feature
            doa: Batch x 2 (cos(theta), sin(theta))
        return
            mask_speech: Batch x Time
            mask_noise: Batch x Time
        """
        if self.use_DAN:
            x, doa = batch
        else:
            x = batch
        x = self.preprocess(x)

        if self.use_DAN:
            directional_feature = torch.sigmoid(self.dan(doa))
            x, _ = self.lstm(x * directional_feature.unsqueeze(1))
        else:
            x, _ = self.lstm(x)

        mask_speech_BT = torch.sigmoid(self.output_mask_speech(x).squeeze())
        return mask_speech_BT

        # mask_noise_BT = torch.sigmoid(self.output_mask_noise(x).squeeze())
        # return mask_speech_BT, mask_noise_BT


if __name__ == "__main__":
    n_mic = 5
    n_freq = 10
    input_dim = n_mic * 2 * n_freq
    hidden_lstm = 3
    batch_size = 2
    model = LSTM_LOC(use_BF=True, use_DAN=True, use_SV=False, n_mic=n_mic, hidden_lstm=hidden_lstm, n_freq=n_freq)

    x = torch.arange(batch_size * 189 * input_dim).to(torch.float32).reshape(batch_size, 189, input_dim)
    doa = torch.rand([batch_size, 2])
    mask_s, mask_n = model.forward((x, doa))
    print(mask_s, mask_n)
