#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from opt_einsum import contract
import speechbrain as sb

from utils.sp_utils import calc_sv


class FourierFeatures(nn.Module):
    def __init__(self, pos_dim, f_dim, sigma=10, train=False):
        super(FourierFeatures, self).__init__()
        assert f_dim % 2 == 0, 'number of channels must be divisible by 2.'
        enc_dim = int(f_dim / 2)
        self.B = torch.randn([pos_dim, enc_dim]) * sigma
        if train:
            self.B = nn.Parameter(self.B)

    def forward(self, pos):
        # pos: (B L C), (B H W C), (B H W T C)
        pos_enc = torch.matmul(pos, self.B.to(pos.device).to(pos.dtype))
        pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)
        return pos_enc


class DOAAwareLSTM(nn.Module):
    def __init__(self, input_lstm=1024, hidden_lstm=512, n_layer=3, n_freq=513, n_mics=5, dropout=0.2, **kwargs):
        super().__init__()

        input_dim = 2 * n_freq * n_mics
    
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
            nn.Linear(32, input_lstm // 4),
            nn.SiLU(),  
            nn.Linear(input_lstm // 4, input_lstm // 2),
            nn.SiLU(),
            nn.Linear(input_lstm // 2, input_lstm),
            nn.SiLU(), 
            nn.Sigmoid(),
        )

        self.lstm = nn.LSTM(
            input_size=input_lstm,
            hidden_size=hidden_lstm,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.output_mask = nn.Sequential(
            nn.Linear(hidden_lstm * 2, n_freq),
            nn.Sigmoid(),
        )
        
        self.PE = FourierFeatures(pos_dim=1, f_dim=32, sigma=30)

    @staticmethod
    def stft_feature(mix_stft, doa, mic_shape, eps=1e-8):  # [B, M, F, T]
        B, _, F, T = mix_stft.shape
        
        log_magnitude = torch.log(torch.abs(mix_stft).mean(dim=1) + eps)  # [B, F, T]
        log_magnitude = log_magnitude.permute(0, 2, 1)  # [B, T, F]
        
        IPD = mix_stft[:, 1:] / (mix_stft[:, :1] + eps)  # [B, M-1, F, T]
        IPD = IPD / torch.abs(IPD) + eps
        interchannel_phase_diff = torch.view_as_real(IPD.permute(0, 3, 2, 1)).reshape(B, T, F, -1)  # [B, T, F, 2*(M-1)]

        sv = calc_sv(doa, mic_shape, F)  # [B, F, M]
        dsbf = torch.log(torch.abs(contract("bfm,bmft->btf", sv.conj(), mix_stft)) + eps)  # [B, T, F]
        
        feature = torch.concat((log_magnitude[..., None], interchannel_phase_diff, dsbf[..., None]), dim=-1)  # [B, T, F, 2*M]
        feature = feature.flatten(-2, -1)  # [B, T, 2*F*M]
        
        return feature

    def forward(self, mix_stft, batch):
        _, _, doa, mic_shape = batch

        # feature extraction
        mix_feature = self.stft_feature(mix_stft, doa, mic_shape)  # [B, T, _D]
        doa_rad = torch.deg2rad(doa)  # [B,]
        doa_feature = self.PE(doa_rad[:, None])  # [B, 32]
        
        x1 = self.preprocess(mix_feature)  # [B, T, D]
        x2 = self.dan(doa_feature)[:, None]  # [B, 1, D]
        
        z, _ = self.lstm(x1 * x2)

        mask = self.output_mask(z)

        return mask


if __name__ == "__main__":
    input_dim = 5
    hidden_lstm = 3
    batch_size = 1
    model = DOAAwareLSTM(input_dim=input_dim, hidden_lstm=hidden_lstm).to("cuda").to(torch.float64)

    mix_stft = torch.ones((batch_size, 5, 513, 189), dtype=torch.complex128, device="cuda")
    doa = torch.rand(batch_size, device="cuda", dtype=torch.float64)
    mic_shape = torch.rand([batch_size, 2, 5], device="cuda", dtype=torch.float64)
    model.forward(mix_stft, (mix_stft, mix_stft, doa, mic_shape))
