#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchaudio
import pytorch_lightning as pl

from utils.import_utils import instantiate

from utils.utility import calc_SI_SDR, MVDR


class Lightning_CSS(pl.LightningModule):
    def __init__(self, separator_args, save_hparam=False, finetune=False, threshold=-10, lr=1e-3, **kwargs):
        super().__init__()
        
        if save_hparam:
            self.save_hyperparameters()

        self.lr = lr
        self.finetune = finetune
        self.threshold = threshold

        self.separation_net = instantiate(separator_args)
        
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(self.device)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=1)
        self.max_pool_1d = nn.MaxPool1d(27, 18)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        feature, image, mixture_BMTF, doa_B2 = batch
        input = (feature, doa_B2)

        mask_est_BTF = self.separation_net(input)

        sep = MVDR(mixture_BMTF, mask_est_BTF, 1 - mask_est_BTF)

        sep = self.istft(sep)
        si_sdr_list = calc_SI_SDR(sep, image)
        
        with torch.no_grad():
            effective_data_ratio = (si_sdr_list > self.threshold).to(torch.float32).mean()
            
        sep_loss = -1 * (si_sdr_list * (si_sdr_list > self.threshold)).mean()
        
        return sep_loss, effective_data_ratio

    def training_step(self, batch, batch_idx):
        sep_loss, effective_data_ratio = self.step(batch)

        self.log("train_sep_loss", sep_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_loss", sep_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_effective_data_ratio", effective_data_ratio, on_epoch=True, on_step=False, sync_dist=True)
        return sep_loss

    def validation_step(self, batch, batch_idx):
        sep_loss, effective_data_ratio = self.step(batch)

        self.log("valid_sep_loss", sep_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss", sep_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_effective_data_ratio", effective_data_ratio, on_epoch=True, on_step=False, sync_dist=True)
        return sep_loss

    def __str__(self):
        return f"{self.separation_net.__class__.__name__}_MVDR"
