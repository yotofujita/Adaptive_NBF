#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging 
import h5py
import numpy as np

import torch
from torch import nn
import torchaudio
import pytorch_lightning as pl

from hydra.utils import instantiate

from utils.utility import make_feature, calc_SI_SDR, MVDR, WPD


class BaseModule(pl.LightningModule):
    def on_after_backward(self):
        super().on_after_backward()

        if hasattr(self, 'skip_nan_grad') and self.skip_nan_grad:
            device = next(self.parameters()).device
            valid_gradients = torch.tensor([1], device=device, dtype=torch.float32)

            for param_name, param in self.named_parameters():
                if param.grad is not None:
                    is_not_nan_or_inf = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    if not is_not_nan_or_inf:
                        valid_gradients = valid_gradients * 0
                        break

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(valid_gradients, op=torch.distributed.ReduceOp.MIN)

            if valid_gradients < 1:
                logging.warning(f'detected inf or nan values in gradients! Setting gradients to zero.')
                self.zero_grad()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def __str__(self):
        return f"{self.__class__.__name__}"


class DirectionAwareMVDR(BaseModule):
    def __init__(self, separator, save_hparam=False, threshold=-10, lr=1e-3, skip_nan_grad=False, **kwargs):
        super().__init__()
        
        if save_hparam:
            self.save_hyperparameters()

        self.lr = lr
        self.skip_nan_grad = skip_nan_grad
        
        self.threshold = threshold

        self.separation_net = instantiate(separator)
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)

        with h5py.File("./data/SV_for_HL2.h5", "r") as f: 
            # SV: [E, A, F, M], azim: [A,], elev: [E,]
            SV = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
            norm_EAF = torch.linalg.norm(SV, axis=3)
            SV /= norm_EAF[..., None]
            self.SV = SV

    def step(self, batch):
        mixture, image, elev_idx, azim_idx = batch  # [B, M, S], [B, S,], [B,], [B,]
        B, *_ = mixture.shape

        stft = self.stft.to(self.device)
        istft = self.istft.to(self.device)
        sv = self.SV.to(self.device)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # STFT
            mixture = stft(mixture).permute(0, 1, 3, 2)  # [B, M, T, F]

            # Feature extraction
            sv = sv.flatten(0, 1)
            sv_indices = (elev_idx + 1) * azim_idx
            _, F, M  = sv.shape
            sv = torch.gather(sv, 0, sv_indices.long()[:, None, None].expand(B, F, M))
            feature, doa = make_feature(mixture.to(torch.cfloat), sv, azim_idx)
            input = (feature, doa)

            # Mask estimation 
            mask_est_BTF = self.separation_net(input)

            # MVDR
            sep = MVDR(mixture.to(torch.cdouble), mask_est_BTF, 1 - mask_est_BTF)

            # Inverse STFT
            sep = istft(sep)
        
            # Calculate SI_SDR
            si_sdr_list = calc_SI_SDR(sep, image)
        
            with torch.no_grad():
                effective_data_ratio = (si_sdr_list > self.threshold).to(torch.float32).mean()
            
            loss = -1 * (si_sdr_list * (si_sdr_list > self.threshold)).mean()
            # loss = - si_sdr_list.mean()
        
        return loss, effective_data_ratio

    def training_step(self, batch, batch_idx):
        loss, effective_data_ratio = self.step(batch)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_effective_data_ratio", effective_data_ratio, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, effective_data_ratio = self.step(batch)

        self.log("valid_loss", loss, sync_dist=True)
        self.log("valid_effective_data_ratio", effective_data_ratio, sync_dist=True)
        return loss


class DirectionAwareWPD(BaseModule):
    def __init__(self, separator1, separator2, delay, tap, save_hparam=False, threshold=-10, lr=1e-3, skip_nan_grad=False, **kwargs):
        super().__init__()
        
        if save_hparam:
            self.save_hyperparameters()

        self.lr = lr
        self.skip_nan_grad = skip_nan_grad
        
        self.threshold = threshold

        self.separation_net1 = instantiate(separator1)
        self.separation_net2 = instantiate(separator2)
        
        self.delay = delay
        self.tap = tap
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)

        with h5py.File("./data/SV_for_HL2.h5", "r") as f: 
            # SV: [E, A, F, M], azim: [A,], elev: [E,]
            SV = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
            norm_EAF = torch.linalg.norm(SV, axis=3)
            SV /= norm_EAF[..., None]
            self.SV = SV

    def step(self, batch):
        mixture, image, elev_idx, azim_idx = batch  # [B, M, S], [B, S,], [B,], [B,]
        B, *_ = mixture.shape

        stft = self.stft.to(self.device)
        istft = self.istft.to(self.device)
        sv = self.SV.to(self.device)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # STFT
            mixture = stft(mixture).permute(0, 1, 3, 2)  # [B, M, T, F]

            # Feature extraction
            sv = sv.flatten(0, 1)
            sv_indices = (elev_idx + 1) * azim_idx
            _, F, M  = sv.shape
            sv = torch.gather(sv, 0, sv_indices.long()[:, None, None].expand(B, F, M))
            feature, doa = make_feature(mixture.to(torch.cfloat), sv, azim_idx)
            input = (feature, doa)

            # Mask estimation 
            mask1 = self.separation_net1(input)
            mask2 = self.separation_net2(input)

            # MVDR
            sep = WPD(mixture.to(torch.cdouble), mask1, mask2, delay=self.delay, tap=self.tap)

            # Inverse STFT
            sep = istft(sep)
        
            # Calculate SI_SDR
            si_sdr_list = calc_SI_SDR(sep, image)
            
            loss = -1 * (si_sdr_list * (si_sdr_list > self.threshold)).mean()
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log("valid_loss", loss, sync_dist=True)
        return loss