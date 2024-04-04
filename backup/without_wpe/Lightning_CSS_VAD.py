#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import mixture
import torch
from torch import nn
import torchaudio
import numpy as np

from torch.nn import functional as F
import pytorch_lightning as pl


def calc_SI_SDR(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    sisdr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return sisdr


class Lightning_CSS_VAD(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, model_name="lstm"):
        parser = parent_parser.add_argument_group("Lightning_CSS_VAD")
        parser.add_argument("--input_dim", type=int, default=9234)
        parser.add_argument("--n_freq", type=int, default=513)
        parser.add_argument("--lr", type=float, default=1e-3)
        if model_name == "lstm":
            from LSTM_CSS_VAD import LSTM_CSS_VAD

            parser = LSTM_CSS_VAD.add_model_specific_args(parser)
        elif model_name == "conformer":
            from Conformer_CSS import Conformer_CSS

            parser = Conformer_CSS.add_model_specific_args(parser)
        return parent_parser

    def __init__(self, model_name="lstm", input_dim=9234, n_freq=513, lr=1e-3, save_hparam=False, **kwargs):
        super().__init__()
        if save_hparam:
            self.save_hyperparameters()
        self.model_name = model_name.lower()
        self.lr = lr
        self.n_freq = n_freq

        if self.model_name == "conformer":
            raise NotImplementedError

        elif self.model_name == "lstm":
            from LSTM_CSS_VAD import LSTM_CSS_VAD

            self.model = LSTM_CSS_VAD(
                input_dim=input_dim,
                n_freq=n_freq,
                input_lstm=kwargs["input_lstm"],
                hidden_lstm=kwargs["hidden_lstm"],
                n_layer=kwargs["n_layer"],
            )
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(self.device)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=1)
        self.max_pool_1d = nn.MaxPool1d(27, 18)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        feature, image, mixture_BMTF = batch
        mask_est_BTF, vad_BT = self.forward(feature)

        with torch.no_grad():
            image_spec_mean_BT = self.stft(image).mean(dim=1)
            vad = (self.max_pool_1d(image_spec_mean_BT) > 0.1).to(torch.float32)

        mvdr = torchaudio.transforms.MVDR(solution="ref_channel")
        sep = mvdr(
            specgram=mixture_BMTF.permute(0, 1, 3, 2).to(torch.cdouble),
            mask_s=mask_est_BTF.permute(0, 2, 1),
            mask_n=1 - mask_est_BTF.permute(0, 2, 1),
        )

        if self.current_epoch > 200:
            sep = sep * self.model.expand_vad(vad_BT)[:, None]
            with torch.no_grad():
                vad_loss = nn.functional.binary_cross_entropy(vad_BT, vad)
        else:
            vad_loss = nn.functional.binary_cross_entropy(vad_BT, vad)

        sep = self.istft(sep)
        si_sdr_list = calc_SI_SDR(sep, image)
        sep_loss = -1 * (si_sdr_list * (si_sdr_list > -10)).mean()

        return vad_loss, sep_loss

    def training_step(self, train_batch, batch_idx):
        vad_loss, sep_loss = self.step(train_batch)
        self.log("train_vad_loss", vad_loss, on_epoch=True, on_step=False)
        self.log("train_sep_loss", sep_loss, on_epoch=True, on_step=False)
        self.log("train_loss", vad_loss + sep_loss, on_epoch=True, on_step=False)
        return vad_loss + sep_loss

    def validation_step(self, valid_batch, batch_idx):
        vad_loss, sep_loss = self.step(valid_batch)
        self.log("valid_vad_loss", vad_loss, on_epoch=True, on_step=False)
        self.log("valid_sep_loss", sep_loss, on_epoch=True, on_step=False)
        self.log("valid_loss", vad_loss + sep_loss, on_epoch=True, on_step=False)

        # for i, si_sdr in enumerate(si_sdr_list):
        #     if si_sdr < -50:
        #         tensorboard = self.logger.experiment
        #         tensorboard.add_audio(f"sep - {si_sdr}", sep[i], sample_rate=16000)
        #         tensorboard.add_audio(f"image - {si_sdr}", image[i], sample_rate=16000)
        return vad_loss + sep_loss

    def separate(self, idx):
        if not hasattr(self, "json_data"):
            import json
            import h5py

            self.n_spk = 2
            json_fname = f"../data/valid_wsj0_chime3_{self.n_spk}spk.json"
            self.json_data = json.load(open(json_fname, "r"))
            self.id_list = list(self.json_data.keys())
            self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_WSJ0_CHiME3/"

            f = h5py.File("../data/SV_for_HL2.h5", "r")
            self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
            norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
            self.SV_EAFM /= norm_EAF[..., None]

        with torch.no_grad():
            data_id = self.id_list[idx]
            mixture_fname = f"{self.root}/{self.n_spk}spk/valid/mixture_{data_id}.wav"

            from make_feature import make_feature

            feature, mixture_BMTF = make_feature(
                self.json_data[data_id], mixture_fname=mixture_fname, SV_EAFM=self.SV_EAFM, spk_id=1
            )
            print("mixture_BMTF : ", mixture_BMTF.shape)
            exit()

            mask_est_BTF = self.forward(feature)
            mvdr = torchaudio.transforms.MVDR(solution="ref_channel")
            sep = mvdr(
                specgram=mixture_BMTF.permute(0, 1, 3, 2).to(torch.cdouble),
                mask_s=mask_est_BTF.permute(0, 2, 1),
                mask_n=1 - mask_est_BTF.permute(0, 2, 1),
            )
            sep = self.istft(sep)
            return sep
