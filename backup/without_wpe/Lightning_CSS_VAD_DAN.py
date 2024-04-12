#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchaudio
import numpy as np
import pytorch_lightning as pl

from utils.utility import calc_SI_SDR, MVDR


class Lightning_CSS_VAD_DAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, model_name="lstm"):
        parser = parent_parser.add_argument_group("Lightning_CSS_VAD_DAN")
        parser.add_argument("--input_dim", type=int, default=5130)
        parser.add_argument("--n_freq", type=int, default=513)
        parser.add_argument("--lr", type=float, default=1e-3)
        if model_name == "lstm":    
            from LSTM_CSS_VAD_DAN import LSTM_CSS_VAD_DAN

            parser = LSTM_CSS_VAD_DAN.add_model_specific_args(parser)
        elif model_name == "conformer":
            from Conformer_CSS import Conformer_CSS

            parser = Conformer_CSS.add_model_specific_args(parser)
        return parent_parser

    def __init__(self, model_name="lstm", input_dim=9234, n_freq=513, lr=1e-3, save_hparam=False, fine_tune=False, **kwargs):
        super().__init__()
        if save_hparam:
            self.save_hyperparameters()
        self.model_name = model_name.lower()
        self.lr = lr
        self.n_freq = n_freq
        self.fine_tune = fine_tune

        if self.model_name == "conformer":
            raise NotImplementedError

        elif self.model_name == "lstm":
            from LSTM_CSS_VAD_DAN import LSTM_CSS_VAD_DAN

            self.model = LSTM_CSS_VAD_DAN(
                input_dim=input_dim,
                n_freq=n_freq,
                input_lstm=kwargs["input_lstm"],
                hidden_lstm=kwargs["hidden_lstm"],
                n_layer=kwargs["n_layer"],
            )
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(self.device)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=1)
        self.max_pool_1d = nn.MaxPool1d(27, 18)

    def forward(self, x, doa):
        return self.model(x, doa)

    def configure_optimizers(self):
        if self.fine_tune:
            return torch.optim.Adam(self.model.dan.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        feature, image, mixture_BMTF, doa_B2 = batch
        mask_est_BTF, vad_BT = self.forward(feature, doa_B2)

        with torch.no_grad():
            image_spec_mean_BT = self.stft(image).mean(dim=1)
            vad = (self.max_pool_1d(image_spec_mean_BT) > 0.1).to(torch.float32)

        sep = MVDR(mixture_BMTF, mask_est_BTF, 1 - mask_est_BTF)

        if torch.isnan(vad).any():
            print("vad is nan ", vad_BT ,vad)
            print("mask_est_BTF : ", mask_est_BTF.sum())

        if torch.isnan(sep).any():
            print("sep is nan ", sep)
            print("mask_est_BTF : ", mask_est_BTF.sum())
            print("vad_BT : ", vad_BT.sum())
            print("feature : ", feature.sum())
            print("image : ", image.sum())
            print("mixture_BMTF : ", mixture_BMTF.sum())
            print("doa_B2 : ", doa_B2.sum())

        if torch.isinf(sep).any():
            print("sep is inf ", sep)
            print("mask_est_BTF : ", mask_est_BTF.sum())


        if self.current_epoch > 500:
            sep = sep * self.model.expand_vad(vad_BT)[:, None]
            with torch.no_grad():
                vad_loss = nn.functional.binary_cross_entropy(vad_BT, vad)
        else:
            vad_loss = nn.functional.binary_cross_entropy(vad_BT, vad)

        sep = self.istft(sep)
        si_sdr_list = calc_SI_SDR(sep, image)
        sep_loss = -1 * (si_sdr_list * (si_sdr_list > -10)).mean()

        with torch.no_grad():
            if (self.current_epoch % 20 == 1) and self.flag == False:
                print("logging sep and mask")
                self.flag = True
                tensorboard = self.logger.experiment
                for i in range(3):
                    tensorboard.add_audio(f"sep-{i+1}", sep[i], sample_rate=16000, global_step=self.current_epoch)
                    tensorboard.add_audio(f"image-{i+1}", image[i], sample_rate=16000, global_step=self.current_epoch)
                    tensorboard.add_image(
                        f"mask-{i+1}", mask_est_BTF[i].T, dataformats="HW", global_step=self.current_epoch
                    )
            elif (self.current_epoch % 20 == 0):
                self.flag = False

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

