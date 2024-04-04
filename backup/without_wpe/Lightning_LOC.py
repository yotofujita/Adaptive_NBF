#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bdb import effective
import torch
from torch import nn
import torchaudio
import numpy as np
import pytorch_lightning as pl

from utility import calc_SI_SDR, MVDR, MUSIC


class Lightning_LOC(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, model_name="lstm"):
        parser = parent_parser.add_argument_group("Lightning_LOC")
        parser.add_argument("--n_freq", type=int, default=513)
        parser.add_argument("--n_mic", type=int, default=5)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_mask_loss", type=float, default=1e-3)
        # parser.add_argument("--weight_speech_mask", type=float, default=3)
        parser.add_argument("--mask_overlap_penalty", type=float, default=1)
        if model_name == "lstm":
            from LSTM_LOC import LSTM_LOC

            parser = LSTM_LOC.add_model_specific_args(parser)
        elif model_name == "conformer":
            from Conformer_CSS import Conformer_CSS

            parser = Conformer_CSS.add_model_specific_args(parser)
        return parent_parser

    def __str__(self):
        return f"LOC-{self.model_name}-BF={self.use_BF}-SV={self.use_SV}-DAN={self.use_DAN}"

    def __init__(
        self,
        use_BF,
        use_SV,
        use_DAN,
        save_hparam=False,
        fine_tune=False,
        **kwargs,
    ):
        super().__init__()
        if save_hparam:
            self.save_hyperparameters()
        self.model_name = kwargs["model_name"].lower()
        self.lr = kwargs["lr"]
        self.n_freq = kwargs["n_freq"]
        self.n_mic = kwargs["n_mic"]
        self.weight_mask_loss = kwargs["weight_mask_loss"]
        # self.weight_speech_mask = kwargs["weight_speech_mask"]
        self.mask_overlap_penalty = kwargs["mask_overlap_penalty"]
        self.fine_tune = fine_tune
        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN

        if self.model_name == "conformer":
            raise NotImplementedError

        elif self.model_name == "lstm":
            from LSTM_LOC import LSTM_LOC

            self.separation_net = LSTM_LOC(
                use_BF=self.use_BF,
                use_SV=self.use_SV,
                use_DAN=self.use_DAN,
                n_mic=self.n_mic,
                n_freq=self.n_freq,
                input_lstm=kwargs["input_lstm"],
                hidden_lstm=kwargs["hidden_lstm"],
                n_layer=kwargs["n_layer"],
            )
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(self.device)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=1)
        self.max_pool_1d = nn.MaxPool1d(27, 18)

    def forward(self, input):
        return self.separation_net(input)

    def configure_optimizers(self):
        if self.fine_tune == "dan":
            return torch.optim.Adam(self.separation_net.dan.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        if self.use_DAN:
            feature, image, mixture_BMTF, SV_BFM, doa_B2 = batch
            input = (feature, doa_B2)
        else:
            feature, image, mixture_BMTF, SV_BFM = batch
            input = feature

        with torch.no_grad():
            image_spec_mean_BT = self.stft(image).mean(dim=1)
            vad_ref_B = (self.max_pool_1d(image_spec_mean_BT) > 0.1).to(torch.float32).sum(dim=1)
            vad_ref_B = (vad_ref_B > 1).to(torch.float32)

        # mask_speech_BT, mask_noise_BT = self.forward(input)
        # music_speech = MUSIC(mixture_BMTF, mask_speech_BT, SV_BFM)
        # music_noise = MUSIC(mixture_BMTF, mask_noise_BT, SV_BFM)

        mask_speech_BT = self.forward(input)
        music_speech = MUSIC(mixture_BMTF, mask_speech_BT, SV_BFM)
        # music_noise = MUSIC(mixture_BMTF, 1 - mask_speech_BT, SV_BFM)

        # あまりに確信度が高いところしか使わないとよくなさそうなので、maskはなるべく発話全体を使うようmaskの和を大きく
        # speech_maskもnoise_maskも１になるのは変なので、２つの積を引くことで、重複部分にペナルティ
        # 対象方向のvad結果がほとんど0の発話はあまり役に立たなそうなので2区間以上で発話が検出されたかで使うか決める
        # mask_loss = (
        #     -1
        #     * (
        #         (mask_speech_BT + mask_noise_BT * 0.1 - mask_speech_BT * mask_noise_BT * self.mask_overlap_penalty)
        #         * vad_ref_B[:, None]
        #     ).sum()
        # )
        # mask_loss = -1 * (mask_speech_BT * vad_ref_B[:, None]).sum()
        mask_loss = 0
        # loc_loss = ((music_speech - music_noise) * vad_ref_B).sum()
        loc_loss = ((music_speech) * vad_ref_B).sum()
        print("mask sum =", mask_speech_BT.sum(dim=1))
        with torch.no_grad():
            print("loss = ", mask_loss, loc_loss)
            if torch.isnan(mask_speech_BT).any():
                print(
                    "\n\n check nan : ",
                    torch.isnan(mask_speech_BT).any(),
                    torch.isnan(vad_ref_B).any(),
                    torch.isnan(feature).any(),
                    torch.isnan(image).any(),
                    torch.isnan(SV_BFM).any(),
                    torch.isnan(doa_B2).any(),
                )
                exit()
        return mask_loss, loc_loss

    def training_step(self, train_batch, batch_idx):
        mask_loss, loc_loss = self.step(train_batch)

        self.log("train_mask_loss", mask_loss, on_epoch=True, on_step=False)
        self.log("train_loc_loss", loc_loss, on_epoch=True, on_step=False)
        self.log("train_loss", mask_loss * self.weight_mask_loss + loc_loss, on_epoch=True, on_step=False)
        return mask_loss * self.weight_mask_loss + loc_loss

    def validation_step(self, valid_batch, batch_idx):
        mask_loss, loc_loss = self.step(valid_batch)

        self.log("valid_mask_loss", mask_loss, on_epoch=True, on_step=False)
        self.log("valid_loc_loss", loc_loss, on_epoch=True, on_step=False)
        self.log("valid_loss", mask_loss * self.weight_mask_loss + loc_loss, on_epoch=True, on_step=False)
        return mask_loss * self.weight_mask_loss + loc_loss

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
