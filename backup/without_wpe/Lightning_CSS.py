#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchaudio
import pytorch_lightning as pl

from utils.utility import calc_SI_SDR, MVDR, GEV


class Lightning_CSS(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, model_name="lstm"):
        parser = parent_parser.add_argument_group("Lightning_CSS")
        parser.add_argument("--n_freq", type=int, default=513)
        parser.add_argument("--n_mic", type=int, default=5)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--threshold", type=int, default=-10, help="SI-SDR threshold for calculating loss")
        parser.add_argument("--BF_name", type=str, default="MVDR", help="Beamformer name (MVDR or GEV)")
        # parser.add_argument("--use_BF", action="store_true")
        # parser.add_argument("--use_SV", action="store_true")
        # parser.add_argument("--use_DAN", action="store_true")
        # parser.add_argument("--use_VAD", action="store_true")
        if model_name == "lstm":
            from modules.LSTM_CSS import LSTM_CSS

            parser = LSTM_CSS.add_model_specific_args(parser)
        elif model_name == "conformer":
            from Conformer_CSS import Conformer_CSS

            parser = Conformer_CSS.add_model_specific_args(parser)
        return parent_parser

    def __str__(self):
        if self.BF_name.lower() == "mvdr":
            return f"{self.model_name}-BF={self.use_BF}-SV={self.use_SV}-DAN={self.use_DAN}-VAD={self.use_VAD}"
        else:
            return f"{self.model_name}_{self.BF_name}-BF={self.use_BF}-SV={self.use_SV}-DAN={self.use_DAN}-VAD={self.use_VAD}"

    def __init__(
        self,
        use_BF,
        use_SV,
        use_DAN,
        use_VAD,
        save_hparam=False,
        finetune=False,
        threshold=-10,
        **kwargs,
    ):
        super().__init__()
        if save_hparam:
            self.save_hyperparameters()
        self.model_name = kwargs["model_name"].lower() if "model_name" in kwargs else "lstm"
        self.BF_name = kwargs["BF_name"] if "BF_name" in kwargs else "MVDR"
        self.lr = kwargs["lr"] if "lr" in kwargs else 1e-3
        self.n_freq = kwargs["n_freq"] if "n_freq" in kwargs else 513
        self.n_mic = kwargs["n_mic"] if "n_mic" in kwargs else 5
        self.finetune = finetune
        self.use_BF = use_BF
        self.use_SV = use_SV
        self.use_DAN = use_DAN
        self.use_VAD = use_VAD
        self.threshold = threshold

        print("\n threshold = ", self.threshold, self.use_BF, self.use_SV, self.use_DAN, self.use_VAD)

        if self.model_name == "conformer":
            raise NotImplementedError

        elif self.model_name == "lstm":
            from modules.LSTM_CSS import LSTM_CSS

            self.separation_net = LSTM_CSS(
                use_BF=self.use_BF,
                use_DAN=self.use_DAN,
                use_SV=self.use_SV,
                use_VAD=self.use_VAD,
                n_mic=self.n_mic,
                n_freq=self.n_freq,
                input_lstm=kwargs["input_lstm"] if "input_lstm" in kwargs else 1024,
                hidden_lstm=kwargs["hidden_lstm"] if "hidden_lstm" in kwargs else 512,
                n_layer=kwargs["n_layer"] if "n_layer" in kwargs else 3,
            )
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(self.device)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=1)
        self.max_pool_1d = nn.MaxPool1d(27, 18)

    def on_fit_start(self):
        pl.seed_everything(0)

    def forward(self, input):
        return self.separation_net(input)

    def configure_optimizers(self):
        if self.finetune == "dan" and self.use_DAN:
            print("\n---  Start Fine-tuning  ---\n")
            return torch.optim.Adam(self.separation_net.dan.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch):
        if self.use_DAN:
            feature, image, mixture_BMTF, doa_B2 = batch
            input = (feature, doa_B2)
        else:
            feature, image, mixture_BMTF = batch
            input = feature

        if self.use_VAD:
            mask_est_BTF, vad_est_BT = self.forward(input)

            with torch.no_grad():
                image_spec_mean_BT = self.stft(image).mean(dim=1)
                vad_ref = (self.max_pool_1d(image_spec_mean_BT) > 0.1).to(torch.float32)
        else:
            mask_est_BTF = self.forward(input)

        if self.BF_name.lower() == "mvdr":
            sep = MVDR(mixture_BMTF, mask_est_BTF, 1 - mask_est_BTF)
            # bf = torchaudio.transforms.MVDR(solution="ref_channel")
            # sep = bf(
            #     specgram=mixture_BMTF.permute(0, 1, 3, 2).to(torch.cdouble),
            #     mask_s=mask_est_BTF.permute(0, 2, 1),
            #     mask_n=1 - mask_est_BTF.permute(0, 2, 1),
            # )
        elif self.BF_name.lower() == "gev":
            if self.current_epoch < 50:
                sep = MVDR(mixture_BMTF, mask_est_BTF, 1 - mask_est_BTF)
            else:
                sep = GEV(mixture_BMTF, mask_est_BTF, 1 - mask_est_BTF)
        else:
            raise ValueError("BF_name should be MVDR or GEV")

        if self.use_VAD:
            if self.current_epoch > 300:
                sep = sep * self.separation_net.expand_vad(vad_est_BT)[:, None]
                with torch.no_grad():
                    vad_loss = nn.functional.binary_cross_entropy(vad_est_BT, vad_ref)
            else:
                vad_loss = nn.functional.binary_cross_entropy(vad_est_BT, vad_ref)
        else:
            vad_loss = 0.0

        sep = self.istft(sep)
        si_sdr_list = calc_SI_SDR(sep, image)
        with torch.no_grad():
            effective_data_ratio = (si_sdr_list > self.threshold).to(torch.float32).mean()
        sep_loss = -1 * (si_sdr_list * (si_sdr_list > self.threshold)).mean()
        return vad_loss, sep_loss, effective_data_ratio

    def finetune_step(self, batch):
        if self.use_DAN:
            feature, image, mixture_BMTF, doa_B2 = batch
            input = (feature, doa_B2)
        else:
            feature, image, mixture_BMTF = batch
            input = feature

        if self.use_VAD:
            mask_est_BTF, vad_est_BT = self.forward(input)
        else:
            mask_est_BTF = self.forward(input)

        sep = MVDR(mixture_BMTF, mask_est_BTF, 1 - mask_est_BTF)
        sep = self.istft(sep)
        si_sdr_list = calc_SI_SDR(sep, image)
        with torch.no_grad():
            effective_data_ratio = (si_sdr_list > self.threshold).to(torch.float32).mean()
        sep_loss = -1 * (si_sdr_list * (si_sdr_list > self.threshold)).mean()
        vad_loss = 0.0
        return vad_loss, sep_loss, effective_data_ratio

    def training_step(self, batch, batch_idx):
        if self.finetune is False:
            vad_loss, sep_loss, effective_data_ratio = self.step(batch)
        else:
            vad_loss, sep_loss, effective_data_ratio = self.finetune_step(batch)

        if self.use_VAD:
            self.log("train_vad_loss", vad_loss, on_epoch=True, on_step=False)
        self.log("train_sep_loss", sep_loss, on_epoch=True, on_step=False)
        self.log("train_loss", vad_loss + sep_loss, on_epoch=True, on_step=False)
        self.log("train_effective_data_ratio", effective_data_ratio, on_epoch=True, on_step=False)
        return vad_loss + sep_loss

        # return {
        #     "vad_loss": vad_loss,
        #     "sep_loss": sep_loss,
        #     "effective_data_ratio": effective_data_ratio,
        #     "loss": vad_loss + sep_loss,
        # }

    def validation_step(self, batch, batch_idx):
        if self.finetune is False:
            vad_loss, sep_loss, effective_data_ratio = self.step(batch)
        else:
            vad_loss, sep_loss, effective_data_ratio = self.finetune_step(batch)

        if self.use_VAD:
            self.log("valid_vad_loss", vad_loss, on_epoch=True, on_step=False)
        self.log("valid_sep_loss", sep_loss, on_epoch=True, on_step=False)
        self.log("valid_loss", vad_loss + sep_loss, on_epoch=True, on_step=False)
        self.log("valid_effective_data_ratio", effective_data_ratio, on_epoch=True, on_step=False)
        return vad_loss + sep_loss

    # def training_epoch_end(self, training_step_outputs):
    #     loss = torch.as_tensor([data["loss"] for data in training_step_outputs]).mean()
    #     vad_loss = torch.as_tensor([data["vad_loss"] for data in training_step_outputs]).mean()
    #     sep_loss = torch.as_tensor([data["sep_loss"] for data in training_step_outputs]).mean()
    #     effective_data_ratio = torch.as_tensor([data["effective_data_ratio"] for data in training_step_outputs]).mean()

    #     tensorboard = self.logger.experiment
    #     if self.use_VAD:
    #         tensorboard.add_scalar("train_vad_loss", vad_loss, global_step=self.current_epoch)
    #     tensorboard.add_scalar("train_sep_loss", sep_loss, global_step=self.current_epoch)
    #     tensorboard.add_scalar("train_loss", loss, global_step=self.current_epoch)
    #     tensorboard.add_scalar("train_effective_data_ratio", effective_data_ratio, global_step=self.current_epoch)

    # def validation_epoch_end(self, validation_step_outputs):
    #     loss = torch.as_tensor([data["loss"] for data in validation_step_outputs]).mean()
    #     vad_loss = torch.as_tensor([data["vad_loss"] for data in validation_step_outputs]).mean()
    #     sep_loss = torch.as_tensor([data["sep_loss"] for data in validation_step_outputs]).mean()
    #     effective_data_ratio = torch.as_tensor([data["effective_data_ratio"] for data in validation_step_outputs]).mean()

    #     tensorboard = self.logger.experiment
    #     if self.use_VAD:
    #         tensorboard.add_scalar("valid_vad_loss", vad_loss, global_step=self.current_epoch)
    #     tensorboard.add_scalar("valid_sep_loss", sep_loss, global_step=self.current_epoch)
    #     tensorboard.add_scalar("valid_loss", loss, global_step=self.current_epoch)
    #     tensorboard.add_scalar("valid_effective_data_ratio", effective_data_ratio, global_step=self.current_epoch)

    # with torch.no_grad():
    #     if (self.current_epoch % 20 == 1) and self.flag == False:
    #         print("logging sep and mask")
    #         self.flag = True
    #         tensorboard = self.logger.experiment
    #         for i in range(3):
    #             tensorboard.add_audio(f"sep-{i+1}", sep[i], sample_rate=16000, global_step=self.current_epoch)
    #             tensorboard.add_audio(f"image-{i+1}", image[i], sample_rate=16000, global_step=self.current_epoch)
    #             tensorboard.add_image(
    #                 f"mask-{i+1}", mask_est_BTF[i].T, dataformats="HW", global_step=self.current_epoch
    #             )
    #     elif self.current_epoch % 20 == 0:
    #         self.flag = False

    # def separate(self, idx):
    #     if not hasattr(self, "json_data"):
    #         import json
    #         import h5py

    #         self.n_spk = 2
    #         json_fname = f"../data/valid_wsj0_chime3_{self.n_spk}spk.json"
    #         self.json_data = json.load(open(json_fname, "r"))
    #         self.id_list = list(self.json_data.keys())
    #         self.root = "/n/work3/sekiguch/dataset/Hololens2_SimData_WSJ0_CHiME3/"

    #         f = h5py.File("../data/SV_for_HL2.h5", "r")
    #         self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
    #         norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
    #         self.SV_EAFM /= norm_EAF[..., None]

    #     with torch.no_grad():
    #         data_id = self.id_list[idx]
    #         mixture_fname = f"{self.root}/{self.n_spk}spk/valid/mixture_{data_id}.wav"

    #         from make_feature import make_feature

    #         feature, mixture_BMTF = make_feature(
    #             self.json_data[data_id], mixture_fname=mixture_fname, SV_EAFM=self.SV_EAFM, spk_id=1
    #         )
    #         print("mixture_BMTF : ", mixture_BMTF.shape)
    #         exit()

    #         mask_est_BTF = self.forward(feature)
    #         mvdr = torchaudio.transforms.MVDR(solution="ref_channel")
    #         sep = mvdr(
    #             specgram=mixture_BMTF.permute(0, 1, 3, 2).to(torch.cdouble),
    #             mask_s=mask_est_BTF.permute(0, 2, 1),
    #             mask_n=1 - mask_est_BTF.permute(0, 2, 1),
    #         )
    #         sep = self.istft(sep)
    #         return sep
