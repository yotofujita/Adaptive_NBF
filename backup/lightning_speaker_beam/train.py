#! /usr/bin/env python3
# coding:utf-8

import os, sys, json
import soundfile as sf
import numpy as np
import librosa
import torch
from pytorch_lightning import LightningDataModule, Trainer

# from pytorch_lightning.metric.functional import accuracy
from torch import nn
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
import speechbrain as sb
from utils.utility import segmentation, overlap_add, calculate_SCM
from model_utility import nn_segmentation, nn_over_add, nn_padding
from pytorch_lightning.loggers import MLFlowLogger

class ClNoEnMa_Dataset(torch.utils.data.Dataset):
    def __init__(self, json_fname, segment_size=128, overlap_ratio=0.5):
        self.data = json.load(open(json_fname, "r"))
        self.key_list = list(self.data.keys())
        self.segment_size = segment_size
        self.overlap_ratio = overlap_ratio

    def __len__(self):
        return 30
        # return len(self.data)

    def __getitem__(self, idx):
        data = self.data[self.key_list[idx]]
        tgt_fname = data["target"]
        noise_fname_list = data["noise_list"]

        clean_sig = sf.read(tgt_fname)[0]
        noise_sig_list = []
        N = len(noise_fname_list)
        for noise_path in noise_fname_list:
            noise_sig_list.append(sf.read(noise_path)[0])

        clean_sig = (clean_sig / np.abs(clean_sig).max() * np.random.rand()).astype(np.float32)
        if len(clean_sig) < (self.segment_size - 1) * 256 + 1024:
            print("---Warning---", len(clean_sig), (self.segment_size - 1) * 256 + 1024)
            tmp = np.zeros((self.segment_size - 1) * 256 + 1024, dtype=clean_sig.dtype)
            tmp[: len(clean_sig)] = clean_sig
            clean_sig = tmp
        clean_spec = segmentation(
            np.abs(librosa.core.stft(clean_sig, n_fft=1024, hop_length=256)) ** 2,
            self.segment_size,
            self.overlap_ratio,
        )

        noise_sig_list = [x / np.abs(x).max() for x in noise_sig_list]
        noise = np.zeros_like(clean_sig)
        noise_enhanced = np.zeros_like(clean_sig)
        for n in range(N):
            len_diff = len(clean_sig) - len(noise_sig_list[n])
            if len_diff > 0:
                start_idx = int(np.random.rand() * len_diff)
                noise[start_idx : start_idx + len(noise_sig_list[n])] += noise_sig_list[n] * np.random.rand()
                noise_enhanced[start_idx : start_idx + len(noise_sig_list[n])] += noise_sig_list[n] * np.random.rand()
            else:
                noise += noise_sig_list[n][: len(clean_sig)] * np.random.rand()
                noise_enhanced += noise_sig_list[n][: len(clean_sig)] * np.random.rand()

        noisy_SNR = np.random.rand() * 20 - 15  # -15 ~ 5
        SNR = 10 * np.log10((np.abs(clean_sig) ** 2).mean() / (np.abs(noise) ** 2).mean())
        scale = np.sqrt(10 ** ((SNR - noisy_SNR) / 10))
        scaled_noise = (scale * noise).astype(np.float32)
        scaled_noise_spec = segmentation(
            np.abs(librosa.core.stft(scaled_noise, n_fft=1024, hop_length=256)) ** 2,
            self.segment_size,
            self.overlap_ratio,
        )
        IBM = (clean_spec > scaled_noise_spec).to(torch.float32)

        noisy_spec = segmentation(
            np.abs(librosa.core.stft(clean_sig + scaled_noise, n_fft=1024, hop_length=256)) ** 2,
            self.segment_size,
            self.overlap_ratio,
        )

        enhanced_SNR = noisy_SNR + (np.random.rand() * 10 + 5)  # noisy_SNR + (5~15)
        SNR = 10 * np.log10((np.abs(clean_sig) ** 2).mean() / (np.abs(noise_enhanced) ** 2).mean())
        scale = np.sqrt(10 ** ((SNR - enhanced_SNR) / 10))
        noisy_enhanced = (clean_sig + scale * noise_enhanced).astype(np.float32)
        noisy_enhanced = segmentation(
            np.abs(librosa.core.stft(noisy_enhanced, n_fft=1024, hop_length=256)) ** 2,
            self.segment_size,
            self.overlap_ratio,
        )
        return clean_spec, noisy_spec, noisy_enhanced, IBM


class CustomBatch:
    def __init__(self, data):
        clean, noisy, enhanced, IBM = list(zip(*data))
        self.spk_label = []
        for x in range(len(data)):
            self.spk_label += [x] * len(clean[x])
        self.spk_label = torch.tensor(self.spk_label)
        self.clean = torch.cat(clean, axis=0)
        self.noisy = torch.cat(noisy, axis=0)
        self.enhanced = torch.cat(enhanced, axis=0)
        self.IBM = torch.cat(IBM, axis=0)

    def pin_memory(self):
        self.clean = self.clean.pin_memory()
        self.noisy = self.noisy.pin_memory()
        self.enhanced = self.enhanced.pin_memory()
        self.IBM = self.IBM.pin_memory()
        self.spk_label = self.spk_label.pin_memory()


class LibrispeechDataModule(LightningDataModule):
    def __init__(
        self, segment_size, overlap_ratio, batch_size, json_dir="../data/", num_workers=2,
    ):
        super().__init__()
        self.json_dir = json_dir
        self.segment_size = segment_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        print("---prepare_data---")

    def setup(self, stage=None):
        print("---setup---")

    def train_dataloader(self):
        train_dataset = ClNoEnMa_Dataset(
            f"{self.json_dir}/train_librispeech.json", self.segment_size, self.overlap_ratio
        )
        return DataLoader(
            train_dataset, self.batch_size, collate_fn=CustomBatch, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        val_dataset = ClNoEnMa_Dataset(
            f"{self.json_dir}/valid_librispeech.json", self.segment_size, self.overlap_ratio
        )
        return DataLoader(
            val_dataset, self.batch_size, collate_fn=CustomBatch, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        test_dataset = ClNoEnMa_Dataset(
            f"{self.json_dir}/test_librispeech.json", self.segment_size, self.overlap_ratio
        )
        return DataLoader(
            test_dataset, self.batch_size, collate_fn=CustomBatch, num_workers=self.num_workers, pin_memory=True
        )


if __name__ == "__main__":
    from model import lightning_SB

    model = lightning_SB()

    dm = LibrispeechDataModule(128, 0.5, 4)

    mlf_logger = MLFlowLogger(experiment_name="default")#, run_name="defalut")
    trainer = Trainer(max_epochs=3, progress_bar_refresh_rate=20, gpus=1, logger=mlf_logger)

    trainer.fit(model, dm)
