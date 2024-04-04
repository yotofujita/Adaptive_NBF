#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torchaudio
import os
import torch
from glob import glob
from tqdm import tqdm
import soundfile as sf

from minibatch_wpe import Minibatch_WPE
import cupy as cp


if __name__ == "__main__":
    wpe = Minibatch_WPE(n_tap=5, n_delay=3, xp=cp)
    tuning_data_dir = (
        "/n/work3/sekiguch/data_for_paper/IROS2022/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset/"
    )
    save_data_dir = (
        "/n/work3/sekiguch/data_for_paper/IROS2022/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset_wpe/"
    )

    sep_fname_list = sorted(glob(tuning_data_dir + "/sep*.wav"))
    for idx, sep_fname in tqdm(enumerate(sep_fname_list)):
        mix_fname = sep_fname.replace("sep", "mix").split("_azim")[0] + ".wav"
        save_fname = mix_fname.replace(tuning_data_dir, save_data_dir)
        if os.path.isfile(save_fname):
            continue

        mix, fs_sep = torchaudio.load(mix_fname)
        mix_MFT = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)(mix)
        dereverb_MFT = torch.as_tensor(wpe.step(cp.asarray(mix_MFT.permute(1, 2, 0)))).permute(2, 0, 1)
        sig_MT = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)(dereverb_MFT)
        sf.write(save_fname, sig_MT.T, 16000)
