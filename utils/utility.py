#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
import torchaudio
from opt_einsum import contract


def make_feature(mixture, SV_FM, azim_idx):
    """
    Parameters
    ----------
    mixture: torch.tensor (n_mic, n_time, n_freq) or (n_batch, n_mic, n_time, n_freq)
        Frequency domain multichannel spectrogam.
        If the dimension is three, n_batch is set to 1.
    SV_FM: torch.tensor (n_freq, n_mic)
        The steering vector
    azim_idx: float
        azimuth [deg] is calculated as azim_idx * 5

    Returns
    -------
    feature: torch.tensor (n_time, n_feature) or (n_batch,  n_time, n_feature)
    doa: torch.tensor (2) or (n_batch,  2)
    """

    def _make_feature(mixture_MTF, SV_FM, azim_idx):
        n_mic, n_time, n_freq = mixture_MTF.shape

        input_dim = 2 * n_mic
        feature = torch.zeros([n_time, n_freq, input_dim], device=mixture_MTF.device)

        # Log-magnitude spectrogram of mixture 
        feature[..., 0] = torch.log(torch.abs(mixture_MTF).mean(axis=0) + 1e-8)

        # Inter-channel phase differences
        IPD = mixture_MTF[1:] / (mixture_MTF[0, None] + 1e-8)  # M-1 x T x F
        IPD /= torch.abs(IPD) + 1e-8
        feature[..., 1 : 2 * n_mic - 1] = torch.view_as_real(IPD.permute(1, 2, 0)).reshape(n_time, n_freq, -1)  # phasal feature 

        # Log-magnitude spectrogram of DSBF
        feature[..., 2 * n_mic - 1] = torch.log(
            torch.abs(torch.einsum("fm, mtf -> tf", SV_FM.conj(), mixture_MTF)) + 1e-8)

        feature = feature.reshape(n_time, -1)

        azim_rad = azim_idx * 5 / 180 * np.pi
        doa = torch.tensor([torch.cos(azim_rad), torch.sin(azim_rad)], dtype=torch.float32)
        return feature, doa

    if mixture.ndim == 3:
        return _make_feature(mixture, SV_FM, azim_idx)

    elif mixture.ndim == 4:
        n_batch = mixture.shape[0]
        for b in range(n_batch):
            feature_tmp, doa_tmp = _make_feature(mixture[b], SV_FM[b], azim_idx[b])
            if b == 0:
                feature = torch.zeros(
                    [n_batch, *feature_tmp.shape], dtype=feature_tmp.dtype, device=mixture.device
                )
                doa = torch.zeros([n_batch, *doa_tmp.shape], dtype=doa_tmp.dtype, device=mixture.device)
            feature[b] = feature_tmp
            doa[b] = doa_tmp
        return feature, doa


def split_sig(sig_MT, window_len=48128, shift=8000, stft=False):
    """
    Parameters
    ----------
    sig_MT: torch.tensor (n_mic, n_time)
        Time domain multichannel signal.
        If the dimension is one, n_mic is set to 1.
    window_len: int
        The number of sample in each window (default 48128)
    shift: int
        The shift sizse (default 8000 = 0.5 sec)
    stft: bool
        If True, the output is STFT result, else time domain signals (default False)

    Returns
    -------
    If stft is False, returns a torch.tensor of (n_window, n_mic, window_len),
    else returns a torch.tensor of (n_window, n_mic, n_frame, n_freq).
    """
    if sig_MT.ndim == 1:
        sig_MT = sig_MT.unsqueeze(0)
    n_mic, n_sample = sig_MT.shape
    n_window = (n_sample - window_len) // shift + 1

    if n_window == 0:
        raise ValueError("data is too short : len = ", sig_MT.shape)

    sig_split_WMT = torch.as_strided(sig_MT, size=(n_mic, n_window, window_len), stride=(n_sample, shift, 1)).permute(
        1, 0, 2
    )

    if not stft:
        return sig_split_WMT
    else:
        return torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)(sig_split_WMT).permute(
            0, 1, 3, 2
        )


def overlap_add(spec_BFT, shift=8000):
    sig_BT = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(spec_BFT.device)(spec_BFT)
    n_window, n_time = sig_BT.shape

    sig_T = torch.zeros(int((n_window - 1) * shift + n_time)).to(spec_BFT.device)
    for i in range(n_window):
        sig_T[i * shift : i * shift + n_time] += sig_BT[i]
        if i > 0:
            sig_T[i * shift : i * shift + (n_time - shift)] /= 2
    return sig_T


def shift_concat(spec_BFT, shift=8000, use_first_all=True):
    sig_BT = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).to(spec_BFT.device)(spec_BFT)
    n_window, n_time = sig_BT.shape

    if use_first_all:
        sig_T = torch.zeros(int((n_window - 1) * shift + n_time)).to(spec_BFT.device)
        sig_T[:n_time] = sig_BT[0]
        for i in range(1, n_window):
            sig_T[n_time + (i - 1) * shift : n_time + i * shift] = sig_BT[i, -shift:]
    else:
        sig_T = torch.zeros(int(n_window * shift)).to(spec_BFT.device)
        for i in range(n_window):
            sig_T[i * shift : (i + 1) * shift] = sig_BT[i, -shift:]
    return sig_T


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


def MVDR(mixture_BMTF, mask_speech_BTF, mask_noise_BTF, ref_channel=0, eps=1e-7, return_filter=False):
    if mixture_BMTF.ndim == 3:
        mixture_BMTF = mixture_BMTF[None]
        mask_speech_BTF = mask_speech_BTF[None]
        mask_noise_BTF = mask_noise_BTF[None]
    n_mic = mixture_BMTF.shape[1]
    speech_est_BMTF = mixture_BMTF * mask_speech_BTF.unsqueeze(1)
    speech_SCM_BFMM = (
        torch.einsum("bitf, bjtf -> bfij", speech_est_BMTF, speech_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    noise_est_BMTF = mixture_BMTF * mask_noise_BTF.unsqueeze(1)
    noise_SCMinv_BFMM = torch.linalg.inv(
        torch.einsum("bitf, bjtf -> bfij", noise_est_BMTF, noise_est_BMTF.conj())
        + eps * torch.eye(n_mic, device=mixture_BMTF.device)[None, None]
    )

    W_BFM = torch.einsum("bfij, bfj -> bfi", noise_SCMinv_BFMM, speech_SCM_BFMM[..., ref_channel]) / torch.sum(
        torch.diagonal(noise_SCMinv_BFMM @ speech_SCM_BFMM, dim1=-2, dim2=-1), dim=-1
    ).unsqueeze(-1)

    if return_filter:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj()), W_BFM
    else:
        return torch.einsum("bitf, bfi -> bft", mixture_BMTF, W_BFM.conj())