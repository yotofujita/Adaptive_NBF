#! /usr/bin/env python3
# coding:utf-8

import numpy as np
import torch
import librosa
import sys, os

def MultiSTFT(wav_TM, n_fft=1024, hop_length=None):
    """
    Multichannel STFT
    
    Arguments
    ---------
    wav_TM: np.ndarray (T x M)
    n_fft: int
    hop_length: int
    """
    if wav_TM.ndim == 1:
        wav_TM = wav_TM[:, None]

    T, M = wav_TM.shape
    if hop_length is None:
        hop_length = n_fft // 4

    for m in range(M):
        tmp = librosa.core.stft(wav_TM[:, m].copy(), n_fft, hop_length)
        if m == 0:
            spec_FTM = np.zeros([*tmp.shape, M], dtype=tmp.dtype)
        spec_FTM[:, :, m] = tmp
    return torch.as_tensor(spec_FTM.squeeze(), dtype=torch.complex64)


def MultiISTFT(spec_FTM, hop_length=None):
    if type(spec_FTM) == torch.Tensor:
        spec_FTM = spec_FTM.cpu().numpy()
    
    if spec_FTM.ndim == 2:
        spec_FTM = spec_FTM[:, :, None]
    
    if hop_length is None:
        hop_length = int((len(spec_FTM) - 1) // 2)

    F, T, M = spec_FTM.shape
    for m in range(M):
        wav_T = librosa.core.istft(spec_FTM[:, :, m], hop_length=hop_length)
        if m == 0:
            wav_TM = np.zeros([len(wav_T), M], dtype=wav_T.dtype)
        wav_TM[:, m] = wav_T
    return wav_TM.squeeze()


def segmentation(spec_FT, segment_size, overlap_ratio=0.5):
    """ 
    Convert F x T to S x F x T' (T = T' * segment_size)
    """
    spec_FT = torch.as_tensor(spec_FT)
    F, T = spec_FT.shape
    overlap = int(segment_size * overlap_ratio)
    shift = segment_size - overlap
    n_segment = (T - segment_size) // shift + 1
    spec_SFT = torch.as_strided(
        spec_FT, 
        size=(F, n_segment, segment_size), 
        stride=(T, shift, 1)
    ).permute(1, 0, 2)
    return spec_SFT


def overlap_add(spec_SFT, overlap_ratio=0.5, mode="mean"):
    """
    Convert S x F x T' to F x T  (T= T' * S)

    Arguments
    ---------
    mode: str ('mean', 'front', 'back')
        'mean': overlap segments are averaged
        'front': the value of the first segment is used for overlap segment
        'back': the value of the last segment is used for overlap segment
    """
    n_segment, F, segment_size = spec_SFT.shape
    overlap = int(segment_size * overlap_ratio)
    shift = segment_size - overlap
    T = (n_segment - 1) * shift + segment_size
    spec_FT = torch.zeros([F, T], dtype=spec_SFT.dtype, device=spec_SFT.device)

    if mode == "mean":
        weight_T = torch.zeros(T, dtype=spec_SFT.dtype, device=spec_SFT.device)
    for s in range(n_segment):
        if mode == "mean":
            spec_FT[:, s*shift:s*shift+segment_size] += spec_SFT[s]
            weight_T[s*shift:s*shift+segment_size] += 1
        elif mode == "forward":
            if s == 0:
                spec_FT[:, s*shift:s*shift+segment_size] = spec_SFT
            else:
                spec_FT[:, s*shift+overlap:s*shift+segment_size] = spec_SFT[s, :, overlap:]
        elif mode == "backward":
            if s < n_segment - 1:
                spec_FT[:, s*shift:(s+1)*shift] = spec_SFT[s, :, :shift]
            else:
                spec_FT[:, s*shift:s*shift+segment_size] = spec_SFT

    if mode == "mean":
        spec_FT /= weight_T[None]
    
    return spec_FT


def MPDR(X_FTM, steeringVector):
    if steeringVector.ndim == 2:
        steeringVector = steeringVector.unsqueeze(-1)

    F, _, M = X_FTM.shape
    cov_inv_FMM = torch.linalg.inv(
        torch.einsum("fti, ftj -> fij", X_FTM, X_FTM.conj())
    )
    cov_inv_FMM[0] += torch.eye(M, device=X_FTM.device) * 1e-5

    filter_FMN = torch.einsum("fim, fmk -> fik", cov_inv_FMM, steeringVector)
    filter_FMN /= torch.einsum(
        "fmn, fmn -> fn", steeringVector.conj(), filter_FMN
    ).unsqueeze(1)
    return torch.einsum("fmn, ftm -> ftn", filter_FMN.conj(), X_FTM).squeeze()


def MVDR( X_FTM, noiseSCM_FMM, steeringVector):
    filter_FM = torch.einsum("fim, fm -> fi", torch.linalg.inv(noiseSCM_FMM), steeringVector)
    filter_FM /= torch.einsum(
        "fm, fm -> f", steeringVector.conj(), filter_FM
    ).unsqueeze(1)
    return torch.einsum("fm, ftm -> ft", filter_FM.conj(), X_FTM)


def calculate_SNR(speech, noise):
    SNR = 10 * np.log10( (np.abs(speech)**2).mean() / (np.abs(noise)**2).mean())
    return SNR


def load_Hololens2_SV(n_fft=1024, elevation_list=[-15, 0, 15]):
    """
    Load steering vector in av-suara/common/steeringVector

    Arguments
    ---------
    n_fft: int
        The nummber of samples in STFT

    Returns
    -------
    steeringVector_EAFM: numpy.ndarray
        E (elevation) x A (azimuth) x F (frequency) x M (mic)
    """
    import pickle as pic
    for e, elevation in enumerate(elevation_list):
        fname_steeringVector = "/n/work2/sekiguch/Hololens2/steeringVector/" + \
            f"steeringVector-sync_add-el{elevation:+03d}-{n_fft}.pic"
        steeringVector_DFM = pic.load(open(fname_steeringVector, "rb"))
        norm_DF = np.linalg.norm(steeringVector_DFM, axis=2)
        steeringVector_DFM /= norm_DF[:, :, None]
        if e == 0:
            tmp = [len(elevation_list), *steeringVector_DFM.shape]
            steeringVector_EAFM = np.zeros([len(elevation_list), \
                *steeringVector_DFM.shape], dtype=steeringVector_DFM.dtype)
        steeringVector_EAFM[e] = steeringVector_DFM
    return torch.as_tensor(steeringVector_EAFM).squeeze()


def calculate_SV_from_fname(fname, n_fft=1024, hop_length=None):
    import soundfile as sf
    wav_TM, sr = sf.read(fname)
    X_FTM = MultiSTFT(wav_TM, n_fft, hop_length)
    return calculate_SV(X_FTM)


def calculate_SV(X_FTM):
    X_FTM = torch.as_tensor(X_FTM)
    eig_val, eig_vec = torch.linalg.eigh(torch.einsum("fti, ftj -> fij", X_FTM, X_FTM.conj()))
    return eig_vec[..., -1]


def calculate_SV_geometric(mic_pos_M3=None, distance=1.5, angle=0, n_freq=513, sr=16000, C=340.0):
    if mic_pos_M3 is None:
        M = 8
        radius = 0.05# [m]

        rad_list = np.linspace(0, 360, M+1)[:-1] / 360 * (2 * np.pi)
        mic_pos_M3 = np.array(
                [np.cos(rad_list)*radius, np.sin(rad_list)*radius, np.zeros(M)]
            ).T

    M = len(mic_pos_M3)
    sound_pos_3 = np.array(
        [np.cos(angle) * distance, np.sin(angle) * distance, mic_pos_M3[:, -1].mean()]
    )
    TDOA_M = np.linalg.norm(mic_pos_M3 - sound_pos_3[None], axis=1) / C
    F_list = np.linspace(0, sr//2, n_freq)
    SV_FM = np.exp(-1j * 2 * np.pi * F_list[:, None] * TDOA_M[None])
    return SV_FM


def calculate_SCM(X_FTM):
    X_FTM = torch.as_tensor(X_FTM)
    return torch.einsum("fti, ftj -> fij", X_FTM, X_FTM.conj()) / X_FTM.shape[1]


def split_wav(wav, length_list_fname):
    start_idx_list = []
    with open(length_list_fname, "r") as f:
        for line in f.readlines():
            start_idx_list.append(int(line.strip()))

    if wav.ndim == 2:
        if wav.shape[0] < wav.shape[1]: # M x Tならば
            wav = wav.T

    split_wav = []
    for i in range(len(start_idx_list)-1):
        start_idx = start_idx_list[i]
        end_idx = start_idx_list[i+1]
        split_wav.append(wav[start_idx:end_idx])
    return split_wav


def eval_SDR(est, clean, length_list_fname):
    import mir_eval
    from tqdm import tqdm

    split_est = split_wav(est, length_list_fname)
    split_clean = split_wav(clean, length_list_fname)

    SDR_list = []
    for i in tqdm(range(len(split_est))):
        min_length = min(len(split_clean[i]), len(split_est[i]))
        SDR, _, _, _ = mir_eval.separation.bss_eval_sources(
            split_clean[i][:min_length], split_est[i][:min_length]
        )
        SDR_list.append(float(SDR))
    return SDR_list


def make_test_dataset(data_dir, N, RT60, save_fname):
    from glob import glob   
    with open(save_fname, "w") as f:
        for fname in glob(f"{data_dir}/mixture_RT60={RT60}*.wav"):
            fname = fname.replace(data_dir, "{data_dir}/")
            clean_fname = fname.split("_N=")[0].replace(f"mixture_", "clean_") + ".wav"
            f.write(f"{fname} {clean_fname}\n")

def make_small_eval_dataset(output_fname):
    if not os.path.isfile(output_fname):
        fname_list = []
        for RT60 in [0, 100, 200]:
            flist = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/test_data_{RT60}.flist"
            FF, FM, Fmix, MM, MF, Mmix = 0, 0, 0, 0, 0, 0
            ngen_list = ["n=F", "n=M"]
            flag_list = [0] * 4
            for line in open(flist, "r").readlines():
                mix_fname, clean_fname = line.strip().split(" ")
                _, _, tgen, id, _, ngen = mix_fname.split("/")[-1].split(".wav")[0].split("_")
                print(tgen, ngen)
                if ngen != "n=mix":
                    idx = (tgen == "M") * 2 + ngen_list.index(ngen)
                    if flag_list[idx] == 0:
                        flag_list[idx] = 1
                        fname_list.append([mix_fname, clean_fname])
                
        with open(output_fname, "w") as f:
            for mix_fname, clean_fname in fname_list:
                f.write(f"{mix_fname} {clean_fname}\n")
