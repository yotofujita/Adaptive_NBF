#!/usr/bin/env python3
# coding:utf-8
import numpy as np
import torch
import speechbrain as sb
import soundfile as sf
import resampy
import os, sys
import pdb
import librosa
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided
from utils.utility import *

EPS = 1e-7


def test(model, SV_FM, hparams, device, fname, data_dir, save_dir, SNR=40,
    iteration=1, localization_loss=False, speaker_loss=False, queue_size=10, bp_freq=3
):
    save_dir += f"/{str(model)}_with_clean_spk_emb_it={iteration}"
    if localization_loss or speaker_loss:
        save_dir += \
            f"_FineTune_loc={localization_loss}_spk={speaker_loss}_" \
            f"queue={queue_size}_bp={bp_freq}"
    save_dir += "/"
    os.system(f"mkdir -p {save_dir}")

    F, M = SV_FM.shape
    np.random.seed(0)

    with open(fname, "r") as f:
        for line in f.readlines():
            line = line.replace("{data_dir}", data_dir)
            noisy_fname, clean_fname = line.strip().split(" ")

            save_MPDR_fname = f"{save_dir}/sep_MPDR__{noisy_fname.split('/')[-1]}"
            save_MVDR_fname = f"{save_dir}/sep_MVDR__{noisy_fname.split('/')[-1]}"
            if os.path.isfile(save_MVDR_fname):
                continue

            noisy, sr = sf.read(noisy_fname)
            clean, sr = sf.read(clean_fname)
            clean = clean[:, 0]
            if sr != 16000:
                print("---Resampling---")
                noisy = resampy.resample(noisy, sr, 16000, axis=0)
                clean = resampy.resample(clean, sr, 16000, axis=0)

            white_noise = np.random.rand(len(noisy), M)
            for m in range(M):
                SNR_tmp = calculate_SNR(noisy[:, m], white_noise[:, m])
                scale = np.sqrt(10 ** ((SNR_tmp - SNR) / 10))
                noisy[:, m] += white_noise[:, m] * scale

            noisy_FTM = MultiSTFT(noisy, n_fft=hparams["n_fft"])
            clean_FT = MultiSTFT(clean, n_fft=hparams["n_fft"])
            clean_pwr_FT = torch.abs(clean_FT) ** 2

            sep_MVDR, sep_MPDR = separate_with_clean_spk_emb(
                model, noisy_FTM, clean_pwr_FT, SV_FM, hparams, device,
                iteration, localization_loss, speaker_loss, queue_size, bp_freq
            )
            sep_MPDR = MultiISTFT(sep_MPDR)
            sep_MVDR = MultiISTFT(sep_MVDR)

            sf.write(save_MVDR_fname, sep_MVDR/(np.abs(sep_MVDR).max()*1.2), 16000)
            sf.write(save_MPDR_fname, sep_MPDR/(np.abs(sep_MPDR).max()*1.2), 16000)


def  separate_with_clean_spk_emb(
    model, noisy_FTM, clean_pwr_FT, steeringVector_FM, hparams, device, 
    iteration=1, localization_loss=False, speaker_loss=False, queue_size=10, bp_freq=3
):
    noisy_FTM = torch.as_tensor(noisy_FTM, dtype=torch.complex64).to(device)
    clean_pwr_FT = torch.as_tensor(clean_pwr_FT, dtype=torch.float32).to(device)
    steeringVector_FM = torch.as_tensor(steeringVector_FM, dtype=torch.complex64).to(device)

    noisy_pwr_SFT = segmentation(
        torch.abs(noisy_FTM[:, :, 0]).to(torch.float32) ** 2, 
        segment_size=hparams["segment_size"],
        overlap_ratio=hparams["overlap_ratio"]
    )
    clean_pwr_SFT = segmentation(
        clean_pwr_FT,
        segment_size=hparams["segment_size"],
        overlap_ratio=hparams["overlap_ratio"]
    )
    S, F, T = noisy_pwr_SFT.shape
    _, _, M = noisy_FTM.shape

    noisy_SFTM = torch.zeros(
        [S, F, T, M], dtype=noisy_FTM.dtype, device=device
    )
    for m in range(M):
        noisy_SFTM[..., m] = segmentation(
            noisy_FTM[:, :, m].contiguous(),
            segment_size=hparams["segment_size"],
            overlap_ratio=hparams["overlap_ratio"]
        )

    enhanced_pwr_SFT = torch.zeros([S, F, T], dtype=torch.float32, device=device)
    sep_MPDR_SFT = torch.zeros([S, F, T], dtype=torch.complex64, device=device)
    for s in range(len(noisy_SFTM)):
        sep_MPDR_SFT[s] = MPDR(noisy_SFTM[s], steeringVector_FM)
        enhanced_pwr_SFT[s] = torch.abs(sep_MPDR_SFT[s]) ** 2
    sep_MPDR_FT = overlap_add(sep_MPDR_SFT, overlap_ratio=hparams["overlap_ratio"])

    n_segment = len(noisy_pwr_SFT)
    speaker_vec = None
    sep_SFT = torch.zeros_like(noisy_pwr_SFT, dtype=noisy_FTM.dtype)

    if localization_loss or speaker_loss:
        queue_noisy_BFTM = torch.zeros(
            [queue_size, F, T, M], dtype=torch.complex64, device=device
        )
        queue_enhanced_pwr_BFT = torch.zeros(
            [queue_size, F, T], dtype=torch.float32, device=device
        )
        queue_index = 0
        full = False
        optimizer = torch.optim.Adam(model.mask_net.parameters(), lr=0.1)

    for s in tqdm(range(n_segment)):
        with torch.no_grad():
            for it in range(iteration):
                mask_FT, speaker_vec = model.estimate_mask_segment_w_clean(
                    noisy_pwr_SFT[s],
                    clean_pwr_SFT[s],
                    enhanced_pwr_SFT[s],
                    speaker_vec
                )
                masked_speech_FTM = noisy_SFTM[s] * mask_FT[:, :, None]
                masked_noise_FTM = noisy_SFTM[s] * (1-mask_FT)[:, :, None]
                SV_FM = calculate_SV(masked_speech_FTM).to(device)
                noise_SCM_FMM = calculate_SCM(masked_noise_FTM) + \
                                EPS * torch.eye(M, device=device)[None]

                enhanced_pwr_SFT[s] = torch.abs(MVDR(noisy_SFTM[s], noise_SCM_FMM, SV_FM))**2
        sep_SFT[s] = MVDR(noisy_SFTM[s], noise_SCM_FMM, SV_FM).cpu()

        if localization_loss or speaker_loss:
            queue_noisy_BFTM[queue_index] = noisy_SFTM[s]
            queue_enhanced_pwr_BFT[queue_index] = torch.abs(sep_SFT[s].to(device)) ** 2
            queue_index = (queue_index + 1) % queue_size
            if queue_index == 0:
                full = True
            if full and (s % bp_freq == 0):
                print("---Fine-tuning---")
                loss = model.fine_tuning(
                    noisy_BFTM=queue_noisy_BFTM,
                    enhanced_pwr_BFT=queue_enhanced_pwr_BFT,
                    speaker_vec_D=speaker_vec,
                    SV_FM=SV_FM,
                    localization_loss=localization_loss, 
                    speaker_loss=speaker_loss
                )
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.mask_net.parameters(), 1)
                optimizer.step()

    sep_MVDR_FT = overlap_add(sep_SFT, overlap_ratio=hparams["overlap_ratio"])
    return sep_MVDR_FT, sep_MPDR_FT


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    device = run_opts["device"]
    torch.cuda.set_device(device) 

    from hyperpyyaml import load_hyperpyyaml
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    from model_SB import SpeakerBeam

    model = SpeakerBeam(
        input_length=hparams["segment_size"],
        z_dim = hparams["z_dim"],
    ).eval().to(device)
    model.mask_net.train()

    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=f"{hparams['save_folder']}/{hparams['model_name']}",
        recoverables={
            "model": model,
            "counter": hparams["epoch_counter"]
        }
    )

    checkpointer.recover_if_possible(device=device)

    RT60 = 200

    fname = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/test_data_{RT60}.flist"
    data_dir = f"/n/work3/sekiguch/dataset/pyroomacoustics_librispeech/"
    save_dir = f"/n/work3/sekiguch/data_for_paper/ICASSP2022/pyroomacoustics_librispeech/"

    SV_FM = calculate_SV_geometric()
    # SV_FM = calculate_SV_from_fname(data_dir + "clean_RT60=100_F_1995.wav")
    save_dir = "./"

    test(
        model, SV_FM, hparams, device, fname, data_dir, save_dir
    )


