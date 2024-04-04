#! /usr/bin/env python3
# coding:utf-8

import sys, os
import soundfile as sf
import numpy as np
from glob import glob
import argparse, random
import resampy
import copy
import pickle as pic


def calculate_SNR(speech, noise):
    SNR = 10 * np.log10( (np.abs(speech)**2).mean() / (np.abs(noise)**2).mean())
    return SNR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root_dir', 
        type= str, 
        default=f"/n/work2/sekiguch/Hololens2/recording_15th_bldg/"
    )
    parser.add_argument('--N', type= int, default=2)
    parser.add_argument('--SNR', type= int, default=0)
    args = parser.parse_args()

    target_dir = args.data_root_dir + "0deg-2/"
    interf_dir_list = [args.data_root_dir + x + "deg/" for x in ["minus45", "minus90", "minus135", "180", "90", "45"]]
    noise_dir_list = [args.data_root_dir + x + "deg/" for x in ["135"]]

    flist_list = ["target", "interf", "noise"]

    # for target
    target_id_set = set()
    for fname in  glob(target_dir + "*.wav"):
        if "tsp" in fname: continue
        id, gender, known, length = fname.split(".wav")[0].split("/")[-1].split("_")
        spk_id = id.split("-")[0]
        target_id_set.add(spk_id)

    target_fname_dict = {}
    for spk_id in target_id_set:
        target_flist = []
        for fname in  glob(target_dir + f"{spk_id}*.wav"):
            target_flist.append(fname.strip())
        target_fname_dict[spk_id] = target_flist

    # for interf
    interf_flist = []
    for interf_dir in interf_dir_list:
        for fname in  glob(interf_dir + "*.wav"):
            interf_flist.append(fname.strip())

    # for noise
    situation_list = ["BUS", "CAF", "PED", "STR"]
    noise_flist_dict = {situ:[] for situ in situation_list}

    for noise_dir in noise_dir_list:
        for fname in  glob(noise_dir + "*.wav"):
            situation = fname.split(".wav")[0].split("_")[-1]
            noise_flist_dict[situation].append(fname.strip())

# make mixture
    sec_per_spk = 8 * 60

    np.random.seed(0)

    for spk_id in target_fname_dict.keys():
        mixture = np.zeros([int(sec_per_spk * 16000) + 16000 * 150, 5])
        start_idx = 0
        target_len_list = []
        for target_fname in target_fname_dict[spk_id]:
            wav, sr = sf.read(target_fname.strip())
            wav /= np.abs(wav).max() * 1.2
            wav = resampy.resample(wav, sr, 16000, axis=0)
            mixture[start_idx:start_idx+len(wav)] = wav
            start_idx += len(wav)
            target_len_list.append(len(wav))

        with open(f"target_len_{spk_id}.txt", "w") as f:
            for target_len in target_len_list:
                f.write(str(target_len) + "\n")
        sf.write(f"clean_{spk_id}.wav", mixture[:start_idx], 16000)

        random.shuffle(interf_flist)
        interf_flist_copy = copy.copy(interf_flist)
        for n in range(args.N):
            start_interf_idx = 0
            while 1:
                interf_fname = interf_flist_copy.pop()
                wav, sr = sf.read(interf_fname.strip())
                wav /= np.abs(wav).max() * 1.2
                wav = resampy.resample(wav, sr, 16000, axis=0)

                start_interf_idx += int(np.random.rand() * 5 * 16000)
                mixture[start_interf_idx:start_interf_idx+len(wav)] += wav
                start_interf_idx += len(wav)

                if start_interf_idx > start_idx:
                    break

        sf.write(
            f"mixture_{spk_id}_N={args.N}.wav", 
            mixture[:start_idx] / (np.abs(mixture[:start_idx]).max() * 1.2),
            16000
        )

        situation = random.choice(situation_list)
        start_noise_idx = 0
        while 1:
            noise_fname = np.random.choice(noise_flist_dict[situation])
            wav, sr = sf.read(noise_fname.strip())
            wav = resampy.resample(wav, sr, 16000, axis=0)

            SNR_tmp = calculate_SNR(mixture[start_noise_idx:start_noise_idx+len(wav), 0], wav[:, 0])
            scale = np.sqrt(10 ** ((SNR_tmp - args.SNR) / 10))
            scaled_noise = wav * scale

            mixture[start_noise_idx:start_noise_idx+len(wav)] += scaled_noise
            start_noise_idx += len(wav)
            if start_noise_idx > start_idx:
                break

        sf.write(
            f"mixture_noisy_{spk_id}_N={args.N}_SNR={args.SNR}.wav", 
            mixture[:start_idx] / (np.abs(mixture[:start_idx]).max() * 1.2),
            16000
        )

