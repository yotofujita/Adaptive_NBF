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
    parser.add_argument('--data_root_dir', type= str, default=f"{os.environ['HOME']}/dataset/recording_15th_bldg/")
    parser.add_argument('--N', type= int, default=2)
    parser.add_argument('--SNR', type= int, default=0)
    args = parser.parse_args()

    train_sec = 0.75 * 8 * 60
    test_sec = 0.25 * 8 * 60

    target_dir = args.data_root_dir + "0deg-2/"
    interf_dir_list = [args.data_root_dir + x + "deg/" for x in ["minus45", "minus90", "minus135", "180deg", "90deg", "45deg"]]
    noise_dir_list = [args.data_root_dir + x + "deg/" for x in ["135"]]

    flist_list = ["target_known_train", "target_known_test", "target_unknown_test", "interf_train", "interf_test", "noise_train", "noise_test"]
    if not np.array([os.path.isfile(l+".txt") for l in flist_list]).all():
        print("---recreate filename list---")
        # for target
        unknown_spk_id_set = set()
        known_spk_id_set = set()
        for fname in  glob(target_dir + "*.wav"):
            if "tsp" in fname: continue
            id, gender, known, length = fname.split(".wav")[0].split("/")[-1].split("_")
            spk_id = id.split("-")[0]
            if known == "known":
                known_spk_id_set.add(spk_id)
            else:
                unknown_spk_id_set.add(spk_id)

        target_known_train = []
        target_known_test = []
        for spk_id in known_spk_id_set:
            len_sum = 0
            for fname in  glob(target_dir + f"{spk_id}*.wav"):
                id, gender, known, length = fname.split(".wav")[0].split("/")[-1].split("_")
                if len_sum > train_sec or len_sum + float(length) > train_sec + 5:
                    target_known_test.append(fname)
                else:
                    target_known_train.append(fname)
                    len_sum += float(length)

        target_unknown_test = []
        for spk_id in unknown_spk_id_set:
            len_sum = 0
            for fname in  glob(target_dir + f"{spk_id}*.wav"):
                id, gender, unknown, length = fname.split(".wav")[0].split("/")[-1].split("_")
                if len_sum > train_sec or len_sum + float(length) > test_sec + 5:
                    continue
                else:
                    target_unknown_test.append(fname)
                    len_sum += float(length)

      # for interf
        interf_train = []
        interf_test = []

        for interf_dir in interf_dir_list:
            interf_spk_id_set = set()
            for fname in  glob(interf_dir + "*.wav"):
                id, length = fname.split(".wav")[0].split("/")[-1].split("_")
                spk_id = id.split("-")[0]
                interf_spk_id_set.add(spk_id)

            for spk_id in interf_spk_id_set:
                len_sum = 0
                for fname in  glob(interf_dir + f"{spk_id}*.wav"):
                    id, length = fname.split(".wav")[0].split("/")[-1].split("_")
                    if len_sum > train_sec or len_sum + float(length) > train_sec + 5:
                        interf_test.append(fname)
                    else:
                        interf_train.append(fname)
                        len_sum += float(length)

      # for noise
        noise_train = []
        noise_test = []
        for noise_dir in noise_dir_list:
            situation_list = ["BUS", "CAF", "PED", "STR"]
            for fname in  glob(noise_dir + "*.wav"):
                situation = fname.split(".wav")[0].split("_")[-1]
                if situation in situation_list:
                    situation_list[situation_list.index(situation)] = 0
                    noise_test.append(fname.strip())
                else:
                    noise_train.append(fname.strip())


      # shuffle each fname list
        for flist in flist_list:
            random.shuffle(eval(flist))
            with open(f"{flist}.txt", "w") as f:
                for fname in eval(flist):
                    f.write(fname.split("/")[-2] + "/" + fname.split("/")[-1] + "\n")
    else:
        target_known_train, target_known_test, target_unknown_test = [], [], []
        interf_train, interf_test = [], []
        noise_train, noise_test = [], []
        for flist in flist_list:
            with open(f"{flist}.txt", "r") as f:
                for line in f.readlines():
                    eval(flist).append(args.data_root_dir + "/" + line.strip())

# make mixture
    if not (os.path.isfile("clean_train.wav") and os.path.isfile("mixture_train.wav")):
        mixture = np.zeros([int(train_sec * 4 * 16000) + 16000 * 100, 5])
        lim = int(train_sec * 4 * 16000)
        start_idx = 0
        target_len_list = []
        for target_fname in target_known_train:
            wav, sr = sf.read(target_fname.strip())
            wav /= np.abs(wav).max() * 1.2
            wav = resampy.resample(wav, sr, 16000, axis=0)
            mixture[start_idx:start_idx+len(wav)] = wav
            start_idx += len(wav)
            target_len_list.append(len(wav))

        sf.write("clean_train.wav", mixture[:lim], 16000)
        with open("target_train_len.txt", "w") as f:
            for target_len in target_len_list:
                f.write(str(target_len) + "\n")

        np.random.seed(0)
        for n in range(args.N):
            start_interf_idx = 0
            while 1:
                interf_fname = interf_train.pop()
                wav, sr = sf.read(interf_fname.strip())
                wav /= np.abs(wav).max() * 1.2
                wav = resampy.resample(wav, sr, 16000, axis=0)

                start_interf_idx += int(np.random.rand() * 5 * 16000)
                mixture[start_interf_idx:start_interf_idx+len(wav)] += wav * (np.random.rand() + 0.5)
                start_interf_idx += len(wav)

                if start_interf_idx > start_idx:
                    break

        sf.write(f"mixture_train_N={args.N}.wav", mixture[:lim] / (np.abs(mixture[:lim]).max() * 1.2), 16000)

        start_noise_idx = 0
        while 1:
            noise_fname = np.random.choice(noise_train)
            wav, sr = sf.read(interf_fname.strip())
            wav = resampy.resample(wav, sr, 16000, axis=0)

            SNR_tmp = calculate_SNR(mixture[start_noise_idx:start_noise_idx+len(wav), 0], wav[:, 0])
            scale = np.sqrt(10 ** ((SNR_tmp - args.SNR) / 10))
            scaled_noise = wav * scale
            SNR_new = calculate_SNR(mixture[start_noise_idx:start_noise_idx+len(wav), 0], scaled_noise[:, 0])

            mixture[start_noise_idx:start_noise_idx+len(wav)] += scaled_noise
            start_noise_idx += len(wav)
            if start_noise_idx > lim:
                break

        sf.write(f"mixture_noisy_train_N={args.N}_SNR={args.SNR}.wav", mixture[:lim] / (np.abs(mixture[:lim]).max() * 1.2), 16000)



    interf_test_for_unknown = copy.copy(interf_test)

    if not (os.path.isfile("clean_known_test.wav") and os.path.isfile(f"mixture_known_test_N={args.N}.wav") and os.path.isfile(f"mixture_noisy_known_test_N={args.N}_SNR={args.SNR}.wav")):

        mixture = np.zeros([int(test_sec) * 4 * 16000 + 16000 * 100, 5])
        lim = int(test_sec * 4 * 16000)
        start_idx = 0
        target_len_list = []
        for target_fname in target_known_test:
            wav, sr = sf.read(target_fname.strip())
            wav /= np.abs(wav).max() * 1.2
            wav = resampy.resample(wav, sr, 16000, axis=0)
            mixture[start_idx:start_idx+len(wav)] = wav
            start_idx += len(wav)
            target_len_list.append(len(wav))

        sf.write("clean_known_test.wav", mixture[:lim], 16000)
        with open("target_test_known_len.txt", "w") as f:
            for target_len in target_len_list:
                f.write(str(target_len) + "\n")

        np.random.seed(0)
        for n in range(args.N):
            start_interf_idx = 0
            while 1:
                interf_fname = interf_test.pop()
                wav, sr = sf.read(interf_fname.strip())
                wav /= np.abs(wav).max() * 1.2
                wav = resampy.resample(wav, sr, 16000, axis=0)

                start_interf_idx += int(np.random.rand() * 5 * 16000)
                mixture[start_interf_idx:start_interf_idx+len(wav)] += wav * (np.random.rand() + 0.5)
                start_interf_idx += len(wav)

                if start_interf_idx > start_idx:
                    break

        sf.write(f"mixture_known_test_N={args.N}.wav", mixture[:lim] / (np.abs(mixture[:lim]).max() * 1.2), 16000)

        start_noise_idx = 0
        while 1:
            noise_fname = np.random.choice(noise_test)
            wav, sr = sf.read(interf_fname.strip())
            wav = resampy.resample(wav, sr, 16000, axis=0)

            SNR_tmp = calculate_SNR(mixture[start_noise_idx:start_noise_idx+len(wav), 0], wav[:, 0])
            scale = np.sqrt(10 ** ((SNR_tmp - args.SNR) / 10))
            scaled_noise = wav * scale

            mixture[start_noise_idx:start_noise_idx+len(wav)] += scaled_noise
            start_noise_idx += len(wav)
            if start_noise_idx > lim:
                break

        sf.write(f"mixture_noisy_known_test_N={args.N}_SNR={args.SNR}.wav", mixture[:lim] / (np.abs(mixture[:lim]).max() * 1.2), 16000)



    if not (os.path.isfile("clean_unknown_test.wav") and os.path.isfile(f"mixture_unknown_test_N={args.N}.wav") and os.path.isfile(f"mixture_noisy_unknown_test_N={args.N}_SNR={args.SNR}.wav")):
        mixture = np.zeros([int(test_sec) * 4 * 16000 + 16000 * 100, 5])
        lim = int(test_sec * 4 * 16000)
        start_idx = 0
        target_len_list = []
        for target_fname in target_unknown_test:
            wav, sr = sf.read(target_fname.strip())
            wav /= np.abs(wav).max() * 1.2
            wav = resampy.resample(wav, sr, 16000, axis=0)
            mixture[start_idx:start_idx+len(wav)] = wav
            start_idx += len(wav)
            target_len_list.append(len(wav))

        sf.write("clean_unknown_test.wav", mixture[:lim], 16000)
        with open("target_test_unknown_len.txt", "w") as f:
            for target_len in target_len_list:
                f.write(str(target_len) + "\n")

        np.random.seed(0)
        for n in range(args.N):
            start_interf_idx = 0
            while 1:
                interf_fname = interf_test_for_unknown.pop()
                wav, sr = sf.read(interf_fname.strip())
                wav /= np.abs(wav).max() * 1.2
                wav = resampy.resample(wav, sr, 16000, axis=0)

                start_interf_idx += int(np.random.rand() * 5 * 16000)
                mixture[start_interf_idx:start_interf_idx+len(wav)] += wav * (np.random.rand() * 0.5)
                start_interf_idx += len(wav)

                if start_interf_idx > start_idx:
                    break

        sf.write(f"mixture_unknown_test_N={args.N}.wav", mixture[:lim] / (np.abs(mixture[:lim]).max() * 1.2), 16000)

        start_noise_idx = 0
        while 1:
            noise_fname = np.random.choice(noise_test)
            wav, sr = sf.read(interf_fname.strip())
            wav = resampy.resample(wav, sr, 16000, axis=0)

            SNR_tmp = calculate_SNR(mixture[start_noise_idx:start_noise_idx+len(wav), 0], wav[:, 0])
            scale = np.sqrt(10 ** ((SNR_tmp - args.SNR) / 10))
            scaled_noise = wav * scale

            mixture[start_noise_idx:start_noise_idx+len(wav)] += scaled_noise
            start_noise_idx += len(wav)
            if start_noise_idx > lim:
                break

        sf.write(f"mixture_noisy_unknown_test_N={args.N}_SNR={args.SNR}.wav", mixture[:lim] / (np.abs(mixture[:lim]).max() * 1.2), 16000)



