#! /usr/bin/env python3
# coding: utf-8

from ctypes import util
from curses import window
import numpy as np
from sklearn import mixture
import torch
import torchaudio
import h5py
import os
from glob import glob

from Lightning_CSS import Lightning_CSS
import utility


def separate(
    model,
    SV_FM,
    azim_idx,
    fname,
    device,
    window_len=48128,
    shift=8000,
    insert_first=True,
    BF="MVDR",
):
    with torch.no_grad():
        mixture_MT, fs = torchaudio.load(fname)
        n_mic, n_sample = mixture_MT.shape

        if insert_first:
            insert_len = window_len - shift
            new_mixture_MT = torch.zeros(n_mic, n_sample + insert_len, dtype=mixture_MT.dtype)
            new_mixture_MT[:, :insert_len] = mixture_MT[:, -insert_len:]
            new_mixture_MT[:, insert_len:] = mixture_MT
            mixture_MT = new_mixture_MT

        mixture_BMTF = utility.split_sig(mixture_MT, window_len=window_len, shift=shift, stft=True).to(device)

        input = utility.make_feature(
            mixture_BMTF, SV_FM, azim_idx, use_BF=model.use_BF, use_SV=model.use_SV, use_DAN=model.use_DAN
        )

        if model.use_DAN:
            input = [feature.to(device) for feature in input]

        if model.use_VAD:
            mask, vad = model.forward(input)
            vad = model.separation_net.expand_vad(vad)
        else:
            mask = model.forward(input)

        if BF == "MA_MVDR":
            sep = utility.MA_MVDR(mixture_BMTF, mask, 1 - mask)
        elif BF == "MVDR":
            sep = utility.MVDR(mixture_BMTF, mask, 1 - mask)
        elif BF == "MVDR_SV":
            sep = utility.MVDR_SV(mixture_BMTF, mask, 1 - mask)
        elif BF == "GEV":
            sep = utility.GEV(mixture_BMTF, mask, 1 - mask)

        sig = utility.shift_concat(sep, shift=8000, use_first_all=False)
        sig /= torch.abs(sig).max() * 1.2
        return sig.to("cpu")


def main(device, idx, BF="MVDR"):
    f = h5py.File("../data/SV_for_HL2.h5", "r")
    SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64)).to(device)
    norm_EAF = torch.linalg.norm(SV_EAFM, axis=3)
    SV_EAFM /= norm_EAF[..., None]

    data_root_dir = "/n/work3/sekiguch/dataset/Hololens2_RealData_15th_bldg/test/"

    if BF == "MA_MVDR":
        save_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_MA_MVDR_result/test/"
    elif BF == "MVDR":
        save_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_BF_result/test/"
    elif BF == "MVDR_SV":
        save_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_MVDR_SV_result/test/"
    elif BF == "GEV":
        save_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_GEV_result/test/"

    ckpt_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/"
    model_name = "lstm"

    settings = [
        [True, False, True, False],
        # [True, False, False, False],
    ]  # use_BF, use_SV, use_DAN, use_VAD

    # tuning_data_ratio = [0.25, 0.5, 0.75, 1.0][idx]
    tuning_data_ratio = 0.75
    n_epoch = [1, 3, 5, 10, 20, 30, 40, 50][idx]

    for setting in settings:
        use_BF, use_SV, use_DAN, use_VAD = setting

        # test original model (without fine-tuning)
        orig_ckpt_dir = f"{ckpt_root_dir}/{model_name}-BF={use_BF}-SV={use_SV}-DAN={use_DAN}-VAD={use_VAD}/"
        orig_ckpt_list = glob(
            f"{orig_ckpt_dir}/lightning_logs/version_0/checkpoints/epoch=*ckpt"
        )
        if len(orig_ckpt_list) == 0:
            print(f"There are no checkpoint of the original model in {orig_ckpt_dir}")
            continue
        assert len(orig_ckpt_list) == 1, f"There are multiple checkpoints of the original model in {orig_ckpt_dir}"
        orig_ckpt = orig_ckpt_list[0]
        save_dir = f"{save_root_dir}/{model_name}-BF={use_BF}-SV={use_SV}-DAN={use_DAN}-VAD={use_VAD}/original/"
        os.system(f"mkdir -p {save_dir}")

        model = Lightning_CSS(use_BF, use_SV, use_DAN, use_VAD, save_hparam=False, device=device)
        ckpt_dict = torch.load(orig_ckpt)
        model.load_state_dict(ckpt_dict["state_dict"])
        model.to(device)
        model.eval()

        for fname in glob(f"{data_root_dir}/test_mixture*SNR=10.wav"):
            elev_idx, azim_idx = [1, 0]
            save_fname = f"{save_dir}/sep-{fname.split('/')[-1]}"
            if os.path.isfile(save_fname):
                continue
            print("start ", save_fname)
            sep = separate(model, SV_EAFM[elev_idx, azim_idx], azim_idx, fname, device, BF=BF)
            torchaudio.save(save_fname, sep.unsqueeze(0), 16000)

        make_wav_list_original(save_dir, data_root_dir)

        # test fine-tuned model
        for finetune_dir in glob(
            f"{ckpt_root_dir}/{model_name}-BF={use_BF}-SV={use_SV}-DAN={use_DAN}-VAD={use_VAD}/lightning_logs/version_0/checkpoints/minute*ratio={tuning_data_ratio}*"
        ):
            if len(finetune_dir.split("/")[-1].split("-")) < 5:
                print(finetune_dir, " is not acceptable.")
                continue

            save_dir = f"{save_root_dir}/{model_name}-BF={use_BF}-SV={use_SV}-DAN={use_DAN}-VAD={use_VAD}/{finetune_dir.split('/')[-1]}"
            os.system(f"mkdir -p {save_dir}")

            ckpt_list = glob(f"{finetune_dir}/checkpoints/epoch={n_epoch-1}-*.ckpt")

            for ckpt in ckpt_list:
                finetune_epoch = ckpt.split("/")[-1].split("-")[0].split("=")[-1]

                if len(glob(f"{save_dir}/sep_epoch={finetune_epoch}-*.wav")) == 8:
                    print(ckpt, " has already been finished !")
                    continue

                model = Lightning_CSS(use_BF, use_SV, use_DAN, use_VAD, save_hparam=False, device=device)
                ckpt_dict = torch.load(ckpt)
                model.load_state_dict(ckpt_dict["state_dict"])
                model.to(device)
                model.eval()

                for fname in glob(f"{data_root_dir}/test_mixture*SNR=10.wav"):
                    elev_idx, azim_idx = [1, 0]
                    save_fname = f"{save_dir}/sep_epoch={finetune_epoch}-{fname.split('/')[-1]}"
                    if os.path.isfile(save_fname):
                        continue
                    print("start ", save_fname)
                    sep = separate(model, SV_EAFM[elev_idx, azim_idx], azim_idx, fname, device, BF=BF)
                    torchaudio.save(save_fname, sep.unsqueeze(0), 16000)
            make_wav_list(save_dir, data_root_dir)


def make_wav_list(save_dir, data_root_dir):
    epoch_set = set()
    for fname in glob(f"{save_dir}/sep_epoch=*.wav"):
        epoch = fname.split("epoch=")[-1].split("-")[0]
        epoch_set.add(epoch)

    for epoch in epoch_set:
        os.system(f"mkdir -p {save_dir}/asr")
        with open(f"{save_dir}/wav_list_epoch={epoch}.txt", "w") as f:
            for fname in glob(f"{save_dir}/sep_epoch={epoch}*.wav"):
                spk_id = fname.split("_mixture_")[-1].split("_SNR")[0]
                output_transcript_file = fname.replace("sep_epoch=", "/asr/text_epoch=").replace(".wav", ".trn")
                mixture_fname = f"{data_root_dir}/{fname.split('/')[-1].split('-', 1)[-1]}"
                json_fname = f"{data_root_dir}/test_target_0deg_{spk_id}.json"
                f.write(f"{spk_id},{output_transcript_file},{fname},{mixture_fname},{json_fname}\n")


def make_wav_list_original(save_dir, data_root_dir):
    os.system(f"mkdir -p {save_dir}/asr")
    with open(f"{save_dir}/wav_list_original.txt", "w") as f:
        for fname in glob(f"{save_dir}/sep-*.wav"):
            spk_id = fname.split("_mixture_")[-1].split("_SNR")[0]
            output_transcript_file = fname.replace("sep-", "/asr/text-").replace(".wav", ".trn")
            mixture_fname = f"{data_root_dir}/{fname.split('/')[-1].split('-', 1)[-1]}"
            json_fname = f"{data_root_dir}/test_target_0deg_{spk_id}.json"
            f.write(f"{spk_id},{output_transcript_file},{fname},{mixture_fname},{json_fname}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--BF", type=str, default="MVDR")
    args, _ = parser.parse_known_args()
    # torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda" if args.gpu >= 0 else "cpu")
    main(device, idx=args.gpu if args.idx is None else args.idx, BF=args.BF)
