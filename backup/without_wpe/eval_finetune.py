#! /usr/bin/env python3
# coding: utf-8

from operator import sub
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import csv

from pytest import param

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--total", type=int, default=1)
    parser.add_argument("--BF", type=str, default="MVDR")
    args, _ = parser.parse_known_args()

    if args.BF == "MA_MVDR":
        data_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_MA_MVDR_result/test/"
        save_WER_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_MA_MVDR_result/test/WER/"
        output_csv_fname = "wer_MA_MVDR.csv"
    elif args.BF == "MVDR":
        data_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_BF_result/test/"
        save_WER_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_BF_result/test/WER/"
        output_csv_fname = "wer_MVDR.csv"
    elif args.BF == "MVDR_SV":
        data_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_MVDR_SV_result/test/"
        save_WER_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_MVDR_SV_result/test/WER/"
        output_csv_fname = "wer_MVDR_SV.csv"
    elif args.BF == "GEV":
        data_root_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_GEV_result/test/"
        save_WER_dir = "/n/work3/sekiguch/data_for_paper/IROS2022/DNN_GEV_result/test/WER/"
        output_csv_fname = "wer_GEV.csv"
    else:
        raise ValueError("BF should be MA_MVDR, MVDR, MVDR_SV, GEV")

    os.system(f"mkdir -p {save_WER_dir}")

    ref_trn_fname = "/n/work3/sekiguch/dataset/Hololens2_RealData_15th_bldg/test/ref_test_target_0deg.trn"

    args.idx = args.gpu if args.idx is None else args.idx

    # minute_fin_list = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0]
    minute_fin_list = []
    # epoch_list = [5, *range(10, 51, 10)]
    epoch_list = [1, 3, 5, *range(10, 51, 10)]
    tuning_data_ratio = 0.75

    for data_parent_dir in glob(f"{data_root_dir}/lstm*"):
        data_dir_list_tmp = sorted(glob(f"{data_parent_dir}/minute*ratio={tuning_data_ratio}*"))
        data_dir_list_tmp2 = []
        for data_dir in data_dir_list_tmp:
            for minute in minute_fin_list:
                if f"minute={minute}" in data_dir:
                    break
            else:
                data_dir_list_tmp2.append(data_dir)
        data_dir_list_tmp = [f"{data_parent_dir}/original", *data_dir_list_tmp2]

        if len(data_dir_list_tmp) <= args.total:
            if args.idx >= len(data_parent_dir):
                continue
            data_dir_list = [data_dir_list_tmp[args.idx % len(data_dir_list_tmp)]]
        else:
            data_dir_list = []
            for i in range(int(np.ceil(len(data_dir_list_tmp) / args.total))):
                if args.total * i + args.idx < len(data_dir_list_tmp):
                    data_dir_list.append(data_dir_list_tmp[args.total * i + args.idx])

        print("gpu = ", args.gpu, "  idx = ", args.idx, " : ", data_dir_list)
        for data_dir in tqdm(data_dir_list):
            for wav_list in glob(f"{data_dir}/wav_list*.txt"):
                if "epoch=" in wav_list:
                    epoch = int(wav_list.rsplit("epoch=", 1)[-1].split(".")[0]) + 1
                    if epoch not in epoch_list:
                        continue
                os.system(f"CUDA_VISIBLE_DEVICES={args.gpu} python3 asr_speechbrain_noSync.py --wav_list {wav_list}")

            wer_fname = save_WER_dir + "---".join(data_dir.split("/")[-2:]) + ".txt"
            with open(wer_fname, "w") as f_wer:
                for hyp_fname in glob(f"{data_dir}/asr/hyp*transformer*"):
                    output_fname = hyp_fname.replace("hyp_", "sclite_").replace(".trn", ".txt")
                    if not os.path.isfile(output_fname):
                        if "epoch" in output_fname:
                            epoch = int(output_fname.rsplit("epoch=", 1)[-1].split("-")[0]) + 1
                            if epoch not in epoch_list:
                                continue

                        os.system(
                            f"sclite -r '{ref_trn_fname}' trn -h '{hyp_fname}' trn -i rm -o all stdout > '{output_fname}'"
                        )

                    with open(output_fname, "r") as f:
                        for line in f.readlines():
                            if "Sum/Avg" in line:
                                break
                        else:
                            print("\n ---Error--- Sum/Avg not found in ", output_fname)
                            exit()
                        WER = [d for d in line.strip().split(" ") if len(d) >= 1][-3]

                        if "epoch=" in output_fname:
                            epoch = int(output_fname.rsplit("epoch=", 1)[-1].split("-")[0]) + 1
                        else:
                            epoch = 0
                        spk_id = output_fname.rsplit("_mixture_", 1)[-1].split(".")[0]
                        if "transformer" in output_fname:
                            asr = "transformer"
                        else:
                            asr = "crdnn"
                        f_wer.write(f"{epoch},{spk_id},{asr},{WER},{output_fname}\n")

    n_test_data = 8
    data_dict_original = {}
    for wer_fname in glob(f"{save_WER_dir}/*original*.txt"):
        key = wer_fname.split("/")[-1].split("---")[0]
        data_dict_original[key] = {"crdnn": 0, "transformer": 0}
        with open(wer_fname, "r") as f:
            for line in f.readlines():
                _, spk_id, asr, WER, _ = line.strip().split(",")
                data_dict_original[key][asr] += float(WER) / n_test_data
    print("data_dict_original: \n", data_dict_original)
    data_dict = {}
    for wer_fname in glob(f"{save_WER_dir}/*---minute*.txt"):
        with open(wer_fname, "r") as f:
            key, sub_key = wer_fname.split("/")[-1].split(".txt")[0].split("---")

            if key not in data_dict:
                data_dict[key] = {}
            if sub_key not in data_dict[key]:
                data_dict[key][sub_key] = {epoch: {"crdnn": 0, "transformer": 0} for epoch in epoch_list}

            for line in f.readlines():
                epoch, spk_id, asr, WER, _ = line.strip().split(",")
                if int(epoch) in epoch_list:
                    data_dict[key][sub_key][int(epoch)][asr] += float(WER) / n_test_data

    with open(output_csv_fname, "w") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "model",
                "BF",
                "SV",
                "DAN",
                "VAD",
                "minute",
                "ratio",
                "finetune",
                "batch",
                "lr",
                "input_BF",
                "asr",
                *[f"epoch={epoch}" for epoch in [5, *range(0, 51, 10)]],
            ]
        )
        print(data_dict_original, "\n\n", data_dict)
        for key in data_dict_original.keys():
            params = key.split("-")
            print(params, key)
            model = params[0]
            BF, SV, DAN, VAD = [param.split("=")[1] for param in params[1:]]

            for sub_key in data_dict[key].keys():
                transformer_wer_list = [data_dict_original[key]["transformer"]]
                transformer_wer_list.extend([data_dict[key][sub_key][epoch]["transformer"] for epoch in epoch_list])

                if len([param.split("=")[1] for param in sub_key.split("-")]) == 5:
                    minute, ratio, finetune, batch, lr = [param.split("=")[1] for param in sub_key.split("-")]
                    input_BF = "DSBF"
                else:
                    minute, ratio, finetune, batch, lr, input_BF = [
                        param.split("=")[1] for param in sub_key.split("-")
                    ]

                writer.writerow(
                    [
                        model,
                        BF,
                        SV,
                        DAN,
                        VAD,
                        minute,
                        ratio,
                        finetune,
                        batch,
                        lr,
                        input_BF,
                        "transformer",
                        *transformer_wer_list,
                    ]
                )
