#! /usr/bin/env python3
# coding:utf-8

import numpy as np
import csv
import matplotlib.pyplot as plt
import itertools as it
import matplotlib as mat
import seaborn as sns

# sns.set_style("whitegrid")
sns.set_style("ticks")
mat.rcParams['pdf.fonttype'] = 42
mat.rcParams['ps.fonttype'] = 42
del mat.font_manager.weight_dict['roman']
# mat.font_manager._rebuild()
mat.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# csv_fname = "wer2.csv"
csv_fname = "wer_incremental_MVDR_wpe.csv"

n_epoch_list = [1, 3, 5, 10]
# n_epoch_list = [3]
intv_minute_list = [3]
minute_list = [12]
ratio_list = [0.75]

filter_param = ["n_epoch", "intv_minute", "ratio", "minute"]
color_list = ["#2c3e50", "#3498db", "#e74c3c", "#1abc9c"]
linestyle_list = ["solid", "dashed", "dotted", "dashdot"]
marker_list = ["o", "^", "*", "s"]

count = 0
plt.figure(figsize=(25, 15))
with open(csv_fname, "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        (
            model_name,
            BF,
            SV,
            DAN,
            VAD,
            n_epoch,
            intv_minute,
            minute,
            ratio,
            finetune,
            batch_size,
            lr,
            asr,
        ) = row[:13]
        wer_est = [float(wer) for wer in row[13:]]

        if not eval(BF):
            continue

        n_epoch, intv_minute, minute, ratio = int(n_epoch), int(intv_minute), int(minute), float(ratio)

        # label = f"epoch={n_epoch}  intv={intv_minute}min  total={minute}min"
        label = f"epoch={n_epoch if len(str(n_epoch)) == 2 else f'  {n_epoch}'}"

        if len(wer_est) > 1:
            if np.array([eval(param) in eval(param + "_list") for param in filter_param]).all():
                color = color_list[n_epoch_list.index(n_epoch)]
                linestyle = linestyle_list[n_epoch_list.index(n_epoch)]
                marker = marker_list[n_epoch_list.index(n_epoch)]

                ave = np.mean(wer_est[1:])
                min = np.min(wer_est[1:])
                print(f"n_epoch = {n_epoch}  average = {ave} min = {min}")
                plt.hlines(ave, 0, 49, color=color, linestyle=linestyle, linewidth=3)

                n_data = len(wer_est)
                plt.plot([0, *range(intv_minute, 49, intv_minute)], wer_est, marker, color=color, linestyle=linestyle, linewidth=4, markersize=25, label=label)
                count += 1
    plt.xlabel("The elapsed time [min]", fontsize=60)
    plt.ylabel("WER [%]", fontsize=60)
    plt.ylim(20, 36.5)
    plt.xlim(-0.3, 48.5)
    plt.xticks([0, *range(intv_minute, 49, intv_minute)], fontsize=50)
    plt.yticks(fontsize=50)
    plt.legend(fontsize=58)
    plt.tight_layout()
    plt.savefig("incremental.pdf")
    plt.show()

