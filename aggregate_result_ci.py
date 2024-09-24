import re
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams


def plot_row(ax, df, y1, y2, var="n_spks", domain=[2, 3, 4], put_xlabel=True, put_title=True):
    x = [0, 30, 60, 120, 240]
    fontsize = 20
    labelsize = 18
    ax[0].plot(x, df[df[var]==domain[0]]["WER"], color=(0, 0, 1, 1), marker="o", linestyle="-", label=f"{var}={domain[0]}")
    ax[0].plot(x, df[df[var]==domain[1]]["WER"], color=(0, 1, 0, 1), marker="^", linestyle="--", label=f"{var}={domain[1]}")
    ax[0].plot(x, df[df[var]==domain[2]]["WER"], color=(1, 0, 0, 1), marker="d", linestyle=":", label=f"{var}={domain[2]}")
    ax[0].fill_between(x, y1[y1[var]==domain[0]]["WER"], y2[y2[var]==domain[0]]["WER"], color=(0, 0, 1, 1), alpha=0.15)
    ax[0].fill_between(x, y1[y1[var]==domain[1]]["WER"], y2[y2[var]==domain[1]]["WER"], color=(0, 1, 0, 1), alpha=0.15)
    ax[0].fill_between(x, y1[y1[var]==domain[2]]["WER"], y2[y2[var]==domain[2]]["WER"], color=(1, 0, 0, 1), alpha=0.15)
    ax[0].set_ylim(0, 1)
    if put_xlabel:
        ax[0].set_xlabel('Amount of finetune data [sec]', fontsize=fontsize)
    if put_title:
        ax[0].set_title(r'$\text{WER}\downarrow$', fontsize=fontsize)
    ax[0].set_xticks(x)
    ax[0].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[0].tick_params(axis='both', which='minor', labelsize=labelsize)
    ax[0].legend()
    ax[1].plot(x, df[df[var]==domain[0]]["SDR"], color=(0, 0, 1, 1), marker="o", linestyle="-", label=f"{var}={domain[0]}")
    ax[1].plot(x, df[df[var]==domain[1]]["SDR"], color=(0, 1, 0, 1), marker="^", linestyle="--", label=f"{var}={domain[1]}")
    ax[1].plot(x, df[df[var]==domain[2]]["SDR"], color=(1, 0, 0, 1), marker="d", linestyle=":", label=f"{var}={domain[2]}")
    ax[1].fill_between(x, y1[y1[var]==domain[0]]["SDR"], y2[y2[var]==domain[0]]["SDR"], color=(0, 0, 1, 1), alpha=0.15)
    ax[1].fill_between(x, y1[y1[var]==domain[1]]["SDR"], y2[y2[var]==domain[1]]["SDR"], color=(0, 1, 0, 1), alpha=0.15)
    ax[1].fill_between(x, y1[y1[var]==domain[2]]["SDR"], y2[y2[var]==domain[2]]["SDR"], color=(1, 0, 0, 1), alpha=0.15)
    ax[1].set_ylim(-2, 9)
    if put_xlabel:
        ax[1].set_xlabel('Amount of fine-tune data [sec]', fontsize=fontsize)
    if put_title:
        ax[1].set_title(r'$\text{SDR}\uparrow$', fontsize=fontsize)
    ax[1].set_xticks(x)
    ax[1].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[1].tick_params(axis='both', which='minor', labelsize=labelsize)
    ax[1].legend()
    ax[2].plot(x, df[df[var]==domain[0]]["STOI"], color=(0, 0, 1, 1), marker="o", linestyle="-", label=f"{var}={domain[0]}")
    ax[2].plot(x, df[df[var]==domain[1]]["STOI"], color=(0, 1, 0, 1), marker="^", linestyle="--", label=f"{var}={domain[1]}")
    ax[2].plot(x, df[df[var]==domain[2]]["STOI"], color=(1, 0, 0, 1), marker="d", linestyle=":", label=f"{var}={domain[2]}")
    ax[2].fill_between(x, y1[y1[var]==domain[0]]["STOI"], y2[y2[var]==domain[0]]["STOI"], color=(0, 0, 1, 1), alpha=0.15)
    ax[2].fill_between(x, y1[y1[var]==domain[1]]["STOI"], y2[y2[var]==domain[1]]["STOI"], color=(0, 1, 0, 1), alpha=0.15)
    ax[2].fill_between(x, y1[y1[var]==domain[2]]["STOI"], y2[y2[var]==domain[2]]["STOI"], color=(1, 0, 0, 1), alpha=0.15)
    ax[2].set_ylim(0.6, 1)
    if put_xlabel:
        ax[2].set_xlabel('Amount of fine-tune data [sec]', fontsize=fontsize)
    if put_title:
        ax[2].set_title(r'$\text{STOI}\uparrow$', fontsize=fontsize)
    ax[2].set_xticks(x)
    ax[2].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[2].tick_params(axis='both', which='minor', labelsize=labelsize)
    ax[2].legend()
    ax[3].plot(x, df[df[var]==domain[0]]["PESQ"], color=(0, 0, 1, 1), marker="o", linestyle="-", label=f"{var}={domain[0]}")
    ax[3].plot(x, df[df[var]==domain[1]]["PESQ"], color=(0, 1, 0, 1), marker="^", linestyle="--", label=f"{var}={domain[1]}")
    ax[3].plot(x, df[df[var]==domain[2]]["PESQ"], color=(1, 0, 0, 1), marker="d", linestyle=":", label=f"{var}={domain[2]}")
    ax[3].fill_between(x, y1[y1[var]==domain[0]]["PESQ"], y2[y2[var]==domain[0]]["PESQ"], color=(0, 0, 1, 1), alpha=0.15)
    ax[3].fill_between(x, y1[y1[var]==domain[1]]["PESQ"], y2[y2[var]==domain[1]]["PESQ"], color=(0, 1, 0, 1), alpha=0.15)
    ax[3].fill_between(x, y1[y1[var]==domain[2]]["PESQ"], y2[y2[var]==domain[2]]["PESQ"], color=(1, 0, 0, 1), alpha=0.15)
    ax[3].set_ylim(1.2, 2.5)
    if put_xlabel:
        ax[3].set_xlabel('Amount of fine-tune data [sec]', fontsize=fontsize)
    if put_title:
        ax[3].set_title(r'$\text{PESQ}\uparrow$', fontsize=fontsize)
    ax[3].set_xticks(x)
    ax[3].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[3].tick_params(axis='both', which='minor', labelsize=labelsize)
    ax[3].legend()
    ax[4].plot(x, df[df[var]==domain[0]]["SRMR"], color=(0, 0, 1, 1), marker="o", linestyle="-", label=f"{var}={domain[0]}")
    ax[4].plot(x, df[df[var]==domain[1]]["SRMR"], color=(0, 1, 0, 1), marker="^", linestyle="--", label=f"{var}={domain[1]}")
    ax[4].plot(x, df[df[var]==domain[2]]["SRMR"], color=(1, 0, 0, 1), marker="d", linestyle=":", label=f"{var}={domain[2]}")
    ax[4].fill_between(x, y1[y1[var]==domain[0]]["SRMR"], y2[y2[var]==domain[0]]["SRMR"], color=(0, 0, 1, 1), alpha=0.15)
    ax[4].fill_between(x, y1[y1[var]==domain[1]]["SRMR"], y2[y2[var]==domain[1]]["SRMR"], color=(0, 1, 0, 1), alpha=0.15)
    ax[4].fill_between(x, y1[y1[var]==domain[2]]["SRMR"], y2[y2[var]==domain[2]]["SRMR"], color=(1, 0, 0, 1), alpha=0.15)
    ax[4].set_ylim(5, 10)
    if put_xlabel: 
        ax[4].set_xlabel('Amount of fine-tune data [sec]', fontsize=fontsize)
    if put_title:
        ax[4].set_title(r'$\text{SRMR}\uparrow$', fontsize=fontsize)
    ax[4].set_xticks(x)
    ax[4].tick_params(axis='both', which='major', labelsize=labelsize)
    ax[4].tick_params(axis='both', which='minor', labelsize=labelsize)
    ax[4].legend()


name = "finetune_mask-based-wpd_doa-aware-lstm_iter-wpe-fastmnmf-doaest"
wpd_dir = f"/n/work3/fujita/research/NeuralBF-IROS2022/experiments/{name}_LibriMixDemandTest/"

dirs_none = [p for p in Path(wpd_dir).glob("**/metrics.csv") if re.search("None", str(p))]
dirs_finetune = [p for p in Path(wpd_dir).glob("**/metrics.csv") if not re.search("None", str(p))]

df = pd.DataFrame()

for path in dirs_none:
    csv = pd.read_csv(path)
    try:
        df = pd.concat((df, pd.DataFrame({
            "n_spks": [int(str(path).split("n_spks=")[1].split("_")[0])],
            # "room_size": [str(path).split("room_size=")[1].split("_")[0]],
            "rt60": [float(str(path).split("rt60=")[1].split("_")[0])],
            "SNR": [float(str(path).split("noise-snr=")[1].split("_")[0])],
            # "src_distance": [float(str(path).split("src-distance=")[1].split("_")[0])],
            "total_s": [0],
            "id": [int(str(path).split("id=")[1].split("/")[0])],
            "WER": [csv.iloc[0]["test_WER"].item()],
            "SI-SDR": [csv.iloc[0]["test_SI-SDR"].item()],
            "SDR": [csv.iloc[0]["test_SDR"].item()],
            "STOI": [csv.iloc[0]["test_STOI"].item()],
            "PESQ": [csv.iloc[0]["test_PESQ"].item()],
            "SRMR": [csv.iloc[0]["test_SRMR"].item()]
        })))
    except:
        continue

for path in dirs_finetune:
    csv = pd.read_csv(path)
    try:
        df = pd.concat((df, pd.DataFrame({
            "n_spks": [int(str(path).split("n_spks=")[1].split("_")[0])],
            # "room_size": [str(path).split("room_size=")[1].split("_")[0]],
            "rt60": [float(str(path).split("rt60=")[1].split("_")[0])],
            "SNR": [float(str(path).split("noise-snr=")[1].split("_")[0])],
            # "src_distance": [float(str(path).split("src-distance=")[1].split("_")[0])],
            "total_s": [int(str(path).split("total_s=")[1].split("_")[0])],
            "id": [int(str(path).split("id=")[1].split("/")[0])],
            "WER": [csv[csv.epoch==1]["test_WER"].item()],
            "SI-SDR": [csv[csv.epoch==1]["test_SI-SDR"].item()],
            "SDR": [csv[csv.epoch==1]["test_SDR"].item()],
            "STOI": [csv[csv.epoch==1]["test_STOI"].item()],
            "PESQ": [csv[csv.epoch==1]["test_PESQ"].item()],
            "SRMR": [csv[csv.epoch==1]["test_SRMR"].item()]
        })))
    except:
        continue

# compute 95% confidence interval
z_score = 1.96
mean = df.groupby(["n_spks", "rt60", "SNR", "total_s"]).mean()
sem = df.groupby(["n_spks", "rt60", "SNR", "total_s"]).sem()
margin_of_error = z_score * sem
y1, y2 = mean - margin_of_error, mean + margin_of_error
mean = mean.reset_index()
y1 = y1.reset_index()
y2 = y2.reset_index()

fig, ax = plt.subplots(3, 5, figsize=(26, 12))
# fig.suptitle("Adaptation result based on matching between target DOA and separated signal", size=25)
# fig.suptitle("Adaptation result based on filtering and localization of separated signals", size=25)
# fig.suptitle("Adaptation result based on distance between the WPD prediction and separated speeches", size=25)
# fig.tight_layout()
plt.rcParams.update({'font.size': 15})

default = pd.Series(mean.n_spks==2) & pd.Series(mean.rt60==0.5) & pd.Series(mean.SNR==30.0)
condition = default | pd.Series(mean.n_spks==3) | pd.Series(mean.n_spks==4)
plot_row(ax[0], mean[condition], y1[condition], y2[condition], var="n_spks", domain=[2, 3, 4], put_xlabel=False, put_title=True)
condition = default | pd.Series(mean.rt60==0.8) | pd.Series(mean.rt60==1.2)
plot_row(ax[1], mean[condition], y1[condition], y2[condition], var="rt60", domain=[0.5, 0.8, 1.2], put_xlabel=False, put_title=False)
condition = default | pd.Series(mean.SNR==5.0) | pd.Series(mean.SNR==-5.0)
plot_row(ax[2], mean[condition], y1[condition], y2[condition], var="SNR", domain=[30.0, 5.0, -5.0], put_xlabel=True, put_title=False)
fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.06, hspace=0.1, wspace=0.15)

rcParams['pdf.fonttype'] = 42
fig.savefig(f"{name}_ci.pdf")