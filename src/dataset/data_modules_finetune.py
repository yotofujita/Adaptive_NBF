import os
import h5py 
import numpy as np
from tqdm import tqdm
import glob
import bz2
import pickle as pkl
import torchaudio

import torch 
import pytorch_lightning as pl

import pandas as pd


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print("loading finetuning dataset")
        self.train_dataset = Dataset(step="train")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tuning_data_dir="./data/FastMNMF-M=5-Ns=5-Nn=0-K=8-it=100-itIVA=50-bit=64_reset_wpe/",
        tuning_data_minute=3,
        window_len=48128,
        step=["train", "valid"][0],
    ):
        self.step = step
        self.window_len = window_len
        self.shift = int(self.window_len * (1 - 0.25))

        self.tuning_mix = []
        self.tuning_sep = []
        self.tuning_angle = []
        second_per_wav = 9
        n_sample_per_wav = window_len * 3
        self.tuning_data_dir = tuning_data_dir
        self.tuning_data_minute = tuning_data_minute
        self.tuning_data_ratio = 0.5
        self.n_mic = 5
        
        datadict = {"image": [], "mixture": [], "angle": []}
        
        fname_list = sorted(glob.glob(self.tuning_data_dir + "/sep*.wav"))
        
        n_wav = int(np.ceil(self.tuning_data_minute * 60 / second_per_wav))
        np.random.seed(1)
        perm = np.random.permutation(len(fname_list))

        sep_fname_list = np.array(fname_list)[perm[:n_wav]]
        for idx, sep_fname in enumerate(tqdm(sep_fname_list)):
            sep, _ = torchaudio.load(sep_fname)
            mix_fname = sep_fname.replace("sep", "mix").split("_azim")[0] + ".wav"
            mix, _ = torchaudio.load(mix_fname)
            angle = int(sep_fname.split("azim=")[1].split("_")[0])
            angle = int(np.round(angle / 5))

            if idx == len(sep_fname_list) - 1:  # If last iteration
                second_remainder = self.tuning_data_minute * 60 - idx * second_per_wav
                n_sample_last = int(second_remainder * n_sample_per_wav / second_per_wav)
                sep = sep[:, :n_sample_last]
                mix = mix[:, :n_sample_last]

            n_window = (sep.shape[1] - self.window_len) // self.shift + 1
            if n_window == 0:
                continue
            
            mix = torch.as_strided(mix, size=(self.n_mic, n_window, window_len), stride=(n_sample_per_wav, self.shift, 1))
            datadict["mixture"].extend(np.array(mix.permute(1, 0, 2)))
            sep = torch.as_strided(sep, size=(n_window, window_len), stride=(self.shift, 1))
            datadict["image"].extend(np.array(sep))
            datadict["angle"].extend([np.int16([1, angle]) for _ in range(n_window)])

        self.root = "./data/Hololens2_SimData_WSJ0_CHiME3"
        hdf5_path = f"{self.root}/h5/2spk/{self.step}/wsj0_chime3_{self.step}_{self.window_len}_0.25-full.h5"
        
        if step == "train":
            n_trains = int((1 - self.tuning_data_ratio) / self.tuning_data_ratio * len(datadict["image"]))
            with h5py.File(hdf5_path, "r") as f:
                perm = np.random.permutation(len(f))
                for key in tqdm(perm[:n_trains]):
                    datadict["image"].append(pkl.loads(bz2.decompress(f[f"{key}/image"][:])))
                    datadict["mixture"].append(pkl.loads(bz2.decompress(f[f"{key}/mixture"][:])))
                    datadict["angle"].append(f[f"{key}/angle"][:])
        
        self.dataset = pd.DataFrame(datadict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):       
        mixture = self.dataset.iloc[idx]["mixture"]
        image = self.dataset.iloc[idx]["image"]
        
        mixture = torch.as_tensor(mixture)
        image = torch.as_tensor(image)
        
        elev_idx, azim_idx = self.dataset.iloc[idx]["angle"]
        elev_idx = torch.as_tensor(elev_idx)
        azim_idx = torch.as_tensor(azim_idx)
        
        return mixture, image, elev_idx, azim_idx
