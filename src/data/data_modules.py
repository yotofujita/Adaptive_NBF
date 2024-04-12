import numpy as np
import h5py 
import json 
from tqdm import tqdm

import torch 
import pytorch_lightning as pl

import pandas as pd

from data.utils import HDF5RawDataset
from utils.utility import make_feature


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=2, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MultichannelDataset(step="train")
        self.valid_dataset = MultichannelDataset(step="valid")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )


class MultichannelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        window_len=48128,
        step=["train", "valid"][0],
    ):
        self.step = step
        self.window_len = window_len
        self.shift = int(self.window_len * (1 - 0.25))

        with h5py.File("./data/SV_for_HL2.h5", "r") as f: 
            # SV_EAFM: [E, A, F, M], azim: [A,], elev: [E,]
            self.SV_EAFM = torch.from_numpy(np.asarray(f["SV_EAFM"], dtype=np.complex64))
            norm_EAF = torch.linalg.norm(self.SV_EAFM, axis=3)
            self.SV_EAFM /= norm_EAF[..., None]

        self.root = "./data/Hololens2_SimData_WSJ0_CHiME3"
        fname = f"{self.root}/{self.step}_wsj0_chime3_2spk.json"

        self.flist = json.load(open(fname, "r"))
        self.id_list = np.random.permutation(list(self.flist.keys()))
        
        datadict = {"image": [], "mixture": [], "angle": []}
        path = f"{self.root}/h5/2spk/{self.step}/wsj0_chime3_{self.step}_{self.window_len}_0.25-full.h5"
        with h5py.File(path, "r") as f:
            for key in tqdm(f.keys()):
                datadict["image"].append(f[f"{key}/image"][:])
                datadict["mixture"].append(f[f"{key}/mixture"][:])
                datadict["angle"].append(f[f"{key}/angle"][:])
        
        self.dataset = pd.DataFrame(datadict)

        # self.dataset = {"image": [], "mixture": [], "angle": []}
        # for i in range(24):
        #     idx = i + 1
        #     path = f"{self.root}/h5/2spk/{self.step}/wsj0_chime3_{self.step}_{self.window_len}_0.25-{idx}.h5"
        #     print("Start loading ", path)
        #     with h5py.File(path, "r") as f:
        #         print(f["image"][:].shape)
        #         self.dataset["image"].append(torch.as_tensor(f["image"][:]))
        #         self.dataset["mixture"].append(torch.as_tensor(f["mixture"][:]))
        #         self.dataset["angle"].append(torch.as_tensor(f["angle"][:]))
        
        # self.dataset["image"] = torch.cat(self.dataset["image"])
        # self.dataset["mixture"] = torch.cat(self.dataset["mixture"])
        # self.dataset["angle"] = torch.cat(self.dataset["angle"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):        
        mixture_MTF = self.dataset.iloc[idx]["mixture"]  # [M, T, F]
        
        elev_idx, azim_idx = self.dataset.iloc[idx]["angle"]
        
        image = self.dataset.iloc[idx]["image"]  # [S,]

        # mixture = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)(mixture).permute(
        #     0, 1, 3, 2
        # )  # B x M x T x F

        feature, doa = make_feature(
            mixture_MTF,
            self.SV_EAFM[elev_idx, azim_idx],
            azim_idx
        )
        
        return feature, image, mixture_MTF.to(torch.cdouble), doa