import os
import h5py 
import numpy as np
from tqdm import tqdm
import bz2
import pickle as pkl

import torch 
import pytorch_lightning as pl

import pandas as pd

from numpy.random import Generator, PCG64

from utils import read_manifest


def scatter_indices(dataset_size, rank, world_size, permute_fn=Generator(PCG64(0)).permutation):
    total_size = int(np.ceil(dataset_size / world_size)) * world_size

    indices = permute_fn(np.arange(dataset_size))
    repeated_indices = np.concatenate([indices, indices[:total_size - dataset_size]])

    split_indices = np.split(repeated_indices, world_size)

    return split_indices[rank]


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, tr_path, vl_path, n_gpus=-1, tr_size=-1, vl_size=-1, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # ignore=["overwrite", "hoge"]
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_gpus = n_gpus
        
        self.tr_path = tr_path
        self.vl_path = vl_path
        
        self.tr_size = tr_size
        self.vl_size = vl_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print(self.n_gpus)
        if self.n_gpus < 2:
            self.train_dataset = Dataset(step="train", path=self.tr_path, size=self.tr_size)
            self.valid_dataset = Dataset(step="valid", path=self.vl_path, size=self.vl_size)
        else:
            local_rank, world_size = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
            self.train_dataset = Dataset(step="train", path=self.tr_path, rank=local_rank, world_size=world_size, size=self.tr_size)
            self.valid_dataset = Dataset(step="valid", path=self.vl_path, rank=local_rank, world_size=world_size, size=self.vl_size)

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


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        window_len=48128,
        step=["train", "valid"][0],
        rank=0,
        world_size=1,
        size=-1
    ):
        self.step = step
        self.window_len = window_len
        self.shift = int(self.window_len * (1 - 0.25))
        
        datadict = {"image": [], "mixture": [], "angle": []}
        with h5py.File(path, "r") as f:
            indices = scatter_indices(len(f), rank, world_size)
            for i, key in enumerate(tqdm(indices)):
                datadict["image"].append(f[f"{key}/image"][:])
                datadict["mixture"].append(f[f"{key}/mixture"][:])
                datadict["angle"].append(f[f"{key}/angle"][:])
                
                if i == size:
                    break
        
        self.dataset = pd.DataFrame(datadict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):       
        mixture = self.dataset.iloc[idx]["mixture"]
        image = self.dataset.iloc[idx]["image"]
        
        mixture = pkl.loads(bz2.decompress(mixture))
        image = pkl.loads(bz2.decompress(image))
        
        mixture = torch.as_tensor(mixture)
        image = torch.as_tensor(image)
        
        elev_idx, azim_idx = self.dataset.iloc[idx]["angle"]
        elev_idx = torch.as_tensor(elev_idx)
        azim_idx = torch.as_tensor(azim_idx)
        
        return mixture, image, elev_idx, azim_idx
