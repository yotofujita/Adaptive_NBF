import os
import json
import torch 
import torchaudio

from utils.data_utils import read_manifest, scatter_indices
from utils.lightning_utils import BaseDataModule


class LightningDataModule(BaseDataModule):
    def __init__(self, batch_size, num_workers, tt_path, tt_size=-1, n_gpus=-1, **kwargs):
        super().__init__(batch_size, num_workers)
        
        self.n_gpus = n_gpus

        self.tt_path = tt_path
        self.tt_size = tt_size
        
    def setup(self, stage=None):
        if stage == "fit":
            print("This dataset is only for test.")
            assert False

        if stage == "test":
            if self.n_gpus < 2:
                self.test_dataset = Dataset(self.tt_path, size=self.tt_size)
            else:
                print("Not supporting multiple GPUs.")
                assert False


class Dataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, rank=0, world_size=1, size=-1):
        self.size = size
        
        with open(self.tt_path, "r") as f:
            data = json.load(f)
        
        indices = scatter_indices(len(_manifest), rank, world_size)
        self.manifest = []
        for idx in indices:
            self.manifest.append(_manifest[idx])
        del _manifest

    def __len__(self):
        return len(self.manifest) if self.size < 1 else self.size

    def __getitem__(self, idx):
        mix, _ = torchaudio.load(self.manifest[idx]["mix_path"])
        src, _ = torchaudio.load(self.manifest[idx]["source_path"])
        spk_doa = torch.tensor(self.manifest[idx]["spk_doa"])
        mic_shape = torch.tensor(self.manifest[idx]["mic_shape"]).T
        
        return mix.to(torch.float64), src.to(torch.float64), spk_doa.to(torch.float64), mic_shape.to(torch.float64)


if __name__ == "__main__":
    json_path = "/n/work1/fujita/json/libri_mix_demand_test_n_spks=4_room_size=small_rt60=1.5_noise-snr=5.0_src-distance=1.5.json"
    datamodule = LightningDataModule(2, 1, json_path)
    datamodule.setup(stage="test")
    dataloader = datamodule.train_dataloader()
    for mix, src, spk_doa, mic_shape in dataloader:
        print(mix.shape, mix.dtype)
        print(src.shape, src.dtype)
        print(spk_doa.shape, spk_doa.dtype)
        print(mic_shape.shape, mic_shape.dtype)
        break