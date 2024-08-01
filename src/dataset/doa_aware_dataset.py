import os
import torch 
import torchaudio

from utils.data_utils import read_manifest, scatter_indices
from utils.lightning_utils import BaseDataModule


class LightningDataModule(BaseDataModule):
    def __init__(self, batch_size, num_workers, tr_path, vl_path, tt_path, n_gpus=-1, tr_size=-1, vl_size=-1, tt_size=-1, **kwargs):
        super().__init__(batch_size, num_workers)
        
        self.n_gpus = n_gpus
        
        self.tr_path = tr_path
        self.vl_path = vl_path
        self.tt_path = tt_path
        
        self.tr_size = tr_size
        self.vl_size = vl_size
        self.tt_size = tt_size

    def setup(self, stage=None):
        if stage == "fit":
            if self.n_gpus < 2:
                self.train_dataset = Dataset(self.tr_path, size=self.tr_size)
                self.valid_dataset = Dataset(self.vl_path, size=self.vl_size)
            else:
                local_rank, world_size = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
                self.train_dataset = Dataset(self.tr_path, rank=local_rank, world_size=world_size, size=self.tr_size)
                self.valid_dataset = Dataset(self.vl_path, rank=local_rank, world_size=world_size, size=self.vl_size)

        if stage == "test":
            if self.n_gpus < 2:
                self.test_dataset = Dataset(self.tt_path, size=self.tt_size)
            else:
                local_rank, world_size = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
                self.test_dataset = Dataset(self.tt_path, rank=local_rank, world_size=world_size, size=self.tt_size)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, rank=0, world_size=1, size=-1):
        self.size = size
        
        _manifest = read_manifest(manifest_path)
        
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
    

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, manifest_path, rank=0, world_size=1, size=-1):
#         _manifest = read_manifest(manifest_path)
        
#         indices = scatter_indices(len(_manifest), rank, world_size)
#         manifest = []
#         for idx in indices:
#             manifest.append(_manifest[idx])
#         del _manifest
        
#         datadict = {"mix": [], "src": [], "doa": []}
#         for m in tqdm(manifest):
#             mix, _ = torchaudio.load(m["mix_path"])
#             datadict["mix"].append(mix)
#             src, _ = torchaudio.load(m["source_path"])
#             datadict["src"].append(src.flatten())
#             datadict["doa"].append(m["spk_doa"])
        
#         self.dataset = pd.DataFrame(datadict)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         mix = self.dataset.iloc[idx]["mix"]
#         src = self.dataset.iloc[idx]["src"]
#         doa = self.dataset.iloc[idx]["doa"]
        
#         return mix, src, doa


if __name__ == "__main__":
    manifest_path = "/n/work1/fujita/manifest/libri_mix_demand_dual_2sec_1m_7ch_short.manifest"
    datamodule = LightningDataModule(2, 1, manifest_path, manifest_path, manifest_path)
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()
    for mix, src, spk_doa, mic_shape in dataloader:
        print(mix.shape, mix.dtype)
        print(src.shape, src.dtype)
        print(spk_doa.shape, spk_doa.dtype)
        print(mic_shape.shape, mic_shape.dtype)
        break