import os
import json
import pandas as pd

import torch 
import torchaudio

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
    def __init__(self, json_path, duration=6.0, sr=16000, rank=0, world_size=1, size=-1):
        self.size = size
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        mix, _ = torchaudio.load(data["mix_path"]); M, _ = mix.shape
        speech, _ = torchaudio.load(data["source_path"])
        spk_doa = torch.tensor(data["spk_doa"])
        mic_shape = torch.tensor(data["mic_shape"]).T
        
        unit_n_samples = int(duration * sr)
        
        mix = mix[:, :mix.shape[-1]-mix.shape[-1]%(unit_n_samples)]
        speech = speech[:, :mix.shape[-1]]

        mix_blocked = mix.reshape(M, mix.shape[-1]//(unit_n_samples), unit_n_samples)
        mix_blocked = mix_blocked.permute(1, 0, 2)
        speech_blocked = speech.reshape(M, speech.shape[-1]//(unit_n_samples), unit_n_samples)
        speech_blocked = speech_blocked.permute(1, 0, 2)
        
        B, *_ = mix_blocked.shape
        self.df = pd.DataFrame({
            "mix": [mb for mb in mix_blocked],
            "src": [sb for sb in speech_blocked],
            "spk_doa": [spk_doa for _ in range(B)],
            "mic_shape": [mic_shape for _ in range(B)]
        })
        self.asr_target = data["text"]

    def __len__(self):
        return len(self.df) if self.size < 1 else self.size

    def __getitem__(self, idx):
        mix = self.df.iloc[idx]["mix"]
        src = self.df.iloc[idx]["src"]
        spk_doa = self.df.iloc[idx]["spk_doa"]
        mic_shape = self.df.iloc[idx]["mic_shape"]
        
        return mix.to(torch.float64), src.to(torch.float64), spk_doa.to(torch.float64), mic_shape.to(torch.float64), self.asr_target


if __name__ == "__main__":
    json_path = "/n/work1/fujita/json/libri_mix_demand_test_n_spks=4_room_size=small_rt60=1.5_noise-snr=5.0_src-distance=1.5.json"
    datamodule = LightningDataModule(2, 1, json_path)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()
    for mix, src, spk_doa, mic_shape in dataloader:
        print(mix.shape, mix.dtype)
        print(src.shape, src.dtype)
        print(spk_doa.shape, spk_doa.dtype)
        print(mic_shape.shape, mic_shape.dtype)
        break