import json
import random
import pandas as pd

import torch 
import torchaudio

from utils.data_utils import read_manifest
from utils.lightning_utils import BaseDataModule
from src.modules.iter_wpe_fastmnmf_doaest import FastMNMFDOAEst


class LightningDataModule(BaseDataModule):
    def __init__(self, batch_size, num_workers, json_path, manifest_path, duration=30.0, n_srcs=5, total_s=60, test_duration=2.0, n_gpus=-1, size=-1, **kwargs):
        super().__init__(batch_size, num_workers)
        
        self.n_srcs = n_srcs
        self.total_s = total_s 
        self.test_duration = test_duration
        
        self.n_gpus = n_gpus
        
        self.json_path = json_path
        self.manifest_path = manifest_path
        self.duration = duration
        
        self.size = size

    def setup(self, stage=None):
        if stage == "fit":
            if self.n_gpus < 2:
                self.train_dataset = TrainDataset(self.json_path, self.manifest_path, duration=self.duration, n_srcs=self.n_srcs, total_s=self.total_s, size=self.size)
                self.valid_dataset = self.train_dataset
            else:
                print("Not supporting multiple GPUs.")
                assert False

        if stage == "test":
            if self.n_gpus < 2:
                self.test_dataset = TestDataset(self.json_path, duration=self.test_duration, size=self.size)
            else:
                print("Not supporting multiple GPUs.")
                assert False



class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, manifest_path, duration=30.0, sr=16000, total_s=60, n_srcs=5, threshold=5.0, size=-1):
        assert total_s <= 240
        
        self.size = size
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        mix, _ = torchaudio.load(data["mix_path"]); M, _ = mix.shape
        mic_shape = torch.tensor(data["mic_shape"]).T
        
        mix = mix[:, :int(total_s*sr)]
    
        unit_n_samples = int(duration * sr)
    
        mix = mix[:, :mix.shape[-1]-mix.shape[-1]%(unit_n_samples)]
    
        mix_blocked = mix.reshape(M, mix.shape[-1]//(unit_n_samples), unit_n_samples)
        mix_blocked = mix_blocked.permute(1, 0, 2)  # [B, M, S]
        B, *_ = mix_blocked.shape
        
        # apply fastmnmf 
        print("Apply FastMNMF to obtain pseudo target speech.")
        model = FastMNMFDOAEst(n_srcs=n_srcs, threshold=threshold).to("cuda")
        with torch.inference_mode():
            speech_blocked, doa_blocked = model.step((mix_blocked.to("cuda"), None, None, [mic_shape.to("cuda").to() for _ in range(B)]))
        
        # reshape to the duration of pretraining dataset
        _manifest = read_manifest(manifest_path)
        unit_n_samples = int(_manifest[0]["duration"] * sr)
        
        self.df = pd.DataFrame()
        for mx, sp, doa in zip(mix_blocked, speech_blocked, doa_blocked):
            N = sp.shape[0]
            n_unit = sp.shape[-1]//(unit_n_samples)
            
            sp = sp[:, None].expand(N, M, sp.shape[1])  # [N, M, S]
            sp = sp.reshape(N, M, n_unit, unit_n_samples) 
            sp = sp.permute((0, 2, 1, 3)).flatten(0, 1).cpu()  # [B', M, S]
            
            mx = mx[None, :].expand(N, M, mx.shape[-1])  # [N, M, S]
            mx = mx.reshape(N, M, n_unit, unit_n_samples)
            mx = mx.permute((0, 2, 1, 3)).flatten(0, 1)  # [B', M, S]
            
            doa = doa[:, None].expand(N, n_unit).flatten().cpu()
        
            self.df = pd.concat((self.df, pd.DataFrame({
                "mix": [mb for mb in mx],
                "src": [sb for sb in sp.cpu()],
                "spk_doa": [d for d in doa.cpu()],
                "mic_shape": [mic_shape for _ in range(mx.shape[0])]
            })))
        
        B = len(self.df)
        
        # merge with pretraining data
        random.seed(0); random.shuffle(_manifest)
        manifest = _manifest[:B]
        for m in manifest:
            self.df = pd.concat((self.df, pd.DataFrame({
                "mix": [torchaudio.load(m["mix_path"])[0]],
                "src": [torchaudio.load(m["source_path"])[0]],
                "spk_doa": [torch.tensor(m["spk_doa"])],
                "mic_shape": [torch.tensor(m["mic_shape"]).T]
            })))

    def __len__(self):
        return len(self.df) if self.size < 1 else self.size

    def __getitem__(self, idx):
        mix = self.df.iloc[idx]["mix"]
        src = self.df.iloc[idx]["src"]
        spk_doa = self.df.iloc[idx]["spk_doa"]
        mic_shape = self.df.iloc[idx]["mic_shape"]
        
        return mix.to(torch.float64), src.to(torch.float64), spk_doa.to(torch.float64), mic_shape.to(torch.float64)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, duration=2.0, sr=16000, size=-1):
        self.size = size
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        mix, _ = torchaudio.load(data["mix_path"]); M, _ = mix.shape
        speech, _ = torchaudio.load(data["source_paths"][0])
        spk_doa = torch.tensor(data["spk_doas"][0])
        mic_shape = torch.tensor(data["mic_shape"]).T

        mix = mix[:, int(240*sr):]
        speech = speech[:, int(240*sr):]
        
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
        self.asr_target = data["texts_test"]    

    def __len__(self):
        return len(self.df) if self.size < 1 else self.size

    def __getitem__(self, idx):
        mix = self.df.iloc[idx]["mix"]
        src = self.df.iloc[idx]["src"]
        spk_doa = self.df.iloc[idx]["spk_doa"]
        mic_shape = self.df.iloc[idx]["mic_shape"]
        
        return mix.to(torch.float64), src.to(torch.float64), spk_doa.to(torch.float64), mic_shape.to(torch.float64), self.asr_target



if __name__ == "__main__":
    json_path = "/n/work1/fujita/json/libri_mix_demand_test_n_spks=4_rt60=0.5_noise-snr=30.0/2.json"
    manifest_path = "/n/work1/fujita/manifest/libri_mix_demand_dual_2sec_20h_7ch_train.manifest"

    # datamodule = LightningDataModule(2, 1, json_path, json_path)
    # datamodule.setup(stage="test")
    # dataloader = datamodule.test_dataloader()
    # print(len(dataloader.dataset))
    # for mix, src, spk_doa, mic_shape, asr_target in dataloader:
    #     print(mix.shape, mix.dtype)
    #     print(src.shape, src.dtype)
    #     print(spk_doa.shape, spk_doa.dtype)
    #     print(mic_shape.shape, mic_shape.dtype)
    #     print(asr_target[0][:100])
    #     break

    datamodule = LightningDataModule(2, 1, json_path, manifest_path)
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()
    print(len(dataloader.dataset))
    for mix, src, spk_doa, mic_shape in dataloader:
        print(mix.shape, mix.dtype)
        print(src.shape, src.dtype)
        print(spk_doa.shape, spk_doa.dtype)
        print(mic_shape.shape, mic_shape.dtype)
        break
