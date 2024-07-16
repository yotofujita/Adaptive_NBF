import torch
import torchaudio

from opt_einsum import contract

from utils.import_utils import instantiate
from utils.lightning_utils import BaseModule
from utils.FastMNMF2 import FastMNMF2
from utils.sp_utils import calc_sv


class FastMNMF(BaseModule):
    def __init__(self, delay=3, tap=5, n_srcs=2, lr=1e-3, skip_nan_grad=False, eval_asr=False, **kwargs):
        super().__init__(lr=lr, skip_nan_grad=skip_nan_grad, eval_asr=eval_asr)
        
        self.delay = delay
        self.tap = tap
        
        self.n_srcs = n_srcs
        
        self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None)
        self.istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256)
        
        self.separater = FastMNMF2(
            n_source=n_srcs+1,
            n_basis=16,
            device="cuda",
            init_SCM="twostep",
            n_bit=64,
            algo="IP",
            n_iter_init=30,
            g_eps=5e-2,
        )

    def step(self, batch):
        mix, _, doa, mic_shape = batch  # [B, M, S], [B, M, S], [B,]

        stft = self.stft.to(self.device)
        istft = self.istft.to(self.device)

        # STFT
        mix_stft = stft(mix)  # [B, M, F, T]

        # Separation
        sep_stft = []
        for mx, d, mc in zip(mix_stft, doa, mic_shape):
            sep_stft.append(self.separate(mx, d, mc))
        sep_stft = torch.stack(sep_stft, dim=0)  # [B, F, T]

        # Inverse STFT
        sep = istft(sep_stft)  # [B, S]
        
        return sep, sep_stft

    def separate(self, mix_stft, doa, mic_shape):
        self.separater.load_spectrogram(mix_stft.permute((1, 2, 0)), sample_rate=16000)
        est = self.separater.solve(n_iter=200)  # [N, F, T]
    
        G = self.separater.get_SCM()  # [N, F, M, M]
        _, v = torch.linalg.eigh(G)  # [N, F, M], [N, F, M, M]
        sv = calc_sv(doa, mic_shape, est.shape[1]).to(G.device).to(G.dtype)  # [F, M]
        l = (contract("fl,nflm->nfm", sv, v[:, :, :, :-1]).abs()**2).sum(dim=-1).sum(dim=-1)  # [N]
        target_idx = torch.argmin(l).item()
        
        return est[target_idx]
        

if __name__ == "__main__":
    from omegaconf import OmegaConf 
    cfg1 = OmegaConf.create()
    cfg1.n_mics = 7
    cfg1.batchsize = 5
    
    device = "cuda"
    module = FastMNMF().to(device).to(torch.float64)
    mix = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    src = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    doa = torch.rand((cfg1.batchsize), dtype=torch.float64).to(device)
    mic_shape = torch.rand((cfg1.batchsize, 2, cfg1.n_mics), dtype=torch.float64).to(device)
    sep = module.step((mix, src, doa, mic_shape))
    