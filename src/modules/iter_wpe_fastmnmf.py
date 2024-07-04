import torch
import torchaudio

from opt_einsum import contract

from utils.import_utils import instantiate
from utils.lightning_utils import BaseBSSModule
from utils.FastMNMF2 import FastMNMF2


class IterWPEFastMNMF(BaseBSSModule):
    def __init__(self, delay=3, tap=5, n_srcs=2, lr=1e-3, skip_nan_grad=False, **kwargs):
        super().__init__(lr, skip_nan_grad)
        
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
        mix, *_ = batch  # [B, M, S], [B, M, S], [B,]

        stft = self.stft.to(self.device)
        istft = self.istft.to(self.device)

        # STFT
        mix_stft = stft(mix)  # [B, M, F, T]

        # Dereverberation
        mix_stft_dereverberated = self.dereverberate(mix_stft, delay=self.delay, tap=self.tap)  # [B, F, T]
        
        # Separation
        sep_stft = []
        for msd in mix_stft_dereverberated:
            sep_stft.append(self.separate(msd))
        sep_stft = torch.stack(sep_stft, dim=0)  # [B, N, F, T]

        # Inverse STFT
        sep = istft(sep_stft)  # [N, S]
        
        return sep, sep_stft
    
    @staticmethod
    def dereverberate(mix_stft, delay=3, tap=5, eps=1e-4):

        B, M, F, T = mix_stft.shape
        D, K, L = delay, tap, delay+tap
    
        pad = torch.zeros((B, M, F, L-1), device=mix_stft.device)
        mix_padded = torch.concatenate((pad, mix_stft), dim=-1)

        mix_tilde = torch.concatenate([mix_padded[..., (L-i-1):T+L-1-i] for i in range(D, L)], dim=1)  # [B, M*K, F, T]
    
        w = torch.zeros((B, F, M, M*K), device=mix_stft.device, dtype=mix_stft.dtype)

        for _ in range(3):
            est = mix_stft - contract("bfkl, blft -> bkft", w.conj(), mix_tilde)
            sigma = torch.mean(torch.abs(est) ** 2, dim=1)
            sigma = sigma[:, None].expand((B, M*K, F, T))  # [B, M*K, F, T]
    
            R = contract("bkft,blft->bklf", mix_tilde / (sigma + eps), mix_tilde.conj())  # [B, M*K, M*K, F]
            R = R.permute((0, 3, 1, 2))  # [B, F, M*K, M*K]   
    
            P = contract("bkft,blft->bklf", mix_tilde / (sigma + eps), mix_stft.conj())  # [B, M*K, M*K, F]
            P = P.permute((0, 3, 1, 2))  # [B, F, M*K, M]   

            w = torch.linalg.solve(R, P)  # [B, F, M*K, M]
            w = w.permute((0, 1, 3, 2))  # [B, F, M, M*K]

            est = mix_stft - contract("bfkl,blft->bkft", w.conj(), mix_tilde)
        
        return est   

    def separate(self, mix_stft):
        self.separater.load_spectrogram(mix_stft.permute((1, 2, 0)), sample_rate=16000)
        est = self.separater.solve(n_iter=100)  # [N, F, T]
        
        return est
        

if __name__ == "__main__":
    from omegaconf import OmegaConf 
    cfg1 = OmegaConf.create()
    cfg1.n_mics = 7
    cfg1.batchsize = 5
    
    device = "cuda"
    module = IterWPEFastMNMF().to(device).to(torch.float64)
    mix = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    src = torch.rand((cfg1.batchsize, cfg1.n_mics, 32000), dtype=torch.float64).to(device)
    doa = torch.rand((cfg1.batchsize), dtype=torch.float64).to(device)
    mic_shape = torch.rand((cfg1.batchsize, 2, cfg1.n_mics), dtype=torch.float64).to(device)
    metrics = module.test_step((mix, src, doa, mic_shape), None)
    for key in metrics.keys():
        print(key, metrics[key][0].item())
    