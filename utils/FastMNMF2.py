# Based on https://github.com/sekiguchi92/SoundSourceSeparation/blob/master/src_torch/separation/FastMNMF2.py


import sys
import os
import torch
import torchaudio
from opt_einsum import contract
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

EPS = 1e-10
MIC_INDEX = 0


def MultiSTFT(wav_TM: "torch.tensor", n_fft=1024, hop_length=None) -> torch.tensor:
    """
    Multichannel STFT

    Parameters
    ---------
    wav_TM: torch.tensor (T x M) or (T)
    n_fft: int
        The window size (default 1024)
    hop_length: int
        The shift length (default None)
        If None, n_fft // 4

    Returns
    -------
    spec_FTM: np.ndarray (F x T x M) or (F x T)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if wav_TM.ndim == 1:
        wav_TM = wav_TM[:, None]

    return torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)(wav_TM.T).permute(1, 2, 0)


def MultiISTFT(spec, hop_length=None, shape="FTM"):
    """
    Multichannel inverse STFT

    Parameters
    ---------
    spec: torch.tensor
        If shape = 'MFT', (M x F x T) or (F x T).
        If shape = 'FTM', (F x T x M) or (F x T).
    hop_length: int
        The shift length (default None)
        If None, (F-1) * 4
    shape: str
        Shape of the spec. FTM or MFT

    Returns
    -------
    wav_TM: torch.tensor ((M x T') or T')
    """
    if spec.ndim == 2:
        spec = spec[None]
        shape = "MFT"

    if shape == "FTM":
        spec = spec.permute(2, 0, 1)

    _, F, _ = spec.shape
    n_fft = (F - 1) * 2
    if hop_length is None:
        hop_length = n_fft // 4

    return torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)(spec.cpu())


class Base:
    "Base Class for Source Separation Methods"

    def __init__(self, device="cpu", seed=0, n_bit=64):
        torch.manual_seed(seed)
        self.device = device
        self.n_bit = n_bit
        if self.n_bit == 64:
            self.TYPE_FLOAT = torch.float64
            self.TYPE_COMPLEX = torch.complex128
        elif self.n_bit == 32:
            self.TYPE_FLOAT = torch.float32
            self.TYPE_COMPLEX = torch.complex64
        self.method_name = "Base"
        self.save_param_list = ["n_bit"]

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        """load complex spectrogram

        Parameters:
        -----------
            X_FTM: np.ndarray F x T x M
                Spectrogram of observed signals
        """
        self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = torch.as_tensor(X_FTM, dtype=self.TYPE_COMPLEX, device=self.device)
        self.sample_rate = sample_rate
        self.start_idx = 0

    def solve(
        self,
        n_iter=100,
        mic_index=MIC_INDEX,
        init=True,
    ):
        """
        Parameters:
            n_iter: int
            save_dir: str
            save_wav: bool
                Save the separated signals only after the last iteration
            save_wav_all: bool
                Save the separated signals at every 'interval_save' iterations
            save_param: bool
            save_param_all: bool
            save_likelihood: bool
            interval_save: int
                interval of saving wav, parameter, and log-likelihood
        """
        self.n_iter = n_iter
        if init:
            self.init_source_model()
            self.init_spatial_model()

        # print(f"Update {self.method_name}-{self}  {self.n_iter-self.start_idx} times ...")

        self.log_likelihood_dict = {}
        for self.it in range(self.start_idx, n_iter):
            self.update()

        self.separate(mic_index=mic_index)
        
        return self.separated_spec

    def save_to_wav(self, spec, save_fname, shape=["FTM", "MFT"][0]):
        assert not torch.isnan(spec).any(), "spec includes NaN"
        separated_signal = MultiISTFT(spec, shape=shape).to(torch.float32)
        torchaudio.save(save_fname, separated_signal, self.sample_rate)

    def save_param(self, fname):
        import h5py

        with h5py.File(fname, "w") as f:
            for param in self.save_param_list:
                data = getattr(self, param)
                f.create_dataset(param, data=data.cpu())
            f.flush()

    def load_param(self, fname):
        import h5py

        with h5py.File(fname, "r") as f:
            for key in f.keys():
                data = torch.as_tensor(f[key], device=self.device)
                setattr(self, key, data)

            if "n_bit" in f.keys():
                if self.n_bit == 64:
                    self.TYPE_COMPLEX = torch.complex128
                    self.TYPE_FLOAT = torch.float64
                else:
                    self.TYPE_COMPLEX = torch.complex64
                    self.TYPE_FLOAT = torch.float32


class FastMNMF2(Base):
    """
    The blind souce separation using FastMNMF2

    X_FTM: the observed complex spectrogram
    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
    G_NM: diagonal elements of the diagonalized SCMs
    W_NFK: basis vectors
    H_NKT: activations
    PSD_NFT: power spectral densities
    Qx_power_FTM: power spectra of Q_FMM times X_FTM
    Y_FTM: sum of (PSD_NFT x G_NM) over all sources
    """

    def __init__(
        self,
        n_source,
        n_basis=8,
        init_SCM="twostep",
        algo="IP",
        n_iter_init=10,
        g_eps=5e-2,
        interval_norm=10,
        n_bit=64,
        device="cpu",
        seed=0,
    ):
        """Initialize FastMNMF2

        Parameters:
        -----------
            n_source: int
                The number of sources.
            n_basis: int
                The number of bases for the NMF-based source model.
            init_SCM: str ('circular', 'obs', 'twostep')
                How to initialize SCM.
                'obs' is for the case that one speech is dominant in the mixture.
            algo: str (IP, ISS)
                How to update Q.
            n_iter_init: int
                The number of iteration for the first step in 'twostep' initialization.
            device : 'cpu' or 'cuda' or 'cuda:[id]'
        """
        super().__init__(device=device, n_bit=n_bit, seed=seed)
        self.n_source = n_source
        self.n_basis = n_basis
        self.init_SCM = init_SCM
        self.g_eps = g_eps
        self.algo = algo
        self.interval_norm = interval_norm
        self.n_iter_init = n_iter_init
        self.save_param_list += ["W_NFK", "H_NKT", "G_NM", "Q_FMM"]

        if self.algo == "IP":
            self.method_name = "FastMNMF2_IP"
        elif "ISS" in algo:
            self.method_name = "FastMNMF2_ISS"
        else:
            raise ValueError("algo must be IP or ISS")

    def __str__(self):
        init = f"twostep_{self.n_iter_init}it" if self.init_SCM == "twostep" else self.init_SCM
        filename_suffix = (
            f"M={self.n_mic}-S={self.n_source}-F={self.n_freq}-K={self.n_basis}"
            f"-init={init}-g={self.g_eps}-bit={self.n_bit}-intv_norm={self.interval_norm}"
        )
        if hasattr(self, "file_id"):
            filename_suffix += f"-ID={self.file_id}"
        return filename_suffix

    def load_spectrogram(self, X_FTM, sample_rate=16000):
        super().load_spectrogram(X_FTM, sample_rate=sample_rate)
        if self.algo == "IP":
            self.XX_FTMM = contract("fti, ftj -> ftij", self.X_FTM, self.X_FTM.conj())

    def init_source_model(self):
        self.W_NFK = torch.rand(self.n_source, self.n_freq, self.n_basis, dtype=self.TYPE_FLOAT, device=self.device)
        self.H_NKT = torch.rand(self.n_source, self.n_basis, self.n_time, dtype=self.TYPE_FLOAT, device=self.device)

    def init_spatial_model(self):
        self.start_idx = 0
        self.Q_FMM = torch.tile(
            torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device), [self.n_freq, 1, 1]
        )
        self.G_NM = self.g_eps * torch.ones([self.n_source, self.n_mic], dtype=self.TYPE_FLOAT, device=self.device)
        for m in range(self.n_mic):
            self.G_NM[m % self.n_source, m] = 1

        if "circular" in self.init_SCM:
            pass
        elif "obs" in self.init_SCM:
            if hasattr(self, "XX_FTMM"):
                XX_FMM = self.XX_FTMM.sum(axis=1)
            else:
                XX_FMM = contract("fti, ftj -> fij", self.X_FTM, self.X_FTM.conj())
            _, eig_vec_FMM = torch.linalg.eigh(XX_FMM)
            eig_vec_FMM = eig_vec_FMM[:, :, range(self.n_mic - 1, -1, -1)]
            self.Q_FMM = eig_vec_FMM.permute(0, 2, 1).conj()
        elif "twostep" == self.init_SCM:
            if self.n_iter_init >= self.n_iter:
                print(
                    "\n------------------------------------------------------------------\n"
                    f"Warning: n_iter_init must be smaller than n_iter (= {self.n_iter}).\n"
                    f"n_iter_init is changed from {self.n_iter_init} to {self.n_iter // 3}"
                    "\n------------------------------------------------------------------\n"
                )
                self.n_iter_init = self.n_iter // 3

            self.start_idx = self.n_iter_init

            separater_init = FastMNMF2(
                n_source=self.n_source,
                n_basis=2,
                device=self.device,
                init_SCM="circular",
                n_bit=self.n_bit,
                g_eps=self.g_eps,
            )
            separater_init.load_spectrogram(self.X_FTM, self.sample_rate)
            separater_init.solve(n_iter=self.start_idx)

            self.Q_FMM = separater_init.Q_FMM
            self.G_NM = separater_init.G_NM
        else:
            raise ValueError("init_SCM should be circular, obs, or twostep.")

        self.G_NM /= self.G_NM.sum(axis=1)[:, None]
        self.normalize()

    def calculate_Qx(self):
        self.Qx_FTM = contract("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qx_power_FTM = torch.abs(self.Qx_FTM) ** 2

    def calculate_PSD(self):
        self.PSD_NFT = self.W_NFK @ self.H_NKT + EPS

    def calculate_Y(self):
        self.Y_FTM = contract("nft, nm -> ftm", self.PSD_NFT, self.G_NM) + EPS

    def update(self):
        self.update_WH()
        self.update_G()
        if self.algo == "IP":
            self.update_Q_IP()
        else:
            self.update_Q_ISS()
        if self.it % self.interval_norm == 0:
            self.normalize()
        else:
            self.calculate_Qx()

    def update_WH(self):
        tmp1_NFT = contract("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = contract("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = contract("nkt, nft -> nfk", self.H_NKT, tmp1_NFT)
        denominator = contract("nkt, nft -> nfk", self.H_NKT, tmp2_NFT)
        self.W_NFK *= torch.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

        tmp1_NFT = contract("nm, ftm -> nft", self.G_NM, self.Qx_power_FTM / (self.Y_FTM**2))
        tmp2_NFT = contract("nm, ftm -> nft", self.G_NM, 1 / self.Y_FTM)
        numerator = contract("nfk, nft -> nkt", self.W_NFK, tmp1_NFT)
        denominator = contract("nfk, nft -> nkt", self.W_NFK, tmp2_NFT)
        self.H_NKT *= torch.sqrt(numerator / denominator)
        self.calculate_PSD()
        self.calculate_Y()

    def update_G(self):
        numerator = contract("nft, ftm -> nm", self.PSD_NFT, self.Qx_power_FTM / (self.Y_FTM**2))
        denominator = contract("nft, ftm -> nm", self.PSD_NFT, 1 / self.Y_FTM)
        self.G_NM *= torch.sqrt(numerator / denominator)
        self.calculate_Y()

    def update_Q_IP(self):
        for m in range(self.n_mic):
            V_FMM = (
                contract("ftij, ft -> fij", self.XX_FTMM, (1 / self.Y_FTM[..., m]).to(self.TYPE_COMPLEX))
                / self.n_time
            )
            tmp_FM = torch.linalg.solve(
                self.Q_FMM @ V_FMM, torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device)[None]
            )[..., m]
            self.Q_FMM[:, m] = (
                tmp_FM / torch.sqrt(contract("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[:, None]
            ).conj()

    def update_Q_ISS(self):
        for m in range(self.n_mic):
            QxQx_FTM = self.Qx_FTM * self.Qx_FTM[:, :, m, None].conj()
            V_tmp_FxM = (QxQx_FTM[:, :, m, None] / self.Y_FTM).mean(axis=1)
            V_FxM = (QxQx_FTM / self.Y_FTM).mean(axis=1) / V_tmp_FxM
            V_FxM[:, m] = 1 - 1 / torch.sqrt(V_tmp_FxM[:, m])
            self.Qx_FTM -= contract("fm, ft -> ftm", V_FxM, self.Qx_FTM[:, :, m])
            self.Q_FMM -= contract("fi, fj -> fij", V_FxM, self.Q_FMM[:, m])

    def normalize(self):
        phi_F = contract("fij, fij -> f", self.Q_FMM, self.Q_FMM.conj()).real / self.n_mic
        self.Q_FMM /= torch.sqrt(phi_F)[:, None, None]
        self.W_NFK /= phi_F[None, :, None]

        mu_N = self.G_NM.sum(axis=1)
        self.G_NM /= mu_N[:, None]
        self.W_NFK *= mu_N[:, None, None]

        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        self.H_NKT *= nu_NK[:, :, None]

        self.calculate_Qx()
        self.calculate_PSD()
        self.calculate_Y()

    def separate(self, mic_index=MIC_INDEX):
        Y_NFTM = contract("nft, nm -> nftm", self.PSD_NFT, self.G_NM).to(self.TYPE_COMPLEX)
        self.Y_FTM = Y_NFTM.sum(axis=0)
        self.Qx_FTM = contract("fmi, fti -> ftm", self.Q_FMM, self.X_FTM)
        self.Qinv_FMM = torch.linalg.solve(
            self.Q_FMM, torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device)[None]
        )

        self.separated_spec = contract(
            "fj, ftj, nftj -> nft", self.Qinv_FMM[:, mic_index], self.Qx_FTM / self.Y_FTM, Y_NFTM
        )
        return self.separated_spec

    def get_SCM(self):
        if not hasattr(self, "Qinv_FMM"):
            self.Qinv_FMM = torch.linalg.solve(
                self.Q_FMM, torch.eye(self.n_mic, dtype=self.TYPE_COMPLEX, device=self.device)[None]
            )

        G_NFMM = contract("fmk,nk,flk->nfml", self.Qinv_FMM, self.G_NM, self.Qinv_FMM.conj())
        
        return G_NFMM

    def calculate_log_likelihood(self):
        log_likelihood = (
            -(self.Qx_power_FTM / self.Y_FTM + torch.log(self.Y_FTM)).sum()
            + self.n_time * (torch.log(torch.linalg.det(self.Q_FMM @ self.Q_FMM.transpose(0, 2, 1).conj()))).sum()
        ).real
        return log_likelihood

    def load_param(self, filename):
        super().load_param(filename)

        self.n_source, self.n_freq, self.n_basis = self.W_NFK.shape
        _, _, self.n_time = self.H_NKT


if __name__ == "__main__":
    import soundfile as sf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_fname", type=str, help="filename of the multichannel observed signals")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_fft", type=int, default=1024, help="number of frequencies")
    parser.add_argument("--n_source", type=int, default=3, help="number of noise")
    parser.add_argument("--n_basis", type=int, default=16, help="number of basis")
    parser.add_argument("--n_iter_init", type=int, default=30, help="nujmber of iteration used in twostep init")
    parser.add_argument(
        "--init_SCM",
        type=str,
        default="twostep",
        help="circular, obs (only for enhancement), twostep",
    )
    parser.add_argument("--n_iter", type=int, default=100, help="number of iteration")
    parser.add_argument("--g_eps", type=float, default=5e-2, help="minumum value used for initializing G_NM")
    parser.add_argument("--n_mic", type=int, default=8, help="number of microphone")
    parser.add_argument("--n_bit", type=int, default=64, help="number of microphone")
    parser.add_argument("--algo", type=str, default="IP", help="the method for updating Q")
    args = parser.parse_args()

    if args.gpu < 0:
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"

    separater = FastMNMF2(
        n_source=args.n_source,
        n_basis=args.n_basis,
        device=device,
        init_SCM=args.init_SCM,
        n_bit=args.n_bit,
        algo=args.algo,
        n_iter_init=args.n_iter_init,
        g_eps=args.g_eps,
    )

    wav, sample_rate = torchaudio.load(args.input_fname, channels_first=False)
    wav /= torch.abs(wav).max() * 1.2
    M = min(len(wav), args.n_mic)
    spec_FTM = MultiSTFT(wav[:, :M], n_fft=args.n_fft)

    separater.file_id = args.input_fname.split("/")[-1].split(".")[0]
    separater.load_spectrogram(spec_FTM, sample_rate)
    separater.solve(
        n_iter=args.n_iter,
        save_dir="./",
        save_likelihood=False,
        save_param=False,
        save_wav=True,
        interval_save=5,
    )
