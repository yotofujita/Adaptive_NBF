#! /usr/bin/env python3
# coding: utf-8

import torch
import torchaudio
from tqdm import tqdm

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

        print(f"Update {self.method_name}-{self}  {self.n_iter-self.start_idx} times ...")

        self.log_likelihood_dict = {}
        for self.it in tqdm(range(self.start_idx, n_iter)):
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
