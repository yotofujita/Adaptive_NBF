#! /usr/bin/env python3
# coding:utf-8

import os, sys, time
import numpy as np
from tqdm import tqdm
import resampy


EPS = 1e-8

__all__ = ["Minibatch_WPE", "get_window_sum"]


class Minibatch_WPE:
    def __init__(self, n_tap=8, n_delay=3, xp=np, context=0):
        self.n_tap, self.n_delay = n_tap, n_delay
        self.buffer_idx = 0  # index to indicate the end of buffer_data
        self.alpha = 0.98  # fogetting coefficient
        self.n_delta = 2  # # of time frame that is used for calculating PSD
        self.context = context
        self.xp = xp

    def step(self, X_FTM, n_iteration=3):
        X_FTM = self.xp.asarray(X_FTM)
        n_freq, n_time, n_mic = X_FTM.shape

        if not hasattr(self, "buffer_FLM"):
            self.buffer_FLM = self.xp.zeros([n_freq, self.n_tap + self.n_delay - 1, n_mic], dtype=X_FTM.dtype)
        if not hasattr(self, "B_FxMxML"):
            self.B_FxMxML = self.xp.zeros([n_freq, n_mic, n_mic * self.n_tap], dtype=X_FTM.dtype)

        X_shifted_FTM = self.xp.concatenate([self.buffer_FLM, X_FTM[:, : -self.n_delay]], axis=1).copy()

        X_shifted_FxTxML = self.xp.lib.stride_tricks.as_strided(
            X_shifted_FTM, shape=(n_freq, n_time, n_mic * self.n_tap), strides=X_shifted_FTM.strides
        )

        for it in tqdm(range(n_iteration)):
            Y_FTM = X_FTM - self.xp.einsum("fmi, fti -> ftm", self.B_FxMxML, X_shifted_FxTxML)
            if self.context == 0:
                PSD_FT = (self.xp.abs(Y_FTM) ** 2).mean(axis=2) + 1e-5
            else:
                # padding -> as_strided -> mean
                raise NotImplementedError

            R_FxMLxML = self.xp.einsum(
                "fti, ftj -> fij", X_shifted_FxTxML / PSD_FT[..., None], X_shifted_FxTxML.conj()
            )  # + 1e-5 * self.xp.eye(self.n_tap * n_mic)

            P_FxMLxM = self.xp.einsum("fti, ftj -> fij", X_shifted_FxTxML / PSD_FT[..., None], X_FTM.conj())

            self.B_FxMxML = self.xp.linalg.solve(R_FxMLxML, P_FxMLxM).transpose(0, 2, 1).conj()

        Y_FTM = X_FTM - self.xp.einsum("fmi, fti -> ftm", self.B_FxMxML, X_shifted_FxTxML)
        self.buffer_FLM = X_FTM[:, -(self.n_delay + self.n_tap - 1) :]
        return Y_FTM


def get_window_sum(n_fft, shift, length, first: bool):
    assert n_fft % shift == 0, "shift shoud be a divisor of n_fft"
    assert length > n_fft, "len should be larger than n_fft"
    n = n_fft // shift

    window_sum = np.zeros(length + n_fft)
    for i in range(int(np.ceil((length + n_fft) / shift))):
        l = len(window_sum[i * shift : i * shift + n_fft])
        window_sum[i * shift : i * shift + n_fft] += np.hamming(n_fft)[:l] ** 2

    if first:
        return window_sum[:length]
    else:
        return window_sum[n_fft:]


buffer_data = []
buffer_direct = []
n_fft = 1024
shift = 256
buffer_size = 16000 * 10
window_len = n_fft
buffer_start_idx = 0
flag_play_audio = 0
last_callback_time = np.inf
n_sample_out = 4096
window_sum_first = get_window_sum(n_fft, shift, n_sample_out, True)
window_sum = get_window_sum(n_fft, shift, n_sample_out, False)
flag_first = True


def callback_audio(data, args: "[wpe, publisher]"):
    wpe, pub = args
    global buffer_data, buffer_start_idx, buffer_direct, last_callback_time, flag_first
    last_callback_time = time.time()
    n_channel, sample_rate = data.n_channel, data.sample_rate
    n_sample = len(data.data) // n_channel // 4

    if len(buffer_data) == 0:
        buffer_data = np.zeros([buffer_size, n_channel])

    buffer_data[buffer_start_idx : buffer_start_idx + n_sample] = np.frombuffer(data.data, dtype=np.float32).reshape(
        -1, n_channel
    )
    buffer_start_idx += n_sample

    if buffer_start_idx >= window_len * 3:
        tmp = resampy.resample(buffer_data[: window_len * 3], sample_rate, 16000, axis=0)
        X_FM = np.fft.rfft(tmp * np.hamming(window_len)[:, None], axis=0)
        direct_TM = np.fft.irfft(wpe.step(X_FM), axis=0)

        buffer_data[: buffer_start_idx - shift * 3] = buffer_data[shift * 3 : buffer_start_idx]
        buffer_start_idx = buffer_start_idx - shift * 3
        buffer_direct = concat_sep(direct_TM, buffer_direct, n_fft - shift, init=(len(buffer_direct) == 0))

        if len(buffer_direct) > n_sample_out + n_fft:
            if flag_first:
                tmp = buffer_direct[:n_sample_out] / window_sum_first[:, None]
                flag_first = False
            else:
                tmp = buffer_direct[:n_sample_out] / window_sum[:, None]
                print("publish ", tmp.shape, tmp, buffer_direct[:n_sample_out])
            buffer_direct = buffer_direct[n_sample_out:]
            pub.publish(n_channel, 16000, tmp.flatten().astype(np.float32).tobytes())
