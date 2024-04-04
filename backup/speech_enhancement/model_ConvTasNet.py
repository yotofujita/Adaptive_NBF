#!/usr/bin/env python3
# coding:utf-8

import os
import numpy as np
import soundfile as sf
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split

import pdb

class ConvBlock(torch.nn.Module):
    def __init__(self, 
        io_channels:int,
        hidden_channels:int,
        kernel_size:int,
        padding:int,
        dilation: int = 1,
        no_residual: bool=False
    ):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=io_channels,
                out_channels=hidden_channels,
                kernel_size=1
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-8),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-8)
        )

        self.res_out = (
            None if no_residual
            else torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=io_channels,
                kernel_size=1
            )
        )
        self.skip_out = torch.nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=io_channels,
            kernel_size=1
            )
        
    def forward(self, input: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int, # 入力の次元=513*2
        num_sources: int,
        kernel_size: int,
        num_feats: int, # 1D-convでinput-dim->num_featsに最初に変換
        num_hidden: int,
        num_layers: int,
        num_stacks: int
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(
            num_groups=2, num_channels=input_dim, eps=1e-08
        )
        # self.input_conv = torch.nn.Conv1d(
        #     in_channels=input_dim, out_channels=num_feats, kernel_size=1
        # )

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=input_dim,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        no_residual=(l == (num_layers-1) and s == (num_stacks-1))
                    )
                )
                self.receptive_field += \
                    kernel_size if s == 0 and l == 0 else (kernel_size-1)*multi
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim // 2 * num_sources,
            kernel_size=1
        )

        # self.conv_layers2 = torch.nn.ModuleList([])
        # for s in range(num_stacks):
        #     for l in range(num_layers):
        #         multi = 2 ** l
        #         self.conv_layers2.append(
        #             ConvBlock(
        #                 io_channels=num_feats,
        #                 hidden_channels=num_hidden,
        #                 kernel_size=kernel_size,
        #                 dilation=multi,
        #                 padding=multi,
        #                 no_residual=(l == (num_layers-1) and s == (num_stacks-1))
        #             )
        #         )
        #         self.receptive_field += \
        #             kernel_size if s == 0 and l == 0 else (kernel_size-1)*multi
        # self.output_prelu = torch.nn.PReLU()
        # self.output_conv = torch.nn.Conv1d(
        #     in_channels=num_feats,
        #     out_channels=input_dim // 2 * num_sources,
        #     kernel_size=1
        # )

    def forward(self, input):
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        # feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = torch.sigmoid(output)
        return output.view(batch_size, self.num_sources, self.input_dim//2, -1)



class ConvTasNet(torch.nn.Module):
    def __init__(
        self,
        num_sources: int = 1,
        enc_kernel_size: int = 16,
        enc_num_feats: int = 513*2,
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 4,
        msk_num_stacks: int = 3
    ):
        super().__init__()
        self.model_name = "ConvTasNet"
        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        # input = B x Cin=1 x T (Cin*Cout個のkernelを持つ)
        # output = B x Cout x T' (窓幅16,shift=8のSTFTの結果みたいなもの)
        # self.encoder = torch.nn.Conv1d( 
        #     in_channels=1,
        #     out_channels=enc_num_feats,
        #     kernel_size=enc_kernel_size,
        #     stride=self.enc_stride,
        #     padding=self.enc_stride,
        #     bias=False
        # )

        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks
        )

        # self.decoder = torch.nn.ConvTranspose1d(
        #     in_channels=enc_num_feats,
        #     out_channels=1,
        #     kernel_size=enc_kernel_size,
        #     stride=self.enc_stride,
        #     padding=self.enc_stride,
        #     bias=False
        # )

    # def forward(self, input):
    #     if input.ndim == 2:
    #         input = input.unsqueeze(1)
    #     if input.ndim != 3 or input.shape[1] != 1:
    #         raise ValueError(
    #             f"Expected 3D tensor (batch, channel=1, frames). found: {input.shape}"
    #         )
    def forward(self, noisy_pwr_spec, enhanced_pwr_spec):
        x = torch.log1p(torch.cat([noisy_pwr_spec, enhanced_pwr_spec], dim=1)) # B x 2F x T
        # pdb.set_trace()        
        # padded, num_pads = self._align_num_frames_with_strides(input)
        # batch_size, _, num_padded_frames = padded.shape
        # feats = self.encoder(padded)
        return self.mask_generator(x).squeeze(1) * noisy_pwr_spec
        # masked = masked.view(
        #     batch_size * self.num_sources, self.enc_num_feats, -1
        # )
        # decoded = self.decoder(masked)
        # output = decoded.view(batch_size, self.num_sources, num_padded_frames)
        # if num_pads > 0:
        #     output = output[..., :-num_pads]
        # return output


    def _align_num_frames_with_strides(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0
        
        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device
        )
        return torch.cat([input, pad], 2), num_paddings


    def training_step(self, batch, batch_idx):
        mix_BT, src_BNT, len_list_B = batch
        mix_BT = mix_BT.to(self.device)
        src_BNT = src_BNT.to(self.device)
        output_BNT = self.forward(mix_BT)
        loss = self.calculate_loss(output_BNT, src_BNT, len_list_B)
        return loss


    def calculate_SISNR(self, est, src):
        scale = (est @ src) / (src @ src)
        return ((scale * src) ** 2).sum() / ((scale * src - est) ** 2).sum()


    def calculate_loss(self, est_BNT, src_BNT, len_list_B):
        loss = 0
        for b in range(len(est_BNT)):
            for i in range(len(est_BNT[b])):
                max_SNR = -np.inf
                for j in range(len(src_BNT[b])):
                    SNR = self.calculate_SISNR(est_BNT[b, i, :len_list_B[b]], src_BNT[b, j, :len_list_B[b]])
                    if max_SNR < SNR:
                        max_SNR = SNR
                loss -= max_SNR
        return loss                    




if __name__ == "__main__":
    model = ConvTasNet()
    print(model.__dir__)
    # for name, param in model.named_parameters():
    #     print(name)
    from torchinfo import summary
    summary(
        model
    )
    
    input = torch.rand(3, 1, 1024)
    model(input)
