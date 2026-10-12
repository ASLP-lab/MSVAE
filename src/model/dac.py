# Acoustic encoder blocks adapted from https://github.com/descriptinc/descript-audio-codec under the MIT license.
#   License: incl_licenses/LICENSE.dac
import json
import math
from typing import List

import numpy as np
import torch
import torchaudio
from torch import nn

from muq import MuQ
from .bigvgan import BigVGAN
from src.model.utils import AttrDict
from src.nn.layers import Snake1d
from src.nn.layers import WNConv1d


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        in_channels: int = 1,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(in_channels, d_model, kernel_size=7, padding=3)]
        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DAC(nn.Module):
    """MuQ posterior with gated acoustic residuals and a BigVGAN decoder.

    encode_latents() consumes input_sample_rate audio; decode() emits sample_rate audio.
    Latents use [batch, frames, vae_dim] throughout the model.
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        encoder_rates: List[int] = [4, 5, 6, 8],
        vae_dim: int = 64,
        sample_rate: int = 48000,
        input_sample_rate: int = 24000,
        bigvgan_conf: str = "src/config/bigvgan_conf/bigvgan_v2_48khz_25hz_base_vae64.json",
        muq_model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.sample_rate = sample_rate
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.hop_length = int(np.prod(encoder_rates))
        self.input_sample_rate = input_sample_rate
        self.muq_dim = kwargs.get("muq_dim", 1024)
        self.muq_hidden_state_layer = kwargs.get("muq_hidden_state_layer", 6)
        self.muq_channel_fusion = kwargs.get("muq_channel_fusion", "mean")
        self.acoustic_residual_scale_max = kwargs.get("acoustic_residual_scale_max", 1.0)
        self.acoustic_gate_logit_clip = kwargs.get("acoustic_gate_logit_clip", 8.0)
        self.acoustic_gate_temperature = float(kwargs.get("acoustic_gate_temperature_end", 1.0))
        if self.acoustic_residual_scale_max <= 0.0:
            raise ValueError(
                f"acoustic_residual_scale_max must be positive, got {self.acoustic_residual_scale_max}."
            )
        if self.acoustic_gate_logit_clip <= 0.0:
            raise ValueError(
                f"acoustic_gate_logit_clip must be positive, got {self.acoustic_gate_logit_clip}."
            )

        # Both encoders receive input_sample_rate audio; avoid downsampling MuQ twice.
        self.muq_sample_rate = kwargs.get("muq_sample_rate", 24000)
        self.muq_resampler = (
            torchaudio.transforms.Resample(self.input_sample_rate, self.muq_sample_rate)
            if self.input_sample_rate != self.muq_sample_rate else None
        )
        self.encoder = MuQ.from_pretrained(muq_model_name)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.vae_dim = vae_dim

        self.muq_projector = nn.Linear(self.muq_dim, self.vae_dim)
        self.acoustic_encoder = Encoder(encoder_dim, encoder_rates, vae_dim, in_channels=in_channels)

        self.fc_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.fc_var = nn.Linear(self.vae_dim, self.vae_dim)
        self.acoustic_mu = nn.Linear(self.vae_dim, self.vae_dim)
        self.acoustic_var = nn.Linear(self.vae_dim, self.vae_dim)
        self.gate_sem_norm = nn.LayerNorm(self.vae_dim)
        self.gate_acoustic_norm = nn.LayerNorm(self.vae_dim)
        gate_hidden_dim = kwargs.get("acoustic_gate_hidden_dim", self.vae_dim * 2)
        if gate_hidden_dim <= 0:
            raise ValueError(f"acoustic_gate_hidden_dim must be positive, got {gate_hidden_dim}.")
        self.acoustic_gate_mlp = nn.Sequential(
            nn.Linear(self.vae_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.acoustic_gate = nn.Linear(gate_hidden_dim, self.vae_dim)

        residual_scale_init = kwargs.get("acoustic_residual_scale_init", 0.1)
        if residual_scale_init < 0.0:
            raise ValueError(
                f"acoustic_residual_scale_init must be non-negative, got {residual_scale_init}."
            )
        scale_ratio = residual_scale_init / self.acoustic_residual_scale_max
        scale_ratio = min(max(scale_ratio, 1e-4), 1.0 - 1e-4)
        scale_logit = math.log(scale_ratio / (1.0 - scale_ratio))
        self.acoustic_mu_scale = nn.Parameter(torch.tensor(scale_logit, dtype=torch.float32))
        self.acoustic_var_scale = nn.Parameter(torch.tensor(scale_logit, dtype=torch.float32))

        residual_init_gain = kwargs.get("acoustic_residual_init_gain", 0.1)
        gate_init_gain = kwargs.get("acoustic_gate_init_gain", 0.1)
        nn.init.xavier_uniform_(self.acoustic_mu.weight, gain=residual_init_gain)
        nn.init.zeros_(self.acoustic_mu.bias)
        nn.init.xavier_uniform_(self.acoustic_var.weight, gain=residual_init_gain)
        nn.init.zeros_(self.acoustic_var.bias)
        for module in self.acoustic_gate_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=gate_init_gain)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.acoustic_gate.weight, gain=gate_init_gain)
        gate_init = kwargs.get("acoustic_gate_init", kwargs.get("gate_init", 0.3))
        if not 0.0 < gate_init < 1.0:
            raise ValueError(f"acoustic_gate_init must be in (0, 1), got {gate_init}.")
        gate_bias = kwargs.get("acoustic_gate_bias")
        if gate_bias is None:
            gate_bias = math.log(gate_init / (1.0 - gate_init))
        nn.init.constant_(self.acoustic_gate.bias, gate_bias)

        self.bigvgan_conf = bigvgan_conf
        with open(self.bigvgan_conf, encoding="utf-8") as f:
            h = AttrDict(json.load(f))
        h.out_channels = out_channels

        self.decoder = BigVGAN(h)

    def _compute_acoustic_gate(self, z_sem: torch.Tensor, acoustic_hidden: torch.Tensor):
        gate_input = torch.cat(
            [
                self.gate_sem_norm(z_sem),
                self.gate_acoustic_norm(acoustic_hidden),
            ],
            dim=-1,
        )
        gate_logits = self.acoustic_gate(self.acoustic_gate_mlp(gate_input))
        gate_logits = torch.clamp(
            gate_logits,
            min=-self.acoustic_gate_logit_clip,
            max=self.acoustic_gate_logit_clip,
        )
        temperature = max(float(self.acoustic_gate_temperature), 1e-4)
        return torch.sigmoid(gate_logits / temperature)

    def _extract_muq_hidden(self, output):
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("MuQ output_hidden_states is required but hidden_states is None.")
        if not (-len(hidden_states) <= self.muq_hidden_state_layer < len(hidden_states)):
            raise IndexError(
                f"MuQ hidden state index {self.muq_hidden_state_layer} is out of range for {len(hidden_states)} layers."
            )

        hidden = hidden_states[self.muq_hidden_state_layer]
        if hidden.shape[-1] != self.muq_dim:
            raise RuntimeError(
                f"MuQ hidden dim mismatch: expected {self.muq_dim}, got {hidden.shape[-1]}."
            )
        return hidden

    def _run_muq_encoder(self, audio_data: torch.Tensor):
        if self.muq_resampler is not None:
            audio_data = self.muq_resampler(audio_data.float())
        encoder_ctx = torch.no_grad()

        if audio_data.dim() == 2:
            muq_audio = audio_data.float()
            with encoder_ctx:
                output = self.encoder(muq_audio, output_hidden_states=True)
            return self._extract_muq_hidden(output)

        if audio_data.dim() != 3:
            raise ValueError(f"Expected audio shape [B, T] or [B, C, T], got {tuple(audio_data.shape)}.")

        if audio_data.size(1) == 1:
            muq_audio = audio_data[:, 0].float()
            with encoder_ctx:
                output = self.encoder(muq_audio, output_hidden_states=True)
            return self._extract_muq_hidden(output)

        if audio_data.size(1) != 2:
            raise ValueError(f"MuQ path only supports mono or stereo audio, got {audio_data.size(1)} channels.")

        if self.muq_channel_fusion == "mono":
            muq_audio = audio_data.mean(dim=1).float()
            with encoder_ctx:
                output = self.encoder(muq_audio, output_hidden_states=True)
            return self._extract_muq_hidden(output)

        if self.muq_channel_fusion != "mean":
            raise ValueError(f"Unsupported MuQ channel fusion: {self.muq_channel_fusion}.")

        batch_size, num_channels, num_samples = audio_data.shape
        flat_audio = audio_data.reshape(batch_size * num_channels, num_samples).float()
        with encoder_ctx:
            output = self.encoder(flat_audio, output_hidden_states=True)
        hidden = self._extract_muq_hidden(output)
        hidden = hidden.reshape(batch_size, num_channels, hidden.shape[1], hidden.shape[2])
        return hidden.mean(dim=1)

    def _match_time_length(self, hidden: torch.Tensor, target_len: int):
        if hidden.shape[1] == target_len:
            return hidden
        if abs(hidden.shape[1] - target_len) <= 1:
            min_len = min(hidden.shape[1], target_len)
            return hidden[:, :min_len, :]
        raise RuntimeError(
            f"MuQ and acoustic encoder should both be 25Hz, but got lengths "
            f"{hidden.shape[1]} and {target_len}."
        )

    def _run_acoustic_encoder(self, audio_data: torch.Tensor):
        if audio_data.dim() == 2:
            audio_data = audio_data.unsqueeze(1)
        return self.acoustic_encoder(audio_data).transpose(1, 2)

    def _encode_posteriors(self, audio_data: torch.Tensor):
        """Return the fused posterior parameters on matched encoder frames."""
        acoustic_hidden = self._run_acoustic_encoder(audio_data)
        muq_hidden = self._run_muq_encoder(audio_data)
        muq_hidden = self._match_time_length(muq_hidden, acoustic_hidden.shape[1])
        acoustic_hidden = acoustic_hidden[:, :muq_hidden.shape[1], :]
        z_sem = self.muq_projector(muq_hidden)
        mu_sem = self.fc_mu(z_sem)
        log_var_sem = self.fc_var(z_sem)

        mu_scale = self.acoustic_residual_scale_max * torch.sigmoid(self.acoustic_mu_scale)
        var_scale = self.acoustic_residual_scale_max * torch.sigmoid(self.acoustic_var_scale)
        delta_mu = mu_scale * torch.tanh(self.acoustic_mu(acoustic_hidden))
        delta_log_var = var_scale * torch.tanh(self.acoustic_var(acoustic_hidden))
        acoustic_gate = self._compute_acoustic_gate(z_sem, acoustic_hidden)

        mu = mu_sem + acoustic_gate * delta_mu
        log_var = log_var_sem + acoustic_gate * delta_log_var
        return mu, log_var.clamp(-12, 12)

    def encode_latents(self, audio_data: torch.Tensor, latent_mode="sample"):
        """Encode stereo input into the complete fused latent."""
        if latent_mode not in ("sample", "mean"):
            raise ValueError(f"Unknown latent_mode: {latent_mode}")
        mu, log_var = self._encode_posteriors(audio_data)
        if latent_mode == "sample":
            eps = torch.randn_like(mu)
            return eps * torch.exp(0.5 * log_var) + mu
        return mu

    def decode(self, z: torch.Tensor):
        return self.decoder(z.transpose(1, 2))
