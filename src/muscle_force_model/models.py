"""PyTorch model definitions.

Architecture summary
---------------------
Each muscle (and each cross-validation fold) has its own trained pair of
models:

1. ``AutoEncoder``: ``Encoder`` compresses a single target muscle's
   ground-truth force time series into a low-dimensional latent code
   (``encode_size``), and ``Decoder`` reconstructs all ``n_muscle`` force
   time series from that code. Trained with real (EMG-informed or static
   optimization) muscle force labels.

2. ``CNN``: learns to predict that same latent code directly from
   muscle-agnostic biomechanical inputs (segment kinematics, normalized
   muscle-tendon kinematics, external moments, muscle moment arms,
   anthropometry, cross-sectional area), with no muscle force or EMG input.

At inference, the CNN's predicted code goes straight into the frozen
decoder (skipping the encoder, since no ground-truth force is available),
producing all ``n_muscle`` force estimates; only the column for the muscle
that model was trained for is kept. Fold outputs are averaged for the
final estimate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_MUSCLE = 10


class Encoder(nn.Module):
    """1D causal-ish convolutional encoder: (B, T, 1) -> (B, encode_size, T/4)."""

    def __init__(self, in_features: int, n_features: int, encode_size: int, k_size: int = 3):
        super().__init__()
        self.k_size = k_size
        self.conv1 = nn.Conv1d(in_features, n_features, k_size, stride=1, padding=0)
        self.conv2 = nn.Conv1d(n_features, n_features, k_size, stride=1, padding=0)
        self.conv3 = nn.Conv1d(n_features, encode_size, 1, stride=1, padding=0)
        self.act = nn.Softmax(dim=1)

    @staticmethod
    def _same_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
        padding = (kernel_size - 1) * dilation
        start = x[:, :, 0:1].repeat(1, 1, padding // 2)
        end = x[:, :, -1:].repeat(1, 1, padding // 2)
        return torch.cat([start, x, end], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        x = self._same_pad(x, self.k_size, 1)
        x = torch.relu(self.conv1(x))

        x = F.avg_pool1d(x, 2, 2)
        x = self._same_pad(x, self.k_size, 1)
        x = torch.relu(self.conv2(x))

        x = F.avg_pool1d(x, 2, 2)
        x = self.conv3(x)

        x = x / np.sqrt(x.shape[1])
        return self.act(x)


class Decoder(nn.Module):
    """Upsampling convolutional decoder: (B, encode_size, T/4) -> (B, T, n_muscle)."""

    def __init__(self, n_features: int, encode_size: int, n_muscle: int = N_MUSCLE, k_size: int = 3):
        super().__init__()
        self.k_size = k_size
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_u2 = nn.Conv1d(encode_size, n_features, k_size, stride=1, padding=0)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_u1 = nn.Conv1d(n_features, n_features, k_size, stride=1, padding=0)
        self.out = nn.Conv1d(n_features, n_muscle, 1, stride=1, padding=0)

    @staticmethod
    def _same_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
        padding = (kernel_size - 1) * dilation
        start = x[:, :, 0:1].repeat(1, 1, padding // 2)
        end = x[:, :, -1:].repeat(1, 1, padding // 2)
        return torch.cat([start, x, end], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up2(x)
        x = self._same_pad(x, self.k_size, 1)
        x = torch.relu(self.conv_u2(x))

        x = self.up1(x)
        x = self._same_pad(x, self.k_size, 1)
        x = torch.relu(self.conv_u1(x))

        x = self.out(x)
        return x.permute(0, 2, 1)


class AutoEncoder(nn.Module):
    """Per-muscle encoder/decoder pair with optional z-score normalization."""

    def __init__(
        self,
        encode_size: int = 2,
        encode_feature: int = 150,
        n_muscle: int = N_MUSCLE,
        norm: bool = True,
    ):
        super().__init__()
        self.encoder_label = Encoder(1, encode_feature, encode_size)
        self.decoder_label = Decoder(encode_feature, encode_size, n_muscle)
        self.norm = norm

        self.Y_mean = nn.Parameter(torch.zeros((1, 1, n_muscle)), requires_grad=False)
        self.Y_std = nn.Parameter(torch.ones((1, 1, n_muscle)), requires_grad=False)

    def forward(self, x: torch.Tensor, muscle: int, training: bool = True):
        """
        Parameters
        ----------
        x:
            If ``training=True``: ground-truth force tensor (B, T, n_muscle);
            only column ``muscle`` is encoded.
            If ``training=False``: a pre-computed latent code (B, encode_size, T')
            coming from the CNN, used directly by the decoder.
        muscle:
            Index of the target muscle (only used when ``training=True``).
        """
        if training:
            if self.norm:
                x = (x - self.Y_mean) / self.Y_std
            enc = self.encoder_label(x[:, :, muscle : muscle + 1])
        else:
            enc = x

        out = self.decoder_label(enc)

        if self.norm:
            out = out * self.Y_std + self.Y_mean

        return out, enc


class FeatureExtract(nn.Module):
    """Shared convolutional trunk used by the CNN to produce a latent code per timestep."""

    def __init__(self, in_features: int, n_features: int, encode_size: int):
        super().__init__()
        self.k_size = 3
        self.conv1 = nn.Conv1d(in_features, n_features * 2, self.k_size, stride=1, padding=0)
        self.drop1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(n_features * 2, n_features * 4, self.k_size, stride=1, padding=0)
        self.drop2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(n_features * 4, encode_size, 1, stride=1, padding=0)

    @staticmethod
    def _same_pad(x: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
        padding = (kernel_size - 1) * dilation
        start = x[:, :, 0:1].repeat(1, 1, padding // 2)
        end = x[:, :, -1:].repeat(1, 1, padding // 2)
        return torch.cat([start, x, end], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        x = self._same_pad(x, self.k_size, 1)
        x = torch.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.avg_pool1d(x, 2, 2)

        x = self._same_pad(x, self.k_size, 1)
        x = torch.relu(self.conv2(x))
        x = self.drop2(x)
        x = F.avg_pool1d(x, 2, 2)

        x = self.conv3(x)
        return x.permute(0, 2, 1)


class CNN(nn.Module):
    """Predicts a latent muscle-force code from muscle-agnostic biomechanical features.

    Per-timestep input layout (assembled in ``preprocessing.py``):
        - normalized muscle-tendon lengths & velocities: 2 * n_muscle
        - pelvis + chest Euler angles, velocities, accelerations: 3 * 2 * 3
        - external moment, soft-assigned across spine levels: 3
        - muscle moment arms, soft-assigned across spine levels: n_muscle * 3
        - demographics (age, height, weight) + muscle CSA: 3 + n_muscle
    """

    def __init__(
        self,
        encode_size: int,
        encode_features: int,
        n_muscle: int = N_MUSCLE,
        n_disc: int = 6,
        norm: bool = True,
    ):
        super().__init__()
        self.n_muscle = n_muscle
        self.n_disc = n_disc
        self.act = nn.Softmax(dim=-1)
        self.norm = norm

        # in_features = kinematics(nmk 2*M + chest/pelvis 1+6*3) + moment(3)
        #               + moment_arms(M*3) + demographics(3) + csa(M)
        in_features = (2 * n_muscle) + (1 + 6 * 3) + 3 + (n_muscle * 3) + 3 + n_muscle
        self.feature_extract = FeatureExtract(in_features, encode_features, encode_size)

        self.estimate_spine_level = nn.Sequential(
            nn.Linear(1 + 6 * 3, 64), nn.Tanh(), nn.Linear(64, n_disc)
        )

        self.MCA_mean = nn.Parameter(torch.zeros((1, 1, 1)), requires_grad=False)
        self.MCA_std = nn.Parameter(torch.ones((1, 1, 1)), requires_grad=False)

        self.D_mean = nn.Parameter(torch.zeros((1, 3)), requires_grad=False)
        self.D_std = nn.Parameter(torch.ones((1, 3)), requires_grad=False)

        self.MOM_std = nn.Parameter(torch.ones((1, 1, n_disc, 3)), requires_grad=False)
        self.MOM_ARMS_std = nn.Parameter(torch.ones((1, 1, n_muscle, n_disc, 3)), requires_grad=False)
        self.CHEST_PELVIS_std = nn.Parameter(torch.ones((1, 1, 1 + 6 * 3)), requires_grad=False)
        self.NMK_std = nn.Parameter(torch.ones((1, 1, 2 * n_muscle)), requires_grad=False)

    def forward(
        self,
        chest_pelvis: torch.Tensor,
        nmk: torch.Tensor,
        mom: torch.Tensor,
        mom_arms: torch.Tensor,
        dem: torch.Tensor,
        mca: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shapes
        ------
        chest_pelvis: (B, T, 1 + 6*3)   [time, pelvis xyz, chest xyz] * {pos,vel,acc}
        nmk:          (B, T, 2*n_muscle)
        mom:          (B, T, n_disc, 3)
        mom_arms:     (B, T, n_muscle, n_disc, 3)
        dem:          (B, 3)            [age, height, weight]
        mca:          (B, n_muscle, 1)
        """
        if self.norm:
            chest_pelvis = chest_pelvis / (self.CHEST_PELVIS_std + 1e-8)
            nmk = nmk / (self.NMK_std + 1e-8)
            mom = mom / (self.MOM_std + 1e-8)
            mom_arms = mom_arms / (self.MOM_ARMS_std + 1e-8)
            dem = (dem - self.D_mean) / (self.D_std + 1e-8)
            mca = (mca - self.MCA_mean) / (self.MCA_std + 1e-8)

        # Estimate a soft assignment of each muscle's moment arm to a spine level,
        # using the mean moment arm (excluding edge timepoints) + muscle CSA.
        avg_mom_arm = torch.mean(mom_arms[:, 10:-10, ...], dim=1)
        avg_mom_arm = avg_mom_arm.view(avg_mom_arm.shape[0], self.n_muscle, -1)

        spine_level = torch.softmax(
            self.estimate_spine_level(torch.cat([mca, avg_mom_arm], dim=-1)), dim=-1
        )
        spine_level_mom = torch.mean(spine_level, dim=1).view(chest_pelvis.shape[0], 1, -1, 1)
        spine_level_mom_arm = spine_level.view(chest_pelvis.shape[0], 1, self.n_muscle, -1, 1)

        mca = mca.squeeze(-1)
        dem = torch.cat([dem, mca], dim=-1)
        dem = dem.unsqueeze(1).repeat(1, chest_pelvis.shape[1], 1)

        mom = torch.sum(mom * spine_level_mom, dim=2)
        mom_arms = torch.sum(mom_arms * spine_level_mom_arm, dim=3)
        mom_arms = mom_arms.reshape(mom_arms.shape[0], mom_arms.shape[1], -1)

        x = torch.cat([nmk, chest_pelvis, mom, mom_arms, dem], dim=-1)
        x = self.feature_extract(x.float())
        return self.act(x)
