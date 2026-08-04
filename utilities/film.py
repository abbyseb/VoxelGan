"""FiLM conditioning: continuous phase MLP + per-scale γ/β heads.

Phase indices (0..n_phases-1) are mapped to φ/n_phases ∈ [0, 1) and encoded
with a shared MLP so unseen leave-phase-out IDs still get a valid code.
Context is [e_ref, e_target, e_target - e_ref] → trunk → c.
Each head maps c → (Δγ, β) with γ = 1 + tanh(Δγ).
"""

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    def __init__(self, n_phases=10, embed_dim=8, channel_list=None):
        super().__init__()
        if channel_list is None:
            # bottleneck, Dec4, Dec3, Dec2, Dec1
            channel_list = [256, 128, 64, 32, 16]

        self.n_phases = n_phases

        # shared continuous phase encoder: φ/n_phases → embed_dim
        self.phase_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, embed_dim),
        )

        # concat[e_ref, e_target, delta] → dim 3 * embed_dim
        in_dim = 3 * embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
        )

        self.heads = nn.ModuleList(
            [nn.Linear(64, 2 * c) for c in channel_list]
        )
        self.channel_list = list(channel_list)

    def _encode_phase(self, phase):
        # phase: (B,) long/float indices → (B, embed_dim)
        phi = phase.float().unsqueeze(-1) / float(self.n_phases)
        return self.phase_mlp(phi)

    def forward(self, ref_phase, target_phase):
        """Return list of (gamma, beta), one pair per conditioned scale."""
        e_ref = self._encode_phase(ref_phase)
        e_target = self._encode_phase(target_phase)
        delta = e_target - e_ref
        c = self.trunk(torch.cat([e_ref, e_target, delta], dim=-1))

        params = []
        for head, n_ch in zip(self.heads, self.channel_list):
            gb = head(c)
            delta_gamma, beta = gb[:, :n_ch], gb[:, n_ch:]
            gamma = 1.0 + torch.tanh(delta_gamma)
            # (B, C, 1, 1, 1) for broadcast over spatial dims
            params.append((gamma[:, :, None, None, None], beta[:, :, None, None, None]))
        return params


def apply_film(x, gamma, beta):
    return gamma * x + beta
