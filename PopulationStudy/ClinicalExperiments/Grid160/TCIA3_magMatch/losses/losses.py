"""MAE + direct magnitude match (not under-move hinge)."""
from __future__ import annotations

import torch


class DVFMAELoss:
    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = torch.abs(target_dvf - predict_dvf) * m
        return error.sum() / m.sum().clamp_min(1.0)


class DVFMagMatchLoss:
    """Lung-masked L1 between ||u_pred|| and ||u_el|| (direct size match)."""

    def loss(self, target_dvf, predict_dvf, mask):
        m_el = target_dvf.norm(dim=1, keepdim=True)
        m_pr = predict_dvf.norm(dim=1, keepdim=True)
        w = mask if mask.shape[1] == 1 else mask[:, :1]
        err = torch.abs(m_el - m_pr) * w
        return err.sum() / w.sum().clamp_min(1.0)


class DVFSIMagMatchLoss:
    """Lung-masked L1 on |SI| component (channel 1)."""

    def __init__(self, channel: int = 1):
        self.channel = int(channel)

    def loss(self, target_dvf, predict_dvf, mask):
        el = target_dvf[:, self.channel].abs()
        pr = predict_dvf[:, self.channel].abs()
        w = mask[:, 0] if mask.ndim == 5 else mask
        err = torch.abs(el - pr) * w
        return err.sum() / w.sum().clamp_min(1.0)


class DVFMAEMagMatchLoss:
    """total = MAE + λ_mag * ||u|| match + λ_si * |SI| match."""

    def __init__(
        self,
        lambda_mag: float = 1.0,
        lambda_si: float = 0.5,
        si_channel: int = 1,
    ):
        self.mae = DVFMAELoss()
        self.mag = DVFMagMatchLoss()
        self.si = DVFSIMagMatchLoss(channel=si_channel)
        self.lambda_mag = float(lambda_mag)
        self.lambda_si = float(lambda_si)

    def loss_parts(self, target_dvf, predict_dvf, mask):
        l_mae = self.mae.loss(target_dvf, predict_dvf, mask)
        l_mag = self.mag.loss(target_dvf, predict_dvf, mask)
        l_si = self.si.loss(target_dvf, predict_dvf, mask)
        total = l_mae + self.lambda_mag * l_mag + self.lambda_si * l_si
        return total, l_mae, l_mag, l_si

    def loss(self, target_dvf, predict_dvf, mask):
        return self.loss_parts(target_dvf, predict_dvf, mask)[0]
