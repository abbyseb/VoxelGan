"""MAE + under-move hinge for TCIA3 mag fine-tune."""
from __future__ import annotations

import torch


class DVFMAELoss:
    """Lung-masked L1 (same family as TCIA3)."""

    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = torch.abs(target_dvf - predict_dvf) * m
        return error.sum() / m.sum().clamp_min(1.0)


class DVFUnderMoveLoss:
    """max(0, ||u_el|| - ||u_pred|| - eps) lung-masked (vector magnitude)."""

    def __init__(self, eps: float = 0.25):
        self.eps = float(eps)

    def loss(self, target_dvf, predict_dvf, mask):
        m_el = target_dvf.norm(dim=1, keepdim=True)
        m_pr = predict_dvf.norm(dim=1, keepdim=True)
        w = mask if mask.shape[1] == 1 else mask[:, :1]
        gap = (m_el - m_pr - self.eps).clamp_min(0.0)
        return (gap * w).sum() / w.sum().clamp_min(1.0)


class DVFSIUnderMoveLoss:
    """Under-move on SI component only (channel 1 in our DVF layout)."""

    def __init__(self, eps: float = 0.25, channel: int = 1):
        self.eps = float(eps)
        self.channel = int(channel)

    def loss(self, target_dvf, predict_dvf, mask):
        el = target_dvf[:, self.channel].abs()
        pr = predict_dvf[:, self.channel].abs()
        w = mask[:, 0] if mask.ndim == 5 else mask
        gap = (el - pr - self.eps).clamp_min(0.0)
        return (gap * w).sum() / w.sum().clamp_min(1.0)


class DVFMAEUnderMoveLoss:
    def __init__(
        self,
        lambda_under: float = 1.0,
        under_eps: float = 0.25,
        mode: str = "si",
        si_channel: int = 1,
    ):
        self.mae = DVFMAELoss()
        if mode == "si":
            self.under = DVFSIUnderMoveLoss(eps=under_eps, channel=si_channel)
        else:
            self.under = DVFUnderMoveLoss(eps=under_eps)
        self.lambda_under = float(lambda_under)
        self.mode = mode

    def loss_parts(self, target_dvf, predict_dvf, mask):
        l_mae = self.mae.loss(target_dvf, predict_dvf, mask)
        l_under = self.under.loss(target_dvf, predict_dvf, mask)
        total = l_mae + self.lambda_under * l_under
        return total, l_mae, l_under

    def loss(self, target_dvf, predict_dvf, mask):
        return self.loss_parts(target_dvf, predict_dvf, mask)[0]
