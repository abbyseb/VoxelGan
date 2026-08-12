"""Losses for Dan 2.0 (MSE-only training; L1 kept for QC compare)."""

import torch


class DVFMSELoss:
    """Lung-masked MSE between predicted and Elastix DVF (train objective)."""

    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = (target_dvf - predict_dvf) ** 2 * m
        return error.sum() / m.sum().clamp_min(1.0)


class DVFLoss:
    """Lung-masked L1 — used in QC only (same metric as Dan'sPaperGan)."""

    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = torch.abs(target_dvf - predict_dvf) * m
        return error.sum() / m.sum().clamp_min(1.0)
