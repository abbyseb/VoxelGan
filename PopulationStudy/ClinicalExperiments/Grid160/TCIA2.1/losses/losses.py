"""TCIA2.1 losses: MSE + under-move hinge (restored)."""
from __future__ import annotations
import torch

class DVFMSELoss:
    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = (target_dvf - predict_dvf) ** 2 * m
        return error.sum() / m.sum().clamp_min(1.0)

class DVFUnderMoveLoss:
    def __init__(self, eps=0.25):
        self.eps = float(eps)
    def loss(self, target_dvf, predict_dvf, mask):
        m_el = target_dvf.norm(dim=1, keepdim=True)
        m_pr = predict_dvf.norm(dim=1, keepdim=True)
        w = mask
        if w.shape[1] != 1:
            w = w[:, :1]
        gap = (m_el - m_pr - self.eps).clamp_min(0.0)
        return (gap * w).sum() / w.sum().clamp_min(1.0)

class DVFMSEUnderMoveLoss:
    def __init__(self, lambda_under=1.0, under_eps=0.25):
        self.mse = DVFMSELoss()
        self.under = DVFUnderMoveLoss(eps=under_eps)
        self.lambda_under = float(lambda_under)
    def loss_parts(self, target_dvf, predict_dvf, mask):
        l_mse = self.mse.loss(target_dvf, predict_dvf, mask)
        l_under = self.under.loss(target_dvf, predict_dvf, mask)
        return l_mse + self.lambda_under * l_under, l_mse, l_under
    def loss(self, target_dvf, predict_dvf, mask):
        return self.loss_parts(target_dvf, predict_dvf, mask)[0]
