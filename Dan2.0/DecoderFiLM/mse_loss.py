"""Lung-masked MSE (train) for DecoderFiLM Dan 2.0 experiments."""


class DVFMSELoss:
    def loss(self, target_dvf, predict_dvf, mask):
        m = mask.expand_as(target_dvf)
        error = (target_dvf - predict_dvf) ** 2 * m
        return error.sum() / m.sum().clamp_min(1.0)
