"""Loss functions."""

from .focal import FocalLoss, inverse_frequency_alpha

__all__ = ["FocalLoss", "inverse_frequency_alpha"]
