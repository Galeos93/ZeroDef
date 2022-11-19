"""Module that contains various losses."""
import torch.nn.functional as F
from torch import nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    loss = nn.CrossEntropyLoss()
    return loss(output, target)


binary_cross_entropy = nn.BCEWithLogitsLoss()
