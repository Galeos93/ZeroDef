"""Module that contains various metrics."""
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F


def f1_score_metric(output, target):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        return f1_score(target.cpu(), output.cpu(), average="macro")
