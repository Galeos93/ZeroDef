from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F


def accuracy(output, target):
    with torch.no_grad():
        threshold = 0.5
        pred = torch.sigmoid(output)
        pred = torch.where(pred > threshold, 1, 0)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1_score_metric(output, target):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        return f1_score(target.cpu(), output.cpu(), average="macro")
