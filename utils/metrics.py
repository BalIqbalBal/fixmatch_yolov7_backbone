from collections import namedtuple

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

evaluation_metrics = namedtuple(
    "evaluation_metrics",
    [
        "loss",
        "top1",
        "top5",
        "prec",
        "rec",
        "f1",
        "prec_macro",
        "rec_macro",
        "f1_macro",
        "prec_weighted",
        "rec_weighted",
        "f1_weighted",
        "tp",
        "fp",
        "tn",
        "fn",
    ],
)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = len(target)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def precision(output: torch.Tensor, target: torch.Tensor, average: str = "weighted"):
    return precision_score(target.cpu(), output.cpu(), average=average)


def recall(output: torch.Tensor, target: torch.Tensor, average: str = "weighted"):
    return recall_score(target.cpu(), output.cpu(), average=average)


def f1(output: torch.Tensor, target: torch.Tensor, average: str = "weighted"):
    return f1_score(target.cpu(), output.cpu(), average=average)


def confusion_matrix_metrics(output: torch.Tensor, target: torch.Tensor):
    cm = confusion_matrix(target.cpu(), output.cpu())
    # For multi-class, extract metrics from the confusion matrix
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    return tp, fp, tn, fn


def write_metrics(writer: SummaryWriter, epoch: int, metrics: evaluation_metrics, descriptor: str = "val"):
    writer.add_scalar("Loss/{}_total".format(descriptor), metrics.loss, epoch)
    writer.add_scalar("Accuracy/{}_acc_1".format(descriptor), metrics.top1, epoch)
    writer.add_scalar("Accuracy/{}_acc_5".format(descriptor), metrics.top5, epoch)
    writer.add_scalar(
        "Classification_metrics/{}_precision".format(descriptor), metrics.prec, epoch
    )
    writer.add_scalar(
        "Classification_metrics/{}_recall".format(descriptor), metrics.rec, epoch
    )
    writer.add_scalar(
        "Classification_metrics/{}_f1score".format(descriptor), metrics.f1, epoch
    )
    writer.add_scalar(
        "Classification_metrics/{}_precision_macro".format(descriptor),
        metrics.prec_macro,
        epoch,
    )
    writer.add_scalar(
        "Classification_metrics/{}_recall_macro".format(descriptor),
        metrics.rec_macro,
        epoch,
    )
    writer.add_scalar(
        "Classification_metrics/{}_f1score_macro".format(descriptor),
        metrics.f1_macro,
        epoch,
    )
    writer.add_scalar(
        "Classification_metrics/{}_precision_weighted".format(descriptor),
        metrics.prec_weighted,
        epoch,
    )
    writer.add_scalar(
        "Classification_metrics/{}_recall_weighted".format(descriptor),
        metrics.rec_weighted,
        epoch,
    )
    writer.add_scalar(
        "Classification_metrics/{}_f1score_weighted".format(descriptor),
        metrics.f1_weighted,
        epoch,
    )
    writer.add_scalar(
        "Confusion_matrix/{}_true_positives".format(descriptor),
        metrics.tp,
        epoch,
    )
    writer.add_scalar(
        "Confusion_matrix/{}_false_positives".format(descriptor),
        metrics.fp,
        epoch,
    )
    writer.add_scalar(
        "Confusion_matrix/{}_true_negatives".format(descriptor),
        metrics.tn,
        epoch,
    )
    writer.add_scalar(
        "Confusion_matrix/{}_false_negatives".format(descriptor),
        metrics.fn,
        epoch,
    )