# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:35 下午
# @Author  : lizhen
# @FileName: metric.py
# @Description:
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def tag_accuracy(preds, labels):
    scores = []
    for pred, label in zip(preds, labels):
        scores.append(accuracy_score(pred, label))
    return sum(scores) / len(scores)


def batch_precision(tp, fp, ap):
    return tp / (tp + fp+1e-6)


def batch_recall(tp, fp, ap):
    return tp / (ap+1e-6)


def batch_f1(tp, fp, ap):
    p = batch_precision(tp, fp, ap)
    r = batch_recall(tp, fp, ap)
    return (2 * p * r) / (p + r+1e-6)



def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return (correct.sum() / torch.FloatTensor([y.shape[0]])).item()
