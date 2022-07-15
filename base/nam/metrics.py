"""the calculation of matrices"""

import torch.nn.functional as F
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.datasets import load_gbsg2
import pandas as pd
import torch
import numpy as np


_, y = load_gbsg2()
y = pd.DataFrame(y)
# y["cens"] = y["cens"].astype(int)
time, cum_hazard = nelson_aalen_estimator(y["cens"].to_numpy(), y["time"].to_numpy())
# h0 = np.unique(cum_hazard)
# cum_hazard = torch.tensor(cum_hazard)


def feature_loss(fnn_out, lambda_=0.):
    """

    :param fnn_out:
    :param lambda_:
    :return:
    """
    return lambda_ * (fnn_out ** 2).sum() / fnn_out.shape[1]


def penalized_cross_entropy(logits, truth, fnn_out, feature_penalty=0.):
    """

    :param logits:
    :param truth:
    :param fnn_out:
    :param feature_penalty:
    :return:
    """
    # regression loss + L2 regularization loss
    return F.binary_cross_entropy_with_logits(logits.view(-1), truth.view(-1)) + feature_loss(fnn_out, feature_penalty)


def penalized_mse(logits, truth, fnn_out, feature_penalty=0.):
    """

    :param logits:
    :param truth:
    :param fnn_out:
    :param feature_penalty:
    :return:
    """
    # regression loss + L2 regularization loss
    return F.mse_loss(logits.view(-1), truth.view(-1)) + feature_loss(fnn_out, feature_penalty)


def survnam_loss(logits, rsf_preds, event_times):
    rsf_preds = np.squeeze(rsf_preds)

    loss = 0
    limit = event_times.shape[0]
    # print(limit)
    for i, j in enumerate(time):
        flag = 0
        while event_times[flag] < j:
            flag += 1
            if flag > limit - 1:
                flag = limit - 1
                break
        duration = j - time[i - 1] if i > 0 else j
        # print("rsf_preds[flag]", rsf_preds[flag])
        # print("cum_hazard[i]", cum_hazard[i])
        # print("logits", logits)
        # print("duration", duration)
        loss += torch.mul(torch.pow(torch.tensor(rsf_preds[flag]) -
                                    torch.mul(torch.tensor(cum_hazard[i]), torch.exp(logits)), 2),
                          torch.tensor(duration / 3000))
    # print(loss)
    return loss


def calculate_metric(logits, truths, regression=True):
    """Calculates the evaluation metric."""
    if regression:
        # root mean squared error
        # return torch.sqrt(F.mse_loss(logits, truths, reduction="none")).mean().item()
        # mean absolute error
        return "MAE", ((logits.view(-1) - truths.view(-1)).abs().sum() / logits.numel()).item()
    else:
        # return sklearn.metrics.roc_auc_score(truths.view(-1).tolist(), torch.sigmoid(logits.view(-1)).tolist())
        return "accuracy", accuracy(logits, truths)


def accuracy(logits, truths):
    """

    :param logits:
    :param truths:
    :return:
    """
    return (((truths.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / truths.numel()).item()
