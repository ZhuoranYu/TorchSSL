import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def consistency_loss(logits_s, logits_w, loss_form, feats_ulb, centroids, sharpening=False, use_hard_labels=True):
    # normalize unlabeled feature
    feats_ulb = F.normalize(feats_ulb.detach())
    d2c = torch.cdist(feats_ulb.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    d2c = (d2c / d2c.sum(dim=-1, keepdim=True)).detach()

    if loss_form == 'mse':
        scores_s = torch.softmax(logits_s, dim=-1)
        scores_w = torch.softmax(logits_w, dim=-1).detach()

        if sharpening:
            scores_w = scores_w ** (1 / 0.5)
            scores_w = (scores_w / scores_w.sum(dim=-1, keepdim=True)).detach()

        loss = F.mse_loss(scores_s, scores_w, reduction='none')
        loss = d2c * loss
        return loss.mean()

    else:
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none')
        weight = d2c[torch.arange(pseudo_label.shape[0]), max_idx]

        loss = weight * loss

        return loss.mean()

