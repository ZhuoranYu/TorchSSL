import numpy as np
import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def interpolation(x1, y1, x2, y2, k):
    beta = np.log(y1 / y2) / (x1 ** k - x2 ** k)
    alpha = y1 / np.exp(beta * (x1 ** k))

    return alpha, beta

def debiasing(current_logit, qhat, tau=0.5):
    debiased_logits = current_logit - tau * torch.log(qhat)
    return debiased_logits

def consistency_loss(logits_s, logits_w, qhat, p_cutoff=0.95, e_cutoff=-8.75, weight=1.0, debias=False, tau=0.5, use_hard_labels=True):
    logits_w = logits_w.detach()

    # add debiasing before computing energy
    if debias:
        logits_w = debiasing(logits_w, qhat, tau)

    energy = -torch.logsumexp(logits_w, dim=1)
    pseudo_label = torch.softmax(logits_w, dim=-1)

    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    if e_cutoff is not None:
        if isinstance(e_cutoff, float):
            mask_raw_e = energy < e_cutoff
        else:
            mask_raw_e = energy < e_cutoff[max_idx] # class-specific energy threshold
    else:
        mask_raw_e = energy > 0 # warmup: make it all false

    mask_raw_p = max_probs > p_cutoff

    mask_raw = torch.logical_or(mask_raw_p, mask_raw_e)

    mask_e = mask_raw_e.float()
    mask_p = mask_raw_p.float()
    mask = mask_raw.float()

    select = max_probs[mask_raw]

    # adaptive marginal loss
    if debias:
        delta_logits = torch.log(qhat)
        logits_s = logits_s + tau * delta_logits

    if use_hard_labels:
        masked_loss_e = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask_e
        masked_loss_p = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask_p
    else:
        T = 0.5
        pseudo_label = torch.softmax(logits_w / T , dim=-1)
        masked_loss_e = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask_e
        masked_loss_p = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask_p

    masked_loss = masked_loss_e * weight + (1 - weight) * masked_loss_p

    if e_cutoff is None:
        masked_loss = masked_loss * 0
        mask = mask * 0
    return masked_loss.mean(), mask.sum(), select, max_idx.long(), mask_raw

