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


def consistency_loss(logits_s, logits_w, x1, y1, x2, y2, degree, name='ce', p_cutoff=0.95, e_cutoff=-8.75, joint_conf=False, use_hard_labels=True, K=10):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    alpha, beta = interpolation(x1, y1, x2, y2, degree)

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        energy = -torch.logsumexp(logits_w, dim=1)
        pseudo_label = torch.softmax(logits_w, dim=-1)

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        if joint_conf:
            boundary = alpha * torch.exp(beta * (max_probs ** degree))
            mask_raw = energy < boundary
        else:
            mask_raw = energy < e_cutoff
        mask = mask_raw.float()
        select = max_probs[mask_raw]

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long(), mask_raw

    else:
        assert Exception('Not Implemented consistency_loss')
            
