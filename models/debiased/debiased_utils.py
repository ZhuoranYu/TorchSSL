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

def causal_inference(current_logit, qhat, tau=0.5):
    # de-bias pseudo-labels
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob


def consistency_loss(logits_s, logits_w, qhat, name='ce', T=1.0, p_cutoff=0.0, e_cutoff=-8, use_hard_labels=True, use_debias=True, use_marginal_loss=True, tau=0.5):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        # whether we want to use debias
        if use_debias:
            pseudo_label = causal_inference(logits_w, qhat, tau)
        else:
            pseudo_label = F.softmax(logits_w, dim=1)

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        energy = -torch.logsumexp(logits_w, dim=1)

        if e_cutoff is not None:
            mask_raw = energy.le(e_cutoff)
        else:
            mask_raw = max_probs.ge(p_cutoff)
        mask = mask_raw.float()
        select = max_probs.ge(p_cutoff).long()


        if use_marginal_loss:
            delta_logits = torch.log(qhat)
            logits_s = logits_s + tau * delta_logits

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long(), mask_raw

    else:
        assert Exception('Not Implemented consistency_loss')
