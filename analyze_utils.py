import numpy as np
import torch

def analyze_prob(scores, predictions, labels):
    res = {}
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for idx, t in enumerate(bins):
        if t < 0.9:
            t = np.round(t, 2)
            t1 = np.round(t + 0.1, 2)
        else:
            t = np.round(t, 2)
            t1 = np.round(t + 0.05, 2)
        mask = torch.logical_and(scores >= t, scores < t1)
        score_bin = scores[mask]

        if torch.numel(score_bin) == 0:
            res[f'precision/{t}-to-{t1}'] = 0
            res[f'count/{t}-to-{t1}'] = 0
            continue
        pred_bin = predictions[mask]
        labels_bin = labels[mask]
        tp_bin = pred_bin[pred_bin == labels_bin]
        res[f'precision/{t}-to-{t1}'] = torch.numel(tp_bin) / torch.numel(pred_bin)
        res[f'count/{t}-to-{t1}'] = pred_bin.shape[0]

    return res

def analyze_pseudo(pseudo_labels, true_labels, all_true_labels, num_classes):
    pseudo_labels = torch.cat(pseudo_labels)
    true_labels = torch.cat(true_labels)
    all_true_labels = torch.cat(all_true_labels)

    pr_dict = analyze_pseudo_pr(pseudo_labels, true_labels, all_true_labels, num_classes)

    del pseudo_labels, true_labels, all_true_labels
    return pr_dict


def analyze_pseudo_pr(pseudo_labels, true_labels, all_true_labels, num_classes):
    pr_dict = {}

    overall_tp = torch.sum((pseudo_labels == true_labels).float())
    pr_dict[f'pseudo-precision/overall'] = overall_tp.cpu().item() / (pseudo_labels.shape[0] + 1e-7)
    pr_dict[f'pseudo-recall/overall'] = overall_tp.cpu().item() / (all_true_labels.shape[0] + 1e-7)

    for c in range(num_classes):
        c_mask = pseudo_labels == c
        c_pseudo = pseudo_labels[c_mask]
        c_true = true_labels[c_mask]
        c_true_all = all_true_labels[all_true_labels == c]
        tp_c = torch.sum((c_pseudo == c_true).float())
        pr_dict[f'pseudo-precision/class_{c}'] = tp_c.cpu().item() / (c_pseudo.shape[0] + 1e-7)
        pr_dict[f'pseudo-recall/class_{c}'] = tp_c.cpu().item() / (c_true_all.shape[0] + 1e-7)

    return pr_dict
