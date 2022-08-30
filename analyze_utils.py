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
        mask = torch.logical_or(scores >= t, scores < t1)
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
    precision_overall = overall_tp.cpu().item() / (pseudo_labels.shape[0] + 1e-7)
    recall_overall = overall_tp.cpu().item() / (all_true_labels.shape[0] + 1e-7)
    f1_overall = (2 * precision_overall * recall_overall) / (precision_overall + recall_overall + 1e-7)
    pr_dict[f'pseudo-precision/overall'] = precision_overall
    pr_dict[f'pseudo-recall/overall'] = recall_overall
    pr_dict[f'pseudo-f1/overall'] = f1_overall
    pr_dict['count/overall'] = pseudo_labels.shape[0]

    #print(overall_tp.cpu().item(), pseudo_labels.shape[0], precision_overall)


    head_mask_pseudo = torch.logical_or(torch.logical_or(pseudo_labels==0, pseudo_labels==1), pseudo_labels==2)
    body_mask_pseudo = torch.logical_or(torch.logical_or(torch.logical_or(pseudo_labels == 3, pseudo_labels == 4), pseudo_labels == 5), pseudo_labels == 6)
    tail_mask_pseudo = torch.logical_or(torch.logical_or(pseudo_labels == 7, pseudo_labels == 8), pseudo_labels == 9)

    head_mask_real = torch.logical_or(torch.logical_or(all_true_labels==0, all_true_labels==1), all_true_labels==2)
    body_mask_real = torch.logical_or(torch.logical_or(torch.logical_or(all_true_labels == 3, all_true_labels == 4), all_true_labels == 5), all_true_labels == 6)
    tail_mask_real = torch.logical_or(torch.logical_or(all_true_labels == 7, all_true_labels == 8), all_true_labels == 9)

    head_pseudo = pseudo_labels[head_mask_pseudo]
    head_true = true_labels[head_mask_pseudo]
    head_true_all = all_true_labels[head_mask_real]
    tp_head = torch.sum((head_pseudo == head_true).float())

    precision_head = tp_head.cpu().item() / (head_pseudo.shape[0] + 1e-7)
    recall_head = tp_head.cpu().item() / (head_true_all.shape[0] + 1e-7)
    f1_head = (2 * precision_head * recall_head) / (precision_head + recall_head + 1e-7)
    pr_dict[f'pseudo-precision/head'] = precision_head
    pr_dict[f'pseudo-recall/head'] = recall_head
    pr_dict[f'pseudo-f1/head'] = f1_head

    body_pseudo = pseudo_labels[body_mask_pseudo]
    body_true = true_labels[body_mask_pseudo]
    body_true_all = all_true_labels[body_mask_real]
    tp_body = torch.sum((body_pseudo == body_true).float())

    precision_body = tp_body.cpu().item() / (body_pseudo.shape[0] + 1e-7)
    recall_body = tp_body.cpu().item() / (body_true_all.shape[0] + 1e-7)
    f1_body = (2 * precision_body * recall_body) / (precision_body + recall_body + 1e-7)
    pr_dict[f'pseudo-precision/body'] = precision_body
    pr_dict[f'pseudo-recall/body'] = recall_body
    pr_dict[f'pseudo-f1/body'] = f1_body

    tail_pseudo = pseudo_labels[tail_mask_pseudo]
    tail_true = true_labels[tail_mask_pseudo]
    tail_true_all = all_true_labels[tail_mask_real]
    tp_tail = torch.sum((tail_pseudo == tail_true).float())

    precision_tail = tp_tail.cpu().item() / (tail_pseudo.shape[0] + 1e-7)
    recall_tail = tp_tail.cpu().item() / (tail_true_all.shape[0] + 1e-7)
    f1_tail = (2 * precision_tail * recall_tail) / (precision_tail + recall_tail + 1e-7)
    pr_dict[f'pseudo-precision/tail'] = precision_tail
    pr_dict[f'pseudo-recall/tail'] = recall_tail
    pr_dict[f'pseudo-f1/tail'] = f1_tail

    for c in range(10):
        c_mask = pseudo_labels == c
        c_pseudo = pseudo_labels[c_mask]
        c_true = true_labels[c_mask]
        c_true_all = all_true_labels[all_true_labels == c]
        tp_c = torch.sum((c_pseudo == c_true).float())
        pr_dict[f'pseudo-precision/class_{c}'] = tp_c.cpu().item() / (c_pseudo.shape[0] + 1e-7)
        pr_dict[f'pseudo-recall/class_{c}'] = tp_c.cpu().item() / (c_true_all.shape[0] + 1e-7)
        pr_dict[f'count/class_{c}'] = c_pseudo.shape[0]


    return pr_dict
