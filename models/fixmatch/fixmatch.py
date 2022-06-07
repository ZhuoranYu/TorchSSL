import pickle

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib
from train_utils import AverageMeter

from .fixmatch_utils import consistency_loss, Get_Scalar
from train_utils import ce_loss, wd_loss, EMA, Bn_Controller
from analyze_utils import *

from sklearn.metrics import *
from copy import deepcopy


class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u, \
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        self.it = 0
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_energy_pseudo(self, dataset, scores_ulb, label_ulb, energy_ulb):
        scores_ulb = torch.cat(scores_ulb, dim=0)
        label_ulb = torch.cat(label_ulb, dim=0)
        max_score, max_index = torch.max(scores_ulb, dim=-1)
        incorrect_mask = (max_index != label_ulb)

        pred_column = torch.zeros_like(label_ulb)  # to store correctness
        pred_column[incorrect_mask] = 1  # incorrectly predicted as head
        pred_column = pred_column

        energy_ulb = torch.cat(energy_ulb, dim=0)

        # from pseudo label's perspective
        data = np.vstack([max_score.detach().cpu().numpy(), energy_ulb.detach().cpu().numpy(),
                          pred_column.detach().cpu().numpy().astype(np.float)]).transpose()
        df = pd.DataFrame(data, columns=['score', 'energy', 'correct'])
        os.makedirs(f'{dataset}_track/{self.it // 1000}k', exist_ok=True)
        df.to_csv(f'{dataset}_track/{self.it // 1000}k/overall_pseudo.csv', index=False)
        for c in range(self.num_classes):
            c_mask = max_index == c
            c_score = max_score[c_mask].detach().cpu().numpy()
            c_energy = energy_ulb[c_mask].detach().cpu().numpy()
            c_pred_column = pred_column[c_mask].detach().cpu().numpy().astype(np.float)

            data = np.vstack([c_score, c_energy, c_pred_column]).transpose()
            df = pd.DataFrame(data, columns=['score', 'energy', 'correct'])
            os.makedirs(f'{dataset}_track/{self.it // 1000}k', exist_ok=True)
            df.to_csv(f'{dataset}_track/{self.it // 1000}k/class_{c}_pseudo.csv', index=False)

    def save_energy_real(self, dataset, scores_ulb, label_ulb, energy_ulb):
        scores_ulb = torch.cat(scores_ulb, dim=0)
        label_ulb = torch.cat(label_ulb, dim=0)
        max_score, max_index = torch.max(scores_ulb, dim=-1)
        incorrect_mask = (max_index != label_ulb)

        pred_column = torch.zeros_like(label_ulb)  # to store correctness
        pred_column[incorrect_mask] = 1  # incorrectly predicted as head
        pred_column = pred_column

        energy_ulb = torch.cat(energy_ulb, dim=0)

        # from real label's perspective
        data = np.vstack([max_score.detach().cpu().numpy(), energy_ulb.detach().cpu().numpy(),
                          pred_column.detach().cpu().numpy().astype(np.float)]).transpose()
        df = pd.DataFrame(data, columns=['score', 'energy', 'correct'])
        os.makedirs(f'{dataset}_track/{self.it // 1000}k', exist_ok=True)
        df.to_csv(f'{dataset}_track/{self.it // 1000}k/overall_real.csv', index=False)
        for c in range(self.num_classes):
            c_mask = label_ulb == c
            c_score = max_score[c_mask].detach().cpu().numpy()
            c_energy = energy_ulb[c_mask].detach().cpu().numpy()
            c_pred_column = pred_column[c_mask].detach().cpu().numpy().astype(np.float)

            data = np.vstack([c_score, c_energy, c_pred_column]).transpose()
            df = pd.DataFrame(data, columns=['score', 'energy', 'correct'])
            os.makedirs(f'{dataset}_track/{self.it // 1000}k', exist_ok=True)
            df.to_csv(f'{dataset}_track/{self.it // 1000}k/class_{c}_real.csv', index=False)

    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)

        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

        scores_ulb = []
        label_ulb = []
        energy_ulb = []

        pseudo_labels_acc = []
        true_labels_acc = []
        all_true_labels_acc = []

        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s, y_ulb) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb, y_ulb = y_lb.cuda(args.gpu), y_ulb.cuda(args.gpu)

            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # inference and calculate sup/unsup losses
            with amp_cm():
                logits = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                # out_prob = []
                # out_prob_nl = []
                # with torch.no_grad():
                #     for _ in range(10):
                #         outputs = self.model(x_ulb_w)
                #         out_prob.append(F.softmax(outputs, dim=1)) #for selecting positive pseudo-labels
                # out_prob = torch.stack(out_prob)
                # out_std = torch.std(out_prob, dim=0)
                # out_prob = torch.mean(out_prob, dim=0)
                # max_value, max_idx = torch.max(out_prob, dim=1)
                # max_std = out_std.gather(1, max_idx.view(-1,1))


                unsup_loss, mask, select, pseudo_lb, mask_raw = consistency_loss(logits_x_ulb_s,
                                                                       logits_x_ulb_w,
                                                                       'ce', T, p_cutoff, #max_std.squeeze(1),
                                                                       use_hard_labels=args.hard_label)
                pseudo_labels_acc.append(pseudo_lb[mask_raw])
                true_labels_acc.append(y_ulb[mask_raw])
                all_true_labels_acc.append(y_ulb)


                total_loss = sup_loss + self.lambda_u * unsup_loss


            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.

            # Save model for each 10K steps and best model for each 1K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model('latest_model.pth', save_path)

            if self.it % 50 == 0:
                pr_dict = analyze_pseudo(pseudo_labels_acc, true_labels_acc, all_true_labels_acc, self.num_classes)

                tb_dict.update(pr_dict)
                pseudo_labels_acc = []
                true_labels_acc = []
                all_true_labels_acc = []


            if self.it % self.num_eval_iter == 0:

                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")
                total_time = 0

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch

        minority_mask = torch.logical_or(torch.logical_or(torch.tensor(y_true) == 9, torch.tensor(y_true) == 8), torch.tensor(y_true) == 7)
        y_mino = torch.tensor(y_true)[minority_mask].tolist()
        pred_mino = torch.tensor(y_pred)[minority_mask].tolist()
        minority_acc = accuracy_score(y_mino, pred_mino)

        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5, 'eval/minority-acc': minority_acc}

    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()

        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['model'])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
