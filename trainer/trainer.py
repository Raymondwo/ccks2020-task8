# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 5:47 下午
# @Author  : lizhen
# @FileName: trainer.py
# @Description:
import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from time import time
from utils.data_process_utils import get_evalute_param
import json

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, train_iter, valid_iter, test_iter=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.train_iter, self.valid_iter, self.test_iter = train_iter, valid_iter, test_iter
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_iter)
        else:
            # iteration-based training
            self.data_loader = inf_loop(train_iter)
            self.len_epoch = len_epoch

        self.do_validation = self.valid_iter is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_iter.batch_size))

        self.train_metrics = MetricTracker('tag_loss','crf_loss','total_loss','p','r','f', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('tag_loss', 'crf_loss','total_loss','p','r','f',*[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # self.cross_entropy_weight_ = [1.0] * schema.class_tag_num[class_id]
        self.cross_entropy_weight_ = [1.0] * 9
        for i in range(1, 9):
            if i % 2 == 1:
                self.cross_entropy_weight_[i] = 1.5
        self.cross_entropy_weight_[0] = 0.1

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        t1 = time()
        self.model.train()
        self.train_metrics.reset()
        tps = 0
        fps = 0
        aps = 0
        for batch_idx, batch_data in enumerate(self.train_iter):
            self.optimizer.zero_grad()

            text_token_ids, bert_masks, tag_ids, tag_masks, text_lengths, raw_tag_ids,texts = batch_data
            text_token_ids = text_token_ids.cuda()
            bert_masks = bert_masks.cuda()
            tag_ids = tag_ids.cuda()
            tag_masks = tag_masks.cuda()
            text_lengths = text_lengths.cuda()

            pred_tags = self.model(text_token_ids,bert_masks, text_lengths).squeeze(1)
            tag_loss = self.criterion[0](pred_tags,tag_ids,self.cross_entropy_weight_)
            crf_loss = self.model.crf(emissions=pred_tags, mask=tag_masks, tags=tag_ids, reduction='mean')
            total_loss = tag_loss + crf_loss
            total_loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('tag_loss', tag_loss.item())
            self.train_metrics.update('crf_loss', crf_loss.item())
            self.train_metrics.update('total_loss', total_loss.item())

            # crf 解码
            scores, best_path = self.model.crf.decode(emissions=pred_tags, mask=tag_masks)
            tp, fp, ap = get_evalute_param(best_path,raw_tag_ids,texts)
            tps += tp
            fps += fp
            aps += ap
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(tp,fp,ap))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_loss.item()))

            if batch_idx == self.len_epoch:
                break
        p = tps/(tps+fps)
        r = tps/aps
        f = (2*p*r)/(p+r)
        self.train_metrics.update('p',p)
        self.train_metrics.update('r',r)
        self.train_metrics.update('f',f)
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        print('spending time:', time() - t1)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        tps = 0
        fps = 0
        aps = 0
        pred_gold_file = open('data/military_ner/valid_epoches/{}.json'.format(epoch),'w',encoding='utf8')
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_iter):

                text_token_ids, bert_masks, tag_ids, tag_masks, text_lengths, raw_tag_ids, texts = batch_data
                text_token_ids = text_token_ids.cuda()
                bert_masks = bert_masks.cuda()
                tag_ids = tag_ids.cuda()
                tag_masks = tag_masks.cuda()
                text_lengths = text_lengths.cuda()

                pred_tags = self.model(text_token_ids, bert_masks, text_lengths).squeeze(1)
                tag_loss = self.criterion[0](pred_tags, tag_ids, self.cross_entropy_weight_)
                crf_loss = self.model.crf(emissions=pred_tags, mask=tag_masks, tags=tag_ids, reduction='mean')
                total_loss = tag_loss + crf_loss

                self.writer.set_step((epoch - 1) * len(self.valid_iter) + batch_idx, 'valid')
                self.valid_metrics.update('tag_loss', tag_loss.item())
                self.valid_metrics.update('crf_loss', crf_loss.item())
                self.valid_metrics.update('total_loss', total_loss.item())
                scores, best_path = self.model.crf.decode(emissions=pred_tags, mask=tag_masks)
                tp, fp, ap = get_evalute_param(best_path, raw_tag_ids, texts)
                tps += tp
                fps += fp
                aps += ap
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(tp, fp, ap))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_iter, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)







