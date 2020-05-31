# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 4:47 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from operator import itemgetter
from transformers import BertModel, XLNetModel
from utils.model_utils import prepare_pack_padded_sequence, matrix_mul, element_wise_mul
from .torch_crf_r import CRF
import numpy as np





class Bert_CRF(BaseModel):

    def __init__(self, bert_path, bert_train, num_tags, dropout, restrain):
        super(Bert_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.crf = CRF(num_tags, batch_first=True, restrain_matrix=restrain, loss_side=2.5)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train

        self.fc_tags = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, mask_bert,seq_len):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        bert_sentence, bert_cls = self.bert(context, attention_mask=mask_bert)
        sentence_len = bert_sentence.shape[1]

        # bert_cls = bert_cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        # bert_sentence = bert_sentence + bert_cls
        pred_tags = self.fc_tags(self.dropout(bert_sentence))[:, 1:, :]
        return pred_tags