# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 9:37 上午
# @Author  : lizhen
# @FileName: model_utils.py
# @Description:

import torch
import numpy as np

def matrix_mul(input, weight, bias=False):
    """
    for HAN model
    :param input:
    :param weight:
    :param bias:
    :return:
    """
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, dim=0).squeeze(-1)
def element_wise_mul(input1, input2):
    """
    for HAN model
    :param input1:
    :param input2:
    :return:
    """
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 1)



def prepare_pack_padded_sequence( inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices

def get_restrain_crf(class_tag_num):
    restrain = []
    restrain_start = np.zeros(class_tag_num)
    restrain_end = np.zeros(class_tag_num)
    restrain_trans = np.zeros((class_tag_num, class_tag_num))

    for i in range(0, class_tag_num):
        if i > 0 and i % 2 == 0:
            restrain_start[i] = -1000.0
        if i == 0:
            for j in range(0, class_tag_num):
                if j > 0 and j % 2 == 0:
                    restrain_trans[i][j] = -1000.0
        if i % 2 == 1:
            for j in range(0, class_tag_num):
                if j > 0 and j % 2 == 0 and not j == (i + 1):
                    restrain_trans[i][j] = -1000.0

    restrain.append(torch.from_numpy(restrain_start).float().cuda())
    restrain.append(torch.from_numpy(restrain_end).float().cuda())
    restrain.append(torch.from_numpy(restrain_trans).float().cuda())
    return restrain