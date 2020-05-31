# -*- coding: utf-8 -*-
# @Time    : 2020/5/21 9:37 上午
# @Author  : lizhen
# @FileName: data_process_utils.py
# @Description:
import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from collections import defaultdict
import jieba
import tqdm
from pyltp import SentenceSplitter, Postagger, NamedEntityRecognizer, Parser
import os
import pydotplus
import numpy as np


def load_pretrained_wordembedding(word_embedding_path):
    """
    加载预训练的词向量，并添加 'PAD'，'UNK' 以及生成对应的随机向量
    :return:
    """
    if not os.path.exists(word_embedding_path + '.pkl'):
        wv_from_text = KeyedVectors.load_word2vec_format(word_embedding_path, binary=False, encoding='utf-8',
                                                         unicode_errors='ignore')
        with open(word_embedding_path + '.pkl', 'wb') as f:
            pickle.dump(wv_from_text, f)
    else:
        with open(word_embedding_path + '.pkl', 'rb') as f:
            wv_from_text = pickle.load(f)
    wv_from_text.add('PAD', np.random.randn(wv_from_text.vector_size))
    wv_from_text.add('UNK', np.random.randn(wv_from_text.vector_size))

    return wv_from_text


def extract_entities(tag_ids, text):
    """
    用于抽取文本中的实体，以及对应的实体类别
    example:
        tag_ids = [1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 4, 4, 4, 0, 0, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 0]
        text = “小直径炸弹”Ⅱ有三种主要攻击模式,一是常规打击模式(NA),二是激光照射打击模式(LIA),三是坐标打击模式(CA),前两种模式可打击移动或固定目标,第三种模式用于打击固定目标?

    :param tag_ids:
    :param text: 原始文本语句
    :return:(list(dict)):[{'entity_text': '固定目标', 'start_pos': 86, 'end_pos': 89, 'label_type': '性能指标'},...]
    """

    ans = []

    entity_class = {1: '试验要素', 2: '性能指标', 3: '任务场景', 4: '系统组成'}

    entities, start_poses, entity, starting = [], [], [], False
    for idx, tag_id in enumerate(tag_ids):
        if tag_id > 0:
            if tag_id % 2 == 1:
                starting = True
                start_poses.append(idx)
                if entity:
                    entities.append(entity)
                    entity = []
                entity.append(tag_id)


            elif starting and tag_id == tag_ids[start_poses[-1]] + 1:
                entity.append(tag_id)
            else:
                starting = False
                if entity:
                    entities.append(entity)
                entity = []


        else:
            if entity:
                entities.append(entity)
            entity = []
            starting = False
    for entity_ids, start_pos in zip(entities, start_poses):
        entity_len = len(entity_ids)
        ans.append({
            'entity_text': text[start_pos:start_pos + entity_len],
            'start_pos': start_pos + 1,
            'end_pos': start_pos + entity_len,
            'label_type': entity_class[(entity_ids[0] + 1) / 2]

        })

    return ans


def get_evalute_param(preds, labels, texts):
    """
     计算一个batch 里面的 TP,FP,AP
    :param preds:
    :param labels:
    :param texts:
    :return:
    """
    tp = 0  # 一个batch里面预测对了多少个实体
    fp = 0  # 一个batch里面预测错了多少个实体
    ap = 0  # 一个batch里面有多少个 gold 实体
    for pred, label, text in zip(preds, labels, texts):
        pred_entities = extract_entities(pred, text)
        label_entities = extract_entities(label, text)

        cur_tp = 0
        # 计算一个样例文本里预测对了多少个实体
        for pred_entity in pred_entities:
            for label_entity in label_entities:
                if pred_entity['start_pos'] == label_entity['start_pos'] and pred_entity['end_pos'] == label_entity[
                    'end_pos'] and pred_entity['label_type'] == label_entity['label_type']:
                    cur_tp += 1

        tp += cur_tp
        fp += (len(pred_entities) - cur_tp)
        ap += len(label_entities)
    return tp, fp, ap

