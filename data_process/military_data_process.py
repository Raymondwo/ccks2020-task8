# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 10:49 上午
# @Author  : lizhen
# @FileName: military_data_process.py
# @Description:
from base import NLPDataSet
import os
import json
from collections import defaultdict, Counter
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import jieba


class Entity:
    def __init__(self, start_pos, end_pos, label_type, label_text, overlap):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.label_type = label_type
        self.label_text = label_text

        self.overlap = overlap


class Schema:
    def __init__(self):
        self.type_2_id = {'O': 0, 'B-试验要素': 1, 'I-试验要素': 2, 'B-性能指标': 3, 'I-性能指标': 4, 'B-任务场景': 5, 'I-任务场景': 6,
                          'B-系统组成': 7, 'I-系统组成': 8}

        self.id_2_type = {0: 'O', 1: 'B-试验要素', 2: 'I-试验要素', 3: 'B-性能指标', 4: 'I-性能指标', 5: 'B-任务场景', 6: 'I-任务场景',
                          7: 'B-系统组成', 8: 'I-系统组成'}


class InputExample:
    def __init__(self, file_path, text, text_tokens, inputs, entities=None):
        self.schema = Schema()

        self.text = text
        self.text_len = len(text)
        self.text_tokens = text_tokens
        self.inputs = inputs
        self.entities = entities
        if entities is not None:
            self.tag_tokens, self.tag_ids = self._convert_entities_2_tag()
        self.file_path = file_path

    def _convert_entities_2_tag(self):
        tag_tokens = ['O'] * len(self.text)
        tag_ids = [0] * len(self.text)
        for ent in self.entities:
            tag_tokens[ent.start_pos - 1] = 'B-' + ent.label_type
            tag_tokens[ent.start_pos:ent.end_pos] = ['I-' + ent.label_type] * (ent.end_pos - ent.start_pos)

            tag_ids[ent.start_pos - 1] = self.schema.type_2_id['B-' + ent.label_type]
            tag_ids[ent.start_pos:ent.end_pos] = [self.schema.type_2_id['I-' + ent.label_type]] * (
                    ent.end_pos - ent.start_pos)
        return tag_tokens, tag_ids


class MilitaryDataSet(NLPDataSet):

    def __init__(self, data_dir,  bert_path,valid_size=0.15,test=False):
        self.data_dir = data_dir
        self.valid_size = valid_size
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        if not test:
            self._dataset_statistics()
            self.data = self._load_dataset()
            self.train_set, self.valid_set = train_test_split(self.data, test_size=valid_size, random_state=13)
        else:
            self.test_set = self._load_testset()


    def _dataset_statistics(self):
        """
        训练集的统计信息
        :return:
        """
        label_list = defaultdict(list)  # 用于统计每个标签下所有的实体
        train_file_list = os.listdir(self.data_dir)
        for train_file in train_file_list:
            with open(os.path.join(self.data_dir, train_file), 'r', encoding='gbk') as f:
                train_json = json.load(f)
                text = train_json['originalText']
                for ent in train_json['entities']:
                    start_pos = ent['start_pos']
                    end_pos = ent['end_pos']
                    overlap = ent['overlap']
                    label_text = text[start_pos - 1:end_pos]
                    label_type = ent['label_type']

                    label_list[label_type].append(label_text)

        # 每个标签下实体数量
        for label_type in label_list:
            print('{}:{}'.format(label_type, len(label_list[label_type])))

        # 输出出现次数大于两次的实体
        for label_type in label_list:
            output_str = label_type + '\n'
            counter = Counter(label_list[label_type])
            counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            for k, v in counter:
                if v > 2:
                    output_str += k + ':' + str(v) + ', '
            print(output_str + '\n')

    def _load_dataset(self):
        """
        加载训练集
        :return:
        """
        examples = []
        train_file_list = os.listdir(self.data_dir)
        for train_file in train_file_list:
            with open(os.path.join(self.data_dir, train_file), 'r', encoding='gbk') as f:
                train_json = json.load(f)
                text = train_json['originalText']
                assert len(text) < 510, 'text length is over 510'

                # 依存句法
                text_seg = jieba.lcut(text, HMM=False)
                poses = ' '.join(postagger.postag(text_seg)).split()
                arcs = parser.parse(text_seg, poses)
                arcses = ' '.join(
                    "%d:%s" % (arc.head, arc.relation) for arc in arcs).split()

                text_tokens = list(text)
                inputs = self.tokenizer.encode_plus(text_tokens, is_pretokenized=True, add_special_tokens=True)

                entities = []
                for ent in train_json['entities']:
                    start_pos = ent['start_pos']
                    end_pos = ent['end_pos']
                    overlap = ent['overlap']
                    label_text = text[start_pos - 1:end_pos]
                    label_type = ent['label_type']
                    entities.append(Entity(start_pos, end_pos, label_type, label_text, overlap))
            examples.append(
                InputExample(os.path.join(self.data_dir, train_file), text, text_tokens, inputs, entities))
        return examples



    def collate_fn(self, datas):

        max_seq_len = max([data.text_len for data in datas])
        text_token_ids = []
        bert_masks = []
        raw_tag_ids = []
        tag_ids = []
        tag_masks = []
        texts = []
        text_lengths = []

        for data in datas:
            text = data.text
            text_lengths.append(data.text_len)
            texts.append(text)

            # 把 sep 删除掉
            text_token_ids.append(
                data.inputs.data['input_ids'][:-1] + [self.tokenizer.pad_token_id] * (max_seq_len - data.text_len))
            bert_masks.append(data.inputs.data['attention_mask'][:-1] + [0] * (max_seq_len - data.text_len))
            raw_tag_ids.append(data.tag_ids)

            tag_ids.append(data.tag_ids + [0] * (max_seq_len - data.text_len))
            tag_masks.append([1] * data.text_len + [0] * (max_seq_len - data.text_len))
        text_token_ids = torch.LongTensor(np.array(text_token_ids))
        bert_masks = torch.ByteTensor(np.array(bert_masks))
        tag_ids = torch.LongTensor(np.array(tag_ids))
        tag_masks = torch.ByteTensor(np.array(tag_masks))
        text_lengths = torch.LongTensor(np.array(text_lengths))
        return text_token_ids, bert_masks, tag_ids, tag_masks, text_lengths, raw_tag_ids, texts


    def _load_testset(self):
        """
        加载测试集
        :return:
        """
        examples = []
        with open(os.path.join(self.data_dir,'validate_data.json'),'r') as f:
            test_json = json.load(f)
            for file_name in test_json.keys():
                text = test_json[file_name]
                text_tokens = list(text)
                inputs = self.tokenizer.encode_plus(text_tokens, is_pretokenized=True, add_special_tokens=True)
                examples.append(InputExample(file_path=file_name,text=text,text_tokens=text_tokens,inputs=inputs))

        return examples


    def collate_fn_4_inference(self,datas):
        """

        :return:
        """
        max_seq_len = max([data.text_len for data in datas])
        text_token_ids = []
        bert_masks = []
        tag_masks = []
        texts = []
        text_lengths = []
        file_names = []

        for data in datas:
            file_names.append(data.file_path)
            text = data.text
            text_lengths.append(data.text_len)
            texts.append(text)

            # 把 sep 删除掉
            text_token_ids.append(
                data.inputs.data['input_ids'][:-1] + [self.tokenizer.pad_token_id] * (max_seq_len - data.text_len))
            bert_masks.append(data.inputs.data['attention_mask'][:-1] + [0] * (max_seq_len - data.text_len))

            tag_masks.append([1] * data.text_len + [0] * (max_seq_len - data.text_len))
        text_token_ids = torch.LongTensor(np.array(text_token_ids))
        bert_masks = torch.ByteTensor(np.array(bert_masks))
        tag_masks = torch.ByteTensor(np.array(tag_masks))
        text_lengths = torch.LongTensor(np.array(text_lengths))
        return text_token_ids, bert_masks, tag_masks, text_lengths, texts,file_names
