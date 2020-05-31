# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 12:03 下午
# @Author  : lizhen
# @FileName: inference.py
# @Description:
import argparse
import collections
import torch
import numpy as np
from tqdm import tqdm
# import data_process.data_loaders as module_data
from data_process import military_data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
from base import base_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import transformers as optimization
import os
from utils.model_utils import get_restrain_crf
from utils.data_process_utils import extract_entities
from collections import defaultdict
import json


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('test')

    # setup data_set, data_process instances
    data_set = config.init_obj('test1_set', module_data_process)
    # setup data_loader instances
    test_dataloader = config.init_obj('data_loader', module_dataloader, data_set.test_set,
                                      collate_fn=data_set.collate_fn_4_inference)

    restrain = get_restrain_crf(9)
    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch, restrain=restrain)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.cuda()
    model.eval()

    # inference
    f_submit = open('data/military_ner/submit.json', 'w', encoding='utf8')

    with torch.no_grad():
        submit = defaultdict(list)
        for i, batch_data in enumerate(tqdm(test_dataloader)):
            text_token_ids, bert_masks, tag_masks, text_lengths, texts, file_names = batch_data
            text_token_ids = text_token_ids.cuda()
            bert_masks = bert_masks.cuda()
            tag_masks = tag_masks.cuda()
            text_lengths = text_lengths.cuda()

            pred_tags = model(text_token_ids, bert_masks, text_lengths).squeeze(1)
            # crf 解码
            scores, best_path = model.crf.decode(emissions=pred_tags, mask=tag_masks)
            for pred, text, file_name in zip(best_path, texts, file_names):
                pred_entities = extract_entities(pred, text)

                for entity in pred_entities:
                    submit[file_name].append(
                        {
                            "label_type": entity['label_type'],
                            "overlap": 0,
                            "start_pos": entity['start_pos'],
                            "end_pos": entity['end_pos']
                        }
                    )
        json.dump(submit,f_submit,ensure_ascii=False)

    f_submit.close()


def run(config_file, model_path):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=model_path, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())

    main(config)


if __name__ == '__main__':
    run('configs/base_config.json', 'saved/ner/models/seq_label/0530_143426/model_best.pth')
