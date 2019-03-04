
    
import io
import json
import logging
import os
import sys
import time

import torch
import numpy as np
from data_util.data_batcher import get_batch_generator
from data_util.official_eval_helper import get_json_data, generate_answers
# from model import CoattentionModel
# from model_baseline import Baseline


logging.basicConfig(level=logging.INFO)

use_cuda = torch.cuda.is_available()

class Processor(object):
    def __init__(self):
        # self.glove_path = os.path.join(config.data_dir, "glove.6B.{}d.txt".format(config.embedding_size))
        # self.emb_matrix, self.word2id, self.id2word = get_glove(self.glove_path, config.embedding_size)

        self.train_context_path = os.path.join(config.data_dir, "train.context")
        # self.train_qn_path = os.path.join(config.data_dir, "train.question")
        self.train_ans_path = os.path.join(config.data_dir, "train.span")
        # self.dev_context_path = os.path.join(config.data_dir, "dev.context")
        # self.dev_qn_path = os.path.join(config.data_dir, "dev.question")
        # self.dev_ans_path = os.path.join(config.data_dir, "dev.span")

    def get_mask_from_seq_len(self, seq_mask):
        seq_lens = np.sum(seq_mask, 1)
        max_len = np.max(seq_lens)
        indices = np.arange(0, max_len)
        mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
        return mask

    def get_data(self, batch, is_train=True):
        # qn_mask = self.get_mask_from_seq_len(batch.qn_mask)
        # qn_mask_var = torch.from_numpy(qn_mask).long()

        context_mask = self.get_mask_from_seq_len(batch.context_mask)
        context_mask_var = torch.from_numpy(context_mask).long()

        # qn_seq_var = torch.from_numpy(batch.qn_ids).long()
        context_seq_var = torch.from_numpy(batch.context_ids).long()

        if is_train:
            span_var = torch.from_numpy(batch.ans_span).long()

        if use_cuda:
            # qn_mask_var = qn_mask_var.cuda()
            context_mask_var = context_mask_var.cuda()
            # qn_seq_var = qn_seq_var.cuda()
            context_seq_var = context_seq_var.cuda()
            if is_train:
                span_var = span_var.cuda()

        if is_train:
            return context_mask_var, span_var
        else:
            return context_mask_var
