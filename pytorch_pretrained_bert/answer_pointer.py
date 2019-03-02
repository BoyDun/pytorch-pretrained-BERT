#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import math
import h5py
import torch
import torch.nn.functional as F
import numpy as np

def answer_search(answer_prop, mask, max_tokens=15):
    """
    global search best answer for model predict
    :param answer_prop: (batch, answer_len, context_len)
    :return:
    """
    batch_size = answer_prop.shape[0]
    context_len = answer_prop.shape[2]

    # get min length
    lengths = mask.data.eq(1).long().sum(1).squeeze()
    min_length, _ = torch.min(lengths, 0)
    min_length = min_length.item()

    # max move steps
    max_move = max_tokens + context_len - min_length
    max_move = min(context_len, max_move)

    ans_s_p = answer_prop[:, 0, :]
    ans_e_p = answer_prop[:, 1, :]
    b_zero = answer_prop.new_zeros(batch_size, 1)

    # each step, move ans-start-prop matrix to right with one element
    ans_s_e_p_lst = []
    for i in range(max_move):
        tmp_ans_s_e_p = ans_s_p * ans_e_p
        ans_s_e_p_lst.append(tmp_ans_s_e_p)

        ans_s_p = ans_s_p[:, :(context_len - 1)]
        ans_s_p = torch.cat((b_zero, ans_s_p), dim=1)
    ans_s_e_p = torch.stack(ans_s_e_p_lst, dim=2)

    # get the best end position, and move steps
    max_prop1, max_prop_idx1 = torch.max(ans_s_e_p, 1)
    max_prop2, max_prop_idx2 = torch.max(max_prop1, 1)

    ans_e = max_prop_idx1.gather(1, max_prop_idx2.unsqueeze(1)).squeeze(1)
    # ans_e = max_prop_idx1[:, max_prop_idx2].diag()  # notice that only diag element valid, the same with top ways
    ans_s = ans_e - max_prop_idx2

    ans_range = torch.stack((ans_s, ans_e), dim=1)
    return ans_range

def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax

class PointerAttention(torch.nn.Module):
    r"""
    attention mechanism in pointer network
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        Hk_last(batch, hidden_size): the last hidden output of previous time

    Outputs:
        beta(batch, context_len): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(PointerAttention, self).__init__()

        self.linear_wr = torch.nn.Linear(input_size, hidden_size)
        self.linear_wa = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wf = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hr, Hr_mask, Hk_pre):
        wr_hr = self.linear_wr(Hr)  # (context_len, batch, hidden_size)
        wa_ha = self.linear_wa(Hk_pre).unsqueeze(0)  # (1, batch, hidden_size)
        f = F.tanh(wr_hr + wa_ha)  # (context_len, batch, hidden_size)

        beta_tmp = self.linear_wf(f) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, context_len)

        beta = masked_softmax(beta_tmp, m=Hr_mask, dim=1)
        return beta


class UniBoundaryPointer(torch.nn.Module):
    r"""
    Unidirectional Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0(batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
        **hidden** (batch, hidden_size), [(batch, hidden_size)]: rnn last state
    """
    answer_len = 2

    def __init__(self, mode, input_size, hidden_size, enable_layer_norm):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enable_layer_norm = enable_layer_norm

        self.attention = PointerAttention(input_size, hidden_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hr, Hr_mask, h_0=None):
        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = Hr.new_zeros(batch_size, self.hidden_size)

        hidden = (h_0, h_0) if self.mode == 'LSTM' and isinstance(h_0, torch.Tensor) else h_0
        beta_out = []

        for t in range(self.answer_len):
            attention_input = hidden[0] if self.mode == 'LSTM' else hidden
            beta = self.attention.forward(Hr, Hr_mask, attention_input)  # (batch, context_len)
            beta_out.append(beta)

            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)

            if self.enable_layer_norm:
                context_beta = self.layer_norm(context_beta)  # (batch, input_size)

            hidden = self.hidden_cell.forward(context_beta, hidden)  # (batch, hidden_size), (batch, hidden_size)

        result = torch.stack(beta_out, dim=0)
        return result, hidden


class BoundaryPointer(torch.nn.Module):
    r"""
    Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - bidirectional: Bidirectional or Unidirectional
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization

    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm):
        super(BoundaryPointer, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.left_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        if bidirectional:
            self.right_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr_mask = Hr_mask.float()
        Hr = self.dropout.forward(Hr)

        left_beta, _ = self.left_ptr_rnn.forward(Hr, Hr_mask, h_0)
        rtn_beta = left_beta
        if self.bidirectional:
            right_beta_inv, _ = self.right_ptr_rnn.forward(Hr, Hr_mask, h_0)
            right_beta = right_beta_inv[[1, 0], :]

            rtn_beta = (left_beta + right_beta) / 2

        # todo: unexplainable
        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = rtn_beta + new_mask.unsqueeze(0)

        return rtn_beta

