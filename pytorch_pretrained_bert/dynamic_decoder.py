from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()

class DynamicDecoder(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, max_dec_steps, dropout_ratio):
        super(DynamicDecoder, self).__init__()
        self.max_dec_steps = max_dec_steps
        self.decoder = nn.LSTM(4 * hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        init_lstm_forget_bias(self.decoder)

        self.maxout_start = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)
        self.maxout_end = MaxOutHighway(hidden_dim, maxout_pool_size, dropout_ratio)

    def forward(self, U, d_mask, span):
        b, m, _ = list(U.size())

        curr_mask_s,  curr_mask_e = None, None
        results_mask_s, results_s = [], []
        results_mask_e, results_e = [], []
#         step_losses = []

        mask_mult = (1.0 - d_mask.float()) * (-1e30)
        indices = torch.arange(0, b, out=torch.LongTensor(b))

        # ??how to initialize s_i_1, e_i_1
        s_i_1 = torch.zeros(b, ).long()
        e_i_1 = torch.sum(d_mask, 1)
        e_i_1 = e_i_1 - 1

        if use_cuda:
            s_i_1 = s_i_1.cuda()
            e_i_1 = e_i_1.cuda()
            indices = indices.cuda()

        dec_state_i = None
        s_target = None
        e_target = None
        if span is not None:
            s_target = span[:, 0]
            e_target = span[:, 1]
        u_s_i_1 = U[indices, s_i_1, :]  # b x 2l
        for _ in range(self.max_dec_steps):
            u_e_i_1 = U[indices, e_i_1, :]  # b x 2l
            u_cat = torch.cat((u_s_i_1, u_e_i_1), 1)  # b x 4l

            lstm_out, dec_state_i = self.decoder(u_cat.unsqueeze(1), dec_state_i)
            h_i, c_i = dec_state_i

            s_i_1, curr_mask_s, start_logits = self.maxout_start(h_i, U, curr_mask_s, s_i_1,
                                                                u_cat, mask_mult, s_target)
            u_s_i_1 = U[indices, s_i_1, :]  # b x 2l
            u_cat = torch.cat((u_s_i_1, u_e_i_1), 1)  # b x 4l

            e_i_1, curr_mask_e, end_logits = self.maxout_end(h_i, U, curr_mask_e, e_i_1,
                                                              u_cat, mask_mult, e_target)

#             if span is not None:
#                 step_loss = step_loss_s + step_loss_e
#                 step_losses.append(step_loss)

            results_mask_s.append(curr_mask_s)
            results_s.append(s_i_1)
            results_mask_e.append(curr_mask_e)
            results_e.append(e_i_1)

        result_pos_s = torch.sum(torch.stack(results_mask_s, 1), 1).long()
        result_pos_s = result_pos_s - 1
        idx_s = torch.gather(torch.stack(results_s, 1), 1, result_pos_s.unsqueeze(1)).squeeze()

        result_pos_e = torch.sum(torch.stack(results_mask_e, 1), 1).long()
        result_pos_e = result_pos_e - 1
        idx_e = torch.gather(torch.stack(results_e, 1), 1, result_pos_e.unsqueeze(1)).squeeze()

#         loss = None

#         if span is not None:
#             sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
#             batch_avg_loss = sum_losses / self.max_dec_steps
#             loss = torch.mean(batch_avg_loss)

        return idx_s, idx_e, start_logits, end_logits #, loss


class MaxOutHighway(nn.Module):
    def __init__(self, hidden_dim, maxout_pool_size, dropout_ratio):
        super(MaxOutHighway, self).__init__()
        self.hidden_dim = hidden_dim
        self.maxout_pool_size = maxout_pool_size

        self.r = nn.Linear(5 * hidden_dim, hidden_dim, bias=False)
        #self.dropout_r = nn.Dropout(p=dropout_ratio)

        self.m_t_1_mxp = nn.Linear(3 * hidden_dim, hidden_dim*maxout_pool_size)
        #self.dropout_m_t_1 = nn.Dropout(p=dropout_ratio)

        self.m_t_2_mxp = nn.Linear(hidden_dim, hidden_dim*maxout_pool_size)
        #self.dropout_m_t_2 = nn.Dropout(p=dropout_ratio)

        self.m_t_12_mxp = nn.Linear(2 * hidden_dim, maxout_pool_size)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, h_i, U, curr_mask, idx_i_1, u_cat, mask_mult, target=None):
        b, m, _ = list(U.size())

        r = F.tanh(self.r(torch.cat((h_i.view(-1, self.hidden_dim), u_cat), 1)))  # b x 5l => b x l
        #r = self.dropout_r(r)

        r_expanded = r.unsqueeze(1).expand(b, m, self.hidden_dim).contiguous()  # b x m x l

        m_t_1_in = torch.cat((U, r_expanded), 2).view(-1, 3*self.hidden_dim)  # b*m x 3l

        m_t_1 = self.m_t_1_mxp(m_t_1_in)  # b*m x p*l
        #m_t_1 = self.dropout_m_t_1(m_t_1)
        m_t_1, _ = m_t_1.view(-1, self.hidden_dim, self.maxout_pool_size).max(2) # b*m x l

        m_t_2 = self.m_t_2_mxp(m_t_1)  # b*m x l*p
        #m_t_2 = self.dropout_m_t_2(m_t_2)
        m_t_2, _ = m_t_2.view(-1, self.hidden_dim, self.maxout_pool_size).max(2)  # b*m x l

        alpha_in = torch.cat((m_t_1, m_t_2), 1)  # b*m x 2l
        alpha = self.m_t_12_mxp(alpha_in)  # b * m x p
        alpha, _ = alpha.max(1)  # b*m
        alpha = alpha.view(-1, m) # b x m

        logits = alpha + mask_mult  # b x m
        alpha = F.log_softmax(logits, 1)  # b x m
        _, idx_i = torch.max(alpha, dim=1)

        if curr_mask is None:
            curr_mask = (idx_i == idx_i)
        else:
            idx_i = idx_i*curr_mask.long()
            idx_i_1 = idx_i_1*curr_mask.long()
            curr_mask = (idx_i != idx_i_1)

#         step_loss = None

#         if target is not None:
#            step_loss = self.loss(alpha, target)
#            step_loss = step_loss * curr_mask.float()

        return idx_i, curr_mask, logits #logits size b x m, step_loss
