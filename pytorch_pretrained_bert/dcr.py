import math
import h5py
import torch
import torch.nn.functional as F
import numpy as np

class DCR(torch.nn.Module):
	def __init__(self, max_ans_len):
		super(DCR, self).__init__()
		self.max_ans_len = max_ans_len
		self.cos_sim = torch.nn.CosineSimilarity(dim=0)

	#returns logits
	def forward(self, sequence_outputs, idxs):

		seq_len = sequence_outputs.size()[1]
		hidden_sz = sequence_outputs.size()[2]

		#iterate example by example over batch
		for idx, single_ex in enumerate(sequence_outputs.split(1)):
			seq_full = single_ex.squeeze(0)
			sep_idxs = idxs[idx] # in form [idx of first sep token, idx of last sep token]
			question_cat_repr = torch.cat((seq_full[1], seq_full[sep_idxs[0]-1])) # concatenate first and last hidden state of question
			# question_cat_repr = 2*786
			# seq_full is [384,786]
			# dict of max end ind values (format: idx:val)
			max_end_cos_vals = {}

			#iterate over each passage token
			for i in range(sep_idxs[0]+1, sep_idxs[1]):
				ans_start_hidden = seq_full[i]
				max_val = -10000
				end_ind = None
				#iterate starting from start token to max answer len or end
				for j in range(i, min(sep_idxs[1], i+self.max_ans_len)):
					sim = self.cos_sim(torch.cat((ans_start_hidden, seq_full[j])), question_cat_repr) 
					if sim > max_val:
						end_ind = j
						max_val = dim
				# if end_ind in max_end_cos_vals:
				# 	if max_end_cos_vals[end_ind] > max_val:

				# else:
				# 	max_end_cos_vals[end_ind] = max_val
