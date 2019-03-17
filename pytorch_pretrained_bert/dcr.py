import math
import h5py
import torch
import torch.nn.functional as F
import numpy as np

class DCR(torch.nn.Module):
	def __init__(self, max_ans_len):
		super(DCR, self).__init__()
		self.max_ans_len = max_ans_len

	#returns logits
	def forward(self, sequence_outputs, idxs):

		seq_len = sequence_outputs.size()[1]
		hidden_sz = sequence_outputs.size()[2]

		#iterate example by example over batch
		for idx, single_ex in enumerate(sequence_outputs.split(1)):
			squeeze_ex = single_ex.squeeze(0)
			sep_idxs = idxs[idx] # in form [idx of first sep token, idx of last sep token]
			
			# squeeze_ex is [384,786]
			for i in range(sep_idxs[0]):

				for j in range(i, min(len(word), i+max_len)):
					sub.append((i,j+1))