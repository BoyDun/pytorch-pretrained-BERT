import math
import h5py
import torch
import torch.nn.functional as F
import numpy as np

class DCR(torch.nn.Module):
	def __init__(self, max_ans_len, null_cosine_thresh):
		super(DCR, self).__init__()
		self.max_ans_len = max_ans_len
		self.cos_sim = torch.nn.CosineSimilarity(dim=0)
		self.null_cosine_thresh = null_cosine_thresh

	#returns logits
	def forward(self, sequence_outputs, idxs):
		batch_sz = sequence_outputs.size()[0]
		seq_len = sequence_outputs.size()[1]
		hidden_sz = sequence_outputs.size()[2]

		cuda0 = torch.device('cuda:0')

		start_logits = torch.zeros((batch_sz, seq_len)).cuda()
		end_logits = start_logits = torch.zeros((batch_sz, seq_len)).cuda()

		#iterate example by example over batch
		for idx, single_ex in enumerate(sequence_outputs.split(1)):
			seq_full = single_ex.squeeze(0)
			sep_idxs = idxs[idx] # in form [idx of first sep token, idx of last sep token]
			question_cat_repr = torch.cat((seq_full[1], seq_full[sep_idxs[0]-1])) # concatenate first and last hidden state of question
			# question_cat_repr = 2*786
			# seq_full is [384,786]
			# dict of max end ind values (format: idx:val)
			max_end_cos_vals = {}
			start_logits_ex = torch.zeros(seq_len).cuda()
			end_logits_ex = torch.zeros(seq_len).cuda()

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
						max_val = sim

				start_logits_ex[i] = max_val

				# case where end ind is taken in eng_logits_ex
				if end_ind in max_end_cos_vals:
					# replace in dict and in end_logits_ex
					if max_end_cos_vals[end_ind] < max_val:
						max_end_cos_vals[end_ind] = max_val
						end_logits_ex[end_ind] = max_val 
				else:
					max_end_cos_vals[end_ind] = max_val
					end_logits_ex[end_ind] = max_val

			#max value in start logits
			max_logit = torch.max(start_logits_ex)
			#print("STD:")
			#print(std)
			#print("MAX LOGIT:")
			#print(max_logit)
			# if max logit value is under thresh (i.e. .75), then set 0th index to 1 (max val.) for both start and end logits to signify null.
			
			if max_logit < (start_logits_ex.std() + start_logits_ex.mean()) or max_logit < (end_logits_ex.std() + end_logits_ex.mean()):
				start_logits_ex *= -1
				end_logits_ex *= -1
				start_logits_ex[start_logits_ex==0] = -0.001
				end_logits_ex[eng_logits_ex==0] = -0.001
			# if float(max_logit) < self.null_cosine_thresh:
				# start_logits_ex[0] = 1.0
				# end_logits_ex[0] = 1.0
				
			start_logits[idx] = start_logits_ex
			end_logits[idx] = end_logits_ex

		return start_logits, end_logits
