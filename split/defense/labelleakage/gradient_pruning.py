import torch
import numpy as np
import pdb

def grad_pruning(grads, compress_ratio):
  flatten_grads = grads.flatten()
  keep_ratio = 1 - compress_ratio
  keep_num = int(np.ceil(keep_ratio * len(flatten_grads)))
  keep_indices = torch.abs(flatten_grads).topk(keep_num)[1]

  new_grads = torch.zeros_like(flatten_grads)
  new_grads[keep_indices] = flatten_grads[keep_indices]
  new_grads = new_grads.reshape(grads.shape)
  return new_grads
