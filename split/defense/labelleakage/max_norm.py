import torch
import pdb

def max_norm(grad):
  eps = torch.finfo(torch.float32).eps

  g_norm = grad.pow(2).sum(dim=list(range(1, len(grad.shape)))).sqrt()
  g_max = torch.max(g_norm) 

  # the standard deviation to be determined
  sigma = torch.sqrt(torch.maximum(g_max ** 2 / (g_norm ** 2+ eps) - 1, torch.zeros(len(g_norm)).to(g_norm.device)))

  # gausian noise
  std_gaus_noice = torch.normal(torch.zeros_like(sigma), 1.0)
  gaus_noice = std_gaus_noice * sigma

  # perturbed gradient
  dims = [1] * len(grad.shape)
  pertubated_grad = grad + grad * gaus_noice.view(len(grad), *dims[1:]) 

  return pertubated_grad
