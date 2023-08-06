#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pdb

def rrtop_k(y, k, p, epsilon):
    '''
    y: real label
    k: top k
    p: prior probability
    epsilon: privacy argument
    '''
    topk = np.argsort(p)[::-1][:k]
    if y in topk:
        # random response
        rr_probs = np.ones(k)
        rr_probs[topk == y] = np.exp(epsilon)
        rr_probs = rr_probs / (np.exp(epsilon) + k - 1)

        y_hat = np.random.choice(topk, p=rr_probs)
    else:
        # uniform pick
        y_hat = np.random.choice(np.arange(len(p)))

    return y_hat

def rrwith_prior(y, p, epsilon):
    '''
    y: real label
    p: prior probability
    '''
    w_k = []
    sort_p = np.argsort(p)[::-1]
    for k in range(1, len(p) + 1):
        topk = sort_p[:k]
        prob = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        w_k.append(prob * np.sum([p[i] for i in topk]))

    k_star = np.argmax(w_k) + 1
    return rrtop_k(y, k_star, p, epsilon) 

