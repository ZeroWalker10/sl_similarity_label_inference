#!/usr/bin/env python
# coding: utf-8
import sys
import os
import datetime
import time
import random
# import logging
import torch
import numpy as np
from .solver import solve_isotropic_covariance
import pdb

def gradient_KL_perturb(grads, batch_labels, s=0.6):
    ngrads = grads.reshape((grads.shape[0], -1))

    uv_choice = "uv"  # "uv"
    # init_scale = 1.0
    init_scale = s

    p_frac = 'pos_frac'
    # dynamic = True
    dynamic = False
    error_prob_lower_bound = None
    sumKL_threshold = 0.16 # 0.25 #0.81 #0.64#0.16 #0.64
    # sumKL_threshold = threshold # 0.25 #0.81 #0.64#0.16 #0.64

    if dynamic and (error_prob_lower_bound is not None):
        sumKL_threshold = (2 - 4 * error_prob_lower_bound) ** 2

    def grad_fn(g):
        pos_g = g[batch_labels == 1]
        neg_g = g[batch_labels == 0]

        y = batch_labels.float().reshape([-1, 1])
        # pos_g = g[y==1]
        '''
        pos_g = tf.boolean_mask(
            g, tf.tile(
                tf.cast(
                    y, dtype=tf.int32), [
                    1, tf.shape(g)[1]]))
        pos_g = tf.reshape(pos_g, [-1, tf.shape(g)[1]])
        '''

        pos_g_mean = torch.mean(
            pos_g, dim=0, keepdim=True)  # shape [1, d]
        pos_coordinate_var = torch.mean(
            torch.square(
                pos_g - pos_g_mean),
            dim=0)  # use broadcast

        # neg_g = g[y==0]
        '''
        neg_g = tf.boolean_mask(g, tf.tile(
            1 - tf.cast(y, dtype=tf.int32), [1, tf.shape(g)[1]]))
        neg_g = tf.reshape(neg_g, [-1, tf.shape(g)[1]])
        '''

        neg_g_mean = torch.mean(
            neg_g, dim=0, keepdim=True)  # shape [1, d]
        neg_coordinate_var = torch.mean(
            torch.square(neg_g - neg_g_mean), dim=0)

        avg_pos_coordinate_var = torch.mean(pos_coordinate_var)
        avg_neg_coordinate_var = torch.mean(neg_coordinate_var)

        if torch.isnan(avg_pos_coordinate_var) or torch.isnan(
                avg_neg_coordinate_var):
            # no negative/positive instances in this batch
            return g
        g_diff = pos_g_mean - neg_g_mean
        # g_diff_norm = float(tf.norm(tensor=g_diff).numpy())
        g_diff_norm = torch.norm(g_diff)
        if uv_choice == 'uv':
            u = avg_neg_coordinate_var
            v = avg_pos_coordinate_var
        elif uv_choice == 'same':
            u = (avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
            v = (avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
        elif uv_choice == 'zero':
            u, v = 0.0, 0.0
        # d = tf.cast(tf.shape(g)[1], dtype=tf.float32)
        d = float(g.shape[1])
        if p_frac == 'pos_frac':
            p = torch.mean(y)
        else:
            p = float(p_frac)

        scale = init_scale
        g_norm_square = g_diff_norm ** 2

        def compute_lambdas(
            u,
            v,
            scale,
            d,
            g_norm_square,
            p,
            sumKL_threshold,
            pos_g_mean,
            neg_g_mean,
                g_diff):

            lam10, lam20, lam11, lam21 = None, None, None, None
            while True:
                P = scale * g_norm_square
                lam10, lam20, lam11, lam21, sumKL = \
                                    solve_isotropic_covariance(u=u,
                                               v=v,
                                               d=d,
                                               g_norm_square=g_norm_square,
                                               p=p,
                                               P=P,
                                               lam10_init=lam10,
                                               lam20_init=lam20,
                                               lam11_init=lam11,
                                               lam21_init=lam21)
                if not dynamic or sumKL <= sumKL_threshold:
                    break

                scale *= 1.5  # loosen the power constraint
            return lam10, lam20, lam11, lam21, sumKL

        lam10, lam20, lam11, lam21, sumKL = compute_lambdas(
            u.cpu(), v.cpu(), scale, d, g_norm_square.cpu(), p.cpu(), 
            sumKL_threshold, pos_g_mean,
            neg_g_mean.cpu(), g_diff.cpu())

        '''
        lam10, lam20, lam11, lam21, sumKL = torch.reshape(
            lam10, shape=[1]), torch.reshape(
            lam20, shape=[1]), torch.reshape(
            lam11, shape=[1]), torch.reshape(
                lam21, shape=[1]), torch.reshape(
                    sumKL, shape=[1])
        '''

        perturbed_g = g
        # y_float = tf.cast(y, dtype=tf.float32)

        noise_1 = torch.reshape(torch.multiply(torch.randn(size=\
                            np.shape(y.cpu())), y.cpu()), \
                             shape=(-1, 1)) * g_diff.cpu() * \
                    (np.sqrt(np.abs(lam11 - lam21)) / g_diff_norm.cpu())

        noise_2 = torch.randn(size=np.shape(
            g.cpu())) * torch.reshape(y.cpu(), shape=(-1, 1)) * \
            np.sqrt(np.maximum(lam21, 0.0))

        noise_3 = torch.reshape(torch.multiply(torch.randn(size=\
                            np.shape(y.cpu())), 1 - y.cpu()),
                             shape=(-1, 1)) * g_diff.cpu() * \
                    (np.sqrt(np.abs(lam10 - lam20)) / g_diff_norm.cpu())

        noise_4 = torch.randn(size=np.shape(
            g.cpu())) * torch.reshape(1 - y.cpu(), shape=(-1, 1)) * \
            np.sqrt(np.maximum(lam20, 0.0))

        perturbed_g += (noise_1 + noise_2 + noise_3 + noise_4).to(g.device)
        return perturbed_g

    ngrads = grad_fn(ngrads)
    return ngrads.reshape(grads.shape)

def gradient_KL_perturb_mul(grads, batch_labels, s=0.6):
    unq_labels = torch.unique(batch_labels)
    flatten_grads = grads.view(grads.shape[0], -1)

    labels_grads = flatten_grads.clone()
    num = len(unq_labels)
    labels_grads = 0.0
    for i in range(num):
        lab = unq_labels[i]
        labels = batch_labels.clone()
        labels[labels == lab] = 1
        labels[labels != lab] = 0
        
        lab_grads = gradient_KL_perturb(flatten_grads, labels, s)
        labels_grads = labels_grads + lab_grads
    labels_grads = labels_grads / num
    return labels_grads.view(grads.shape)
    '''
    indexes = np.arange(num)
    np.random.shuffle(indexes)
    i = 0
    while i + 1 < len(indexes):
        labels = batch_labels.clone()

        poses = flatten_grads[batch_labels == unq_labels[indexes[i]]]
        negs = flatten_grads[batch_labels == unq_labels[indexes[i+1]]]
        plabs = labels[batch_labels == unq_labels[indexes[i]]]
        nlabs = labels[batch_labels == unq_labels[indexes[i+1]]]
        plabs[:] = 1
        nlabs[:] = 0

        part_grads = torch.cat([poses, negs])
        part_labs = torch.cat([plabs, nlabs])
        lab_grads = gradient_KL_perturb(part_grads, part_labs, s)

        flatten_grads[batch_labels == unq_labels[indexes[i]]] = lab_grads[part_labs == 1]
        flatten_grads[batch_labels == unq_labels[indexes[i+1]]] = lab_grads[part_labs == 0]
        i = i + 2
    return flatten_grads.view(grads.shape)
    '''
def gradient_KL_perturb_rnd(grads, batch_labels, s=0.6):
    unq_labels = torch.unique(batch_labels)
    flatten_grads = grads.view(grads.shape[0], -1)

    labels_grads = flatten_grads.clone()
    num = len(unq_labels)

    pick_lab = np.random.choice(unq_labels.cpu().numpy())
    labels = batch_labels.clone()
    labels[labels == pick_lab] = 1
    labels[labels != pick_lab] = 0
    lab_grads = gradient_KL_perturb(flatten_grads, labels, s)
    return labels_grads.view(grads.shape)
