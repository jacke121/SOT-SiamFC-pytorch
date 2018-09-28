import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable


def set_optimizer(model, lr_base, lr_mult = opts['lr_mult'], momentum = opts['momentum'], w_decay = opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startwith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum = momentum, weight_decay = w_decay)
    return optimizer

def train():

    model = SiamFC()