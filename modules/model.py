import os
import sys
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
def Alexnet():
    model = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4)
        nn.LocalResponseNorm(size = 96)
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1)
        nn.LocalResponseNorm(256)
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1)
        nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1)
        nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1)
        nn.MaxPool2d(kernel_size = 3, stride = 2)
        nn.Linear()
    )
'''

class SiamFC(nn.Module):
    def __init__(self, model_path = None):
        super(SiamFC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True)
        )

        self.branch = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unknow model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderdDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)

    def set_learnable_params(self, layers):
        for k,p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, z, x):
        z = self.branch(z)
        x = self.branch(x)

        out = self.xcorr(z, x)

        return out

    def xcorr(self, z, x):
        return F.conv2d(z, x)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]

        loss = pos_loss.sum() + neg_loss.sum()
        return loss

class Accuracy():
    def __call__(self, pos_score, neg_score):

        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]
        
class Precision():
    def __call__(self, pos_score, neg_score):

        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        return prec.data[0]


