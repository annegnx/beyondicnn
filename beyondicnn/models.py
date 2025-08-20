import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# adapted from https://github.com/arturs-berzins/relu_edge_subdivision/blob/master/NN.py


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim=2, widths=[2, 2], residuals=False):
        super(FeedForwardNet, self).__init__()
        self.layers = [nn.Linear(input_dim, widths[0], bias=True)]
        for i in range(len(widths)-1):
            self.layers.append(nn.Linear(widths[i], widths[i+1], bias=True))
        self.layers.append(nn.Linear(widths[-1], 1, bias=True))
        self.residuals = residuals

        # for layer in self.layers:
        #     torch.nn.init.normal_(layer.weight.data)
        #     torch.nn.init.normal_(layer.bias.data)

        # with torch.no_grad():
        #     self.layers[-1].weight.data = torch.abs(self.layers[-1].weight.data)

        self.fcs = nn.ModuleList(self.layers)
        if residuals:
            self.residual_layers = [nn.Linear(input_dim, w, bias=True)
                                    for w in widths + [1]]
            self.res_fcs = nn.ModuleList(self.residual_layers)

        # Depth
        self.D = len(self.layers)
        self.ks = [input_dim] + widths + [1]
        # Placeholder for intermediate values
        self.fs = [None]*(self.D+1)
        self.ls = list(range(1, self.D+1))
        self.Ks = [sum(self.ks[1:l]) for l in self.ls] + \
            [sum(self.ks[1:])]  # similar to neuron_idx

    def forward(self, x):
        x_ = x.clone()
        self.fs[1] = self.fcs[0](x)
        for i in range(2, self.D+1):
            if self.residuals:
                self.fs[i] = self.fcs[i -
                                      1](F.relu(self.fs[i-1] + self.res_fcs[i-2](x_)))
            else:
                # self.fs[i] = self.fcs[i-1](F.relu(self.fs[i-1]))
                self.fs[i] = self.fcs[i -
                                      1](F.relu(self.fs[i-1]))
        return self.fs[-1]

    def eval_block(self, x, device='cpu'):
        '''Like forward but preallocates and slices the tensor for storing values.'''
        Ks = self.Ks
        x_ = x.clone()
        # print("KS", Ks)
        fs = torch.empty(len(x), Ks[-1], device=device)
        fs[:, Ks[0]:Ks[1]] = self.fcs[0](x)
        for i in range(1, self.D):
            # print(fs[:, Ks[i]:Ks[i+1]].shape)
            # print(self.fcs[i](F.relu(fs[:, Ks[i-1]:Ks[i]])).shape)
            if self.residuals:
                fs[:, Ks[i]:Ks[i+1]
                   ] = self.fcs[i](F.relu(fs[:, Ks[i-1]:Ks[i]] + self.res_fcs[i-1](x_)))
            else:
                fs[:, Ks[i]:Ks[i+1]
                   ] = self.fcs[i](F.relu(fs[:, Ks[i-1]:Ks[i]]))
        return fs

    def init_weights(self, mean=0.0, std=1.0):
        with torch.no_grad():
            for layer in self.fcs:
                layer.weight.data.normal_(mean, std)
                layer.bias.data.normal_(0.0, 0.001)

    def w_clip(self):
        with torch.no_grad():
            for layer in self.fcs[1:]:
                layer.weight.data.clamp_(0)
