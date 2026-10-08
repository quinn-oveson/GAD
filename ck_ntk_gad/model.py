import torch
import torch.nn as nn

import config

torch.set_default_dtype(torch.float64)


def make_mlp(width=config.WIDTH, depth=config.DEPTH, act=config.ACTIVATION):
    layers, d_in = [], 1
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), act()]
        d_in = width
    layers.append(nn.Linear(d_in, 1))
    return nn.Sequential(*layers)
