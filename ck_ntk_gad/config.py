"""Settings for one experiment. Edit this file, then run `python gad.py`.

Every run of gad.py saves a verbatim copy of this file next to the plot it made
(see the docstring of gad.py), so there is no need to keep notes on what was set.
"""
import torch
import torch.nn as nn


# ---- data ----
def f1(x):
    return torch.exp(torch.sin(2 * torch.pi * x))


TARGET = f1                 # function that creates the data, torch tensor -> torch tensor
INTERVAL = (-1.0, 1.0)
TRAIN_SPACING = "even"      # "even": evenly spaced on INTERVAL, "random": uniform random draws
N_TRAIN = 201
N_TEST = 601                # test points are always evenly spaced

# ---- model ----
WIDTH = 128
DEPTH = 3                   # number of hidden layers
ACTIVATION = nn.Tanh        # any nn.Module class, e.g. nn.Tanh, nn.ReLU, nn.GELU

# ---- training ----
EPOCHS = 2400
LR = 1e-3

# ---- GAD ----
RCOND = 1e-14               # relative cutoff for singular values
SEED = 0                    # network init, random training points, column order
