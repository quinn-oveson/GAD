"""Settings for one experiment. Edit this file, then run `python gad.py`.

Every run of gad.py saves a verbatim copy of this file next to the plot it made
(see the docstring of gad.py), so there is no need to keep notes on what was set.
"""
import torch
import torch.nn as nn


# ---- data ----
def f1(x):
    return torch.exp(torch.sin(2 * torch.pi * x))
f1.tex = r"e^{\sin(2\pi x)}"  # LaTeX for the plot titles


def f2(x):
    return torch.exp(3 * x)
f2.tex = r"e^{3x}"


def f3(x):
    return torch.cos(torch.exp(3 * x))
f3.tex = r"\cos(e^{3x})"

def f3coef3pt5(x):
    return torch.cos(torch.exp(3.5 * x))
f3coef3pt5.tex = r"\cos(e^{3.5x})"

def f3coef4(x):
    return torch.cos(torch.exp(4 * x))
f3coef4.tex = r"\cos(e^{4x})"

def f3coef5(x):
    return torch.cos(torch.exp(5 * x))
f3coef5.tex = r"\cos(e^{5x})"


TARGET = f1                 # function that creates the data, torch tensor -> torch tensor
INTERVAL = (-1.0, 1.0)
TRAIN_SPACING = "even"      # "even": evenly spaced on INTERVAL, "random": uniform random draws
N_TRAIN = 201
N_TEST = 601                # evenly spaced test points; any training point not among
                            # them is added, so the risk covers both test and training

# ---- model ----
WIDTH = 128
DEPTH = 3                   # number of hidden layers
ACTIVATION = nn.Tanh        # any nn.Module class, e.g. nn.Tanh, nn.ReLU, nn.GELU

# ---- training ----
EPOCHS = 2400
LR = 1e-3

# ---- GAD ----
COLUMN_ORDER = "ck_first"   # order parameters become "modeled" in: "ck_first" or "random"
                            # "ck_first": last layer (the CK features), then all earlier layers
RCOND = 1e-14               # relative cutoff for singular values
SEED = 0                    # network init, random training points, column order

# ---- data double descent (data_dd.py) ----
DD_SAMPLING = "uniform"     # where the points of D come from: "uniform": uniform random on
                            # INTERVAL, "grid": without replacement from the N_TEST grid
DD_N_MAX = 400              # largest sample set D; each D is the previous one plus one
                            # random point ("grid" needs DD_N_MAX <= N_TEST)
DD_DRAWS = 10               # independent nested sequences; plots show median and 10-90% band
