"""Builds the data set in config.py and trains the MLP on it.

Run directly (`python train.py`) to train and print the test error.
"""
from types import SimpleNamespace

import torch

import config
from model import make_mlp


def trap_weights(n_points, a=-1.0, b=1.0):
    dx = (b - a) / (n_points - 1)
    w = torch.full((n_points, 1), dx)
    w[0] = w[-1] = dx / 2
    return w


def make_data():
    a, b = config.INTERVAL
    if config.TRAIN_SPACING == "even":
        x_train = torch.linspace(a, b, config.N_TRAIN).unsqueeze(1)
        w_train = trap_weights(config.N_TRAIN, a, b)
    elif config.TRAIN_SPACING == "random":
        g = torch.Generator().manual_seed(config.SEED)
        x_train = (a + (b - a) * torch.rand(config.N_TRAIN, generator=g)).sort().values.unsqueeze(1)
        # random draws get equal (Monte Carlo) weights instead of trapezoid weights
        w_train = torch.full((config.N_TRAIN, 1), (b - a) / config.N_TRAIN)
    else:
        raise ValueError(f"TRAIN_SPACING must be 'even' or 'random', got {config.TRAIN_SPACING!r}")
    x_test = torch.linspace(a, b, config.N_TEST).unsqueeze(1)
    w_test = trap_weights(config.N_TEST, a, b)
    return SimpleNamespace(x_train=x_train, y_train=config.TARGET(x_train), w_train=w_train,
                           x_test=x_test, y_test=config.TARGET(x_test), w_test=w_test)


def train(model, x, y, w, epochs=config.EPOCHS, lr=config.LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = (w * (model(x) - y) ** 2).sum()
        loss.backward()
        opt.step()
    return model


def test_norm(g, w_test):
    return (w_test * g ** 2).sum().sqrt().item()


def run():
    """Train the model described by config.py. Returns (model, data)."""
    data = make_data()
    torch.manual_seed(config.SEED)
    model = train(make_mlp(), data.x_train, data.y_train, data.w_train)
    return model, data


if __name__ == "__main__":
    model, data = run()
    with torch.no_grad():
        print("test error:", test_norm(model(data.x_test) - data.y_test, data.w_test))
