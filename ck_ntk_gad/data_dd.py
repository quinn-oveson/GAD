"""Data double descent: how the aliasing operator A(D) depends on the sample set D.

    python data_dd.py                          # train once, sweep D, plot
    python data_dd.py --config other.py --out f1

The network is trained once on the data in config.py and then held fixed, so the basis
functions (the Jacobian columns) never change; only the sample set D does. The modeled
set is the CK (the last layer), the unmodeled set is every other parameter, and

    A(D) = M_TM(D)^+ M_TU(D)          min-norm, singular values below RCOND cut
    R(D) = M_PU - M_PM A(D)           rows weighted so its norm is in the test norm

with theta_U taken from the NTK machine fit to the training data in config.py. D gets
uniform weights, which cancel out of A(D) and everything below.

For each of DD_DRAWS draws, D grows from 1 to DD_N_MAX random points, one point at a
time: uniform on INTERVAL, without replacement from the N_TEST grid, or from the grid in
bisection order so that every D is roughly evenly spaced (DD_SAMPLING). At each step the SVD of M_TM(D) gives the leverage scores h_i, and since
dropping point i is a rank-one change,

    A(D \\ {i}) - A(D) = -(M_TM^+ e_i) r_i^T / (1 - h_i),    r = (I - U U^T) M_TU

for every i at once. Its norm is reported raw and in function space (M_PM times it, in
the test norm). A point with h_i = 1 changes the rank when dropped; the formula does not
hold for it, so it is left out of the max and the mean.

Each run gets its own folder, runs/<time>_<settings>_data_dd_<sampling>/, holding data_dd.png,
config.py, run.json and results.npz, as in gad.py.
"""
import argparse
import importlib.util
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--config", metavar="FILE", help="use this file instead of config.py")
parser.add_argument("--out", metavar="DIR", default=HERE / "runs", type=Path,
                    help="folder the run folder is created in (default: runs)")
args = parser.parse_args()

if args.config:
    # has to happen before gad.py, model.py and train.py do `import config`
    spec = importlib.util.spec_from_file_location("config", args.config)
    sys.modules["config"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["config"])

import config
import train
from gad import PNG_KEY, config_values, fit_subset, git_state, jacobian, plot_title, stamp

COLUMNS = ["R_norm", "R_theta", "rank", "sigma_min", "h_max", "h_mean",
           "raw_max", "raw_mean", "fun_max", "fun_mean"]


def bisection_order(n_grid):
    """Grid indices ordered so that every prefix is roughly evenly spaced: the midpoint, then
    the two ends, then the quarter points, and so on. Each next point is the one farthest
    from those already chosen (ties go to the left)."""
    idx = torch.arange(n_grid)
    order = [n_grid // 2]
    dist = (idx - order[0]).abs()
    for _ in range(n_grid - 1):
        order.append(int(dist.argmax()))
        dist = torch.minimum(dist, (idx - order[-1]).abs())
    return torch.tensor(order)


def dd_sweep(model, data):
    J_train, J_test = jacobian(model, data.x_train), jacobian(model, data.x_test)
    n_params = J_train.shape[1]
    n_ck = config.WIDTH + 1
    ck, rest = slice(n_params - n_ck, n_params), slice(0, n_params - n_ck)
    gamma = fit_subset(J_train, data.y_train, data.w_train, torch.arange(n_params))
    theta_U = gamma[rest]

    sw_test = data.w_test.sqrt()
    QM = sw_test * J_test[:, ck]            # weighted M_PM
    Q = sw_test * J_test[:, rest]           # weighted M_PU
    Q_theta = Q @ theta_U

    a, b = config.INTERVAL
    stats = np.full((config.DD_DRAWS, config.DD_N_MAX, len(COLUMNS)), np.nan)
    for draw in range(config.DD_DRAWS):
        g = torch.Generator().manual_seed(config.SEED + 1 + draw)
        if config.DD_SAMPLING == "uniform":
            x_pool = a + (b - a) * torch.rand(config.DD_N_MAX, 1, generator=g)
        elif config.DD_SAMPLING in ("grid", "bisect"):
            if config.DD_N_MAX > config.N_TEST:
                raise ValueError(f"DD_N_MAX = {config.DD_N_MAX} is more than the "
                                 f"{config.N_TEST} grid points")
            grid = torch.linspace(a, b, config.N_TEST).unsqueeze(1)
            order = (torch.randperm(config.N_TEST, generator=g) if config.DD_SAMPLING == "grid"
                     else bisection_order(config.N_TEST))
            x_pool = grid[order[:config.DD_N_MAX]]
        else:
            raise ValueError("DD_SAMPLING must be 'uniform', 'grid' or 'bisect', "
                             f"got {config.DD_SAMPLING!r}")
        J_pool = jacobian(model, x_pool)
        T = J_pool[:, rest]                 # M_TU for the whole pool; D is its first n rows
        T_theta = T @ theta_U
        # every row of Q and T lies in the span of Z, so R can be formed in those
        # coordinates: n_test + DD_N_MAX columns instead of all the unmodeled parameters
        Z = torch.linalg.qr(torch.cat([Q, T]).T).Q
        Qc, Tc = Q @ Z, T @ Z

        for n in range(1, config.DD_N_MAX + 1):
            U, s, Vh = torch.linalg.svd(J_pool[:n, ck], full_matrices=False)
            keep = s > config.RCOND * s[0]
            U, s, Vh = U[:, keep], s[keep], Vh[keep]
            B_pinv = (Vh.T / s) @ U.T       # M_TM(D)^+
            K = QM @ B_pinv                 # weighted M_PM M_TM(D)^+

            Rc = Qc - K @ Tc[:n]
            R_norm = torch.linalg.eigvalsh(Rc @ Rc.T)[-1].clamp(min=0).sqrt().item()
            R_theta = (Q_theta - K @ T_theta[:n]).norm().item()

            # leave-one-out changes in A, all points at once
            h = (U ** 2).sum(dim=1)
            r_norm = (Tc[:n] - U @ (U.T @ Tc[:n])).norm(dim=1)
            ok = (1 - h) > 1e-10
            raw = (B_pinv.norm(dim=0) * r_norm / (1 - h))[ok]
            fun = (K.norm(dim=0) * r_norm / (1 - h))[ok]
            loo = ([raw.max().item(), raw.mean().item(), fun.max().item(), fun.mean().item()]
                   if ok.any() else [np.nan] * 4)

            stats[draw, n - 1] = [R_norm, R_theta, len(s), s[-1].item(),
                                  h.max().item(), h.mean().item(), *loo]
        print(f"draw {draw + 1}/{config.DD_DRAWS} done", flush=True)

    summary = {
        "n_ck": n_ck,
        "ck_rank_test": torch.linalg.matrix_rank(QM, rtol=config.RCOND).item(),
        "ck_test_error": train.test_norm(
            J_test[:, ck] @ fit_subset(J_train, data.y_train, data.w_train,
                                       torch.arange(n_params)[ck]) - data.y_test, data.w_test),
        "ntk_test_error": train.test_norm(J_test @ gamma - data.y_test, data.w_test),
    }
    return stats, summary


def band(ax, n, values, style, color, label):
    """Median over the draws, with the 10-90% band shaded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # steps where no draw has a value
        med = np.nanmedian(values, axis=0)
        lo, hi = np.nanpercentile(values, [10, 90], axis=0)
    ax.plot(n, med, style, color=color, label=label)
    ax.fill_between(n, lo, hi, color=color, alpha=0.2, lw=0)


def plot_dd(stats, summary, run_id):
    n = np.arange(1, stats.shape[1] + 1)
    col = {name: stats[:, :, i] for i, name in enumerate(COLUMNS)}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout="constrained")

    band(ax1, n, col["R_theta"], "-", "k", r"$\|R\theta_U\|$")
    band(ax1, n, col["R_norm"], "--", "k", r"$\|R\|$")
    ax1.set_yscale("log")
    ax1.set_ylabel("size at the prediction points")

    dA = r"(A(D\setminus i) - A(D))"
    band(ax2, n, col["raw_max"], "-", "tab:blue", rf"$\mathrm{{max}}_i\,\|{dA}\|$")
    band(ax2, n, col["raw_mean"], "-", "tab:green", rf"$\mathrm{{mean}}_i\,\|{dA}\|$")
    band(ax2, n, col["fun_max"], "--", "tab:blue", rf"$\mathrm{{max}}_i\,\|M_{{PM}}{dA}\|$")
    band(ax2, n, col["fun_mean"], "--", "tab:green", rf"$\mathrm{{mean}}_i\,\|M_{{PM}}{dA}\|$")
    ax2.set_yscale("log")
    ax2.set_ylabel("leave-one-out change in $A$")
    ax2.set_xlabel("number of sample points $|D|$")
    lev = ax2.twinx()
    band(lev, n, col["h_max"], "-", "gray", r"$\mathrm{max}_i\, h_i$")
    band(lev, n, col["h_mean"], ":", "gray", r"$\mathrm{mean}_i\, h_i$")
    lev.set_ylim(0, 1.05)
    lev.set_ylabel("leverage")

    for ax in (ax1, ax2):
        ax.axvline(summary["ck_rank_test"], color="k", ls="-.", label="rank of the CK features")
    ax1.legend(fontsize=8)
    handles = ax2.get_legend_handles_labels()
    lev_handles = lev.get_legend_handles_labels()
    ax2.legend(handles[0] + lev_handles[0], handles[1] + lev_handles[1], fontsize=8)
    fig.suptitle(plot_title("Data double descent", "CK modeled"))
    stamp(fig, run_id)
    return fig


def main():
    # read config.py before the long computation so later edits can't leak into the record
    config_source = Path(config.__file__).read_text()
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (f"{now}_{config.TARGET.__name__}_{config.ACTIVATION.__name__.lower()}"
              f"_w{config.WIDTH}_d{config.DEPTH}_n{config.N_TRAIN}{config.TRAIN_SPACING}"
              f"_data_dd_{config.DD_SAMPLING}")

    model, data = train.run()
    stats, summary = dd_sweep(model, data)

    record = {"run_id": run_id, "config": config_values(), "git": git_state(),
              "summary": summary, "config_source": config_source}
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.py").write_text(config_source)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    np.savez(run_dir / "results.npz", n=np.arange(1, config.DD_N_MAX + 1), stats=stats,
             columns=COLUMNS)
    plot_dd(stats, summary, run_id).savefig(run_dir / "data_dd.png", dpi=200,
                                            metadata={PNG_KEY: json.dumps(record)})

    print(json.dumps(summary, indent=2))
    print("saved", run_dir)


if __name__ == "__main__":
    main()
