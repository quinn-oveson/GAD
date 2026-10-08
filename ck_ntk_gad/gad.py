"""Trains the model in config.py, runs the GAD sweep and saves the log-log plot.

    python gad.py                      # train, decompose, plot
    python gad.py --config-of gad.png  # print the config a saved plot was made with

Each run gets its own folder, runs/<time>_<settings>/, holding

    gad.png       the plot
    config.py     verbatim copy of config.py as it was when the run started
    run.json      the same settings as plain values, the git commit, and summary numbers
    results.npz   the curves in the plot, for replotting without rerunning

The contents of run.json (config.py source included) are also written into the
metadata of gad.png, so a plot that has been copied out of its folder can still be
traced back with --config-of.
"""
import argparse
import inspect
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

import config
import train
from train import test_norm

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
PNG_KEY = "gad_run"


def jacobian(model, x):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def f_single(p, x):
        return functional_call(model, p, (x.unsqueeze(0),)).squeeze()

    jac = vmap(jacrev(f_single), in_dims=(None, 0))(params, x)
    return torch.cat([j.reshape(len(x), -1) for j in jac.values()], dim=1)


def fit_subset(J_tr, target, w, cols, rcond=config.RCOND):
    sw = w.sqrt()
    A = sw * J_tr[:, cols]          # weighted design matrix
    b = sw * target                 # weighted target
    return torch.linalg.lstsq(A, b, driver="gelsd", rcond=rcond).solution


def gad_terms(J_train, J_test, w_train, gamma, cols, rest, rcond=config.RCOND):
    sw = w_train.sqrt()
    gS, gU = gamma[cols], gamma[rest]

    # truncated SVD of the weighted modeled design matrix
    U, s, Vh = torch.linalg.svd(sw * J_train[:, cols], full_matrices=False)
    keep = s > rcond * s[0]
    U, s, Vh = U[:, keep], s[keep], Vh[keep]
    X = (U.T @ (sw * J_train[:, rest])) / s[:, None]
    A_norm = torch.linalg.matrix_norm(X, ord=2).item() if len(rest) > 0 else 0.0
    rank = int(keep.sum())
    # parameter-space pieces
    B_gS = Vh.T @ (Vh @ gS)
    A_gU = Vh.T @ ((U.T @ (sw * (J_train[:, rest] @ gU))) / s[:, None])
    p_data, p_alias, p_model = gS - B_gS, -A_gU, gU

    # their images at the test points
    f_data  = J_test[:, cols] @ p_data
    f_alias = J_test[:, cols] @ p_alias
    f_model = J_test[:, rest] @ p_model
    return (f_data, f_alias, f_model), (p_data, p_alias, p_model), A_norm, rank


def sweep_sizes(n_train, n_params):
    """Every size up to 2 * n_train, then log-spaced out to all the parameters."""
    dense = min(2 * n_train, n_params + 1)
    sizes = np.arange(1, dense)
    if dense <= n_params:
        sizes = np.concatenate([sizes, np.geomspace(dense, n_params, 60).astype(int)])
    return np.unique(sizes)


def gad_sweep(model, data):
    J_train, J_test = jacobian(model, data.x_train), jacobian(model, data.x_test)
    n_params = J_train.shape[1]
    gamma = fit_subset(J_train, data.y_train, data.w_train, torch.arange(n_params))
    f_ntk_test = J_test @ gamma

    g = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(n_params, generator=g)
    sizes = sweep_sizes(len(data.x_train), n_params)
    rows = []
    for m in sizes:
        cols, rest = order[:m], order[m:]
        (fd, fa, fm), (pd, pa, pm), A_norm, rank = gad_terms(
            J_train, J_test, data.w_train, gamma, cols, rest)
        risk = test_norm(fd + fa + fm, data.w_test) ** 2
        rows.append([risk, pd.norm().item(), pa.norm().item(),
                     pm.norm().item(), A_norm, rank])

    with torch.no_grad():
        nn_err = test_norm(model(data.x_test) - data.y_test, data.w_test)
    summary = {
        "n_params": n_params,
        "jacobian_rank": torch.linalg.matrix_rank(
            data.w_train.sqrt() * J_train, rtol=config.RCOND).item(),
        "nn_test_error": nn_err,
        "ntk_test_error": test_norm(f_ntk_test - data.y_test, data.w_test),
    }
    return sizes, np.array(rows), summary


def plot_gad(sizes, rows, summary, run_id):
    fig, ax = plt.subplots()
    ax.loglog(sizes, rows[:, 0], "k-",  label="risk")
    ax.loglog(sizes, rows[:, 1], "g-",  label=r"$\|P_N\theta_M\|$")
    ax.loglog(sizes, rows[:, 2], "b-",  label=r"$\|A\theta_U\|$")
    ax.loglog(sizes, rows[:, 3], "r-",  label=r"$\|\theta_U\|$")
    ax.loglog(sizes, rows[:, 4], "b--", label=r"$\|A\|$")
    ax.axvline(config.N_TRAIN, color="k", ls="--")
    if summary["jacobian_rank"] != config.N_TRAIN:
        ax.axvline(summary["jacobian_rank"], color="gray", ls=":")
    ax.set_xlabel("number of modeled parameters")
    ax.legend()
    fig.text(0.995, 0.005, run_id, ha="right", va="bottom", fontsize=6, color="gray")
    return fig


# ---- keeping track of which config made which plot ----

def config_values():
    """config.py's settings as plain JSON values."""
    out = {}
    for name, value in vars(config).items():
        if not name.isupper():
            continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[name] = value
        elif isinstance(value, (tuple, list)):
            out[name] = list(value)
        elif inspect.isfunction(value):
            out[name] = {"name": value.__name__, "source": inspect.getsource(value)}
        else:
            out[name] = getattr(value, "__name__", repr(value))
    return out


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--", "."))}


def config_of(png_path):
    from PIL import Image
    text = Image.open(png_path).text.get(PNG_KEY)
    if text is None:
        sys.exit(f"{png_path} has no GAD run information in it")
    record = json.loads(text)
    source = record.pop("config_source")
    print(json.dumps(record, indent=2))
    print("\n# ---- config.py ----\n" + source)


def main():
    # read config.py before the long computation so later edits can't leak into the record
    config_source = Path(config.__file__).read_text()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (f"{stamp}_{config.TARGET.__name__}_{config.ACTIVATION.__name__.lower()}"
              f"_w{config.WIDTH}_d{config.DEPTH}_n{config.N_TRAIN}{config.TRAIN_SPACING}")

    model, data = train.run()
    sizes, rows, summary = gad_sweep(model, data)

    record = {"run_id": run_id, "config": config_values(), "git": git_state(),
              "summary": summary, "config_source": config_source}
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.py").write_text(config_source)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    np.savez(run_dir / "results.npz", sizes=sizes, rows=rows,
             columns=["risk", "data", "alias", "model", "A_norm", "rank"])
    fig = plot_gad(sizes, rows, summary, run_id)
    fig.savefig(run_dir / "gad.png", dpi=200, metadata={PNG_KEY: json.dumps(record)})

    print(json.dumps(summary, indent=2))
    print("saved", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config-of", metavar="PNG",
                        help="print the config a saved plot was made with and exit")
    args = parser.parse_args()
    if args.config_of:
        config_of(args.config_of)
    else:
        main()
