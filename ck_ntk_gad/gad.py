"""Trains the model in config.py, runs the GAD sweep and saves the log-log plot.

    python gad.py                      # train, decompose, plot
    python gad.py --config-of gad.png  # print the config a saved plot was made with
    python gad.py --config other.py --out f1   # another config file / output folder

Each run gets its own folder, runs/<time>_<settings>/, holding

    gad.png       the GAD plot
    fit.png       the network, CK and NTK fits against the true function
    config.py     verbatim copy of config.py as it was when the run started
    run.json      the same settings as plain values, the git commit, and summary numbers
    results.npz   the curves in both plots, for replotting without rerunning

The contents of run.json (config.py source included) are also written into the
metadata of both plots, so a plot that has been copied out of its folder can still be
traced back with --config-of.
"""
import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

HERE = Path(__file__).resolve().parent
PNG_KEY = "gad_run"

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--config-of", metavar="PNG",
                    help="print the config a saved plot was made with and exit")
parser.add_argument("--config", metavar="FILE", help="use this file instead of config.py")
parser.add_argument("--out", metavar="DIR", default=HERE / "runs", type=Path,
                    help="folder the run folder is created in (default: runs)")
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

if args.config:
    # has to happen before model.py and train.py do `import config`
    spec = importlib.util.spec_from_file_location("config", args.config)
    sys.modules["config"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["config"])

import config
import train
from train import test_norm


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


def sweep_sizes(n_train, n_params, n_ck):
    """Every size up to 2 * n_train, then log-spaced out to all the parameters."""
    dense = min(2 * n_train, n_params + 1)
    sizes = np.arange(1, dense)
    if dense <= n_params:
        sizes = np.concatenate([sizes, np.geomspace(dense, n_params, 60).astype(int)])
    return np.unique(np.append(sizes, n_ck))


def column_order(n_params, n_ck):
    """Order in which Jacobian columns (parameters) are moved into the modeled set."""
    g = torch.Generator().manual_seed(config.SEED)
    if config.COLUMN_ORDER == "random":
        return torch.randperm(n_params, generator=g)
    if config.COLUMN_ORDER == "ck_first":
        # the readout weights and bias are the last n_ck columns; they are the CK features
        ck = n_params - n_ck + torch.randperm(n_ck, generator=g)
        return torch.cat([ck, torch.randperm(n_params - n_ck, generator=g)])
    raise ValueError(f"COLUMN_ORDER must be 'ck_first' or 'random', got {config.COLUMN_ORDER!r}")


def gad_sweep(model, data):
    J_train, J_test = jacobian(model, data.x_train), jacobian(model, data.x_test)
    n_params = J_train.shape[1]
    n_ck = config.WIDTH + 1
    gamma = fit_subset(J_train, data.y_train, data.w_train, torch.arange(n_params))
    f_ntk_test = J_test @ gamma
    # CK machine: the same fit to the data using only the last layer's columns
    ck_cols = torch.arange(n_params - n_ck, n_params)
    f_ck_test = J_test[:, ck_cols] @ fit_subset(J_train, data.y_train, data.w_train, ck_cols)
    with torch.no_grad():
        f_nn_test = model(data.x_test)
    fits = {"network": f_nn_test, "CK": f_ck_test, "NTK": f_ntk_test}

    order = column_order(n_params, n_ck)
    sizes = sweep_sizes(len(data.x_train), n_params, n_ck)
    rows = []
    for m in sizes:
        cols, rest = order[:m], order[m:]
        (fd, fa, fm), (pd, pa, pm), A_norm, rank = gad_terms(
            J_train, J_test, data.w_train, gamma, cols, rest)
        risk = test_norm(fd + fa + fm, data.w_test) ** 2
        rows.append([risk, pd.norm().item(), pa.norm().item(),
                     pm.norm().item(), A_norm, rank])
    rows = np.array(rows)

    summary = {
        "n_params": n_params,
        "n_ck": n_ck,
        "jacobian_rank": torch.linalg.matrix_rank(
            data.w_train.sqrt() * J_train, rtol=config.RCOND).item(),
        "nn_test_error": test_norm(f_nn_test - data.y_test, data.w_test),
        "ck_test_error": test_norm(f_ck_test - data.y_test, data.w_test),
        "ntk_test_error": test_norm(f_ntk_test - data.y_test, data.w_test),
    }
    if config.COLUMN_ORDER == "ck_first":
        # modeled = CK, unmodeled = rest of the NTK: the test-norm gap between the two machines
        summary["ck_vs_ntk_test_error"] = float(rows[sizes == n_ck, 0][0] ** 0.5)
    return sizes, rows, summary, fits


def stamp(fig, run_id):
    fig.text(0.995, 0.005, run_id, ha="right", va="bottom", fontsize=6, color="gray")


def plot_gad(sizes, rows, summary, run_id):
    fig, ax = plt.subplots()
    ax.loglog(sizes, rows[:, 0], "k-",  label="risk")
    ax.loglog(sizes, rows[:, 1], "g-",  label=r"$\|P_N\theta_M\|$")
    ax.loglog(sizes, rows[:, 2], "b-",  label=r"$\|A\theta_U\|$")
    ax.loglog(sizes, rows[:, 3], "r-",  label=r"$\|\theta_U\|$")
    ax.loglog(sizes, rows[:, 4], "b--", label=r"$\|A\|$")
    ax.axvline(config.N_TRAIN, color="k", ls="--")
    if config.COLUMN_ORDER == "ck_first":
        ax.axvline(summary["n_ck"], color="gray", ls=":")   # modeled set is exactly the CK
    ax.set_xlabel("number of modeled parameters")
    ax.legend()
    stamp(fig, run_id)
    return fig


def plot_fit(data, fits, run_id):
    x = data.x_test.numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(x, data.y_test.numpy(), "k-", lw=3, alpha=0.3, label="true function")
    for (name, f), style in zip(fits.items(), ["-", "--", ":"]):
        ax1.plot(x, f.numpy(), style, label=name)
        ax2.plot(x, (f - data.y_test).abs().numpy(), style, label=name)
    # keep the true function visible when a fit blows up; the error panel shows the full size
    lo, hi = data.y_test.min().item(), data.y_test.max().item()
    ax1.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    ax1.legend()
    ax2.set_yscale("log"); ax2.set_ylabel("|error|"); ax2.set_xlabel("x")
    stamp(fig, run_id)
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
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = (f"{now}_{config.TARGET.__name__}_{config.ACTIVATION.__name__.lower()}"
              f"_w{config.WIDTH}_d{config.DEPTH}_n{config.N_TRAIN}{config.TRAIN_SPACING}"
              f"_{config.COLUMN_ORDER}")

    model, data = train.run()
    sizes, rows, summary, fits = gad_sweep(model, data)

    record = {"run_id": run_id, "config": config_values(), "git": git_state(),
              "summary": summary, "config_source": config_source}
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.py").write_text(config_source)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2))
    np.savez(run_dir / "results.npz", sizes=sizes, rows=rows,
             columns=["risk", "data", "alias", "model", "A_norm", "rank"],
             x_test=data.x_test.numpy(), y_test=data.y_test.numpy(),
             **{f"fit_{name}": f.numpy() for name, f in fits.items()})
    metadata = {PNG_KEY: json.dumps(record)}
    plot_gad(sizes, rows, summary, run_id).savefig(run_dir / "gad.png", dpi=200, metadata=metadata)
    plot_fit(data, fits, run_id).savefig(run_dir / "fit.png", dpi=200, metadata=metadata)

    print(json.dumps(summary, indent=2))
    print("saved", run_dir)


if __name__ == "__main__":
    if args.config_of:
        config_of(args.config_of)
    else:
        main()
