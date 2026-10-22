"""Runs data_dd.py over a grid of settings, one run per combination.

    python run_dd_experiments.py            # every target in GRID
    python run_dd_experiments.py f3 f3coef5 # only these targets

GRID maps a config.py setting to the values to try, written as Python source exactly as
they would appear in config.py (so strings need their quotes). Every combination gets its
own run: config.py with that combination's settings appended at the end, and that combined
file is what gets saved in the run folder. Results go in a folder named after the target
function, e.g. f1/<time>_f1_tanh_..._data_dd_bisect/.
"""
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

GRID = {
    "TARGET": ["f1", "f2", "f3"],                   # functions defined in config.py
    "ACTIVATION": ["nn.Tanh", "nn.ReLU"],
    "DD_SAMPLING": ['"uniform"', '"bisect"'],
    # any other setting works too, e.g.
    # "DD_N_MAX": ["400", "601"],
    # "RCOND": ["1e-14", "1e-10"],
}

base = (HERE / "config.py").read_text()
grid = dict(GRID, TARGET=sys.argv[1:]) if sys.argv[1:] else GRID

with tempfile.TemporaryDirectory() as tmp:
    for values in itertools.product(*grid.values()):
        settings = dict(zip(grid, values))
        cfg = Path(tmp) / "config.py"
        cfg.write_text(base + "\n# ---- set by run_dd_experiments.py ----\n"
                       + "".join(f"{name} = {value}\n" for name, value in settings.items()))
        print("\n=== " + "  ".join(f"{k}={v}" for k, v in settings.items()) + " ===", flush=True)
        out = HERE / settings["TARGET"] if "TARGET" in settings else HERE / "runs"
        subprocess.run([sys.executable, HERE / "data_dd.py", "--config", cfg, "--out", out],
                       check=True)
