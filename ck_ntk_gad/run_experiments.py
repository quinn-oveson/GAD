"""Runs gad.py over a grid of settings, one run per combination.

    python run_experiments.py            # every target below
    python run_experiments.py f3 f3coef5 # only these targets

Each run uses config.py with that combination's settings appended at the end, and
that combined file is what gets saved in the run folder. Results go in a folder
named after the target function, e.g. f1/<time>_f1_tanh_..._ck_first/.
"""
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

TARGETS = ["f1", "f2", "f3", "f3coef5"]     # functions defined in config.py
ACTIVATIONS = ["nn.Tanh", "nn.ReLU"]
COLUMN_ORDERS = ["ck_first", "random"]

base = (HERE / "config.py").read_text()
targets = sys.argv[1:] or TARGETS

with tempfile.TemporaryDirectory() as tmp:
    for target, act, order in itertools.product(targets, ACTIVATIONS, COLUMN_ORDERS):
        cfg = Path(tmp) / "config.py"
        cfg.write_text(base + "\n# ---- set by run_experiments.py ----\n"
                       f"TARGET = {target}\nACTIVATION = {act}\nCOLUMN_ORDER = {order!r}\n")
        print(f"\n=== {target}  {act}  {order} ===", flush=True)
        subprocess.run([sys.executable, HERE / "gad.py", "--config", cfg, "--out", HERE / target],
                       check=True)
