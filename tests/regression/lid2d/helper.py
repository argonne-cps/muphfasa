"""helper.py — Shared helpers for the lid2d convergence / plotting scripts.

Both convergence.py and plot_lid2d.py drive test_lid2d with the opt-in
diagnostics added to test_lid2d.cpp:

    lid2d.csv_prefix=<p>   writes <p>_history.csv and <p>_centerline.csv
    lid2d.sample_int=<n>   diagnostic sampling interval in steps

so the scripts read structured CSV instead of scraping solver stdout.
Requires numpy (and matplotlib for the plots).
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np

CS2 = 1.0 / 3.0  # lattice sound speed squared (D2Q9)

LID2D_DIR = Path(__file__).resolve().parent
REPO_ROOT = LID2D_DIR.parents[2]

# Marchi, Suero & Araki (2009), doi:10.1590/S1678-58782009000300004: grid-converged
# Re=100 centerline extrema normalized by the lid velocity (same values as
# test_lid2d.cpp; see README.md).
BENCH = {"umin": -2.140417e-1, "vmax": 1.79572814e-1, "vmin": -2.53804e-1}

BUILD_HELP = (
    "Build a 2-D test_lid2d first, e.g.\n"
    "    cd tests/regression/lid2d && make -j DIM=2          # -> test_lid2d2d.gnu.ex\n"
    "or\n"
    "    cmake -S . -B /tmp/build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON\n"
    "    cmake --build /tmp/build2d -j --target test_lid2d\n"
    "and pass it with --exe (or set LID2D_EXE)."
)


def find_solver(explicit=None) -> Path:
    """--exe, then $LID2D_EXE, then a GNUmake build in this dir, then a CMake build."""
    if explicit is None and os.environ.get("LID2D_EXE"):
        explicit = os.environ["LID2D_EXE"]
    if explicit is not None:
        exe = Path(explicit).expanduser().resolve()
        if not exe.is_file():
            raise FileNotFoundError(f"solver executable not found: {exe}")
        return exe
    candidates = sorted(LID2D_DIR.glob("test_lid2d*2d*.ex")) + sorted(
        REPO_ROOT.glob("build*/tests/test_lid2d"))
    if not candidates:
        raise FileNotFoundError("no test_lid2d executable found.\n" + BUILD_HELP)
    return candidates[0]


def input_value(inputs_path: Path, key: str, overrides=()) -> list:
    """Whitespace-split value of `key` from an AMReX inputs file; a trailing
    key=value in `overrides` wins (same rule as the command line)."""
    for item in reversed(list(overrides)):
        k, _, v = item.partition("=")
        if k.strip() == key:
            return v.split()
    text = Path(inputs_path).read_text()
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find {key!r} in {inputs_path}")
    return m.group(1).split()


def input_float(inputs_path: Path, key: str, overrides=()) -> float:
    return float(input_value(inputs_path, key, overrides)[0])


def lattice_lid_velocity(Re: float, tau: float, N: int) -> float:
    """u_lat = nu_lat*Re/N: the lid speed that the pinned-tau inputs give."""
    return (tau - 0.5) * CS2 * Re / N


def run_solver(exe: Path, inputs_path: Path, run_dir: Path, overrides) -> int:
    """Run the solver in run_dir; returns the exit code (nonzero just means
    one of the test's own pass/fail checks failed, so it is not fatal here)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run.log", "w") as log:
        proc = subprocess.run([str(exe), str(inputs_path), *overrides],
                              cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_lid_log(log_text: str) -> dict:
    """U_lid / nu_lat (lattice) from the '--- lid2d validation' banner."""
    m = re.search(r"U_lid=([-+0-9.eE]+) lattice, nu=([-+0-9.eE]+)", log_text)
    if not m:
        raise ValueError("no 'lid2d validation' banner in the solver log (did the run finish?)")
    return {"u_lid": float(m.group(1)), "nu": float(m.group(2))}


def parse_check_lines(log_text: str) -> list:
    """The PASS/FAIL lines printed by test_lid2d."""
    return [ln for ln in log_text.splitlines() if ln.startswith(("PASS", "FAIL"))]


def read_history(path: Path) -> dict:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def read_centerline(path: Path) -> dict:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
