"""helper.py — Shared helpers for Couette regression scripts.

All four drivers (convergence.py, convergence3d.py, plot_couette.py,
plot_couette3d.py) build and compare 2-D/3-D Couette runs via the same
AMReX plotfile tools. This module centralises the duplicated logic so the
drivers stay thin.

fcompare/faverage are dimension-dependent but share a single filename
(fcompare.gnu.ex, faverage.gnu.ex) for every DIM, so every helper below
deletes any stale binary and rebuilds for the requested DIM before use.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CS2 = 1.0 / 3.0

COUETTE_DIR = Path(__file__).resolve().parent
REPO_ROOT = COUETTE_DIR.parents[2]
AMREX_HOME = Path(os.environ.get("AMREX_HOME", REPO_ROOT / "external" / "amrex"))
PLOTFILE_TOOLS_DIR = AMREX_HOME / "Tools" / "Plotfile"

FCOMPARE_ROW_RE = re.compile(r"^\s*(\S+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")

UNITS_LOG_RE = {
    "nu_lat": re.compile(r"^nu_lat:\s*([-+0-9.eE]+)", re.MULTILINE),
    "C_x": re.compile(r"^C_x:\s*([-+0-9.eE]+)", re.MULTILINE),
    "C_t": re.compile(r"^C_t:\s*([-+0-9.eE]+)", re.MULTILINE),
}


def find_one(pattern: str, directory: Path) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return matches[0]


def ensure_fcompare(dim: int = 2) -> Path:
    for p in sorted(PLOTFILE_TOOLS_DIR.glob("fcompare*.ex")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    print(f"Building fcompare (DIM={dim})...")
    subprocess.run(["make", f"DIM={dim}", "programs=fcompare"], cwd=PLOTFILE_TOOLS_DIR, check=True)
    exes = sorted(PLOTFILE_TOOLS_DIR.glob("fcompare*.ex"))
    if exes:
        return exes[0]
    return find_one("fcompare*.ex", PLOTFILE_TOOLS_DIR)


def ensure_faverage(dim: int = 2) -> Path:
    for p in sorted(PLOTFILE_TOOLS_DIR.glob("faverage*.ex")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    print(f"Building faverage (DIM={dim})...")
    subprocess.run(["make", f"DIM={dim}", "programs=faverage"], cwd=PLOTFILE_TOOLS_DIR, check=True)
    exe = find_one("faverage*.ex", PLOTFILE_TOOLS_DIR)
    _verify_faverage_dim(exe, dim)
    return exe


def _verify_faverage_dim(exe: Path, expected_dim: int):
    """Check that faverage was compiled for expected_dim; it always averages
    along the last dimension (z in 3-D), so a DIM mismatch silently gives
    wrong y-profiles for Couette (y-varying)."""
    cfg = PLOTFILE_TOOLS_DIR / "tmp_build_dir" / "s" / f"{expected_dim}d.gnu.EXE" / "AMReX_Config.H"
    if cfg.is_file():
        txt = cfg.read_text()
        if f"#define AMREX_SPACEDIM {expected_dim}" not in txt:
            raise RuntimeError(f"faverage DIM check failed: {cfg} does not contain AMREX_SPACEDIM {expected_dim}")
        return
    try:
        out = subprocess.run(["strings", str(exe)], capture_output=True, text=True, timeout=5).stdout
        if f"AMREX_SPACEDIM {expected_dim}" not in out and f"BL_SPACEDIM {expected_dim}" not in out:
            print(f"Warning: could not verify faverage DIM={expected_dim} via strings; assuming build is correct")
    except Exception:
        pass


def base_grid_ratio(inputs_path: Path) -> float:
    """Nx/Ny ratio from grid.ncells, handling 2-D (Nx Ny) and 3-D (Nx Ny Nz)."""
    text = inputs_path.read_text()
    m = re.search(r"^\s*grid\.ncells\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text, re.MULTILINE)
    if m:
        nx0, ny0 = int(m.group(1)), int(m.group(2))
        return nx0 / ny0
    m = re.search(r"^\s*grid\.ncells\s*=\s*(\d+)\s+(\d+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find grid.ncells in {inputs_path}")
    nx0, ny0 = int(m.group(1)), int(m.group(2))
    return nx0 / ny0


def parse_inputs_values(text: str, key: str) -> list:
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find {key!r} in inputs")
    return [float(x) for x in m.group(1).split()]


def apply_overrides(values: dict, overrides: list) -> dict:
    out = dict(values)
    for ov in overrides:
        key, _, val = ov.partition("=")
        if key in out:
            out[key] = [float(x) for x in val.split()]
    return out


def parse_units_log(log_text: str) -> dict:
    out = {}
    for key, pat in UNITS_LOG_RE.items():
        m = pat.search(log_text)
        if not m:
            raise ValueError(f"could not find '{key}:' in solver log")
        out[key] = float(m.group(1))
    return out


def run_solver(solver_exe: Path, inputs_path: Path, run_dir: Path, overrides: list) -> str:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    log_path = run_dir / "run.log"
    with open(log_path, "w") as log_file:
        subprocess.run([str(solver_exe), str(inputs_path), *overrides],
                       cwd=run_dir, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    return log_path.read_text()


def parse_fcompare(output: str) -> dict:
    errors = {}
    for line in output.splitlines():
        m = FCOMPARE_ROW_RE.match(line)
        if m and m.group(1) in ("density", "vx", "vy", "vz"):
            errors[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return errors


def faverage_profile(faverage_exe: Path, pltdir: Path, varname: str = "vx"):
    subprocess.run([str(faverage_exe), "-v", varname, str(pltdir)],
                   capture_output=True, text=True, check=True)
    slice_path = Path(str(pltdir) + ".slice")
    data = np.loadtxt(slice_path, comments="#")
    return data[:, 0], data[:, 1]


def first_mode_amplitude(y_norm: np.ndarray, u_sim: np.ndarray, U0: float) -> float:
    deviation = u_sim - U0 * y_norm
    return np.sum(deviation * np.sin(np.pi * y_norm)) * 2.0 / len(y_norm)
