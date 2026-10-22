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


# ── Couette transient plot (shared by plot_couette.py / plot_couette3d.py) ──

def run_couette_transient_plot(
    *,
    dim: int,
    wall_axis: int,
    bc_key: str,
    inputs_path: Path,
    run_dir: Path,
    out_path: Path,
    solver_exe: Path,
    overrides: list,
    no_run: bool,
    max_profile_err: float | None,
    max_rate_rel_err: float | None,
    steady_label: str,
    profile_title: str,
    decay_title_prefix: str,
    axis_label: str,
) -> int:
    """Collect plt*/analytic_plt* profiles via faverage and render the two-panel
    Couette startup figure. 2-D and 3-D differ only by dim/wall_axis/bc_key/labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faverage_exe = ensure_faverage(dim=dim)

    inputs_text = inputs_path.read_text()
    params = apply_overrides(
        {
            "grid.ncells": parse_inputs_values(inputs_text, "grid.ncells"),
            bc_key: parse_inputs_values(inputs_text, bc_key),
            "geometry.prob_lo": parse_inputs_values(inputs_text, "geometry.prob_lo"),
            "geometry.prob_hi": parse_inputs_values(inputs_text, "geometry.prob_hi"),
        },
        overrides,
    )
    Nwall = int(params["grid.ncells"][wall_axis])
    Uwall0 = params[bc_key][0]
    lo, hi = params["geometry.prob_lo"][wall_axis], params["geometry.prob_hi"][wall_axis]
    domain_height = hi - lo

    if no_run:
        log_text = (run_dir / "run.log").read_text()
    else:
        log_text = run_solver(solver_exe, inputs_path.resolve(), run_dir, overrides)

    units = parse_units_log(log_text)
    nu_phys = units["nu_lat"] * units["C_x"] ** 2 / units["C_t"]
    U0 = Uwall0  # physical wall velocity (plotfiles are physical)
    H = domain_height

    plt_dirs = sorted(
        (p for p in run_dir.glob("plt*") if re.fullmatch(r"plt\d+", p.name)),
        key=lambda p: int(p.name[3:]),
    )
    if not plt_dirs:
        raise FileNotFoundError(f"no plt<step> directories found in {run_dir}")

    times, profiles = [], []
    for pdir in plt_dirs:
        step = int(pdir.name[3:])
        analytic_dir = pdir.with_name(f"analytic_{pdir.name}")
        if not analytic_dir.is_dir():
            continue
        coord_phys, u_sim = faverage_profile(faverage_exe, pdir, "vx")
        _, u_analytic = faverage_profile(faverage_exe, analytic_dir, "vx")
        if len(coord_phys) != Nwall:
            raise RuntimeError(
                f"faverage returned {len(coord_phys)} bins but Nwall={Nwall}: "
                f"faverage DIM mismatch (averages along last dimension). "
                f"Check that faverage was built with DIM={dim}."
            )
        coord_norm = (coord_phys - lo) / domain_height
        times.append((step + 1.0) * units["C_t"])
        profiles.append((coord_norm, u_sim, u_analytic))

    times = np.array(times)
    profile_max_err = max(
        (float(np.max(np.abs(u_sim - u_analytic)))
         for _, u_sim, u_analytic in (profiles[1:] or profiles)),
        default=0.0,
    ) / U0 if U0 != 0 else 0.0

    fig, (ax_profiles, ax_decay) = plt.subplots(1, 2, figsize=(11, 4.5))
    n_total = len(times)
    if n_total <= 8:
        show_idx = np.arange(n_total)
    else:
        show_idx = np.unique(np.round(np.geomspace(1, n_total, 7)).astype(int) - 1)
    cmap = plt.get_cmap("viridis")

    for k, idx in enumerate(show_idx):
        t = times[idx]
        coord_norm, u_sim, u_analytic = profiles[idx]
        order = np.argsort(coord_norm)
        color = cmap(k / max(len(show_idx) - 1, 1))
        ax_profiles.plot(u_sim[order] / U0, coord_norm[order], "o", color=color, ms=4,
                         label=f"t={t:.2f}")
        ax_profiles.plot(u_analytic[order] / U0, coord_norm[order], "-", color=color, lw=1)

    ax_profiles.plot([0, 1], [0, 1], "k--", lw=1, label=steady_label)
    ax_profiles.set_xlabel("u / U0")
    ax_profiles.set_ylabel(axis_label)
    ax_profiles.set_title(profile_title)
    ax_profiles.legend(fontsize=7, loc="lower right")

    log_amps = []
    for coord_norm, u_sim, _ in profiles:
        a1 = first_mode_amplitude(coord_norm, u_sim, U0)
        log_amps.append(np.log(abs(a1)) if abs(a1) > 1e-12 else np.nan)
    log_amps = np.array(log_amps)

    valid = ~np.isnan(log_amps)
    skip = min(3, valid.sum() // 4)
    fit_idx = np.where(valid)[0][skip:]
    analytic_rate = -(np.pi ** 2) * nu_phys / H ** 2

    ax_decay.plot(times[valid], log_amps[valid], "o", ms=4, label="simulation")
    if len(fit_idx) >= 2:
        slope, intercept = np.polyfit(times[fit_idx], log_amps[fit_idx], 1)
        ax_decay.plot(times[fit_idx], slope * times[fit_idx] + intercept, "r-",
                      label=f"fit: rate={slope:.3e}")
        title_rate = f"fit={slope:.4e}  analytic={analytic_rate:.4e}"
    else:
        title_rate = f"analytic={analytic_rate:.4e} (too few points to fit)"
    ax_decay.set_xlabel("t")
    ax_decay.set_ylabel("ln|A1(t)|")
    ax_decay.set_title(f"{decay_title_prefix}: {title_rate}")
    ax_decay.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    print(f"max|u_sim-u_analytic|/U0 over all snapshots: {profile_max_err:.3e}")
    if len(fit_idx) >= 2:
        rate_rel_err = abs(slope - analytic_rate) / abs(analytic_rate)
        print(f"decay rate relative error vs analytic: {rate_rel_err:.3e}")
    else:
        rate_rel_err = None

    ok = True
    if max_profile_err is not None and profile_max_err > max_profile_err:
        print(f"FAIL: profile error {profile_max_err:.3e} > --max-profile-err {max_profile_err}")
        ok = False
    if max_rate_rel_err is not None:
        if rate_rel_err is None:
            print("FAIL: --max-rate-rel-err given but too few points to fit a decay rate")
            ok = False
        elif rate_rel_err > max_rate_rel_err:
            print(f"FAIL: decay rate relative error {rate_rel_err:.3e} > "
                  f"--max-rate-rel-err {max_rate_rel_err}")
            ok = False

    return 0 if ok else 1
