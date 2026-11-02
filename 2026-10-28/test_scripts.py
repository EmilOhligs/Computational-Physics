"""Smoke tests for the plot scripts of 1d-2b: run headless, check printed results and figures."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def run(script, tmp_path):
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path), "PYTHONPATH": str(HERE)}
    return subprocess.run([sys.executable, str(HERE / script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True).stdout


def test_1d_critical_dt_times_largest_rate_is_constant(tmp_path):
    out = run("plot_1d.py", tmp_path)
    products = [float(s) for s in re.findall(r"dt_crit \* max\(k1, k2\) = ([\d.]+)", out)]
    assert len(products) == 15
    assert sorted(set(products)) == pytest.approx([2.0, 2.785], abs=1e-3)
    below = [float(s) for s in re.findall(r"at 0\.95 dt_crit = ([\d.e+-]+)", out)]
    above = [float(s) for s in re.findall(r"at 1\.05 dt_crit = ([\d.e+-]+)", out)]
    assert len(below) == len(above) == 3
    assert max(below) < 1 < min(above)
    assert (tmp_path / "instability_1d.png").exists()


def test_1e_eigenvalues_and_first_unstable_mode(tmp_path):
    out = run("plot_1e.py", tmp_path)
    eigenvalues = sorted(float(s) for s in re.findall(r"lambda = ([-\d.e+]+), eigenvector", out))
    assert eigenvalues == pytest.approx([-10, -1, 0])
    first = [float(s) for s in re.findall(r"first unstable mode lambda = ([-\d.e+]+)", out)]
    assert first == [-10, -10, -10]
    assert (tmp_path / "stability_1e.png").exists() and (tmp_path / "modes_1e.png").exists()


def test_1f_step_counts_explicit_and_implicit_euler(tmp_path):
    out = run("plot_1f.py", tmp_path)
    counts = dict(re.findall(r"(explicit|implicit) Euler: n = (\d+)", out))
    assert counts == {"explicit": "4999", "implicit": "180"}
    assert "dt_crit explicit Euler = 0.00200" in out
    assert (tmp_path / "stiff_1f.png").exists()


def test_2a_robertson_implicit_converges_and_explicit_blows_up(tmp_path):
    out = run("plot_2a.py", tmp_path)
    rows = re.findall(r"implicit Euler, n = (\d+): A, B, C at t = 1e5 = ([\d.]+), ([\d.e+-]+), ([\d.]+), max \|A \+ B \+ C - 1\| = ([\d.e+-]+)", out)
    assert [r[0] for r in rows] == ["400", "1600"]
    a = [float(r[1]) for r in rows]
    assert a[1] == pytest.approx(0.0179, abs=5e-4)
    assert abs(a[0] - a[1]) < 2e-3
    assert all(float(r[4]) < 1e-10 for r in rows)
    blow = re.findall(r"explicit Euler, dt = 0\.001: blows up at t = ([\d.]+)", out)
    assert len(blow) == 1 and 0 < float(blow[0]) < 40
    assert (tmp_path / "robertson_2a.png").exists()


def test_2b_lorenz_separation_time_independent_of_dt(tmp_path):
    out = run("plot_2b.py", tmp_path)
    times = [float(s) for s in re.findall(r"RK4, dt = [\d.e-]+: trajectories separate \(distance > 1\) at t = ([\d.]+)", out)]
    assert len(times) == 2
    assert 20 < times[0] < 35
    assert times[0] == pytest.approx(times[1], abs=0.5)
    assert (tmp_path / "lorenz_2b.png").exists()
