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
