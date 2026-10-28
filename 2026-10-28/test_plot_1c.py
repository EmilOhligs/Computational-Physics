"""Smoke test for plot_1c.py: runs headless and reports the expected orders."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_plot_1c_reports_orders_and_saves_figure(tmp_path):
    script = Path(__file__).with_name("plot_1c.py")
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path), "PYTHONPATH": str(script.parent)}
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True).stdout
    measured = [(int(p), float(m)) for p, m in re.findall(r"expected p = (\d), measured p = ([\d.]+)", out)]
    assert [p for p, _ in measured] == [1, 2, 4, 1]
    for p, m in measured:
        assert m == pytest.approx(p, abs=0.1)
    assert (tmp_path / "accuracy_1c.png").exists()
