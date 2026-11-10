"""Smoke tests for the plot scripts: run headless, check printed results and figures."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def run(script, tmp_path):
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path), "PYTHONPATH": str(HERE)}
    return subprocess.run([sys.executable, str(HERE / script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True).stdout


def test_1b_plots_orbits_and_reports_newton_iterations(tmp_path):
    out = run("plot_1b.py", tmp_path)
    assert (tmp_path / "orbits_1b.png").exists()
    iterations = [int(n) for n in re.findall(r"Newton iterations per step: (\d+)", out)]
    assert iterations and 1 <= min(iterations) <= 5
    for name in ["euler", "implicit_euler", "heun", "symplectic_euler", "velocity_verlet", "rk4"]:
        assert re.search(rf"{name}: r_end / r_M = \d", out)


def test_1c_plots_relative_errors_of_energy_and_angular_momentum(tmp_path):
    out = run("plot_1c.py", tmp_path)
    assert (tmp_path / "conservation_1c.png").exists()
    for name in ["euler", "implicit_euler", "heun", "symplectic_euler", "velocity_verlet", "rk4"]:
        assert re.search(rf"{name}: max \|dE/E\| = \S+, max \|dL/L\| = \S+", out)


def test_1d_plots_errors_and_reports_slopes(tmp_path):
    out = run("plot_1d.py", tmp_path)
    assert (tmp_path / "convergence_1d.png").exists()
    slopes = dict(re.findall(r"(\w+): slope = (-?[\d.]+)", out))
    assert set(slopes) == {"euler", "implicit_euler", "heun", "symplectic_euler", "velocity_verlet", "rk4"}
    assert abs(float(slopes["rk4"]) - 4) < 0.1


def test_2a_plots_vertical_launch_and_finds_escape_thrust(tmp_path):
    out = run("plot_2a.py", tmp_path)
    assert (tmp_path / "vertical_2a.png").exists()
    for thrust in [10, 12, 14, 16, 18]:
        assert re.search(rf"F/m0 = {thrust}: .*v_burnout = \d+ m/s, rocket equation = \d+ m/s, total loss = \d+ m/s, gravity loss = \d+ m/s, .*escapes = (True|False)", out)
    threshold = float(re.search(r"escape needs F/m0 >= ([\d.]+)", out).group(1))
    assert 17.0 < threshold < 17.3
    assert "orbit possible: False" in out


def test_2b_classifies_pitch_programs_and_finds_best_tp(tmp_path):
    out = run("plot_2b.py", tmp_path)
    assert (tmp_path / "pitch_2b.png").exists()
    found = re.findall(r"(east|west) tp = (\d+) s: .*-> (crash|falls back|orbit|escape)", out)
    classes = {(d, int(tp)): c for d, tp, c in found}
    assert len(classes) == 18
    assert classes[("east", 175)] == "orbit"
    assert all(classes[("west", tp)] != "orbit" for tp in range(100, 301, 25))
    assert re.search(r"best tp \(east\) = 175 s", out)


def test_2c_reaches_circular_orbits_with_fuel_left(tmp_path):
    out = run("plot_2c.py", tmp_path)
    assert (tmp_path / "circular_2c.png").exists()
    for h in [200, 500, 1000]:
        m = re.search(
            rf"h = {h} km: burn 1 = ([\d.]+) s, .*burn 2 = ([\d.]+) s, periapsis = ([\d.]+) km, "
            rf"apoapsis = ([\d.]+) km, fuel left = ([\d.]+) % of the fuel \(([\d.]+) m0\)",
            out,
        )
        assert m, h
        burn1, burn2, h_p, h_a, fuel, fuel_m0 = map(float, m.groups())
        assert abs(h_p - h) < 2 and abs(h_a - h) < 2
        assert 0 < fuel < 5
        # constant force F: the fuel burnt depends only on the total burn time, F (t1 + t2) / v_e
        assert abs(1 - 15.0 * (burn1 + burn2) / 4400.0 - 0.12 - fuel_m0) < 3e-4


def test_2d_compares_one_and_two_stages(tmp_path):
    out = run("plot_2d.py", tmp_path)
    assert (tmp_path / "staging_2d.png").exists()
    one = re.search(r"one stage: dv = (\d+) m/s, .*gravity loss = (\d+) m/s.*-> (\w[\w ]*)", out)
    two = re.search(r"two stages: dv = (\d+) m/s, .*gravity loss = (\d+) m/s.*-> (\w[\w ]*)", out)
    assert one and two
    assert abs(int(one.group(1)) - 4400 * np.log(10)) < 2
    assert abs(int(two.group(1)) - 4400 * (np.log(1 / 0.28) + np.log(10))) < 2
    assert one.group(3) == "orbit" and two.group(3) == "escape"
    assert all(300 < int(m.group(2)) < 1500 for m in (one, two))
    # speed gain + gravity loss + steering loss must add up to the rocket equation's dv
    for name in ["one stage", "two stages"]:
        dv, gain, g, steer = map(int, re.search(
            rf"{name}: dv = (\d+) m/s, .*speed gain = (\d+) m/s, gravity loss = (\d+) m/s, steering loss = (\d+) m/s", out
        ).groups())
        assert abs(gain + g + steer - dv) <= 3
