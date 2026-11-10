"""Exercise 4, 2d: one stage (90 % fuel) against two stages (80 % + 20 % of m0, each 90 % fuel)."""

import matplotlib.pyplot as plt
import numpy as np
from rocket import R_EARTH, V_EXHAUST, classify, fly, gravity_loss, launch_state, rocket_rhs

THRUST = 15.0  # F/m at ignition of every stage
TP = 175.0  # best pitch program from 2b
T_END = 3 * 3600.0


def pitch(t):
    return 90.0 * max(0.0, 1 - t / TP)


def stage_burn(y, m_start, m_end, t_ignition):
    """Burn one stage from m_start to m_end (in m0) with F/m_start = THRUST; theta keeps the launch clock."""
    duration = (1 - m_end / m_start) * V_EXHAUST / THRUST
    theta = lambda t: pitch(t_ignition + t)
    t, ys = fly(y, duration, thrust=THRUST, theta=theta)
    return t_ignition + t, ys, theta


def thrust_along_velocity(t, ys, theta, t_ignition):
    """Integral of (F/m) e . v/|v| dt: the part of the thrust that speeds the rocket up.

    dv minus this is the steering loss, because e does not point along v.
    """
    a = [rocket_rhs(s - t_ignition, z, THRUST, 0.0, theta, True, True)[2:] for s, z in zip(t, ys)]
    a_g = np.array([rocket_rhs(s, z, THRUST, 0.0, theta, True, False)[2:] for s, z in zip(t, ys)])
    along = np.sum((np.array(a) - a_g) * ys[:, 2:], axis=1) / np.linalg.norm(ys[:, 2:], axis=1)
    return np.trapezoid(along, t)


def simulate(stages):
    """stages: [(m_start, m_end), ...] in m0, dropping the empty stage between them."""
    y, t0, pieces, dv, g_loss, s_loss = launch_state(), 0.0, [], 0.0, 0.0, 0.0
    for m_start, m_end in stages:
        t, ys, theta = stage_burn(y, m_start, m_end, t0)
        dv += V_EXHAUST * np.log(m_start / m_end)
        g_loss += gravity_loss(t, ys)
        s_loss += V_EXHAUST * np.log(m_start / m_end) - thrust_along_velocity(t, ys, theta, t0)
        pieces.append((t, ys))
        y, t0 = ys[-1], t[-1]
    t_c, ys_c = fly(y, T_END - t0, t0=t0)
    t_all = np.concatenate([p[0] for p in pieces] + [t_c[1:]])
    ys_all = np.vstack([p[1] for p in pieces] + [ys_c[1:]])
    return t_all, ys_all, t0, dv, g_loss, s_loss


if __name__ == "__main__":
    cases = {"one stage": [(1.0, 0.1)], "two stages": [(1.0, 0.28), (0.2, 0.02)]}
    fig, ax = plt.subplots(figsize=(8, 8))
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.fill(R_EARTH / 1e6 * np.cos(phi), R_EARTH / 1e6 * np.sin(phi), color="lightsteelblue", zorder=0)
    for name, stages in cases.items():
        t, ys, tb, dv, g_loss, s_loss = simulate(stages)
        k = np.argmin(np.abs(t - tb))
        v0, vb = np.linalg.norm(ys[0, 2:]), np.linalg.norm(ys[k, 2:])
        kind = classify(t, ys, tb)
        print(
            f"{name}: dv = {dv:.0f} m/s, burnout at {tb:.0f} s, h = {(np.linalg.norm(ys[k, :2]) - R_EARTH) / 1e3:.0f} km, "
            f"speed gain = {vb - v0:.0f} m/s, gravity loss = {g_loss:.0f} m/s, steering loss = {s_loss:.0f} m/s -> {kind}"
        )
        ax.plot(ys[:, 0] / 1e6, ys[:, 1] / 1e6, lw=1, label=f"{name}: dv = {dv / 1e3:.2f} km/s, {kind}")
    ax.set_aspect("equal")
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("x / 1000 km")
    ax.set_ylabel("y / 1000 km")
    ax.legend()
    ax.set_title(f"Launch east, tp = {TP:.0f} s, F/m = 15 m/s² per stage, 3 h")
    fig.tight_layout()
    fig.savefig("staging_2d.png", dpi=150)
