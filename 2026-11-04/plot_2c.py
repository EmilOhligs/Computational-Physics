"""Exercise 4, 2c: circular orbits at h = 200, 500, 1000 km with burn, coast, second burn."""

import matplotlib.pyplot as plt
import numpy as np
from orbit import GM, energy
from rocket import R_EARTH, V_EXHAUST, apoapsis, burn_time, fly, fly_until, launch_state, mass_after, periapsis

THRUST = 15.0  # F/m0, the same engine for both burns
M_DRY = 0.12
TP = 175.0  # best pitch program from 2b
HEIGHTS = [200e3, 500e3, 1000e3]


def pitch(t):
    return 90.0 * max(0.0, 1 - t / TP)


def to_circular_orbit(h):
    """Burn until the apoapsis is at R_E + h, coast to the apoapsis, burn horizontally until circular.

    Returns the three phases [(t, ys), ...] and the mass left (in m0).
    """
    target = R_EARTH + h
    t1, y1 = fly_until(launch_state(), lambda z: apoapsis(z) - target, burn_time(THRUST, M_DRY), thrust=THRUST, theta=pitch)
    m = mass_after(1.0, THRUST, t1[-1])
    assert y1[-1, :2] @ y1[-1, 2:] > 0, "engine cut after the apoapsis"
    # apoapsis: radial velocity r.v turns from + to -
    t2, y2 = fly_until(y1[-1], lambda z: -(z[:2] @ z[2:]), 20000.0, t0=t1[-1])
    # circular when E = -GM / (2r); the engine relights with F/m = THRUST / m
    fuel_time = (1 - M_DRY / m) * V_EXHAUST / (THRUST / m)
    circular = lambda z: energy(z[None, :])[0] + GM / (2 * np.linalg.norm(z[:2]))
    t3, y3 = fly_until(y2[-1], circular, fuel_time, thrust=THRUST / m, theta=lambda t: 0.0)
    m = mass_after(m, THRUST / m, t3[-1])
    return [(t1, y1), (t2, y2), (t2[-1] + t3, y3)], m


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.fill(R_EARTH / 1e6 * np.cos(phi), R_EARTH / 1e6 * np.sin(phi), color="lightsteelblue", zorder=0)
    for h in HEIGHTS:
        phases, m = to_circular_orbit(h)
        (t1, _), (t2, _), (t3, y3) = phases
        y = y3[-1]
        print(
            f"h = {h / 1e3:.0f} km: burn 1 = {t1[-1]:.1f} s, coast = {t2[-1] - t2[0]:.0f} s, burn 2 = {t3[-1] - t3[0]:.2f} s, "
            f"periapsis = {(periapsis(y) - R_EARTH) / 1e3:.1f} km, apoapsis = {(apoapsis(y) - R_EARTH) / 1e3:.1f} km, "
            f"fuel left = {(m - M_DRY) / (1 - M_DRY) * 100:.2f} % of the fuel ({m - M_DRY:.4f} m0)"
        )
        _, orbit = fly(y, 2 * np.pi * np.sqrt(np.linalg.norm(y[:2]) ** 3 / GM))
        (line,) = ax.plot(orbit[:, 0] / 1e6, orbit[:, 1] / 1e6, lw=0.8, label=f"h = {h / 1e3:.0f} km")
        for (_, ys), style in zip(phases, ["-", ":", "-"]):
            ax.plot(ys[:, 0] / 1e6, ys[:, 1] / 1e6, style, color=line.get_color(), lw=2 if style == "-" else 1)
        ax.plot(y3[0, 0] / 1e6, y3[0, 1] / 1e6, "o", color=line.get_color())
    ax.set_aspect("equal")
    ax.set_xlabel("x / 1000 km")
    ax.set_ylabel("y / 1000 km")
    ax.legend()
    ax.set_title(f"Burn (tp = {TP:.0f} s) until apoapsis = R_E + h, coast (dotted), circularize (dot)")
    fig.tight_layout()
    fig.savefig("circular_2c.png", dpi=150)
