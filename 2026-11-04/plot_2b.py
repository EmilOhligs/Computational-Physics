"""Exercise 4, 2b: pitch program theta(t) = 90 max(0, 1 - t/tp), F/m0 = 15 m/s^2, m1 = 0.12 m0, 3 h."""

import matplotlib.pyplot as plt
import numpy as np
from rocket import R_EARTH, burn_time, classify, launch, periapsis

THRUST = 15.0
M1 = 0.12
TPS = range(100, 301, 25)
T_END = 3 * 3600.0
COLORS = {"crash": "tab:red", "falls back": "tab:orange", "orbit": "tab:green", "escape": "tab:purple"}


def pitch(tp):
    """theta(t) in degrees: linear turn from vertical to horizontal within tp."""
    return lambda t: 90.0 * max(0.0, 1 - t / tp)


if __name__ == "__main__":
    tb = burn_time(THRUST, M1)
    print(f"burn time t_b = {tb:.1f} s")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    best = (None, -np.inf)
    for ax, east in zip(axes, [True, False]):
        name = "east" if east else "west"
        for tp in TPS:
            t, ys = launch(THRUST, M1, T_END, theta=pitch(tp), east=east)
            kind = classify(t, ys, tb)
            y_b = ys[np.argmin(np.abs(t - tb))]
            h_p = (periapsis(y_b) - R_EARTH) / 1e3
            print(f"{name} tp = {tp} s: periapsis altitude {h_p:.0f} km, end t = {t[-1]:.0f} s -> {kind}")
            if east and kind == "orbit" and h_p > best[1]:
                best = (tp, h_p)
            ax.plot(ys[:, 0] / 1e6, ys[:, 1] / 1e6, color=COLORS[kind], lw=0.8, alpha=0.5 + tp / 600)
        phi = np.linspace(0, 2 * np.pi, 400)
        ax.fill(R_EARTH / 1e6 * np.cos(phi), R_EARTH / 1e6 * np.sin(phi), color="lightsteelblue", zorder=0)
        for kind, color in COLORS.items():
            ax.plot([], [], color=color, label=kind)
        ax.set_aspect("equal")
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.set_xlabel("x / 1000 km")
        ax.set_ylabel("y / 1000 km")
        ax.set_title(f"launch {name}, tp = 100 ... 300 s")
        ax.legend(loc="lower left", fontsize=8)
    print(f"best tp (east) = {best[0]} s, periapsis altitude {best[1]:.0f} km")
    fig.suptitle("Pitch program, F/m0 = 15 m/s², m1 = 0.12 m0, 3 h")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("pitch_2b.png", dpi=150)
