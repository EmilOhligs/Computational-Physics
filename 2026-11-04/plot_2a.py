"""Exercise 4, 2a: vertical launch, m1 = m0/19, F/m0 = 10 ... 18 m/s^2."""

import matplotlib.pyplot as plt
import numpy as np
from orbit import energy
from rocket import R_EARTH, V_EXHAUST, burn_time, gravity_loss, launch, periapsis

M1 = 1 / 19
THRUSTS = [10, 12, 14, 16, 18]
T_END = 6 * 3600.0


def burnout(thrust):
    """State at burnout."""
    return launch(thrust, M1, t_end=burn_time(thrust, M1))[1][-1]


if __name__ == "__main__":
    dv_rocket = V_EXHAUST * np.log(1 / M1)
    fig, (ax_h, ax_v) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    rp_max = 0.0
    for thrust in THRUSTS:
        tb = burn_time(thrust, M1)
        t, ys = launch(thrust, M1, t_end=T_END)
        radius = np.linalg.norm(ys[:, :2], axis=1)
        v_radial = np.sum(ys[:, :2] * ys[:, 2:], axis=1) / radius
        y_b = ys[np.argmin(np.abs(t - tb))]
        v_b = np.linalg.norm(y_b[2:])
        E_b = energy(y_b[None, :])[0]
        rp_max = max(rp_max, periapsis(y_b))
        print(
            f"F/m0 = {thrust}: t_b = {tb:.0f} s, h_b = {(np.linalg.norm(y_b[:2]) - R_EARTH) / 1e3:.0f} km, "
            f"v_burnout = {v_b:.0f} m/s, rocket equation = {dv_rocket:.0f} m/s, "
            f"total loss = {dv_rocket - (v_b - np.linalg.norm(ys[0, 2:])):.0f} m/s, "
            f"gravity loss = {gravity_loss(*launch(thrust, M1, t_end=tb)):.0f} m/s, "
            f"E_b = {E_b:.3e} J/kg, escapes = {E_b >= 0}"
        )
        label = f"F/m0 = {thrust} m/s²"
        ax_h.plot(t / 3600, (radius - R_EARTH) / 1e6, label=label)
        ax_v.plot(t / 3600, v_radial / 1e3, label=label)

    # bisection on the sign of E at burnout
    lo, hi = 16.0, 18.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if energy(burnout(mid)[None, :])[0] >= 0 else (mid, hi)
    print(f"escape needs F/m0 >= {hi:.3f} m/s^2")
    print(f"largest periapsis after burnout: {rp_max / 1e3:.1f} km (R_E = {R_EARTH / 1e3:.0f} km), orbit possible: {rp_max > R_EARTH}")

    ax_h.set_ylabel("altitude / 1000 km")
    ax_v.axhline(0, color="gray", lw=0.5)
    ax_v.set_ylabel("vertical speed / (km/s)")
    ax_v.set_xlabel("t / h")
    ax_h.legend()
    fig.suptitle("Vertical launch, m1 = m0/19, v_e = 4.4 km/s, 6 h or until impact")
    fig.tight_layout()
    fig.savefig("vertical_2a.png", dpi=150)
