"""All simulations and figures for exercise 6. Run: ../.venv/bin/python figures.py [1a 1b ... 3c]

Each part saves its PNG (and GIF for 3a/3b) next to this file and stores its numbers in results.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from schroedinger import (absorber, eigenstates, free_packet, gaussian, grid, hamiltonian, hamiltonian_2d,  # noqa: E402
                          moments, solenoid, stepper, transmission, well_energies)

HERE = Path(__file__).parent
RESULTS = HERE / "results.json"

# validated categorical palette (blue, orange, aqua, yellow, violet), ink and grid tones
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 11, "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 9.5, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "figure.constrained_layout.use": True, "image.cmap": "inferno",
})


def save(fig, name):
    fig.savefig(HERE / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def store(part, values):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[part] = values
    RESULTS.write_text(json.dumps(data, indent=1, default=float))
    print(part, json.dumps(values, default=float)[:600])


def image_axes(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- 1 stationary

def part_1a():
    x, dx = grid(-8, 8, 1599)
    E, psi = eigenstates(x**2 / 2, dx, 6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), width_ratios=[1, 1.25])
    ax1.plot(x, x**2 / 2, color=INK2, lw=1.5)
    for n in range(6):
        ax1.axhline(E[n], color=GRID, lw=1, zorder=0)
        ax1.fill_between(x, E[n], E[n] + 0.8 * psi[:, n], color=C[n % 5], alpha=0.35, lw=0)
        ax1.plot(x, E[n] + 0.8 * psi[:, n], color=C[n % 5], lw=1.6)
        ax1.text(4.9, E[n] + 0.1, f"$n={n}$:  $E={E[n]:.6f}$", fontsize=9, color=INK2)
    ax1.set(xlim=(-5, 7.6), ylim=(0, 6.4), xlabel="$x$", ylabel=r"$E$,  $\psi_n$ (versetzt)",
            title="Eigenzustände des harmonischen Oszillators")

    out = {}
    nmax = 120
    n = np.arange(nmax)
    for i, (L, h) in enumerate([(10, 0.1), (10, 0.05), (10, 0.02), (6, 0.02)]):
        x, dx = grid(-L, L, round(2 * L / h) - 1)
        E, _ = eigenstates(x**2 / 2, dx, nmax)
        err = np.abs(E - (n + 0.5)) / (n + 0.5)
        ax2.semilogy(n, err, color=C[i], label=fr"$L={L}$, $\Delta x={h}$")
        if L == 10:
            ax2.semilogy(n[:45], dx**2 / 32 * (2 * n[:45]**2 + 2 * n[:45] + 1) / (n[:45] + 0.5),
                         color=C[i], ls=":", lw=1.4)
        good = int(np.argmax(err > 1e-3)) if np.any(err > 1e-3) else nmax
        out[f"L{L}_dx{h}"] = {"E0_err": err[0] * 0.5, "levels_below_1e-3": good, "E": E[:6].tolist()}
    for L, col in [(10, C[0]), (6, C[3])]:
        nt = (L**2 - 1) / 2  # classical turning point sqrt(2n+1) = L
        ax2.axvline(nt, color=col, ls="--", lw=1)
        ax2.text(nt + 1, 3e-8, f"Umkehrpunkt\nerreicht $L={L}$", color=INK2, fontsize=9)
    ax2.plot([], [], color=MUTED, ls=":", lw=1.4, label=r"Störungsrechnung $\Delta x^2\langle p^4\rangle/24$")
    ax2.set(xlabel="$n$", ylabel=r"$|E_n-(n+\frac{1}{2})|\,/\,(n+\frac{1}{2})$", ylim=(1e-9, 10),
            title=r"Relativer Fehler: Gitter $\Delta x$ und Kasten $[-L, L]$")
    ax2.legend(loc="lower right")
    save(fig, "oscillator_1a.png")
    store("1a", out)


def part_1b():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    out = {}
    for i, N in enumerate([31, 127, 511]):
        x, dx = grid(0, 1, N)
        E, _ = eigenstates(np.zeros(N), dx, N)
        n = np.arange(1, N + 1)
        ax1.loglog(n, np.abs(E / (n * np.pi) ** 2 * 2 - 1), color=C[i], label=f"$N={N}$")
    nn = np.logspace(0, np.log10(511), 50)
    ax1.loglog(nn, (nn * np.pi / 512) ** 2 / 12, color=MUTED, ls=":", lw=1.4,
               label=r"$(n\pi\Delta x)^2/12$, $N=511$")
    ax1.set(xlabel="$n$", ylabel=r"$|E_n^{\rm FD}/E_n-1|$", ylim=(1e-7, 2),
            title="Fehler wächst mit $n$: höhere Moden sind schlechter aufgelöst")
    ax1.legend()

    ks = np.arange(3, 13)
    for i, m in enumerate([1, 2, 4, 8]):
        errs, dxs = [], []
        for k in ks:
            x, dx = grid(0, 1, 2**k - 1)
            if 2**k - 1 < m:
                continue
            E, _ = eigenstates(np.zeros_like(x), dx, m)
            errs.append(abs(E[-1] / (m * np.pi) ** 2 * 2 - 1))
            dxs.append(dx)
        ax2.loglog(dxs, errs, "o-", color=C[i], ms=5, label=f"$n={m}$")
        slope = np.polyfit(np.log(dxs[-5:]), np.log(errs[-5:]), 1)[0]
        out[f"n{m}"] = {"slope": slope, "err_dx_1_4096": errs[-1]}
    ax2.set(xlabel=r"$\Delta x$", ylabel=r"$|E_n^{\rm FD}/E_n-1|$",
            title=r"Konvergenz: Steigung 2, Vorfaktor $(n\pi)^2/12$")
    ax2.invert_xaxis()
    ax2.legend()
    save(fig, "well_1b.png")
    store("1b", out)


def part_1c():
    V0, a = 10, 1
    z0 = a * np.sqrt(2 * V0)
    Ea = well_energies(V0, a)
    x, dx = grid(-8, 8, 7999)
    V = np.where(np.abs(x) < a, -V0, 0.0)
    E, psi = eigenstates(V, dx, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    z = np.linspace(1e-3, z0, 2000)
    even, odd = z * np.tan(z), -z / np.tan(z)
    even[np.abs(np.diff(even, append=even[-1])) > 5] = np.nan
    odd[np.abs(np.diff(odd, append=odd[-1])) > 5] = np.nan
    ax1.plot(z, np.sqrt(z0**2 - z**2), color=INK2, lw=1.6, label=r"$\kappa a=\sqrt{z_0^2-z^2}$")
    ax1.plot(z, even, color=C[0], label=r"gerade: $z\tan z$")
    ax1.plot(z, odd, color=C[1], label=r"ungerade: $-z\cot z$")
    zs = a * np.sqrt(2 * (Ea + V0))
    ax1.plot(zs, np.sqrt(z0**2 - zs**2), "o", color=INK, ms=8, mfc="white", mew=2, zorder=5)
    for k in range(1, 3):
        ax1.axvline(k * np.pi / 2, color=MUTED, lw=1, ls="--")
    ax1.set(xlim=(0, z0 + 0.1), ylim=(0, 6), xlabel="$z = ka$", ylabel=r"$\kappa a$",
            title=fr"Grafische Lösung: $z_0=a\sqrt{{2V_0}}={z0:.3f}$ → 3 Schnittpunkte")
    ax1.legend(loc="upper right")

    ax2.plot(x, V, color=INK2, lw=1.5)
    for n in range(3):
        ax2.axhline(Ea[n], color=C[n], lw=1, ls="--")
        ax2.fill_between(x, E[n], E[n] + 1.6 * psi[:, n], color=C[n], alpha=0.3, lw=0)
        ax2.plot(x, E[n] + 1.6 * psi[:, n], color=C[n], lw=1.6)
        ax2.text(2.0, Ea[n] + 0.25, f"$E_{n}={Ea[n]:.4f}$ (FD {E[n]:.4f})", fontsize=9, color=INK2)
    ax2.text(2.0, 0.3, f"4. FD-Zustand: $E={E[3]:.3f}>0$ (Kasten)", fontsize=9, color=MUTED)
    ax2.set(xlim=(-4, 6), ylim=(-11, 1.5), xlabel="$x$", ylabel=r"$E$,  $\psi_n$ (versetzt)",
            title="Gebundene Zustände im endlichen Topf")
    save(fig, "finite_well_1c.png")
    store("1c", {"z0": z0, "n_bound": len(Ea), "E_exact": Ea.tolist(), "E_fd": E[:3].tolist(),
                 "E_fd_4th": E[3], "n_bound_formula": int(np.ceil(z0 / (np.pi / 2)))})


# ---------------------------------------------------------------- 2 time-dependent 1D

def run(H, psi, dt, nsteps, method="cn", every=1, stop=np.inf):
    """Evolve and return snapshots every `every` steps (including t=0); stops when norm^2 * dx > stop."""
    step, frames = stepper(H, dt, method), [psi]
    for i in range(1, nsteps + 1):
        psi = step(psi)
        if i % every == 0:
            frames.append(psi)
        if not np.isfinite(psi).all() or np.vdot(psi, psi).real > stop:
            break
    return np.array(frames)


def part_2a():
    x0, sigma, k0 = -5, 1, 2
    x, dx = grid(-20, 40, 1199)
    H = hamiltonian(np.zeros_like(x), dx)
    psi0 = gaussian(x, x0, sigma, k0)
    dt, T = 0.01, 10
    nsteps = round(T / dt)
    runs = {"Crank–Nicolson": run(H, psi0, dt, nsteps),
            "implizit Euler": run(H, psi0, dt, nsteps, "implicit"),
            "explizit Euler": run(H, psi0, dt, nsteps, "explicit", stop=1e8 / dx)}
    small = run(H, psi0, 1e-4, 30000, "explicit", every=100, stop=1e8 / dx)
    cols = {"Crank–Nicolson": C[0], "implizit Euler": C[1], "explizit Euler": C[2]}

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax = axs[0, 0]
    psi_cn = runs["Crank–Nicolson"]
    for j, t in enumerate([0, 2.5, 5, 7.5, 10]):
        i = round(t / dt)
        m, s = free_packet(t, x0, sigma, k0)
        exact = np.exp(-(x - m) ** 2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)
        ax.fill_between(x, np.abs(psi_cn[i]) ** 2, color=C[0], alpha=0.15 + 0.12 * j, lw=0)
        ax.plot(x, np.abs(runs["implizit Euler"][i]) ** 2, color=C[1], lw=1.3)
        ax.plot(x, exact, color=INK, lw=1, ls="--")
        ax.text(m, exact.max() + 0.015, f"$t={t:g}$", ha="center", fontsize=9, color=INK2)
    ax.plot([], [], color=C[0], lw=6, alpha=0.5, label="Crank–Nicolson")
    ax.plot([], [], color=C[1], lw=1.3, label="implizit Euler")
    ax.plot([], [], color=INK, lw=1, ls="--", label="analytisch")
    ax.set(xlim=(-10, 32), ylim=(0, 0.47), xlabel="$x$", ylabel=r"$|\psi(x,t)|^2$",
           title=r"Freies Paket $\sigma=1$, $k_0=2$, $\Delta t=0.01$, $\Delta x=0.05$")
    ax.legend(loc="upper right")

    out = {}
    ax = axs[0, 1]
    for name, fr in runs.items():
        t = np.arange(len(fr)) * dt
        norm = np.sqrt(np.sum(np.abs(fr) ** 2, axis=1) * dx)
        ax.semilogy(t, norm, color=cols[name], label=name)
        out[name] = {"norm_end": norm[-1], "t_end": t[-1], "norm_dev_max": np.max(np.abs(norm - 1))}
    t = np.arange(len(small)) * 100 * 1e-4
    norm = np.sqrt(np.sum(np.abs(small) ** 2, axis=1) * dx)
    ax.semilogy(t, norm, color=C[2], ls=":", label=r"explizit Euler, $\Delta t=10^{-4}$")
    out["explicit_dt1e-4"] = {"t_norm_2": t[np.argmax(norm > 2)] if np.any(norm > 2) else None, "t_end": t[-1]}
    ax.set(xlabel="$t$", ylabel=r"$\|\psi\|$", ylim=(0.5, 1e4), xlim=(0, T),
           title=r"Norm: nur Crank–Nicolson bleibt bei 1")
    ax.legend(loc="upper right")

    tt = np.linspace(0, T, 200)
    exact = np.array([free_packet(t, x0, sigma, k0) for t in tt])
    for ax, k, label in [(axs[1, 0], 1, r"\langle x\rangle"), (axs[1, 1], 2, r"\sigma")]:
        for name, fr in runs.items():
            mom = np.array([moments(x, p, dx) for p in fr])
            t = np.arange(len(fr)) * dt
            ok = mom[:, 0] < 5
            ax.plot(t[ok], mom[ok, k] - np.interp(t[ok], tt, exact[:, k - 1]), color=cols[name], label=name)
            out[name][f"err_{'mean' if k == 1 else 'std'}_end"] = (mom[ok, k] - np.interp(t[ok], tt, exact[:, k - 1]))[-1]
        ax.axhline(0, color=INK, lw=1, ls="--")
        ax.set(xlabel="$t$", ylabel=f"${label}_{{\\rm num}}-{label}_{{\\rm exakt}}$", xlim=(0, T),
               title=f"Abweichung von ${label}(t)$" + (r" (exakt $x_0+k_0t$)" if k == 1 else
                                                         r" (exakt $\sigma\sqrt{1+t^2/4\sigma^4}$)"))
        ax.set_ylim(-2, 0.2) if k == 1 else ax.set_ylim(-0.6, 0.1)
        ax.legend()
    save(fig, "free_packet_2a.png")

    # how fine must the grid be? error of <x> at t = 5 against points per wavelength
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, kk in enumerate([2, 5]):
        ppw, err, pred = [], [], []
        for h in [0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02]:
            x, dx = grid(-20, 60, round(80 / h) - 1)
            fr = run(hamiltonian(np.zeros_like(x), dx), gaussian(x, -5, 1, kk), 0.002, 2500, every=2500)
            ppw.append(2 * np.pi / kk / dx)
            err.append(abs(moments(x, fr[-1], dx)[1] - free_packet(5, -5, 1, kk)[0]))
            pred.append(abs(kk * 5 * (1 - np.sin(kk * dx) / (kk * dx))))
        ax.loglog(ppw, err, "o-", color=C[i], ms=5, label=f"$k_0={kk}$ gemessen")
        ax.loglog(ppw, pred, ":", color=C[i], lw=1.4, label=r"$k_0t\,(1-\sin(k_0\Delta x)/k_0\Delta x)$")
        out[f"grid_k{kk}"] = {"ppw": ppw, "err": err}
    ax.axvline(10, color=MUTED, ls="--", lw=1)
    ax.text(10.5, 2, "10 Punkte\npro Wellenlänge", fontsize=9, color=INK2)
    ax.set(xlabel=r"Punkte pro Wellenlänge $\lambda/\Delta x = 2\pi/(k_0\Delta x)$",
           ylabel=r"$|\langle x\rangle_{\rm num}-\langle x\rangle|$ bei $t=5$",
           title="Wie fein muss das Gitter sein?")
    ax.legend(fontsize=8.5)
    save(fig, "grid_2a.png")
    store("2a", out)


def part_2b():
    x, dx = grid(-10, 10, 999)
    H = hamiltonian(x**2 / 2, dx)
    dt, T = 0.005, 4 * np.pi
    nsteps, every = round(T / dt), 10
    fr = run(H, gaussian(x, 3, 1 / np.sqrt(2), 0), dt, nsteps, every=every)
    wide = run(H, gaussian(x, 3, 1, 0), dt, nsteps, every=every)
    t = np.arange(len(fr)) * dt * every
    mom = np.array([moments(x, p, dx) for p in fr])
    mom_w = np.array([moments(x, p, dx) for p in wide])

    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(np.abs(fr.T) ** 2, extent=(0, t[-1], x[0], x[-1]), origin="lower", aspect="auto", cmap="inferno")
    tt = np.linspace(0, T, 500)
    ax.plot(tt, 3 * np.cos(tt), color="white", ls="--", lw=1.4, label=r"klassisch: $3\cos t$")
    ax.set(ylim=(-5, 5), xlabel="$t$", ylabel="$x$",
           title=r"$|\psi(x,t)|^2$ im Oszillator: verschobener Grundzustand $x_0=3$ (kohärenter Zustand)")
    ax.set_xticks(np.arange(0, 4.1, 0.5) * np.pi, [f"${v:g}\\pi$" if v else "0" for v in np.arange(0, 4.1, 0.5)])
    ax.text(0.25 * np.pi, -4.4, r"gestrichelt: klassisch $x(t) = 3\cos t$", color="white", fontsize=10)
    image_axes(ax)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, mom[:, 1] - 3 * np.cos(t), color=C[0])
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.set(xlabel="$t$", ylabel=r"$\langle x\rangle - 3\cos t$", title=r"$\langle x\rangle$ folgt dem klassischen Teilchen")
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, mom[:, 2], color=C[0], label=r"$\sigma_0=1/\sqrt{2}$ (Grundzustand)")
    ax.plot(t, mom_w[:, 2], color=C[1], label=r"$\sigma_0=1$ (breiter)")
    ax.axhline(1 / np.sqrt(2), color=INK, lw=1, ls="--")
    ax.set(xlabel="$t$", ylabel=r"$\sigma(t)$", ylim=(0.45, 1.25), title="Breite: konstant bzw. atmend, kein Zerfließen")
    ax.legend(loc="upper center", ncol=2)
    save(fig, "oscillator_2b.png")
    store("2b", {"max_dev_mean": np.max(np.abs(mom[:, 1] - 3 * np.cos(t))), "sigma_min": mom[:, 2].min(),
                 "sigma_max": mom[:, 2].max(), "wide_sigma_min": mom_w[:, 2].min(), "wide_sigma_max": mom_w[:, 2].max(),
                 "norm_end": mom[-1, 0]})


def part_2c():
    x, dx = grid(0, 1, 1999)
    H = hamiltonian(np.zeros_like(x), dx)
    Trev = 4 / np.pi
    nsteps, every = 40000, 50
    dt = Trev / nsteps
    psi0 = gaussian(x, 0.3, 0.05, 0)
    fr = run(H, psi0, dt, nsteps, every=every)
    t = np.arange(len(fr)) * dt * every
    mirror = psi0[::-1]
    fid = np.abs(fr.conj() @ psi0 * dx) ** 2
    fid_m = np.abs(fr.conj() @ mirror * dx) ** 2

    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(np.abs(fr.T) ** 2, extent=(0, t[-1] / Trev, 0, 1), origin="lower", aspect="auto",
              norm=PowerNorm(0.5, vmax=np.abs(psi0).max() ** 2))
    for f, lab in [(0.25, r"$\frac{1}{4}$"), (0.5, r"$\frac{1}{2}$"), (1, "1")]:
        ax.axvline(f, color="white", lw=0.8, ls=":")
    ax.set(xlabel=r"$t\,/\,T_{\rm rev}$", ylabel="$x$",
           title=r"$|\psi(x,t)|^2$ im unendlichen Topf, $x_0=0.3$, $\sigma=0.05$  (Farbe $\propto\sqrt{|\psi|^2}$)")
    image_axes(ax)
    for j, (f, lab) in enumerate([(0, "$t=0$"), (0.25, r"$t=T_{\rm rev}/4$"), (0.5, r"$t=T_{\rm rev}/2$"),
                                  (1, r"$t=T_{\rm rev}$")]):
        a = fig.add_subplot(gs[1, j])
        i = round(f * nsteps / every)
        a.fill_between(x, np.abs(fr[i]) ** 2, color=C[0], alpha=0.35, lw=0)
        a.plot(x, np.abs(fr[i]) ** 2, color=C[0], lw=1.4)
        a.plot(x, np.abs(psi0) ** 2, color=INK, lw=0.9, ls="--")
        a.set(xlabel="$x$", ylim=(0, 8.5), title=lab)
        if j == 0:
            a.set_ylabel(r"$|\psi|^2$")
    save(fig, "revival_2c.png")
    i_half, i_q = nsteps // every // 2, nsteps // every // 4
    store("2c", {"Trev": Trev, "fid_Trev": fid[-1], "fid_mirror_half": fid_m[i_half], "fid_half": fid[i_half],
                 "fid_quarter": fid[i_q], "fid_mirror_quarter": fid_m[i_q], "norm_end": np.sqrt(np.sum(np.abs(fr[-1]) ** 2) * dx)})


def transmitted(E, V0=1.0, b=1.0, sigma=5.0, dx=0.05, cap=20.0):
    """Transmitted probability of a packet: what is right of the barrier plus what the right absorber took."""
    k0 = np.sqrt(2 * E)
    x, dx = grid(-150, 150, round(300 / dx) - 1)
    W = absorber(x, -150, 150, cap, 0.5)
    H = hamiltonian(np.where((x > -dx / 2) & (x < b - dx / 2), V0, 0.0) - 1j * W, dx)  # exactly b / dx points
    dt = 0.05
    step, psi = stepper(H, dt, "cn"), gaussian(x, -60, sigma, k0)
    right, absorbed = x > b, 0.0
    for i in range(int((60 + 6 * sigma + 70) / k0 / dt)):
        new = step(psi)
        absorbed += dt * np.sum(W[right] * (np.abs(psi[right]) ** 2 + np.abs(new[right]) ** 2)) * dx
        psi = new
    return np.sum(np.abs(psi[right]) ** 2) * dx + absorbed


def part_2d():
    V0, b, sigma = 1.0, 1.0, 5.0
    Es = np.linspace(0.2, 2.5, 24)
    Tnum = np.array([transmitted(E) for E in Es])
    Ef = np.linspace(0.05, 2.6, 600)
    k = np.linspace(-6, 6, 4001)

    def averaged(E):  # analytic T averaged over the packet's momentum distribution (std 1/(2 sigma))
        w = np.exp(-(k - np.sqrt(2 * E)) ** 2 * 2 * sigma**2)
        return np.sum(w * transmission(np.maximum(k**2 / 2, 1e-12), V0, b) * (k > 0)) / np.sum(w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), width_ratios=[1.15, 1])
    ax1.plot(Ef, transmission(Ef, V0, b), color=INK2, lw=1.6, label="$T(E)$ analytisch (ebene Welle)")
    ax1.plot(Ef, [averaged(E) for E in Ef], color=C[1], lw=1.4, ls="--", label=r"$T$ gemittelt über $|\phi(k)|^2$")
    ax1.plot(Es, Tnum, "o", color=C[0], ms=7, mfc="white", mew=2, label=r"Paket $\sigma=5$ (CN)")
    ax1.axvline(V0, color=MUTED, ls=":", lw=1)
    ax1.text(V0 + 0.03, 0.05, "$E=V_0$", color=INK2, fontsize=9)
    ax1.set(xlabel=r"$E=k_0^2/2$", ylabel="Transmission $T$", ylim=(0, 1.02),
            title=r"Tunneln durch $V_0=1$, $b=1$")
    ax1.legend(loc="lower right")

    # boundary: hard wall vs complex absorbing potential
    x, dx = grid(-30, 30, 1199)
    psi0 = gaussian(x, -5, 2, 2)
    imgs = []
    for W in (np.zeros_like(x), absorber(x, -30, 30, 10, 2.0)):
        fr = run(hamiltonian(-1j * W, dx), psi0, 0.02, 1500, every=10)
        imgs.append(np.abs(fr.T) ** 2)
    remaining = np.sum(imgs[1][:, -1]) * dx
    ax2.imshow(np.hstack(imgs), extent=(0, 60, -30, 30), origin="lower", aspect="auto", norm=PowerNorm(0.5))
    ax2.axvline(30, color="white", lw=2)
    for s in (-1, 1):
        ax2.axhspan(30 - 10 if s > 0 else -30, 30 if s > 0 else -20, xmin=0.5, xmax=1, color=C[0], alpha=0.25)
    ax2.set_xticks([0, 10, 20, 30, 40, 50, 60], ["0", "10", "20", "30 | 0", "10", "20", "30"])
    ax2.text(1, 26, "harte Wand: Reflexion", color="white", fontsize=9.5, weight="bold")
    ax2.text(31, 26, r"$V-iW(x)$: absorbiert", color="white", fontsize=9.5, weight="bold")
    ax2.set(xlabel="$t$", ylabel="$x$", title=r"Rand des Gebiets: $|\psi(x,t)|^2$, $k_0=2$")
    image_axes(ax2)
    save(fig, "tunnel_2d.png")
    store("2d", {"E": Es.tolist(), "T_num": Tnum.tolist(), "T_plane": transmission(Es, V0, b).tolist(),
                 "T_avg": [averaged(E) for E in Es], "left_after_cap": remaining})


# ---------------------------------------------------------------- 3 two dimensions

def grid2(xa, xb, ya, yb, h):
    x, dx = grid(xa, xb, round((xb - xa) / h) - 1)
    y, _ = grid(ya, yb, round((yb - ya) / h) - 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, X, Y, dx


def cap2(x, y, width, strength):
    return absorber(x, x[0] - (x[1] - x[0]), x[-1] + (x[1] - x[0]), width, strength)[:, None] + \
        absorber(y, y[0] - (y[1] - y[0]), y[-1] + (y[1] - y[0]), width, strength)[None, :]


def gif(frames, extent, overlay, name, fps=15, title=""):
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * (extent[3] - extent[2]) / (extent[1] - extent[0]) + 0.5))
    vmax = np.percentile(frames, 99.9)
    im = ax.imshow(frames[0].T, extent=extent, origin="lower", norm=PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.imshow(np.ma.masked_where(overlay.T == 0, overlay.T), extent=extent, origin="lower", cmap="Greys_r",
              vmin=0, vmax=1, alpha=0.9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    image_axes(ax)
    anim = FuncAnimation(fig, lambda i: [im.set_data(frames[i].T)], frames=len(frames))
    anim.save(HERE / name, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print("saved", name)


def slit_run(slits, frames_every=0):
    x, y, X, Y, dx = grid2(-12, 22, -22, 22, 0.1)
    wall = (X >= 0) & (X < 0.5)
    for s in slits:
        wall &= ~(np.abs(Y - s) < 0.5 - 1e-9)
    V = 200 * wall - 1j * cap2(x, y, 4, 20)
    psi = (gaussian(X, -6, 1.5, 5) * gaussian(Y, 0, 4, 0)).ravel()
    step = stepper(hamiltonian_2d(x, y, V), 0.01, "cn")
    iscreen = np.argmin(np.abs(x - 15.25))
    screen, frames = np.zeros(len(y)), []
    for i in range(700):
        psi = step(psi)
        screen += 0.01 * np.abs(psi.reshape(X.shape)[iscreen]) ** 2
        if frames_every and i % frames_every == 0:
            frames.append((np.abs(psi.reshape(X.shape)) ** 2).astype(np.float32))
    return x, y, wall, screen, frames


def part_3a():
    d, k0, L = 4, 5, 15
    lam = 2 * np.pi / k0
    x, y, wall, both, frames = slit_run((-2, 2), frames_every=10)
    _, _, _, top, _ = slit_run((2,))
    _, _, _, bottom, _ = slit_run((-2,))
    ext = (x[0], x[-1], y[0], y[-1])

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    for j, i in enumerate([15, 35, 55]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(frames[i].T, extent=ext, origin="lower", norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(frames[i], 99.8)))
        ax.imshow(np.ma.masked_where(~wall.T, wall.T * 1.0), extent=ext, origin="lower", cmap="Greys_r", vmin=0, vmax=1)
        ax.axvline(15.25, color=C[2], lw=1.2, ls="--")
        ax.set(xlim=(-10, 20), ylim=(-18, 18), title=f"$t={i * 10 * 0.01:.1f}$", xlabel="$x$")
        if j == 0:
            ax.set_ylabel("$y$")
        image_axes(ax)
    ax = fig.add_subplot(gs[1, :])
    norm = both.max()
    ax.fill_between(y, both / norm, color=C[0], alpha=0.25, lw=0)
    ax.plot(y, both / norm, color=C[0], label="beide Spalte offen")
    ax.plot(y, (top + bottom) / norm, color=C[1], label="Summe der Einzelspalte")
    ax.plot(y, top / norm, color=C[1], lw=1, ls=":", label="Einzelspalte oben / unten")
    ax.plot(y, bottom / norm, color=C[1], lw=1, ls=":")
    peaks, _ = find_peaks(both / norm, prominence=0.05, height=0.03)
    ym = L * np.tan(np.arcsin(np.clip(np.arange(-3, 4) * lam / d, -1, 1)))
    for m, yy in zip(range(-3, 4), ym):
        if abs(yy) < 20:
            ax.axvline(yy, color=MUTED, ls="--", lw=1)
            ax.text(yy, 1.04, f"$m={m}$", ha="center", fontsize=9, color=INK2)
    ax.plot(y[peaks], both[peaks] / norm, "v", color=INK, ms=7)
    ax.set(xlim=(-18, 18), ylim=(0, 1.12), xlabel="$y$ auf dem Schirm ($x=15.25$)",
           ylabel=r"$\int|\psi|^2\,dt$ (normiert)", title=r"Schirmbild; gestrichelt: $\sin\theta_m=m\lambda/d$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    save(fig, "double_slit_3a.png")

    theta = np.arctan(y[peaks] / L)
    gif(frames, ext, wall * 1.0, "double_slit_3a.gif", title="Doppelspalt, $k_0=5$")
    store("3a", {"peaks_y": y[peaks].tolist(), "sin_theta": np.sin(theta).tolist(),
                 "m_lambda_over_d": (np.arange(-3, 4) * lam / d).tolist(), "lambda": lam,
                 "visibility_sum": float(((top + bottom).max() - (top + bottom)[np.abs(y) < 3].min()) / (top + bottom).max())})


# Mach-Zehnder: BS1 at (0,0), mirrors at (Lm, +-Lm), BS2 at (2Lm, 0); all barriers horizontal
Lm, half, H_GRID, K0 = 6.0, 4.5, 0.08, 8.0


def mz_grid():
    return grid2(-11, 27, -12, 12, H_GRID)  # 475 x 300 points, dx = dy exactly


def beam_splitter_height():
    """Barrier height (thickness 3 dx = 0.24) with T = 1/2 for the y-motion of the 45 degree packet."""
    y, dy = grid(-16, 16, round(32 / H_GRID) - 1)
    ky = K0 / np.sqrt(2)

    def T(V):
        fr = run(hamiltonian(np.where(np.abs(y) < 0.12, V, 0.0), dy), gaussian(y, -5, 1, ky), 0.004, 450, every=450)
        return np.sum(np.abs(fr[-1][y > 0]) ** 2) * dy

    return brentq(lambda V: T(V) - 0.5, 10, 40, xtol=1e-3)


def mz_potential(X, Y, Vbs):
    bar = lambda xc: (np.abs(Y) < 0.12) & (np.abs(X - xc) < half)  # noqa: E731
    mirror = lambda s: (s * Y >= Lm) & (s * Y < Lm + 0.5) & (np.abs(X - Lm) < half)  # noqa: E731
    return Vbs * (bar(0) | bar(2 * Lm)) + 1000.0 * (mirror(1) | mirror(-1))


def mz_packet(X, Y):
    return (gaussian(X, -5, 1, K0 / np.sqrt(2)) * gaussian(Y, -5, 1, K0 / np.sqrt(2))).ravel()


def mz_run(V, x, y, X, Y, A=None, frames_every=0):
    """Port probabilities (up, down): what is behind BS2 at the end plus what the absorber took there."""
    W = cap2(x, y, 2.5, 30)
    step, psi = stepper(hamiltonian_2d(x, y, V - 1j * W, A), 0.004, "cn"), mz_packet(X, Y)
    out = X > 2 * Lm
    up, down = (out & (Y > 0.12)).ravel(), (out & (Y < -0.12)).ravel()
    absorbed, frames = np.zeros(2), []
    for i in range(1000):
        new = step(psi)
        dens = W.ravel() * (np.abs(psi) ** 2 + np.abs(new) ** 2) * 0.004 * H_GRID**2  # trapezoid of 2 W |psi|^2
        absorbed += dens[up].sum(), dens[down].sum()
        psi = new
        if frames_every and i % frames_every == 0:
            frames.append((np.abs(psi.reshape(X.shape)) ** 2).astype(np.float32))
    p = np.abs(psi) ** 2 * H_GRID**2
    return p[up].sum() + absorbed[0], p[down].sum() + absorbed[1], frames


def disk(X, Y, c, r=1.0):
    return (X - c[0]) ** 2 + (Y - c[1]) ** 2 < r**2


def mz_figure(frames, ext, geom, name, title, marks=()):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, i in zip(axs, [3, 9, 15, 24]):
        ax.imshow(frames[i].T, extent=ext, origin="lower", norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(frames[i], 99.9)))
        ax.imshow(np.ma.masked_where(geom.T == 0, geom.T), extent=ext, origin="lower", cmap="Greys_r", vmin=0, vmax=1, alpha=0.85)
        for c, col in marks:
            ax.add_patch(plt.Circle(c, 1, fill=False, color=col, lw=1.5))
        ax.set(xlim=(-8, 24), ylim=(-11, 11), title=f"$t={i * 40 * 0.004:.2f}$")
        ax.set_xticks([])
        ax.set_yticks([])
        image_axes(ax)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12)
    save(fig, name)


def part_3b():
    x, y, X, Y, dx = mz_grid()
    Vbs = beam_splitter_height()
    V = mz_potential(X, Y, Vbs)
    up, down, frames = mz_run(V, frames_every=40, x=x, y=y, X=X, Y=Y)
    ext = (x[0], x[-1], y[0], y[-1])
    geom = np.where(V > 500, 1.0, np.where(V > 0, 0.55, 0.0))
    mz_figure(frames, ext, geom, "mach_zehnder_3b.png",
              f"Mach–Zehnder ohne Phasenscheibe: Ausgang oben {up:.3f}, unten {down:.3f}  ($V_{{BS}}={Vbs:.2f}$)")
    gif(frames, ext, geom, "mach_zehnder_3b.gif", fps=10, title="Mach–Zehnder, $k_0=8$")

    # one splitter alone: check T = R = 1/2 in 2D
    bs_only = Vbs * ((np.abs(Y) < 0.12) & (np.abs(X) < half))
    H = hamiltonian_2d(x, y, bs_only - 1j * cap2(x, y, 2.5, 30))
    step, psi = stepper(H, 0.004, "cn"), mz_packet(X, Y)
    for _ in range(320):
        psi = step(psi)
    p = np.abs(psi.reshape(X.shape)) ** 2 * dx**2
    T2, R2 = p[Y > 0.12].sum(), p[Y < -0.12].sum()

    Vs, Vm = np.linspace(-30, 30, 25), np.linspace(-30, 30, 13)  # the arm scan needs the finer steps
    arm = [mz_run(V + Vp * disk(X, Y, (Lm / 2, Lm / 2)), x=x, y=y, X=X, Y=Y)[:2] for Vp in Vs]
    mid = [mz_run(V + Vp * disk(X, Y, (Lm, 0)), x=x, y=y, X=X, Y=Y)[:2] for Vp in Vm]
    arm, mid = np.array(arm), np.array(mid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), width_ratios=[1.35, 1])
    ax1.plot(Vs, arm[:, 0], "o-", color=C[0], label="Ausgang oben, Scheibe im Arm")
    ax1.plot(Vs, arm[:, 1], "o-", color=C[1], label="Ausgang unten, Scheibe im Arm")
    ax1.plot(Vm, mid[:, 0], "s--", color=C[0], mfc="white", lw=1.2, label="oben, Scheibe zwischen den Armen")
    ax1.plot(Vm, mid[:, 1], "s--", color=C[1], mfc="white", lw=1.2, label="unten, Scheibe zwischen den Armen")
    Vf = np.linspace(-30, 30, 300)
    dphi = (np.sqrt(K0**2 - 2 * np.minimum(Vf, K0**2 / 2)) - K0) * 2  # phase of the central ray
    ax1.plot(Vf, (arm[:, 0] + arm[:, 1]).mean() * np.cos(dphi / 2) ** 2, ":", color=MUTED, lw=1.4,
             label=r"Zentralstrahl: $\cos^2(\Delta\varphi/2)$")
    ax1.set(xlabel=r"$V_{\rm phase}$", ylabel="Wahrscheinlichkeit", ylim=(-0.02, 1.02),
            title="Ausgänge über dem Potential der Scheibe")
    ax1.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax2.axis("off")
    mz_figure_geom(ax2, x, y, geom)
    save(fig, "mach_zehnder_ports_3b.png")
    store("3b", {"V_bs": Vbs, "T_bs_2d": T2, "R_bs_2d": R2, "up0": up, "down0": down, "V": Vs.tolist(), "V_mid": Vm.tolist(),
                 "arm": arm.tolist(), "mid": mid.tolist()})


def mz_figure_geom(ax, x, y, geom):
    ax.imshow(np.ma.masked_where(geom.T == 0, geom.T), extent=(x[0], x[-1], y[0], y[-1]), origin="lower",
              cmap="Greys", vmin=0, vmax=1.1)
    ax.add_patch(plt.Circle((Lm / 2, Lm / 2), 1, color=C[2], alpha=0.6))
    ax.add_patch(plt.Circle((Lm, 0), 1, color=C[4], alpha=0.6))
    ax.annotate("", xy=(0, 0), xytext=(-5, -5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    for s in (1, -1):
        ax.annotate("", xy=(2 * Lm + 5, 5 * s), xytext=(2 * Lm, 0), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.text(2 * Lm + 5, 5.8, "oben", ha="center", fontsize=9)
    ax.text(2 * Lm + 5, -6.8, "unten", ha="center", fontsize=9)
    ax.text(0, -1.4, "BS1", ha="center", fontsize=9)
    ax.text(2 * Lm, -1.4, "BS2", ha="center", fontsize=9)
    ax.text(Lm, Lm + 1, "Spiegel", ha="center", fontsize=9)
    ax.text(Lm / 2 - 1.3, Lm / 2 + 0.3, "im Arm", ha="right", fontsize=9, color=C[2])
    ax.text(Lm, -1.8, "zwischen\nden Armen", ha="center", va="top", fontsize=9, color=C[4])
    ax.plot([-5, 0, Lm, 2 * Lm], [-5, 0, Lm, 0], color=MUTED, lw=0.8, ls=":")
    ax.plot([0, Lm, 2 * Lm], [0, -Lm, 0], color=MUTED, lw=0.8, ls=":")
    ax.set(xlim=(-7, 19), ylim=(-8.5, 8.5), aspect="equal", title="Aufbau")


def part_3c():
    x, y, X, Y, dx = mz_grid()
    Vbs = json.loads(RESULTS.read_text()).get("3b", {}).get("V_bs") or beam_splitter_height()
    V = mz_potential(X, Y, Vbs)
    Bs = np.linspace(0, 4, 17)
    ports = []
    for B0 in Bs:
        frames_every = 40 if B0 == 1 else 0
        up, down, frames = mz_run(V, A=solenoid(B0, 1.0, (Lm, 0)), frames_every=frames_every, x=x, y=y, X=X, Y=Y)
        ports.append((up, down))
        if frames:
            geom = np.where(V > 500, 1.0, np.where(V > 0, 0.55, 0.0))
            mz_figure(frames, (x[0], x[-1], y[0], y[-1]), geom, "aharonov_bohm_paths_3c.png",
                      f"Aharonov–Bohm, $B_0=1$ (Fluss $\\Phi=\\pi$): Ausgang oben {up:.3f}, unten {down:.3f}",
                      marks=[((Lm, 0), "#3fd0ff")])
        print("B0", B0, up, down)
    ports = np.array(ports)
    phi = Bs * np.pi
    fig, ax = plt.subplots(figsize=(9, 4.4))
    tot = ports.sum(axis=1).mean()
    pf = np.linspace(0, 4 * np.pi, 400)
    ax.plot(pf / np.pi, tot * np.cos(pf / 2) ** 2, color=C[0], lw=1, ls=":")
    ax.plot(pf / np.pi, tot * np.sin(pf / 2) ** 2, color=C[1], lw=1, ls=":")
    ax.plot(phi / np.pi, ports[:, 0], "o-", color=C[0], label="Ausgang oben")
    ax.plot(phi / np.pi, ports[:, 1], "o-", color=C[1], label="Ausgang unten")
    ax.plot([], [], ":", color=MUTED, label=r"$\cos^2(q\Phi/2)$, $\sin^2(q\Phi/2)$")
    ax.set(xlabel=r"$B_0$  (Fluss $q\Phi = \pi B_0 r_0^2$, $r_0 = 1$)", ylabel="Wahrscheinlichkeit", ylim=(-0.02, 1.02),
           title=r"Aharonov–Bohm: Ausgänge über dem eingeschlossenen Fluss ($B=0$ auf den Wegen)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    save(fig, "aharonov_bohm_3c.png")
    store("3c", {"B0": Bs.tolist(), "ports": ports.tolist()})


PARTS = {"1a": part_1a, "1b": part_1b, "1c": part_1c, "2a": part_2a, "2b": part_2b, "2c": part_2c,
         "2d": part_2d, "3a": part_3a, "3b": part_3b, "3c": part_3c}

if __name__ == "__main__":
    for p in sys.argv[1:] or PARTS:
        PARTS[p]()
