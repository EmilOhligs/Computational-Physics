---
title: Von-Neumann-Stabilitätsanalyse
aliases: Von Neumann, Verstärkungsfaktor, CFL-Bedingung Diffusion, kritischer Zeitschritt PDE
area: dgl
summary: Man setzt eine einzelne Fourier-Mode $g^n e^{i\kappa j\Delta x}$ in das Differenzenschema ein und verlangt $|g| \le 1$ für alle $\kappa$; für FTCS bei Wärmeleitung folgt $\Delta t \le \Delta x^2/2$.
links: linienmethode, stabilitaetsgebiet, finite-differenzen-methode, steife-differentialgleichungen
sources: 2026-11-11/THEORIE.md, 2026-11-11/exercise_5_fd.ipynb, 2026-10-28/ode.py
---
## Idee und Definition

Ansatz $u_j^n = g^n e^{i\kappa j\Delta x}$ in den expliziten Euler (FTCS) für $u_t = u_{xx}$ einsetzen, mit $r = \Delta t/\Delta x^2$:

$$g = 1 - 4r\sin^2\frac{\kappa\Delta x}{2}$$

Stabil heißt $|g| \le 1$ für alle Wellenzahlen. Der schlimmste Fall ist $\sin^2 = 1$, also $4r \le 2$:

$$\Delta t \le \frac{\Delta x^2}{2}$$

Dasselbe sieht man über die Matrix: Die [[Linienmethode]] multipliziert Eigenmode $k$ pro Schritt mit $R(z_k)$, $z_k = -\lambda_k\Delta t$, mit $\lambda_k = \frac{4}{\Delta x^2}\sin^2\frac{k\pi\Delta x}{2}$. Alle $z_k$ liegen auf der negativen reellen Achse. Entscheidend ist der **größte Eigenwert** $\lambda_N \approx 4/\Delta x^2$, denn $z_N$ liegt am weitesten links:

$$\Delta t_\text{krit} = \frac{2}{\lambda_N}\ \text{(Euler, Heun)}, \qquad \frac{2{,}785}{\lambda_N}\ \text{(RK4)}$$

Die Eigenvektoren $\sin(k\pi x_j)$ sind die diskrete Sinusreihe, von Neumann ist also dieselbe Rechnung im Fourier-Raum.

## Algorithmus und Aufwand

Gemessen ($\Delta t_\text{krit}$ Euler aus der Matrix gegen von Neumann $\Delta x^2/2$):

- $N = 10$: $4{,}218\cdot10^{-3}$ gegen $4{,}132\cdot10^{-3}$
- $N = 20$: $1{,}140\cdot10^{-3}$ gegen $1{,}134\cdot10^{-3}$
- $N = 40$: $2{,}979\cdot10^{-4}$ gegen $2{,}974\cdot10^{-4}$
- $N = 80$: $7{,}624\cdot10^{-5}$ gegen $7{,}621\cdot10^{-5}$

Von Neumann liegt immer knapp darunter, weil auf dem Gitter $\sin^2(N\pi\Delta x/2) < 1$ ist. Mit feinerem Gitter nähern sich beide an. RK4 bei $N = 80$: $1{,}062\cdot10^{-4}$.

**Aufwand:** Halbes $\Delta x$ heißt ein Viertel $\Delta t$. Bei $N = 99$ braucht der explizite Euler bis $t = 0{,}1$ mindestens 2000 Schritte, bei $N = 199$ 8000, bei $N = 399$ 32 000. Weil jeder Schritt $O(N)$ kostet: Gesamtaufwand $\propto N^3$.

## Numerische Fallstricke

- **Knapp über der Grenze** ($1{,}02\,\Delta t_\text{krit}$, $N = 20$): Die höchste Mode wird pro Schritt mit $|R| = 1{,}04$ multipliziert und wechselt das Vorzeichen. Sie wächst von Rundungsniveau exponentiell, überholt bei $t \approx 0{,}87$ die korrekt abklingende Grundmode, danach sieht man nur noch Zickzack von Punkt zu Punkt mit Sinus-Einhüllender.
- Instabil wird zuerst die Mode, die physikalisch am schnellsten abklingt und für die Lösung keine Rolle spielt. Das ist [[Steife Differentialgleichungen]] im Reinformat.
- Impliziter Euler und Crank-Nicolson sind für jedes $\Delta t$ stabil (A-stabil): $|1/(1+|z|)| < 1$ und $|(1-|z|/2)/(1+|z|/2)| < 1$ für alle $z < 0$. Siehe [[Stabilitätsgebiet]].
- Stabil heißt nicht genau: Satz von Lax, Konsistenz + Stabilität ⇔ Konvergenz (für lineare Schemata).

## In Python

```python
lam_N = 4 / dx**2 * np.sin(N * np.pi * dx / 2) ** 2
lam_1 = 4 / dx**2 * np.sin(np.pi * dx / 2) ** 2
dt_crit = critical_dt(euler_step, lam_1, lam_N)   # from 2026-10-28/ode.py, = 2 / lam_N
```
