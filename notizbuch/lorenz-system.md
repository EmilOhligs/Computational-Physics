---
title: Lorenz-System
aliases: Lorenz-Attraktor, deterministisches Chaos, Schmetterlingseffekt
area: physmod
summary: Drei gekoppelte nichtlineare DGLs, deren Lösungen chaotisch sind: zwei Bahnen mit Startabstand $10^{-8}$ trennen sich nach $t \approx 29$ vollständig.
links: runge-kutta-verfahren, steife-differentialgleichungen
sources: 2026-10-28/THEORIE.md, 2026-10-28/nonlinear.py, 2026-10-28/plot_2b.py
---
## Idee und Definition

$$
\dot x = \sigma (y - x), \qquad \dot y = x(\rho - z) - y, \qquad \dot z = xy - \beta z
$$

Mit $\sigma = 10$, $\rho = 28$, $\beta = 8/3$ laufen die Lösungen auf einen seltsamen Attraktor (zwei Flügel, $x$ zwischen etwa $-17$ und $20$, $z$ bis $48$). Das System ist deterministisch, aber kleine Unterschiede im Anfangswert wachsen exponentiell.

## Algorithmus und Aufwand

- Gerechnet mit RK4 ([[Runge-Kutta-Verfahren]]), $t \in [0, 40]$, von $(1, 1, 1)$ und $(1, 1, 1 + 10^{-8})$.
- Der Abstand der Bahnen überschreitet 1 bei $t = 28{,}88$, mit $\Delta t = 10^{-3}$ und $5\cdot10^{-4}$ gleich.
- Im semilog-Plot wächst der Abstand im Mittel linear; ein Fit ergibt die Rate $0{,}84$ (Literatur: größter Lyapunov-Exponent $0{,}91$).
- Abschätzung der Trennzeit: $10^{-8}\, e^{0{,}84\, t} = 1$ ergibt $t \approx 22$, die richtige Größenordnung.

## Numerische Fallstricke

- Rundungs- und Diskretisierungsfehler wirken wie eine Störung des Anfangswerts. Dieselbe Bahn mit $\Delta t = 10^{-3}$ und $5\cdot10^{-4}$ trennt sich ebenfalls bei $t \approx 28{,}9$; expliziter Euler gegen RK4 schon bei $t \approx 5{,}1$, Heun gegen RK4 bei $16{,}9$.
- Der einzelnen Bahn kann man also nur bis zur Trennzeit trauen. Form und Wertebereich des Attraktors sind dagegen robust.
- Prüfen ohne exakte Lösung: mit halbem $\Delta t$ wiederholen, Verfahren verschiedener Ordnung vergleichen, statistische Größen statt einzelner Werte betrachten.
- Im Gegensatz zu steifen Problemen ([[Steife Differentialgleichungen]]) begrenzt hier nicht die Stabilität den Schritt, sondern die Genauigkeit.

## In Python

```python
def lorenz(x, sigma=10.0, rho=28.0, beta=8 / 3):
    u, v, w = x
    return np.array([sigma * (v - u), u * (rho - w) - v, u * v - beta * w], dtype=float)
```

Integriert mit `integrate_grid(rk4_f, lorenz, lorenz_jac, x0, t)` aus `nonlinear.py`.
