---
title: Stabilitätsgebiet
aliases: Stabilitätsfunktion, A-Stabilität, kritischer Zeitschritt
area: dgl
summary: Ein Einschrittverfahren multipliziert jede Eigenmode pro Schritt mit $R(\lambda\Delta t)$; es ist stabil, solange $|R(z)| \le 1$ für alle $z = \lambda\Delta t$ gilt.
links: euler-verfahren, runge-kutta-verfahren, steife-differentialgleichungen
sources: 2026-10-28/THEORIE.md, 2026-10-28/ode.py, 2026-10-28/plot_1d.py, 2026-10-28/plot_1e.py
---
## Idee und Definition

Für die Testgleichung $\dot x = \lambda x$ macht jedes Einschrittverfahren $x_{n+1} = R(z)\, x_n$ mit $z = \lambda \Delta t$. Das Stabilitätsgebiet ist die Menge $\{z : |R(z)| \le 1\}$.

- Euler: $R(z) = 1 + z$ (Kreis um $-1$ mit Radius 1)
- Heun: $R(z) = 1 + z + z^2/2$
- RK4: $R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$
- impliziter Euler: $R(z) = 1/(1 - z)$, stabil außerhalb des Kreises um $+1$

Ein lineares System $\dot x = Jx$ zerfällt in Eigenmoden; jede wird mit $R(\lambda_i \Delta t)$ multipliziert. Stabil ist das Verfahren, wenn alle $\lambda_i \Delta t$ im Gebiet liegen.

## Algorithmus und Aufwand

- $R(z)$ lässt sich generisch aus dem Schritt bestimmen: einen Schritt mit $J = 1$, $x = 1$ und $\Delta t = z$ ausführen.
- Den kritischen Zeitschritt findet eine Bisektion auf der negativen reellen Achse.
- Reaktion $A \to B \to C$: Eigenwerte $0, -k_1, -k_2$, also $\Delta t_\text{krit} = c / \max(k_1, k_2)$ mit gemessen $c = 2$ (Euler, Heun) und $c = 2{,}785$ (RK4), für alle getesteten Paare $(k_1, k_2)$ gleich.
- Für $k_1 = 1$, $k_2 = 10$: Eigenmoden $\lambda = 0$ (alles ist C), $\lambda = -1$ (langsamer Zerfall von A), $\lambda = -10$ (schneller Übergang B → C).

## Numerische Fallstricke

- Zuerst wird die schnellste Mode instabil ($\lambda = -10$), auch wenn sie längst abgeklungen ist. Bei $1{,}05\,\Delta t_\text{krit}$: $|R| = 1{,}100$ (Euler), $1{,}105$ (Heun), $1{,}231$ (RK4).
- Knapp über der Grenze ist $R < -1$: die Lösung springt von Schritt zu Schritt das Vorzeichen und wächst. Max. Fehler bei $0{,}95$ gegen $1{,}05\,\Delta t_\text{krit}$: Euler $0{,}118$ gegen $10{,}8$, RK4 $0{,}0821$ gegen $130$.
- A-stabil heißt: die ganze linke Halbebene liegt im Stabilitätsgebiet. Der implizite Euler ist A-stabil, weil $|1 - z| \ge 1$ für $\operatorname{Re} z \le 0$. Kein explizites Verfahren ist A-stabil, darum braucht man für [[Steife Differentialgleichungen]] implizite Verfahren.

## In Python

```python
def stability_function(step, z):
    return np.vectorize(lambda s: step(np.ones((1, 1)), np.ones(1, dtype=complex), s)[0])(z)
```

`critical_dt` in `ode.py` sucht damit per Bisektion die Grenze $|R(-c)| = 1$ und teilt durch $\max(k_1, k_2)$.
