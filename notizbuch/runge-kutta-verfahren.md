---
title: Runge-Kutta-Verfahren
aliases: RK4, Heun, Runge-Kutta
area: dgl
summary: Explizite Einschrittverfahren, die mehrere Steigungen pro Schritt mitteln; Heun hat Ordnung 2, das klassische RK4 Ordnung 4.
links: euler-verfahren, stabilitaetsgebiet
sources: 2026-10-28/THEORIE.md, 2026-10-28/ode.py, 2026-10-28/plot_1c.py
---
## Idee und Definition

Statt nur die Steigung am Anfang zu nehmen wie der explizite Euler ([[Euler-Verfahren]]), wertet ein Runge-Kutta-Verfahren $f$ an mehreren Zwischenpunkten aus und mittelt gewichtet. Die Gewichte sind so gewählt, dass die Taylorreihe der exakten Lösung bis zu einer höheren Ordnung getroffen wird.

Heun (Ordnung 2):

$$
k_1 = f(x_n), \quad k_2 = f(x_n + \Delta t\, k_1), \quad x_{n+1} = x_n + \tfrac{\Delta t}{2}(k_1 + k_2)
$$

RK4 (Ordnung 4):

$$
x_{n+1} = x_n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

mit $k_1 = f(x_n)$, $k_2 = f(x_n + \tfrac{\Delta t}{2} k_1)$, $k_3 = f(x_n + \tfrac{\Delta t}{2} k_2)$, $k_4 = f(x_n + \Delta t\, k_3)$.

## Algorithmus und Aufwand

- Heun: 2 Auswertungen von $f$ pro Schritt, globaler Fehler $\propto \Delta t^2$.
- RK4: 4 Auswertungen pro Schritt, globaler Fehler $\propto \Delta t^4$.
- Gemessen bei $A \to B \to C$ ($k_2 = 10$): Steigungen $2{,}01$ (Heun) und $4{,}09$ (RK4) im log-log-Plot.
- Für einen Fehler von $10^{-6}$ reicht RK4 $\Delta t \approx 0{,}02$, Heun braucht $0{,}0012$, Euler $5\cdot10^{-6}$. Der höhere Aufwand pro Schritt lohnt sich deutlich.

## Numerische Fallstricke

- Die Ordnung gilt erst für kleine $\Delta t$: bei $k_2 \Delta t = 1$ liegen die Fehler noch über der Geraden.
- RK4 erreicht bei $\Delta t \approx 4\cdot10^{-4}$ schon $8\cdot10^{-14}$, also fast die Rundungsgrenze; feiner wird es nicht mehr.
- Explizite Verfahren haben ein beschränktes [[Stabilitätsgebiet]]: $\Delta t_\text{krit} \cdot \max(k_1, k_2) = 2$ für Heun und $2{,}785$ für RK4. Bei steifen Problemen hilft die hohe Ordnung deshalb nicht.
- Bei chaotischen Systemen (Lorenz) bleibt auch RK4 nur bis zur Trennzeit verlässlich, dort aber deutlich länger als Euler ($t \approx 28{,}9$ gegen Abweichung schon bei $5{,}1$).

## In Python

```python
def rk4_step(J, x, dt):
    k1 = J @ x
    k2 = J @ (x + dt / 2 * k1)
    k3 = J @ (x + dt / 2 * k2)
    k4 = J @ (x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
```

Für nichtlineares $f$ dasselbe mit `f(...)` statt `J @ ...` (`rk4_f` in `nonlinear.py`).
