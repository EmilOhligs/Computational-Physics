---
title: Euler-Verfahren
aliases: expliziter Euler, impliziter Euler, Rückwärts-Euler
area: dgl
summary: Einfachstes Einschrittverfahren für $\dot x = f(x)$, explizit $x_{n+1} = x_n + \Delta t\, f(x_n)$ oder implizit $x_{n+1} = x_n + \Delta t\, f(x_{n+1})$; beide haben Ordnung 1, nur der implizite ist A-stabil.
links: runge-kutta-verfahren, stabilitaetsgebiet, steife-differentialgleichungen
sources: 2026-10-28/THEORIE.md, 2026-10-28/ode.py, 2026-10-28/nonlinear.py
---
## Idee und Definition

Die Ableitung wird durch den Differenzenquotienten über einen Schritt $\Delta t$ ersetzt.

$$
x_{n+1} = x_n + \Delta t\, f(x_n) \quad \text{(explizit)}, \qquad
x_{n+1} = x_n + \Delta t\, f(x_{n+1}) \quad \text{(implizit)}
$$

Beim expliziten Euler wird die Steigung am Anfang des Schritts genommen, beim impliziten am Ende. Für ein lineares System $\dot x = Jx$ heißt der implizite Schritt $(I - \Delta t J)\,x_{n+1} = x_n$.

## Algorithmus und Aufwand

- Explizit: eine Auswertung von $f$ pro Schritt.
- Implizit, linear: ein lineares Gleichungssystem pro Schritt.
- Implizit, nichtlinear: Nullstelle von $g(y) = y - x_n - \Delta t\, f(y)$ mit Newton, $(I - \Delta t\, J_f(y))\,\delta = -g(y)$, Start bei $y = x_n$.
- Beide haben Ordnung 1: globaler Fehler $\propto \Delta t$. Gemessen bei der Reaktion $A \to B \to C$ ($k_2 = 10$): Steigung $1{,}00$ für beide im log-log-Plot.
- Der lokale Fehler ist $\mp\tfrac12 \Delta t^2 J^2 x$, die beiden Varianten liegen also fast gleich weit auf entgegengesetzten Seiten der exakten Lösung.

## Numerische Fallstricke

- Stabilitätsfunktion explizit $R(z) = 1 + z$, stabil nur für $\Delta t < 2/\max(k_1, k_2)$; siehe [[Stabilitätsgebiet]].
- Implizit $R(z) = 1/(1 - z)$, $|R| \le 1$ in der ganzen linken Halbebene (A-stabil), also kein kritischer Zeitschritt.
- Bei [[Steife Differentialgleichungen]] ist der Unterschied riesig: für $k_2 = 1000$ und Fehler $10^{-2}$ braucht der explizite Euler 4999 Schritte (genau an der Stabilitätsgrenze $\Delta t = 0{,}002$), der implizite 180.
- Für $10^{-6}$ Genauigkeit braucht Euler etwa $\Delta t = 5\cdot 10^{-6}$, [[Runge-Kutta-Verfahren]] (RK4) nur $0{,}02$. Für Genauigkeit lohnt sich Euler kaum.

## In Python

```python
def euler_step(J, x, dt):
    return x + dt * (J @ x)

def implicit_euler_step(J, x, dt):
    return np.linalg.solve(np.eye(len(x)) - dt * J, x)
```

Die nichtlineare Variante mit Newton steht als `implicit_euler_f` in `nonlinear.py`.
