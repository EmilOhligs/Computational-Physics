---
title: Steife Differentialgleichungen
aliases: Steifheit, Robertson-Modell, Newton für impliziten Euler
area: dgl
summary: Steif ist eine DGL, wenn ihre Zeitskalen weit auseinanderliegen und die schnellste bei expliziten Verfahren winzige Schritte erzwingt, obwohl sie längst abgeklungen ist; implizite Verfahren lösen das.
links: euler-verfahren, stabilitaetsgebiet
sources: 2026-10-28/THEORIE.md, 2026-10-28/nonlinear.py, 2026-10-28/plot_1f.py, 2026-10-28/plot_2a.py
---
## Idee und Definition

Hat die Jacobimatrix Eigenwerte sehr verschiedener Größe, etwa $-1$ und $-1000$, dann bestimmt die langsame Mode, wie die Lösung aussieht, die schnelle aber, wie groß $\Delta t$ bei einem expliziten Verfahren sein darf. Der Schritt wird von der Stabilität begrenzt, nicht von der Genauigkeit. Das nennt man steif.

## Algorithmus und Aufwand

- Reaktion $A \to B \to C$ mit $k_2 = 1000$, Ziel Fehler $10^{-2}$: expliziter Euler 4999 Schritte bei $\Delta t = 0{,}002 = 2/k_2$, impliziter Euler 180 Schritte bei $\Delta t = 0{,}056$. Faktor etwa 28.

Robertson-Modell (stiff chemistry):

$$
\dot A = -k_1 A + k_3 BC, \quad \dot B = k_1 A - k_2 B^2 - k_3 BC, \quad \dot C = k_2 B^2
$$

mit $k_1 = 0{,}04$, $k_2 = 3\cdot10^7$, $k_3 = 10^4$.

- Impliziter Euler bei nichtlinearem $f$: pro Schritt $g(y) = y - x_n - \Delta t\, f(y) = 0$ mit Newton lösen, $(I - \Delta t\, J_f(y))\,\delta = -g(y)$.
- Logarithmisches Zeitgitter von $10^{-6}$ bis $10^5$, weil die Dynamik auf allen Zeitskalen passiert. Ergebnis bei $t = 10^5$ mit 1600 Punkten: $A = 0{,}01808$, $B = 7{,}36\cdot10^{-8}$, $C = 0{,}98192$; mit 400 Punkten $A = 0{,}01871$.

## Numerische Fallstricke

- Beim Robertson-Modell ist der Eigenwert in B-Richtung etwa $-2k_2B - k_3C \approx -2000$. Der explizite Euler mit $\Delta t = 10^{-3}$ explodiert deshalb bei $t = 9{,}35$; bis $t = 10^5$ bräuchte er über $10^8$ Schritte.
- Ein instabiles Ergebnis ist nicht ungenau, sondern unbrauchbar. Ohne exakte Lösung erkennt man das am Blow-up oder an negativen Konzentrationen.
- Kontrolle ohne exakte Lösung: Erhaltung $A + B + C = 1$ (gemessen bis auf $2\cdot10^{-15}$) und feineres Gitter zum Vergleich.
- Newton braucht eine Abbruchbedingung und eine Fehlermeldung, wenn er nicht konvergiert.
- Siehe [[Stabilitätsgebiet]] für den Grund und [[Euler-Verfahren]] für die Formeln.

## In Python

```python
def implicit_euler_f(f, jac, x, dt, tol=1e-12, max_iter=50):
    y = x.copy()
    for _ in range(max_iter):
        g = y - x - dt * f(y)
        delta = np.linalg.solve(np.eye(len(x)) - dt * jac(y), -g)
        y = y + delta
        if np.max(np.abs(delta)) <= tol * (1 + np.max(np.abs(y))):
            return y
    raise RuntimeError("Newton did not converge")
```
