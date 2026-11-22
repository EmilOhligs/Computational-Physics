---
title: Schrödinger-Gleichung mit finiten Differenzen
aliases: Hamilton-Matrix, stationäre Schrödinger-Gleichung, Eigenwertproblem
area: physmod
summary: Mit dem 3-Punkte-Stern wird $H = -\frac12\partial_x^2 + V$ eine tridiagonale Matrix; ihre Eigenwerte nähern die Energien mit relativem Fehler $\approx (k\Delta x)^2/12$.
links: finite-differenzen-methode, crank-nicolson-verfahren, aharonov-bohm-effekt
sources: 2026-11-18/theory.md, 2026-11-18/schroedinger.py, 2026-11-18/figures.py
---
## Idee und Definition

Einheiten $\hbar = m = 1$. Auf $x_j = a + j\Delta x$ mit $\psi = 0$ an beiden Enden (harte Wände) wird

$$H = -\frac{1}{2\Delta x^2}\,\mathrm{tridiag}(1, -2, 1) + \mathrm{diag}(V)$$

Aus $H\psi = E\psi$ wird ein Matrix-Eigenwertproblem. Eine ebene Welle $e^{ikx}$ hat auf dem Gitter die Energie $E_{\rm FD}(k) = (1 - \cos k\Delta x)/\Delta x^2 \approx \frac{k^2}{2}(1 - (k\Delta x)^2/12)$. Fast jeder Genauigkeitsverlust kommt aus dieser Formel: Es zählt, wie viele Punkte pro Wellenlänge ein Zustand hat.

## Algorithmus und Aufwand

- $H$ ist tridiagonal; `scipy.linalg.eigh_tridiagonal(d, e, select="i")` liefert die untersten $k$ Eigenwerte ohne volle Matrix.
- Unendlicher Topf auf $[0, 1]$: exakt $E_n^{\rm FD} = \frac{2}{\Delta x^2}\sin^2\frac{n\pi\Delta x}{2}$, relativer Fehler $\approx (n\pi\Delta x)^2/12$, gemessene Ordnung 2,000.
- Oszillator $V = x^2/2$: Fehler $\Delta E_n \approx -\frac{\Delta x^2}{32}(2n^2 + 2n + 1)$ aus $\langle p^4\rangle$. Mit $\Delta x = 0{,}02$, $L = 10$ sind 47 Niveaus auf $10^{-3}$ genau.
- Endlicher Topf $V_0 = 10$, $a = 1$: $z_0 = a\sqrt{2V_0} = 4{,}47$, also $\lceil z_0/(\pi/2)\rceil = 3$ gebundene Zustände, $E = -9{,}180,\ -6{,}779,\ -3{,}054$.

## Numerische Fallstricke

- Zwei Fehlerquellen beim Oszillator: das Gitter ($\propto \Delta x^2 n^2$) und der Kasten $[-L, L]$, sobald der Umkehrpunkt $\sqrt{2n+1}$ die Wand erreicht. Mit $L = 6$ sind trotz feinem Gitter nur 14 Niveaus gut.
- Ein Potentialsprung, der nicht genau auf dem Gitter liegt, kostet Genauigkeit (endlicher Topf: Fehler bis $10^{-2}$ bei $\Delta x = 0{,}002$).
- Zustände mit $E > 0$ im endlichen Topf sind Kastenzustände des Rechengebiets, keine gebundenen Zustände.
- Barrieren auf dem Gitter genau abzählen: $b/\Delta x$ Punkte. Mit einem Punkt zu wenig war die Transmission um 0,02 zu groß.

## In Python

```python
off = np.full(n - 1, -0.5 / dx**2)
H = sp.diags([off, 1 / dx**2 + V, off], [-1, 0, 1])
E, psi = scipy.linalg.eigh_tridiagonal(1 / dx**2 + V, off, select="i", select_range=(0, k - 1))
psi /= np.sqrt(np.sum(psi**2, axis=0) * dx)
```
