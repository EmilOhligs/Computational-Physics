---
title: Crank-Nicolson-Verfahren
aliases: Crank-Nicolson, Cayley-Transformation, Trapezregel, CN
area: dgl
summary: Implizites Verfahren 2. Ordnung $(1 + \frac{i\Delta t}{2}H)\psi^{n+1} = (1 - \frac{i\Delta t}{2}H)\psi^n$; für hermitesches $H$ exakt unitär und darum das Standardverfahren für die zeitabhängige Schrödinger-Gleichung.
links: euler-verfahren, stabilitaetsgebiet, schroedinger-gleichung-finite-differenzen, absorbierende-randbedingungen
sources: 2026-11-18/theory.md, 2026-11-18/schroedinger.py, 2026-11-18/figures.py
---
## Idee und Definition

Für $\dot\psi = -iH\psi$ ist Crank–Nicolson die Trapezregel. Pro Eigenmode mit Eigenwert $\lambda$ von $H$:

$$g_{\rm CN} = \frac{1 - i\lambda\Delta t/2}{1 + i\lambda\Delta t/2},\qquad |g_{\rm CN}| = 1$$

Zähler und Nenner sind konjugiert komplex. Die Schrittmatrix $(1 + \frac{i\Delta t}{2}H)^{-1}(1 - \frac{i\Delta t}{2}H)$ ist die Cayley-Transformation von $H$ und für jedes $\Delta t$ unitär, genau wie die exakte Entwicklung $e^{-iH\Delta t}$.

Zum Vergleich: explizit Euler $|g| = \sqrt{1 + \lambda^2\Delta t^2} > 1$ (für Schrödinger **immer** instabil), implizit Euler $|g| = 1/\sqrt{1 + \lambda^2\Delta t^2} < 1$ (dämpft).

## Algorithmus und Aufwand

- Die Matrix $1 + \frac{i\Delta t}{2}H$ ändert sich nicht: einmal mit `scipy.sparse.linalg.splu` faktorisieren, danach pro Schritt nur Vorwärts-/Rückwärtseinsetzen.
- 1D (tridiagonal): Mikrosekunden pro Schritt. 2D mit $1{,}3\cdot10^5$ Unbekannten: Faktorisierung 0,8 s, Schritt etwa 17 ms.
- Freies Paket $\sigma = 1$, $k_0 = 2$, $\Delta x = 0{,}05$, $\Delta t = 0{,}01$, 1000 Schritte: Normfehler $< 4\cdot10^{-14}$; implizit Euler hat dann Norm 0,798, explizit Euler ist schon bei $t = 0{,}22$ auf $1{,}9\cdot10^4$.

## Numerische Fallstricke

- Unitär heißt nicht exakt: Die Phase dreht um $2\arctan(\lambda\Delta t/2)$ statt $\lambda\Delta t$, Fehler $(\lambda\Delta t)^3/12$ pro Schritt. Für Revivals im unendlichen Topf brauchte es $\Delta t = 3\cdot10^{-5}$.
- Den größeren Fehler macht meist das Ortsgitter: Pakete laufen mit $v = \sin(k_0\Delta x)/\Delta x$ zu langsam, relativer Wegfehler $(k_0\Delta x)^2/6$. Faustregel: mindestens 20 Punkte pro Wellenlänge.
- Explizit Euler explodiert aus dem Rundungsrauschen der höchsten Gittermode ($\lambda_{\max} \approx 2/\Delta x^2$), auch wenn das Anfangspaket diese Mode gar nicht enthält.
- Mit komplexem Absorber $V - iW$ ist $H$ nicht hermitesch; dann fällt die Norm gewollt.

## In Python

```python
A = (sp.identity(n) + 0.5j * dt * H).tocsc()
B = sp.identity(n) - 0.5j * dt * H
lu = scipy.sparse.linalg.splu(A, permc_spec="MMD_AT_PLUS_A")
for _ in range(nsteps):
    psi = lu.solve(B @ psi)
```
