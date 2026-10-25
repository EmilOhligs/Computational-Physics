---
title: Magnetische Ladungen
aliases: Ladungsformulierung, magnetisches Skalarpotential, Coulomb-Kernel
area: physmod
summary: Statt mit dem Dipol-Tensor rechnet man mit $\rho = -\nabla\cdot\vec m$, $\phi = \frac{1}{4\pi r} * \rho$ und $\vec h = -\nabla\phi$. Im Kontinuum ist das identisch, weil $\partial_i\partial_j\frac{1}{4\pi r}$ der Dipol-Kernel samt Selbstterm $-\tfrac13\delta_{ij}\delta(\vec r)$ ist.
links: entmagnetisierungsfeld, faltung-per-fft
sources: 2026-10-21/THEORIE.md, 2026-10-21/demag.py, 2026-10-21/exercise_2_demag.ipynb
---
## Idee und Definition

$$
\rho = -\nabla\cdot\vec m, \qquad \phi = K_c * \rho, \quad K_c(r) = \frac{1}{4\pi r}, \qquad \vec h = -\nabla\phi
$$

Bei homogener Magnetisierung gibt es keine Volumenladungen, nur Flächenladungen $\sigma = \vec m\cdot\hat n$.

## Warum äquivalent zum Dipol-Kernel

Partielle Integration schiebt die Ableitung von $\vec m$ auf den Kernel:

$$
\vec h = \nabla\nabla\cdot(K_c * \vec m) = (\nabla\nabla^T K_c) * \vec m,
\qquad
\partial_i\partial_j\frac{1}{4\pi r} = \frac{3\hat r_i\hat r_j - \delta_{ij}}{4\pi r^3} - \frac{\delta_{ij}}{3}\delta(\vec r)
$$

Der erste Term ist der Kernel aus [[Entmagnetisierungsfeld]], der Delta-Term der Selbstterm $-\tfrac13 I$. Verschieden ist nur, wo diskretisiert wird.

## Algorithmus und Aufwand

- Divergenz und Gradient als zentrale Differenzen (`np.gradient`), $\phi$ per [[Faltung per FFT]] mit Zero-Padding.
- Selbstterm: $K_c(0) = \frac{1}{4\pi}\int_{\text{Würfel}}\frac{dV}{r} = \frac{3\ln(2+\sqrt3) - \pi/2}{4\pi} \approx 0.1894$.
- Nur ein Skalarfeld: 2 FFTs pro Auswertung statt 6. Gemessen bei $64^3$: 0.05 s gegen 0.25 s.

## Numerische Fallstricke

- Zentrale Differenzen verschmieren Sprünge über zwei Zellen. An der Kugeloberfläche bis 0.31 Unterschied zum Dipol-Kernel; der springt schärfer und trifft außen am Pol $0.61$ (analytisch $0.608$), die Ladungen nur $0.48$.
- Fehler 2. Ordnung: Für eine glatte Gauß-Magnetisierung fällt der Unterschied zum Dipol-Kernel von 7.1 % ($\sigma = 3$) auf 1.9 % ($\sigma = 6$), also wie $1/\sigma^2$.
- Innen und weit außen sind beide gleich gut ($3\cdot10^{-3}$ bzw. $0.015$).

## In Python

```python
rho = -sum(np.gradient(m[a], axis=a) for a in range(3))
phi = np.fft.irfftn(kc_hat * np.fft.rfftn(rho, s=pad), s=pad)[:nx, :ny, :nz]
h = -np.stack(np.gradient(phi))
```
