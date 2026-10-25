---
title: Entmagnetisierungsfeld
aliases: Streufeld, Demagnetizing Field, Dipol-Kernel, Entmagnetisierungsfaktor
area: physmod
summary: Das Streufeld einer Magnetisierung ist die Faltung $\vec h = K * \vec m$ mit dem Dipol-Tensorkernel $K = (3\hat r\hat r^T - I)/(4\pi r^3)$ und Selbstterm $K(0) = -I/3$. Nur in Ellipsoiden ist es bei homogener Magnetisierung gleichmäßig.
links: faltung-per-fft, magnetische-ladungen, fast-fourier-transform
sources: 2026-10-21/THEORIE.md, 2026-10-21/demag.py, 2026-10-21/exercise_2_demag.ipynb
---
## Idee und Definition

Jede Zelle wirkt wie ein Punktdipol:

$$
\vec h(\vec r) = \frac{1}{4\pi r^3}\big(3(\vec m\cdot\hat r)\,\hat r - \vec m\big)
$$

Summiert über alle Zellen ist das eine Faltung, siehe [[Faltung per FFT]]:

$$
h_i = \sum_j K_{ij} * m_j, \qquad K_{ij}(\vec r) = \frac{1}{4\pi r^3}\Big(3\frac{r_i r_j}{r^2} - \delta_{ij}\Big)
$$

$K$ ist symmetrisch (6 unabhängige Komponenten) und für $r \ne 0$ spurfrei. In 1D auf der Achse bleibt $K_{xx}(x) = 1/(2\pi|x|^3)$.

## Selbstterm und Entmagnetisierungsfaktor

Bei $r = 0$ divergiert der Punktdipol. Eine Würfelzelle hat im Mittel $\vec h = -N\vec m$ mit $N = 1/3$ (Symmetrie und $N_x + N_y + N_z = 1$). Deshalb $K(0) = -\tfrac13 I$. Das ist genau der Delta-Anteil von $\partial_i\partial_j\frac{1}{4\pi r}$, siehe [[Magnetische Ladungen]].

## Algorithmus und Aufwand

- 3D auf $n^3$ Zellen: Kernel auf $(2n)^3$ padden, 6 Kernel-FFTs einmal vorab.
- Pro Auswertung 6 FFTs: 3 vorwärts ($m_x, m_y, m_z$), $\hat h_i = \sum_j \hat K_{ij}\hat m_j$, 3 zurück.
- Gemessen: FFT gegen direkte Summe bei $14^3$ Zellen 2.7 ms gegen 0.5 s, Abweichung $\le 13\,\varepsilon$.

## Physikalische Ergebnisse

- **Kette (1D):** in der Mitte $-\tfrac13 + \zeta(3)/\pi = +0.049$, am Ende $-\tfrac13 + \zeta(3)/2\pi = -0.142$ (entmagnetisierend, dort sitzen die Ladungen).
- **Kugel:** innen exakt $-\vec m/3$. Begründung: zwei verschobene homogen geladene Kugeln, deren lineare Innenfelder sich bis auf eine Konstante aufheben. Gilt für alle Ellipsoide.
- **Würfel:** innen nicht gleichmäßig. Gemessen bei $a = 32$: $-0.58$ an den Polflächen, $-1/3$ im Zentrum, $-0.21$ an den Seitenflächen.

## Numerische Fallstricke

- Gerasterte Kugeln haben Treppenstufen. In den Randzellen weicht $h$ bis 0.26 ab, und das wird mit feinerem Gitter **nicht** besser. Im Kern $\approx 10^{-3}$.
- Die Punktdipol-Näherung ist zwischen direkten Nachbarn am ungenauesten (exakter wäre der Würfel-Würfel-Kernel nach Newell).
- Validierung trennt zwei Fehler: Arithmetik (gegen direkte Summe, Rundung) und Physik (Kugel, Fernfeld gegen Punktdipol).

## In Python

```python
k_hat = np.fft.rfftn(kernel_3d(shape), s=pad, axes=(2, 3, 4))
m_hat = np.fft.rfftn(m, s=pad, axes=(1, 2, 3))
h_hat = np.einsum("ij...,j...->i...", k_hat, m_hat)
h = np.fft.irfftn(h_hat, s=pad, axes=(1, 2, 3))[:, :nx, :ny, :nz]
```
