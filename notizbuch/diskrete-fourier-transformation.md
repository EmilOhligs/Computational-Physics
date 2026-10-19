---
title: Diskrete Fourier-Transformation
aliases: DFT
area: fourier
summary: Zerlegt $N$ abgetastete Werte $a_j$ in $N$ komplexe Frequenzanteile $\hat a_k$ – das diskrete Gegenstück zur Fourier-Reihe.
links: fast-fourier-transform, aliasing-und-nyquist, rundungsfehler
sources: 2026-10-14/THEORIE.md, 2026-10-14/exercise_1_fft.ipynb
---
## Idee und Definition

Das Signal wird im Fenster $[0, T)$ an $N$ Punkten abgetastet: $t_j = j\,\Delta t$, $\Delta t = T/N$. Der Endpunkt $t = T$ gehört nicht dazu, weil die DFT das Signal als periodisch behandelt.

$$
\hat a_k = \sum_{j=0}^{N-1} a_j\, e^{-2\pi i\, jk/N}
\qquad
a_j = \frac{1}{N}\sum_{k=0}^{N-1} \hat a_k\, e^{+2\pi i\, jk/N}
$$

Konvention wie `numpy.fft`: der Faktor $1/N$ steht nur bei der Rücktransformation. Der Index $k$ gehört zur Frequenz $f_k = k/T$; für $k > N/2$ ist es die negative Frequenz $(k-N)/T$.

## Wichtige Eigenschaften

- **Reelles Signal:** $\hat a_{N-k} = \hat a_k^{*}$ – die negativen Frequenzen bringen keine neue Information (`np.fft.rfft`).
- **Verschiebung** um $m$ Punkte: $\hat b_k = \hat a_k\, e^{-2\pi i\, km/N}$ – nur die Phase ändert sich.
- **Orthogonalität:** $\sum_j e^{2\pi i\, j(k-k')/N} = N\,\delta_{kk'}$, daraus folgt die Rücktransformation.
- **Parseval:** $\sum_j |a_j|^2 = \frac{1}{N}\sum_k |\hat a_k|^2$.
- Reell und gerade ⇒ Spektrum reell; reell und ungerade ⇒ rein imaginär.

## Skalierung je nach Frage

- Fourier-Reihen-Koeffizient: $c_k \approx \hat a_k / N$
- kontinuierliche Transformation: $\tilde a(\omega_k) \approx \Delta t\, \hat a_k$
- Kosinus-Amplitude (reelles Signal, $0<k<N/2$): $2|\hat a_k|/N$

## Algorithmus und Aufwand

Naiv: $N$ Ausgabewerte, jeder eine Summe über $N$ Terme, also $\mathcal{O}(N^2)$. Die [[Fast Fourier Transform]] liefert dasselbe in $\mathcal{O}(N\log N)$.

## In Python

`np.fft.fft`, `np.fft.ifft`, `np.fft.fftfreq(N, dt)` für die Frequenzen, `np.fft.fftshift` zum Sortieren von negativ nach positiv.
