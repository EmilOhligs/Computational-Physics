---
title: Faltung per FFT
aliases: Faltungssatz, Zero-Padding, lineare Faltung, zyklische Faltung
area: fourier
summary: Eine Faltung wird im Fourierraum zum Produkt, $\hat h_k = \hat K_k \hat m_k$, und kostet damit $O(N\log N)$ statt $O(N^2)$. Weil die DFT zyklisch faltet, braucht man Zero-Padding auf $L \ge 2N-1$ und den Kernel in Wraparound-Anordnung.
links: fast-fourier-transform, diskrete-fourier-transformation, entmagnetisierungsfeld
sources: 2026-10-21/THEORIE.md, 2026-10-21/demag.py, 2026-10-21/exercise_2_demag.ipynb
---
## Idee und Definition

Eine diskrete Faltung $h_i = \sum_j K_{i-j}\, m_j$ kostet direkt $O(N^2)$. Der Faltungssatz macht daraus ein Produkt der Spektren:

$$
h = \mathcal F^{-1}\big(\mathcal F(K)\cdot\mathcal F(m)\big)
$$

Drei FFTs (siehe [[Fast Fourier Transform]]) und ein punktweises Produkt: $O(N\log N)$.

## Zyklisch statt linear

Die [[Diskrete Fourier-Transformation]] behandelt jedes Array als periodisch. Das Produkt der Spektren liefert deshalb

$$
h_i = \sum_j K_{(i-j) \bmod L}\, m_j .
$$

Ohne Padding ($L = N$) wird ein negativer Abstand auf $i - j + N$ gefaltet: Das linke Ende wirkt auf das rechte wie ein direkter Nachbar. Gemessen am 1D-Dipolkernel: Ein Dipol am linken Rand erzeugt am rechten Ende $0.159 = K(1)$ statt $6.4\cdot10^{-7} = K(63)$.

## Zero-Padding und Kernel-Anordnung

- Es gibt $2N-1$ Abstände $-(N-1)\dots N-1$. Ab $L \ge 2N-1$ bekommt jeder einen eigenen Index modulo $L$, dann ist die zyklische Faltung gleich der linearen.
- $m$ mit Nullen auf $L$ auffüllen, vom Ergebnis die ersten $N$ Werte nehmen.
- Kernel in **Wraparound-Anordnung**: Abstände $0, 1, \dots$ vorne, $\dots, -2, -1$ am Ende. Das ist die Reihenfolge von `np.fft.fftfreq(L, 1/L)`.
- In 2D und 3D gilt das in jeder Achse einzeln.

## Algorithmus und Aufwand

Gemessen (1D, numpy): direkte Summe lokal wie $N^{1.9}$, FFT-Faltung etwa $6\,\text{ns}\cdot N\log_2 N$. Vorsprung Faktor 18 bei $N = 64$ und 1150 bei $N = 8192$.

Optimierungen, gemessen zusammen 3- bis 4-mal schneller (bei Primzahl-$N$ 22-mal):

- **reelle FFT** (`rfft`): nur das halbe Spektrum, spart etwa 40 %,
- **Kernel-Spektrum cachen**: hängt nur von $L$ ab, 2 statt 3 FFTs pro Aufruf,
- **glatte Länge** `scipy.fft.next_fast_len(2N-1)` statt $2N$.

## Numerische Fallstricke

- Padding vergessen gibt keine Fehlermeldung, nur falsche Werte an den Rändern.
- Kernel falsch angeordnet (z. B. zentriert statt Wraparound) verschiebt das Ergebnis.
- Das Ergebnis ist bis auf Rundung gleich der direkten Summe ($\le 2\cdot10^{-16}$ in 1D), die FFT ist kein Näherungsverfahren.

## In Python

```python
L = 2 * N
K = kernel(np.fft.fftfreq(L, 1 / L))       # wraparound order
h = np.fft.irfft(np.fft.rfft(K) * np.fft.rfft(m, L), L)[:N]
```
