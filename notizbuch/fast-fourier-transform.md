---
title: Fast Fourier Transform
aliases: FFT, Cooley-Tukey
area: fourier
summary: Berechnet die DFT in $\mathcal{O}(N\log N)$ statt $\mathcal{O}(N^2)$, indem sie rekursiv in gerade und ungerade Indizes zerlegt.
links: diskrete-fourier-transformation, rundungsfehler
sources: 2026-10-14/THEORIE.md, 2026-10-14/exercise_1_fft.ipynb
---
## Idee (Cooley-Tukey, radix-2)

Die Summe der [[Diskrete Fourier-Transformation]] wird in gerade und ungerade Indizes geteilt. Beide Teile sind selbst DFTs der Länge $N/2$:

$$
\hat a_k = E_k + w_N^{-k}\, O_k, \qquad
\hat a_{k+N/2} = E_k - w_N^{-k}\, O_k, \qquad w_N = e^{2\pi i/N}
$$

mit $E = \mathrm{DFT}(a_0, a_2, \dots)$ und $O = \mathrm{DFT}(a_1, a_3, \dots)$, $k = 0, \dots, N/2-1$. Das Paar aus Plus und Minus heißt **Butterfly**: Aus zwei halb so langen DFTs entstehen mit $N/2$ Multiplikationen alle $N$ Werte.

## Algorithmus und Aufwand

Rekursiv bis zur Länge 1 (die DFT eines Werts ist der Wert selbst):

```
fft(a):
    if len(a) == 1: return a
    E, O = fft(a[0::2]), fft(a[1::2])
    w = exp(-2πi k/N),  k = 0 … N/2−1
    return concat(E + w·O, E − w·O)
```

$\log_2 N$ Ebenen mit je $\mathcal{O}(N)$ Arbeit ergeben $\mathcal{O}(N\log N)$. Für $N = 2^{20}$ sind das etwa $2\cdot10^7$ statt $10^{12}$ Operationen. Im log-log-Plot: Steigung knapp über 1 (FFT) gegen 2 (DFT).

## Numerische Fallstricke

- Radix-2 braucht $N = 2^p$; andere Längen brauchen andere Zerlegungen (numpy kann das).
- Rundungsfehler wachsen nur etwa mit $\log N$, siehe [[Rundungsfehler]].
- Eine rekursive Python-FFT ist viel langsamer als `numpy.fft` (C-Code), schlägt aber für großes $N$ die naive DFT.
