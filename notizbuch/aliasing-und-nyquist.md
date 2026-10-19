---
title: Aliasing und Nyquist-Frequenz
aliases: Aliasing, Nyquist-Frequenz, Abtasttheorem
area: fourier
summary: Beim Abtasten mit Schritt $\Delta t$ sind Frequenzen nur bis $f_\text{Nyquist} = 1/(2\Delta t)$ darstellbar; höhere Anteile falten sich auf niedrigere zurück.
links: diskrete-fourier-transformation
sources: 2026-10-14/THEORIE.md
---
## Idee

An den Abtastpunkten gilt $e^{2\pi i (k+N) j/N} = e^{2\pi i kj/N}$. Frequenzen, die sich um ein Vielfaches von $N/T$ unterscheiden, sind deshalb nicht unterscheidbar. Alle Fourier-Koeffizienten $c_{k+lN}$ landen im selben DFT-Wert:

$$
\frac{\hat a_k}{N} = \sum_{l=-\infty}^{\infty} c_{k+lN}
= c_k + \underbrace{c_{k-N} + c_{k+N} + \dots}_{\text{Aliasing-Fehler}}
$$

## Nyquist-Frequenz

$$
f_\text{Nyquist} = \frac{1}{2\Delta t} \qquad (k = N/2)
$$

Nahe $k = N/2$ ist der Fehlerterm $c_{k-N}$ so groß wie der gesuchte Wert. Beim Dreieckssignal mit $N = 64$ steigt der relative Fehler von 1,3 % bei $k=4$ auf 96 % bei $k=28$.

## Was $\Delta t$ und $T$ bestimmen

- Fensterlänge $T$ → Frequenzauflösung $\Delta f = 1/T$
- Abtastschritt $\Delta t$ → größte Frequenz $f_\text{Nyquist}$
- beide → Punktzahl $N = T/\Delta t$

Kleineres $\Delta t$ verringert Aliasing. Ist $T$ kein Vielfaches der Periode, entsteht zusätzlich Leakage.
