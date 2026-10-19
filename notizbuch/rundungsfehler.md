---
title: Rundungsfehler
aliases: Maschinengenauigkeit, Gleitkommafehler
area: numerik
summary: Jede Gleitkommaoperation rundet relativ auf etwa $\varepsilon \approx 2{,}2\cdot10^{-16}$; wie sich diese Fehler ansammeln, hängt vom Rechenweg ab.
links: diskrete-fourier-transformation, fast-fourier-transform
sources: 2026-10-14/THEORIE.md
---
## Idee

Doppelt genaue Gleitkommazahlen haben etwa 16 Dezimalstellen, die Maschinengenauigkeit ist $\varepsilon \approx 2{,}2\cdot 10^{-16}$. Mathematisch gleiche Verfahren liefern daher numerisch leicht verschiedene Ergebnisse.

## Wie Fehler wachsen

- Einfache Schleife `s += …` über $N$ Terme: Fehler $\propto N$
- Paarweise Summation (`np.sum`): Fehler $\propto \log N$
- FFT: jeder Wert durchläuft nur $\log_2 N$ Stufen, Fehler $\propto \log N$
- Große Argumente in $e^{-2\pi i\, jk/N}$ werden ungenau, Abhilfe: $jk \bmod N$ verwenden

Gemessen in der Übung: naive DFT mit `np.sum` und Modulo-Trick bis $N = 4096$ relativ $\sim 4\cdot10^{-16}$, ohne Modulo-Trick $\sim 8\cdot10^{-13}$.

## Faustregel

Unterschiede von $10^{-13}$ bis $10^{-16}$ (relativ) sind normal, $10^{-3}$ wäre ein Fehler. Tests deshalb mit `np.allclose` statt `==`.
