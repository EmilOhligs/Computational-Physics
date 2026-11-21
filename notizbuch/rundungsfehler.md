---
title: Rundungsfehler
aliases: Maschinengenauigkeit, Gleitkommafehler
area: numerik
summary: Jede Gleitkommaoperation rundet relativ auf etwa $\varepsilon \approx 2{,}2\cdot10^{-16}$; wie sich diese Fehler ansammeln, hängt vom Rechenweg ab.
links: diskrete-fourier-transformation, fast-fourier-transform, finite-differenzen-methode
sources: 2026-10-14/THEORIE.md, 2026-11-11/THEORIE.md
---
## Idee

Doppelt genaue Gleitkommazahlen haben etwa 16 Dezimalstellen, die Maschinengenauigkeit ist $\varepsilon \approx 2{,}2\cdot 10^{-16}$. Mathematisch gleiche Verfahren liefern daher numerisch leicht verschiedene Ergebnisse.

## Wie Fehler wachsen

- Einfache Schleife `s += …` über $N$ Terme: Fehler $\propto N$
- Paarweise Summation (`np.sum`): Fehler $\propto \log N$
- FFT: jeder Wert durchläuft nur $\log_2 N$ Stufen, Fehler $\propto \log N$
- Große Argumente in $e^{-2\pi i\, jk/N}$ werden ungenau, Abhilfe: $jk \bmod N$ verwenden

Gemessen in der Übung: naive DFT mit `np.sum` und Modulo-Trick bis $N = 4096$ relativ $\sim 4\cdot10^{-16}$, ohne Modulo-Trick $\sim 8\cdot10^{-13}$.

## Abbruchfehler gegen Rundungsfehler

Bei Differenzenquotienten konkurrieren zwei Fehler: Der Abbruchfehler fällt mit $\Delta x^2$, der Rundungsfehler wächst etwa wie $\varepsilon/\Delta x^2$ (Auslöschung im Stern, Kondition $\sim 1/\Delta x^2$). Bei $-u'' = \pi^2\sin\pi x$ mit dem 3-Punkte-Stern gemessen: Minimum $2{,}3\cdot10^{-10}$ bei $\Delta x = 1{,}5\cdot10^{-5}$, darunter steigt der Fehler unregelmäßig auf $6\cdot10^{-7}$ bei $N \approx 10^6$. Ein feineres Gitter bringt ab dort nichts mehr. Die Schranke $\varepsilon/\Delta x^2$ ist pessimistisch: Bei $\Delta x = 2^{-k}$ ist die Matrix exakt, nur Auswerten und Lösen runden. Siehe [[Finite-Differenzen-Methode]].

## Faustregel

Unterschiede von $10^{-13}$ bis $10^{-16}$ (relativ) sind normal, $10^{-3}$ wäre ein Fehler. Tests deshalb mit `np.allclose` statt `==`.
