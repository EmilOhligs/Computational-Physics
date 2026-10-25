---
title: "Exercise 2: Entmagnetisierungsfeld per FFT"
subtitle: "Computational Physics, WS 2026/27 · Antworten"
author: "Emil Ohligs"
date: "21.10.2026"
lang: de
geometry: "a4paper, margin=2.2cm"
fontsize: 10pt
---

Zellen der Größe 1, Punktdipol-Kernel $K(\vec r) = \frac{1}{4\pi r^3}\big(3\hat r\hat r^T - I\big)$ mit Selbstterm $K(0) = -\tfrac13 I$. Code in `demag.py`, Tests in `test_demag.py` (25 Tests), Abbildungen und Messungen in `exercise_2_demag.ipynb`.

## 1 Entmagnetisierungsfeld in 1D

**a) Dipolfeld und 1D-Kernel.** Auf der x-Achse ist $\hat r = \pm\hat x$, also $(\vec m\cdot\hat r)\hat r = m_x\hat x$ und $h_x = \frac{3-1}{4\pi|x|^3}m_x$. Der Kernel ist $K_{xx}(x) = 1/(2\pi|x|^3)$, gerade und positiv.
*Selbstwechselwirkung:* Der Punktdipol divergiert bei $x = 0$. Die Zelle ist aber ein homogen magnetisierter Würfel mit Entmagnetisierungsfaktor $N = 1/3$ (Symmetrie und $N_x+N_y+N_z = 1$). Deshalb $K_{xx}(0) = -1/3$. Das ist genau der Delta-Anteil von $\partial_i\partial_j\frac{1}{4\pi r}$ (siehe 2d).

**b) Direkte Summe.** Segment aus 32 Dipolen in 64 Zellen.

- *Mitte:* $h_x = +0.049$, wie die unendliche Kette $-\tfrac13 + \zeta(3)/\pi$. Die Nachbarn auf der Achse heben den Selbstterm fast auf.
- *Enden:* $h_x = -0.142$, wie die halbunendliche Kette $-\tfrac13 + \zeta(3)/2\pi$. Das Feld ist entmagnetisierend, hier sitzen die Ladungen $\rho = -\partial_x m_x$.
- *Außerhalb:* Das Streufeld zeigt in $+x$ ($+0.19$ direkt neben dem Ende).

Der Übergang ist auf wenige Zellen beschränkt, weil $K \sim 1/x^3$ schnell abfällt. Das kleine Plus in der Mitte ist ein Fehler der Punktdipol-Näherung; ein echter dünner Stab hätte dort $h \approx 0$.

**c) Faltungssatz.** $\hat h_k = \hat K_k\hat m_k$ liefert eine *zyklische* Faltung: Negative Abstände werden auf $i - j + L$ abgebildet, die Kette wird zum Ring. Ohne Padding erzeugt ein Dipol am linken Rand am rechten Ende $0.159 = K(1)$ statt $6.4\cdot10^{-7}$.
*Lineare Faltung:* Es gibt $2N-1$ Abstände $-(N-1)\dots N-1$. Mit Zero-Padding auf $L \ge 2N-1$ bekommt jeder einen eigenen Index modulo $L$. Der Kernel steht in Wraparound-Anordnung: Abstände $0\dots N-1$ vorne, $-(N-1)\dots-1$ am Ende (Reihenfolge von `np.fft.fftfreq(L, 1/L)`). Vom Ergebnis nimmt man die ersten $N$ Werte. FFT und direkte Summe stimmen dann auf $2\cdot10^{-16}$ überein.

**d) Laufzeit.** Direkt $O(N^2)$ (gemessen lokal $N^{1.9}$), FFT $O(N\log N)$ (etwa $6\,\text{ns}\cdot N\log_2 N$). Faktor 18 bei $N = 64$, 1150 bei $N = 8192$.
*$N \ne 2^m$:* Die FFT braucht keine Zweierpotenz, pocketfft zerlegt in beliebige Primfaktoren. Glatte Längen ($50625 = 3^4 5^4$) sind gleich schnell. Bei einem großen Primfaktor ($N = 65537$ oder 65521, $L = 2p$) ist sie 5-mal langsamer: Bluestein-Algorithmus, weiterhin $O(N\log N)$, aber mehrere FFTs.

**e) Optimierung.** Reelle FFT (`rfft`, spart 40 %), Kernel-Spektrum einmal berechnen und wiederverwenden (2 statt 3 FFTs pro Aufruf), Padding auf `next_fast_len(2N-1)` statt $2N$. Zusammen 3.4- bis 4.1-mal schneller bei $N = 2^k$ und 22-mal bei Primzahl-$N$, bei identischem Ergebnis.
*Weiter:* mehrere Threads (`workers=-1`), FFTW mit geplantem Transform, GPU, Pruned FFTs (die Hälfte der Eingabe ist null, die Hälfte der Ausgabe wird verworfen), `float32`.

## 2 Erweiterung auf 3D

**a) Tensorkernel.** $h_i = \sum_j K_{ij} m_j$ mit $K_{ij} = \frac{1}{4\pi r^3}\big(3 r_i r_j/r^2 - \delta_{ij}\big)$, $K(0) = -\tfrac13 I$. $K$ ist symmetrisch: **6 unabhängige Komponenten**. Die 6 Kernel-FFTs werden einmal vorab berechnet. Pro Auswertung **6 FFTs**: 3 vorwärts ($m_x, m_y, m_z$), 9 punktweise Produkte, 3 zurück, alle reell auf $(2n)^3$.

**b) Validierung.** Zwei Fehlerquellen getrennt prüfen:

- *Arithmetik:* Gegen die direkte Summe auf Gittern bis $14^3$ mit zufälligem $\vec m$: $10^{-16}$ bis $3\cdot10^{-15}$ ($\le 13\,\varepsilon$), wächst langsam mit der Zahl der Terme. Dazu der Einzeldipol gegen `dipole_field`.
- *Physik:* Kugel innen analytisch $-\vec m/3$. Im Kern ($r < R/2$) gemessen $|h_z + 1/3| \approx 10^{-3}$, besser mit wachsendem $R$ ($5\cdot10^{-3}$ bei $R = 4$, $7\cdot10^{-4}$ bei $R = 24$). In den Randzellen bis $0.26$, unabhängig von $R$, weil die Treppenstufen lokal immer gleich rau sind. Im Fernfeld nähert sich eine kleine Kugel dem Punktdipol mit Gesamtmoment (6 % bei Abstand 5.5, 0.4 % bei 19.5).

Der Fehler hängt also von Rundung (FFT), Auflösung, Abstand zur Oberfläche und Geometrie ab, und von der Punktdipol-Näherung zwischen Nachbarn.

**c) Würfel und Kugel** ($\vec m = \hat z$, $R = 16$, $a = 32$). Die Kugel ist innen gleichmäßig bei $-1/3$ (Streuung 0.03, nur aus Randzellen); außen an den Polen $0.61$, analytisch $0.608$. Der Würfel nicht: $-0.58$ an den Polflächen, $-1/3$ im Zentrum, $-0.21$ an den Seitenflächen, $-0.13$ an den Kantenmitten (Streuung 0.10).
*Warum?* Homogenes $\vec m$ hat nur Flächenladungen $\sigma = \vec m\cdot\hat n$. Bei der Kugel ist das dasselbe wie zwei homogen geladene Kugeln $\pm\rho$, um $\vec\delta$ verschoben. Innen ist deren Feld linear ($\rho\vec r/3$), die Differenz also konstant, $-\vec m/3$. Das gilt für jedes Ellipsoid und nur dafür. Beim Würfel sitzen die Ladungen auf zwei endlichen Platten, deren Feld zu den Kanten hin abfällt.

**d) Magnetische Ladungen.** $\rho = -\nabla\cdot\vec m$ und $\vec h = -\nabla\phi$ als zentrale Differenzen, $\phi = K_c*\rho$ per FFT mit $K_c(0) = \frac{3\ln(2+\sqrt3)-\pi/2}{4\pi}$ (Würfelmittelpunkt).

- *Genauigkeit:* Innen stimmen beide auf $3\cdot10^{-3}$ überein, außen auf $0.015$. An der Oberfläche bis $0.31$: Die Differenzen verschmieren die Flächenladung über zwei Zellen ($-1/3 \to +0.48$ statt $-0.39 \to +0.61$ beim Dipol-Kernel, analytisch $-1/3 \to 0.608$). Für glattes $\vec m$ fällt der Unterschied wie $1/\sigma^2$ (7.1 % bei $\sigma = 3$, 1.9 % bei $\sigma = 6$).
- *Aufwand:* 2 statt 6 FFTs pro Auswertung, gemessen 0.05 s gegen 0.25 s bei $n = 64$.
- *Warum äquivalent?* Partielle Integration: $\phi = -\nabla\cdot(K_c*\vec m)$, also $\vec h = (\nabla\nabla^T K_c)*\vec m$. Und $\partial_i\partial_j\frac{1}{4\pi r} = \frac{3\hat r_i\hat r_j - \delta_{ij}}{4\pi r^3} - \frac{\delta_{ij}}{3}\delta(\vec r)$: der Dipol-Kernel samt Selbstterm. Im Kontinuum identisch, verschieden ist nur, wo diskretisiert wird.
