---
title: "Exercise 3: Gewöhnliche Differentialgleichungen"
subtitle: "Computational Physics, WS 2026/27 · Antworten"
author: "Emil Ohligs"
date: "28.10.2026"
lang: de
geometry: "a4paper, margin=2.2cm"
fontsize: 10pt
---

Code in `ode.py` (Teil 1) und `nonlinear.py` (Teil 2), ein Skript pro Teilaufgabe (`plot_1c.py` … `plot_2b.py`) mit den Abbildungen `*_1c.png` … `*_2b.png`, 42 Tests in `test_ode.py`, `test_plot_1c.py`, `test_nonlinear.py`, `test_scripts.py`. Ausführliche Herleitungen in `THEORIE.md`.

## 1 Chemische Reaktion $A \xrightarrow{k_1} B \xrightarrow{k_2} C$

**a) Standardform.** $\dot A = -k_1 A$, $\dot B = k_1 A - k_2 B$, $\dot C = k_2 B$, also $\dot x = Jx$ mit
$J = \begin{pmatrix} -k_1 & 0 & 0 \\ k_1 & -k_2 & 0 \\ 0 & k_2 & 0 \end{pmatrix}$, $x(0) = (1, 0, 0)$.
*Erhaltung:* Jede Spalte von $J$ summiert sich zu null, $(1,1,1)\,J = 0$, also $\frac{d}{dt}(A+B+C) = 0$. Stoff wird nur umgewandelt. Weil jeder Schritt der linearen Verfahren $x_{n+1} = R(\Delta t J)\,x_n$ ist und $(1,1,1)$ Linkseigenvektor zum Eigenwert 0, erhalten alle vier Verfahren die Summe bis auf Rundung.

**b) Verfahren und Kontrolle.** Expliziter Euler, Heun, RK4 und impliziter Euler ($(I - \Delta t J)\,x_{n+1} = x_n$ mit `np.linalg.solve`). Für lineare Systeme ist $R(z)$ die Taylor- bzw. Padé-Näherung von $e^z$: $1+z$, $1+z+z^2/2$, RK4 bis $z^4/24$, $(1-z)^{-1}$.
*Kontrolle:* Vergleich mit der analytischen Lösung $C(t)$; $A+B+C = 1$; $A, B, C \ge 0$ und $C \to 1$; halbes $\Delta t$ verkleinert den Fehler um $2^p$; ein einzelner Schritt gegen die Reihe von $R(\Delta t J)\,x$; bei kleinem $\Delta t$ stimmen alle Verfahren überein.

**c) Genauigkeit** ($k_2 = 10$, max. Fehler von $C$ auf $[0, 10]$). Ein Verfahren der Ordnung $p$ hat globalen Fehler $e \approx c\,\Delta t^p$, im log-log-Plot eine Gerade mit Steigung $p$. Erwartet 1, 2, 4, 1; gemessen $1{,}00$ (Euler), $2{,}01$ (Heun), $4{,}09$ (RK4), $1{,}00$ (impliziter Euler). Bei $\Delta t = 0{,}1$ ($k_2\Delta t = 1$) liegen die Fehler noch über der Geraden. RK4 erreicht bei $\Delta t \approx 4\cdot10^{-4}$ schon $8\cdot10^{-14}$, die Rundungsgrenze. Für Fehler $10^{-6}$: RK4 $\Delta t \approx 0{,}02$, Heun $0{,}0012$, Euler $5\cdot10^{-6}$.

**d) Kritischer Zeitschritt.** Stabil, solange $|R(\lambda\Delta t)| \le 1$ für alle Eigenwerte $0, -k_1, -k_2$. Also $\Delta t_\text{krit} = c/\max(k_1, k_2)$ mit gemessen (Bisektion) $c = 2$ für Euler und Heun, $c = 2{,}785$ für RK4, für alle fünf getesteten Paare $(k_1, k_2)$. Beispiel $k_2 = 10$: $0{,}2$ bzw. $0{,}2785$; impliziter Euler hat keine Grenze.
*Knapp darüber* ($1{,}05\,\Delta t_\text{krit}$) ist $R < -1$: die Lösung springt von Schritt zu Schritt das Vorzeichen, mit wachsender Amplitude. Max. Fehler bei $0{,}95$ / $1{,}05\,\Delta t_\text{krit}$: Euler $0{,}12$ / $10{,}8$, Heun $0{,}087$ / $13{,}4$, RK4 $0{,}082$ / $130$ (`instability_1d.png`).

**e) Eigenwerte und Stabilitätsgebiete** ($k_1 = 1$, $k_2 = 10$). $\lambda = 0$ mit $(0, 0, 1)$ (Endzustand), $\lambda = -1$ mit $(0{,}667;\ 0{,}074;\ -0{,}741)$ (Zerfall von A), $\lambda = -10$ mit $(0;\ 0{,}707;\ -0{,}707)$ (schneller Übergang B → C). Jede Mode wird pro Schritt mit $R(\lambda\Delta t)$ multipliziert (`stability_1e.png`, `modes_1e.png`).
*Zuerst instabil* wird die schnelle Mode $\lambda = -10$: $z = \lambda\Delta t$ verlässt bei ihr zuerst das Gebiet ($|R| = 1{,}100$, $1{,}105$, $1{,}231$ bei $1{,}05\,\Delta t_\text{krit}$), obwohl sie physikalisch nach kurzer Zeit abgeklungen ist.
*Impliziter Euler:* $R(z) = 1/(1-z)$ und $|1 - z| \ge 1$ für $\operatorname{Re} z \le 0$, also $|R| \le 1$ in der ganzen linken Halbebene (A-stabil).

**f) Steifheit** ($k_2 = 1000$, Fehler $\approx 10^{-2}$). Expliziter Euler braucht $n = 4999$ Schritte, $\Delta t = 0{,}002$, genau $2/k_2$: begrenzt durch die **Stabilität** der schnellen Mode. Impliziter Euler braucht $n = 180$, $\Delta t = 0{,}056$: begrenzt nur durch die **Genauigkeit** für die langsame Mode $e^{-t}$ und den Anfangsübergang. Faktor 28 weniger Schritte für ein 3×3-System pro Schritt (`stiff_1f.png`).

## 2 Nichtlineare Gleichungen

Die Verfahren in `nonlinear.py` nehmen $f(x)$ und die Jacobimatrix $J_f(x)$; der implizite Euler löst $g(y) = y - x_n - \Delta t\, f(y) = 0$ mit Newton, $(I - \Delta t\,J_f(y))\,\delta = -g(y)$, bis $\max|\delta| \le 10^{-12}(1+\max|y|)$. Für lineares $f$ liefern sie dasselbe wie Teil 1 (getestet).

**a) Robertson-Modell.** Impliziter Euler auf logarithmischem Gitter $10^{-6} \dots 10^5$, weil die Dynamik auf allen Zeitskalen passiert. Bei $t = 10^5$ mit 1600 Punkten $A = 0{,}01808$, $B = 7{,}36\cdot10^{-8}$, $C = 0{,}98192$, mit 400 Punkten $A = 0{,}01871$. $B$ steigt bis $t \sim 10^{-3}$ auf $3{,}6\cdot10^{-5}$ und bleibt bis $t \approx 1$; $A+B+C = 1$ bis auf $2\cdot10^{-15}$ (`robertson_2a.png`).
*Warum explizit versagt:* der Eigenwert in B-Richtung ist etwa $-2k_2B - k_3C \approx -2000$, also $\Delta t < 10^{-3}$. Gemessen explodiert der explizite Euler mit $\Delta t = 10^{-3}$ bei $t = 9{,}35$; bis $10^5$ wären über $10^8$ Schritte nötig.
*Code wiederverwenden?* Nur teilweise: die expliziten Verfahren mit $f(x)$ statt $Jx$ direkt, der implizite Euler braucht statt eines linearen Systems Newton mit $J_f$, und das gleichmäßige Gitter aus Teil 1 reicht nicht.

**b) Lorenz-System** (RK4, $\Delta t = 10^{-3}$). Der Abstand der Bahnen von $(1,1,1)$ und $(1,1,1+10^{-8})$ wächst im Mittel exponentiell mit Rate $0{,}84$ (Fit; Lyapunov-Exponent laut Literatur $0{,}91$) und überschreitet 1 bei $t = 28{,}88$, mit $\Delta t = 5\cdot10^{-4}$ ebenso. Danach sind die Bahnen unkorreliert, laufen aber auf demselben Attraktor (`lorenz_2b.png`).

**Welchem Verfahren vertrauen?** Ohne exakte Lösung: mit halbem $\Delta t$ bzw. feinerem Gitter wiederholen (Robertson 400 gegen 1600, Lorenz $10^{-3}$ gegen $5\cdot10^{-4}$), Erhaltungsgrößen prüfen ($A+B+C$), Verfahren verschiedener Ordnung vergleichen. Beim steifen Robertson-Modell dem impliziten Euler. Beim Lorenz-System RK4, aber nur bis zur Trennzeit: auch dieselbe Bahn mit $\Delta t = 10^{-3}$ und $5\cdot10^{-4}$ trennt sich bei $t \approx 28{,}9$, Euler gegen RK4 schon bei $5{,}1$. Danach nur Statistik und Attraktor vertrauen.
