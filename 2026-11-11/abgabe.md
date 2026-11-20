---
title: "Exercise 5: Finite Differenzen"
subtitle: "Computational Physics, WS 2026/27 · Antworten"
author: "Emil Ohligs"
date: "11.11.2026"
lang: de
geometry: "a4paper, margin=2.2cm"
fontsize: 10pt
---

Code in `fd.py`, Rechnungen und Abbildungen im Notebook `exercise_5_fd.ipynb` (`matrix_1a.png`, `solution_1a.png`, `error_1b.png`, `stencils_1c.png`, `neumann_1d.png`, `poisson2d_1e.png`, `mol_2a.png`, `accuracy_2b.png`, `stability_2c.png`, `unstable_2c.png`, `stiff_2d.png`, `damping_2d.png`). Für Teil 2 werden die Schritte aus Exercise 3 (`2026-10-28/ode.py`) unverändert importiert. 34 Tests in `test_fd.py`. Ausführliche Herleitungen in `THEORIE.md`.

## 1 Stationäre Wärmeleitung $-u'' = f$

**a) 3-Punkte-Stern.** $-u''(x_j) \approx (-u_{j-1} + 2u_j - u_{j+1})/\Delta x^2$, also $A = \frac{1}{\Delta x^2}\,\mathrm{tridiag}(-1, 2, -1)$ als `scipy.sparse.diags` ($3N - 2$ von $N^2$ Einträgen, `matrix_1a.png`). In der ersten und letzten Zeile fällt der Nachbar weg, weil $u_0 = u_{N+1} = 0$. \
*$A$ vor dem Lösen prüfen:* symmetrisch ($\max|A - A^T| = 0$); innere Zeilensummen 0 (zweite Ableitung einer Konstanten); $A\,x(1-x) = 2$ exakt bis auf $4\cdot10^{-14}$, weil der Abbruchfehler $\frac{\Delta x^2}{12}u''''$ für Polynome vom Grad 2 verschwindet; Eigenwerte gleich $\lambda_k = \frac{4}{\Delta x^2}\sin^2\frac{k\pi\Delta x}{2}$ (relativ $2\cdot10^{-14}$) und alle positiv ($\lambda_1 = 9{,}851$ bei $N = 20$, $\pi^2 = 9{,}870$). Alle Prüfungen stehen als Tests in `test_fd.py`.

**b) Fehler über $\Delta x$** (`error_1b.png`), $N = 2^k - 1$ für $k = 2 \dots 20$, damit $\Delta x$ und alle $x_j$ exakt als `float64` darstellbar sind. \
*Steigung:* erwartet 2 (Abbruchfehler $\frac{\Delta x^2}{12}u''''$). Gemessen **2,00** bis $N = 8191$: Der Fehler viertelt sich pro Halbierung von $\Delta x$ und liegt fast genau auf $\pi^2\Delta x^2/12$. \
*Wiederanstieg:* Minimum $2{,}3\cdot10^{-10}$ bei **$\Delta x \approx 1{,}5\cdot10^{-5}$** ($N = 65\,535$). Darunter wächst der Fehler steil und unregelmäßig bis $6{,}5\cdot10^{-7}$ bei $N \approx 10^6$. Ursache sind Rundungsfehler: Die Einträge von $A$ sind $\sim 1/\Delta x^2$, im Stern ziehen sich fast gleich große Zahlen voneinander ab, und die Kondition $\kappa(A) \approx 4/(\pi^2\Delta x^2)$ wächst. $\varepsilon/\Delta x^2$ ist die grobe Schranke für den schlimmsten Fall; der echte Rundungsfehler liegt darunter.

**c) Stencils höherer Ordnung** (`stencils_1c.png`). Zentrale Sterne für $-u''$: $(-1, 2, -1)$, $\frac{1}{12}(1, -16, 30, -16, 1)$, $\frac{1}{180}(-2, 27, -270, 490, -270, 27, -2)$. \
*Am Rand* reicht der breite Stern über $x_0 = 0$ hinaus. Zwei Varianten: `"reduce"` nimmt in den randnahen Zeilen den größten zentralen Stern, der noch passt; `"mirror"` bestimmt Geisterpunkte durch ungerade Spiegelung $u_{-m} = -u_m$ (richtig, wenn alle geraden Ableitungen am Rand verschwinden, wie bei $\sin\pi x$). Allgemein nimmt man einseitige Sterne gleicher Ordnung, dann wird $A$ unsymmetrisch. \
*Ordnungen* (Raten $\log_2$ des Fehlerverhältnisses, $N = 7 \dots 255$): Ordnung 2: 2,00; Ordnung 4: 4,00 (mirror) bzw. 4,7 → 4,1 (reduce); Ordnung 6: 6,00 (mirror), aber nur **5,00** (reduce). Die verkleinerte Randzeile hat den Abbruchfehler $\tau_1 \approx \frac{\pi^5}{12}\Delta x^3$, der die ganze Lösung um $\sim\Delta x^2\tau_1 \sim \Delta x^5$ verschiebt. Bei Ordnung 4 ist das kleiner als der innere Fehler, bei Ordnung 6 größer. Im rechten Bild ist der Fehler mit `"reduce"` auch in der Intervallmitte größer.
Je höher die Ordnung, desto früher erreicht man den Rundungsboden: Ordnung 6 mit Spiegelung $2\cdot10^{-13}$ schon bei $N = 127$.

**d) Neumann-Rand $u'(1) = 0$** (`neumann_1d.png`), $f = \frac{\pi^2}{4}\sin\frac{\pi x}{2}$, $u = \sin\frac{\pi x}{2}$. $u_{N+1}$ ist jetzt unbekannt und bekommt eine eigene Zeile. \
*Steigung:* einseitige Differenz $(u_{N+1} - u_N)/\Delta x = 0$ gibt **1** (Raten 0,97 → 1,00); Geisterpunkt $u_{N+2} = u_N$ aus der zentralen Differenz, eingesetzt in den Stern ($(-2u_N + 2u_{N+1})/\Delta x^2 = f(1)$, halbiert, damit $A$ symmetrisch bleibt), gibt **2** (Raten 2,00). Alle inneren Zeilen sind gleich, die eine Randzeile entscheidet. \
*Wirkung auf die ganze Lösung:* Der Fehler bleibt nicht am Rand. Taylor gibt $\frac{u_{N+1} - u_N}{\Delta x} = u'(1) - \frac{\Delta x}{2}u''(1)$, die Zeile erzwingt also $u'(1) = -\frac{\pi^2}{8}\Delta x$. Der Fehler ist die Lösung von $-e'' = 0$, $e(0) = 0$ mit dieser Steigung: eine Gerade $-\frac{\pi^2}{8}\Delta x\,x$ durch das ganze Intervall. Eine elliptische Gleichung verteilt jeden Randfehler über das ganze Gebiet.

**e) 2D-Poisson** (`poisson2d_1e.png`). $A_{2D} = A\otimes I + I\otimes A$ (5-Punkte-Stern) mit `scipy.sparse.kron`, $N^2$ Unbekannte. \
*Fehler:* Steigung **2,00** über $\Delta x$, also $\propto N^{-2}$ wie in 1D, genau auf $\pi^2\Delta x^2/12$ (die rechte Seite ist Eigenvektor von $A_{2D}$). Bis $N = 1023$ kein Wiederanstieg; das Rundungsminimum aus 1b läge bei $4\cdot10^9$ Unbekannten. \
*Laufzeit* mit `spsolve`: Steigung **2,8** über $N$, also fast $N^3 = n^{1{,}5}$ für $n = N^2$ Unbekannte ($N = 511$: 1,3\ s, $N = 1023$: 10\ s). Halbes $\Delta x$ kostet etwa achtmal so viel Zeit und viertelt nur den Fehler. Grund ist das Auffüllen des Bandes (Bandbreite $N$) beim Faktorisieren. Für größere Gitter nimmt man iterative Löser (CG, Mehrgitter) oder die FFT.

## 2 Zeitabhängige Wärmeleitung $u_t = u_{xx}$

**a) Linienmethode** (`mol_2a.png`). Raumdiskretisierung mit $A$ aus 1 gibt $\dot u = -Au$, also $\dot x = Jx$ aus Exercise 3 mit $J = -A$. \
*Unverändert wiederverwendbar:* `euler_step`, `heun_step`, `rk4_step` (brauchen nur `J @ x` und laufen mit der sparse-Matrix), `integrate`, `order`, `stability_function` und `critical_dt` (mit $\lambda_1, \lambda_N$ statt $k_1, k_2$). Nur `implicit_euler_step` benutzt `np.eye` und `np.linalg.solve` (dense); `fd.py` hat eine Version mit `spsolve`. Crank–Nicolson ist ein halber expliziter plus ein halber impliziter Schritt: $(I - \frac{\Delta t}{2}J)\,u^{n+1} = (I + \frac{\Delta t}{2}J)\,u^n$. Bei $t = 0{,}1$, $\Delta t = 10^{-4}$: Fehler gegen PDE $6{,}5\cdot10^{-5}$ / $3{,}0\cdot10^{-4}$ / $1{,}2\cdot10^{-4}$ (expl. / impl. Euler / CN).

**b) Genauigkeit** (`accuracy_2b.png`), $t = 0{,}1$. Exakte ODE-Lösung $e^{-\lambda_1 t}\sin\pi x_j$ ($\lambda_1 = 9{,}8665$), PDE-Lösung $e^{-\pi^2 t}\sin\pi x_j$; ihr Abstand $1{,}16\cdot10^{-4}$ ist der reine Raumfehler bei $N = 50$. \
*Steigungen:* erwartet in $\Delta t$ 1 / 1 / 2, gemessen gegen die ODE-Lösung **1,00 / 1,00 / 2,00**. In $\Delta x$ erwartet 2; CN mit $\Delta t = 10^{-4}$ liegt genau auf $\Delta x^2$ (bis $4{,}6\cdot10^{-6}$). \
*Warum der Fehler aufhört zu fallen:* Gegen die PDE bleibt immer der Fehler der anderen Diskretisierung übrig. In der $\Delta t$-Studie ist das der Raumfehler $1{,}16\cdot10^{-4}$; in der $\Delta x$-Studie bleibt der implizite Euler bei $1{,}9\cdot10^{-4}$ stehen, seinem Zeitfehler bei $\Delta t = 10^{-4}$. CN hat dort nur $\sim10^{-9}$ Zeitfehler und fällt weiter.
Expliziter Euler und CN liegen teils *unter* dem Raumfehler (CN mit Knick bei $\Delta t \approx 6\cdot10^{-3}$): Fehlerauslöschung, weil $\lambda_1 < \pi^2$ zu langsam und beide Verfahren pro Schritt etwas zu stark dämpfen. Der explizite Euler ist nur für $\Delta t < 2/\lambda_N = 1{,}92\cdot10^{-4}$ eingezeichnet.

**c) Stabilität** (`stability_2c.png`, `unstable_2c.png`). Eigenvektoren $\sin(k\pi x_j)$, Eigenwerte $\lambda_k = \frac{4}{\Delta x^2}\sin^2\frac{k\pi\Delta x}{2}$; Mode $k$ wird pro Schritt mit $R(z_k)$, $z_k = -\lambda_k\Delta t$ multipliziert. Kritisches $\Delta t$:

- $N = 10$: Euler/Heun $4{,}218\cdot10^{-3}$, RK4 $5{,}874\cdot10^{-3}$, von Neumann $\Delta x^2/2 = 4{,}132\cdot10^{-3}$
- $N = 20$: $1{,}140\cdot10^{-3}$, $1{,}588\cdot10^{-3}$, $1{,}134\cdot10^{-3}$
- $N = 40$: $2{,}979\cdot10^{-4}$, $4{,}148\cdot10^{-4}$, $2{,}974\cdot10^{-4}$
- $N = 80$: $7{,}624\cdot10^{-5}$, $1{,}062\cdot10^{-4}$, $7{,}621\cdot10^{-5}$

*Entscheidender Eigenwert:* der größte, $\lambda_N \approx 4/\Delta x^2$. Alle $z_k$ liegen auf der negativen reellen Achse, $z_N$ am weitesten links; $\Delta t_{\text{krit}} = 2/\lambda_N$ (Euler, Heun) bzw. $2{,}785/\lambda_N$ (RK4). Von Neumann setzt $\sin^2 = 1$ und liegt deshalb knapp darunter, mit feinerem Gitter immer näher. Halbes $\Delta x$ heißt ein Viertel $\Delta t$. \
*Knapp darüber* ($1{,}02\,\Delta t_{\text{krit}}$, expliziter Euler, $N = 20$): Die höchste Mode wächst pro Schritt um $|R| = 1{,}04$ mit wechselndem Vorzeichen von Rundungsniveau an und überholt bei $t \approx 0{,}87$ die korrekt abklingende Grundmode. Danach sieht man einen Zickzack von Punkt zu Punkt mit sinusförmiger Einhüllender, also den Eigenvektor $k = N$. \
*Implizit immer stabil:* Für $z < 0$ ist $|1/(1-z)| < 1$ und $|(1+z/2)/(1-z/2)| < 1$, egal wie groß $|z|$ ist. Beide Stabilitätsgebiete enthalten die ganze linke Halbebene (A-stabil).

**d) Steifheit und Rechteckpuls** (`stiff_2d.png`, `damping_2d.png`), $N = 99$. Die Eigenwerte reichen von 9,87 bis 39 990 (Verhältnis 4052). \
*CN schlechter als impliziter Euler:* Der Sprung enthält alle Moden. Bei $\Delta t = 0{,}01$ ist $z_N = -400$ und $R_{\text{CN}}(z_N) = -0{,}990$: Die schnellen Moden werden kaum gedämpft und wechseln jeden Schritt das Vorzeichen, exakt wäre $e^{-400} \approx 0$. Der implizite Euler hat $R_{\text{IE}}(z_N) = 0{,}0025$ (L-stabil: $R \to 0$ für $z \to -\infty$, bei CN $\to -1$). Fehler gegen die ODE-Lösung bei $t = 0{,}1$: **0,016 (IE) gegen 0,33 (CN)**, CN mit Überschwingern an den Sprungstellen. Erst unter $\Delta t \approx 1{,}6\cdot10^{-3}$ ist CN wieder besser, dann mit Ordnung 2. \
*Verschwinden die schnellen Moden?* Bei CN nur geometrisch: Die höchste Mode fällt in 100 Schritten von 0,10 auf 0,037 ($0{,}99^{100}$) und ist ab $t \approx 0{,}47$ größer als die Grundmode. Beim impliziten Euler ist sie nach 5 Schritten auf Rundungsniveau. \
*Expliziter Euler:* $\Delta t_{\text{krit}} = 2/\lambda_N = 5\cdot10^{-5}$, also mindestens **2000 Schritte** bis $t = 0{,}1$ statt 10. Halbes $\Delta x$ heißt viermal so viele Schritte (8000 bei $N = 199$, 32 000 bei $N = 399$), jeder doppelt so teuer: Gesamtaufwand $\propto N^3$. Deshalb rechnet man Diffusion implizit.
