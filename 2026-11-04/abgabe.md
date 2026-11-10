---
title: "Exercise 4: Symplektische Integratoren für Orbits"
subtitle: "Computational Physics, WS 2026/27 · Antworten"
author: "Emil Ohligs"
date: "04.11.2026"
lang: de
geometry: "a4paper, margin=2.2cm"
fontsize: 10pt
---

Code in `orbit.py` (Teil 1) und `rocket.py` (Teil 2), ein Skript pro Teilaufgabe (`plot_1b.py` … `plot_2d.py`) mit den Abbildungen `orbits_1b.png`, `conservation_1c.png`, `convergence_1d.png`, `vertical_2a.png`, `pitch_2b.png`, `circular_2c.png`, `staging_2d.png`. Tests in `test_orbit.py`, `test_rocket.py`, `test_scripts.py`. Ausführliche Herleitungen in `THEORIE.md`.

## 1 Mond um die Erde

**a) Kreisbahn.** Die Gravitation muss genau die Zentripetalbeschleunigung liefern: $v^2/r = GM/r^2$, also $\mathbf r(0) = (r_M, 0)$, $\mathbf v(0) = (0, \sqrt{GM/r_M}) = (0;\ 1018{,}3\ \mathrm{m/s})$. Periode $T = 2\pi r_M/v = 2\pi\sqrt{r_M^3/GM} = 2{,}372\cdot10^6\ \mathrm s = 27{,}45$ d.
*Differenz zu 27,32 d:* Die Erde ist nicht fest, beide kreisen um den gemeinsamen Schwerpunkt. Im Zweikörperproblem steht $G(M+m)$ statt $GM$, also $T/\sqrt{1 + m/M} = 27{,}29$ d mit $m/M = 0{,}0123$. Der Rest kommt von der elliptischen Bahn (384 400 km ist der mittlere Abstand) und der Störung durch die Sonne.

**b) Verfahren, 10 Umläufe, $\Delta t = 6$ h** (1098 Schritte). Zustand $\mathbf y = (\mathbf r, \mathbf v)$, $\dot{\mathbf y} = (\mathbf v, -GM\mathbf r/|\mathbf r|^3)$. Explizites und implizites Euler, Heun, RK4, symplektisches Euler ($\mathbf v_{n+1} = \mathbf v_n + \Delta t\,\mathbf a(\mathbf r_n)$, $\mathbf r_{n+1} = \mathbf r_n + \Delta t\,\mathbf v_{n+1}$) und Velocity Verlet (halber Kick, Drift, halber Kick).
*Newton:* $(I - \Delta t J)\,\boldsymbol\delta = -(\mathbf z - \mathbf y_n - \Delta t\,\mathbf f(\mathbf z))$ mit $J = \begin{pmatrix} 0 & I \\ A & 0\end{pmatrix}$, $A = -\frac{GM}{|\mathbf r|^3}(I - 3\hat{\mathbf r}\hat{\mathbf r}^T)$. Startwert ist der explizite Euler-Schritt, er liegt nur $O(\Delta t^2)$ neben der Lösung: **3 Iterationen** pro Schritt (Abbruch bei $10^{-14}$ relativ).
*Vorhersage:* Für $\dot z = i\omega z$ mit $x = \omega\Delta t = 0{,}057$ ist $|R|^2 = 1 + x^2$ (Euler) bzw. $1 + x^4/4$ (Heun) $> 1$, also **nach außen**. $1/(1+x^2)$ (impliziter Euler) bzw. $1 - x^6/72$ (RK4) $< 1$, also **nach innen**. Die symplektischen Verfahren erhalten die Phasenraumfläche ($\det = 1$) und spiralen nicht.
*Ergebnis* (`orbits_1b.png`), Endradius / $r_M$: Euler 2,757, Heun 1,006, symplektisches Euler 1,003 (stabile Ellipse), Verlet 1,000000, RK4 0,999999. Der implizite Euler stürzt nach 58 Schritten auf $0{,}30\,r_M$ ab, denn die Dämpfung verstärkt sich selbst, weil $\omega$ auf der engeren Bahn wächst. Danach hat die implizite Gleichung keine nahe Lösung mehr (auch `fsolve` nicht).

**c) Erhaltungsgrößen.** $\dot E = \mathbf v\cdot\dot{\mathbf v} + \frac{GM}{|\mathbf r|^3}\mathbf r\cdot\mathbf v = 0$ und $\dot{\mathbf L} = \mathbf v\times\mathbf v + \mathbf r\times\mathbf a = 0$ (Zentralkraft, $\mathbf a\parallel\mathbf r$).
*Fehler über 100 Umläufe* (`conservation_1c.png`), max. $|\Delta E/E|$ / $|\Delta L/L|$: Euler $0{,}84$ / $1{,}4$; Heun $4{,}8\cdot10^{-2}$ / $2{,}5\cdot10^{-2}$; symplektisches Euler $3{,}3\cdot10^{-3}$ / $4\cdot10^{-15}$; Verlet $2{,}7\cdot10^{-6}$ / $2\cdot10^{-14}$; RK4 $1{,}1\cdot10^{-5}$ / $5\cdot10^{-6}$.
*L exakt erhalten:* nur **symplektisches Euler und Verlet**. Beide bestehen aus Kicks ($\mathbf r$ fest, $\Delta\mathbf v\parallel\mathbf r$: $\mathbf r\times\Delta\mathbf v = 0$) und Drifts ($\mathbf v$ fest, $\Delta\mathbf r\parallel\mathbf v$). Beim Euler bleibt pro Schritt $\Delta\mathbf L = \Delta t^2\,\mathbf v\times\mathbf a \ne 0$.
*Heun schlechter als symplektisches Euler:* Heun pumpt bei jedem Schritt Energie mit demselben Vorzeichen hinein, der Fehler wächst linear ($5{,}6\cdot10^{-3}$ nach 10, $4{,}8\cdot10^{-2}$ nach 100 Umläufen). Das symplektische Euler erhält eine Schatten-Energie $E + O(\Delta t)$, sein Fehler schwankt nur und bleibt bei $3{,}3\cdot10^{-3}$. Ab knapp 6 Umläufen ist Heun schlechter.

**d) Konvergenz bei $T/2$** (`convergence_1d.png`), Fehler $|\mathbf r - (-r_M, 0)|/r_M$ für $\Delta t = T/200 \dots T/6400$. Steigungen: Euler 0,96, impliziter Euler 1,05, symplektisches Euler 0,99, Heun 2,00, Verlet 2,00, RK4 4,01, also die Ordnungen 1, 1, 1, 2, 2, 4. Bei $T/6400$: RK4 $7{,}5\cdot10^{-14}$, Verlet $1{,}1\cdot10^{-6}$.
*1000 Jahre* ($\approx 13\,300$ Umläufe): **Velocity Verlet** (oder ein symplektisches Verfahren höherer Ordnung). $L$ bleibt exakt und $E$ beschränkt, die Bahnform stimmt für beliebig lange Zeiten, und pro Schritt braucht es im Prinzip nur eine Kraftauswertung. RK4 ist auf kurzen Zeiten genauer, driftet aber in der Energie linear ($\approx 1{,}4\cdot10^{-3}$ nach 1000 Jahren bei 6 h). Der Phasenfehler wächst bei allen Verfahren linear.

## 2 Raketenstart

Modell in `rocket.py`: $\ddot{\mathbf r} = -GM\mathbf r/|\mathbf r|^3 + \frac{F/m_0}{1 - (F/m_0)t/v_e}\,\mathbf e$ bis $t_b = (1 - m_1/m_0)\,v_e/(F/m_0)$, RK4 mit $\Delta t = 0{,}1$ s beim Brennen und 1 s im Freiflug. $t_b$ liegt genau auf dem Gitter, damit keine RK4-Stufe über den Sprung im Schub rechnet.

**a) Senkrechter Start** ($m_1 = m_0/19$, `vertical_2a.png`). Raketengleichung $\Delta v = v_e\ln 19 = 12{,}96$ km/s. Bei Brennschluss für $F/m_0 = 10/12/14/16/18$: $t_b = 417/347/298/261/232$ s, $v = 9{,}05/9{,}71/10{,}18/10{,}53/10{,}79$ km/s. Nur bei 18 ist $E \ge 0$ (**entkommt**); bei 10 Aufschlag nach 3,9 h, bei 12 Gipfel 37 000 km.
*Differenz:* radialer Gravitationsverlust $\int g\,dt \approx \Delta v - v_r$, etwas weniger als $g_0 t_b$, weil $g$ mit der Höhe abnimmt: 3,9 km/s bei 10, 2,2 km/s bei 18. Je schneller die Rakete brennt, desto kleiner ist der Verlust. (Gesamtverlust $\Delta v - (|v_b| - |v_0|)$ wie in 2d: 4,4 bzw. 2,6 km/s.)
*Schub zum Entkommen:* $F/m_0 \ge 17{,}13\ \mathrm{m/s^2}$ (Bisektion auf $E(t_b) = 0$).
*Orbit:* nein. Schub und Gravitation zeigen radial, kein Drehmoment, $L = R_E^2\omega_E$ bleibt so klein wie am Start. Damit ist $r_p = L^2/(GM(1+e)) \le L^2/GM = 22$ km, jede gebundene Bahn geht durch die Erde.

**b) Pitch-Programm** ($F/m_0 = 15$, $m_1 = 0{,}12\,m_0$, $t_b = 258$ s, 3 h, `pitch_2b.png`). Einteilung nach dem Zustand bei Brennschluss: crash (Aufschlag beim Brennen), escape ($E \ge 0$), orbit ($r_p > R_E$), sonst falls back.
Osten: $t_p = 100, 125$ crash; $150, 175, 200$ orbit (Periapsis 47, 98, 24 km); $225 \dots 300$ falls back. Westen: $100, 125$ crash, sonst falls back, kein Orbit.
*Bestes $t_p$:* **175 s**. Zu früh flach heißt Aufschlag, zu lange steil heißt zu viel Gravitationsverlust und zu wenig waagerechte Geschwindigkeit.
*Osten:* Die Erdrotation schenkt 465 m/s nach Osten, nach Westen muss die Rakete sie abbauen. Das macht 930 m/s Unterschied, etwa 10 % des $\Delta v = 9{,}33$ km/s (bei 175 s: 8,6 gegen 7,7 km/s bei Brennschluss).

**c) Kreisbahnen** (`circular_2c.png`). Ein Brennvorgang endet auf einer Ellipse durch den Brennschlusspunkt. Für einen Kreis müssten dort Höhe $h$, waagerechte Richtung und $v = \sqrt{GM/(R_E+h)}$ gleichzeitig stimmen, in 258 s kommt die Rakete aber nur auf 100 bis 300 km.
*Strategie:* brennen mit $t_p = 175$ s, bis die Apoapsis $r_a = L^2/(GM(1-e))$ bei $R_E + h$ liegt; abschalten, bis zur Apoapsis gleiten; dort waagerecht nachbrennen, bis $E = -GM/(2r)$.
200 km: 251,1 s + 900 s Gleiten + 1,02 s, Bahn 199,9 bis 200,1 km, **Rest 2,35 %** des Treibstoffs. 500 km: 252,4 s + 2060 s + 1,21 s, **1,74 %**. 1000 km: 253,8 s + 2554 s + 2,18 s, **0,85 %**. Höhere Bahnen brauchen mehr Energie; viel höher reicht der Treibstoff nicht.

**d) Zwei Stufen** (Osten, $t_p = 175$ s, `staging_2d.png`). $\Delta v = v_e\ln 10 = 10{,}13$ km/s für eine Stufe, $v_e(\ln\frac{1}{0{,}28} + \ln\frac{0{,}2}{0{,}02}) = 15{,}73$ km/s für zwei. Eine Stufe erreicht einen Orbit, zwei Stufen entkommen.
*Warum Stufen helfen:* Der leere erste Tank ($0{,}08\,m_0$) wird abgeworfen und nicht weiter beschleunigt. Die Massenverhältnisse multiplizieren sich: $3{,}57 \cdot 10 = 35{,}7$ statt 10.
*Gravitationsverlust* $\int g\sin\gamma\,dt$ (`gravity_loss`): **626 m/s** (eine Stufe) bzw. **620 m/s** (608 + 13, die zweite Stufe brennt waagerecht). Dazu kommen 571 bzw. 572 m/s Lenkverlust. Probe: Geschwindigkeitsgewinn 8935 bzw. 14 541 m/s $= \Delta v$ minus beide Verluste. Gesamtverlust also 1,2 km/s, gegenüber 2,6 bis 4,4 km/s beim senkrechten Start in 2a.
