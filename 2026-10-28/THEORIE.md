# Theorie: Gewöhnliche Differentialgleichungen

Exercise 3 · 28.10.2026

Dieses Dokument erklärt die Theorie zu den Aufgaben 1a bis 2b. Die Verweise
(1a, 1b, …) beziehen sich auf die Angabe in `exercise (1).pdf`, Code in `ode.py`
und `nonlinear.py`, Tests in `test_ode.py`, `test_nonlinear.py` und `test_scripts.py`.

---

## 1. Standardform $\dot x = Jx$ (Aufgabe 1a)

Die Reaktion $A \xrightarrow{k_1} B \xrightarrow{k_2} C$ wird durch drei
Ratengleichungen beschrieben. Jede Reaktion verbraucht ihren Ausgangsstoff
proportional zu dessen Konzentration:

$$
\dot A = -k_1 A, \qquad \dot B = k_1 A - k_2 B, \qquad \dot C = k_2 B.
$$

Mit dem Zustandsvektor $x = (A, B, C)^T$ lässt sich das System in Matrixform
schreiben als $\dot x = Jx$ mit der Jacobimatrix

$$
J = \begin{pmatrix} -k_1 & 0 & 0 \\ k_1 & -k_2 & 0 \\ 0 & k_2 & 0 \end{pmatrix}.
$$

Die Startwerte sind $x(0) = (1, 0, 0)^T$, also $A(0) = 1$, $B(0) = 0$, $C(0) = 0$.
Laut Angabe sind die Raten $k_1 = 1$, $k_2 \in \{1, 10, 100, 1000\}$,
und die Zeit läuft von $t = 0$ bis $t = 10$.

---

## 2. Warum $A + B + C$ konstant bleibt (Aufgabe 1a)

Addiert man die drei Gleichungen, erhält man

$$
\frac{d}{dt}(A + B + C) = -k_1 A + k_1 A - k_2 B + k_2 B = 0.
$$

In Matrixform bedeutet das: jede Spalte von $J$ summiert sich zu 0, also
$(1, 1, 1)\,J = 0$. Damit gilt für jede Lösung

$$
\frac{d}{dt}\,(1, 1, 1)\,x = (1, 1, 1)\,J\,x = 0,
$$

und die Summe $A + B + C$ bleibt konstant. Physikalisch: Stoff wird nur
umgewandelt, nie erzeugt oder vernichtet (Massenerhaltung).

Alle vier numerischen Verfahren erhalten die Summe exakt bis auf Rundungsfehler,
weil jeder Schritt die Form $x_{n+1} = R(\Delta t J)\,x_n$ hat und
$(1, 1, 1)$ ein Linkseigenvektor von $J$ zum Eigenwert 0 ist. Für den
impliziten Euler gilt speziell $(1, 1, 1)(1 - \Delta t J) = (1, 1, 1)$,
also bleibt auch dort die Summe exakt erhalten.

---

## 3. Die vier Verfahren (Aufgabe 1b)

Für $\dot x = f(x) = Jx$ sind die vier Verfahren wie folgt:

### Expliziter Euler

$$
x_{n+1} = x_n + \Delta t\, f(x_n),
$$

Ordnung 1.

### Heun (explizit)

$$
k_1 = f(x_n), \qquad k_2 = f(x_n + \Delta t\,k_1),
$$
$$
x_{n+1} = x_n + \frac{\Delta t}{2}(k_1 + k_2),
$$

Ordnung 2.

### RK4 (Runge-Kutta 4)

$$
\begin{aligned}
k_1 &= f(x_n), \\
k_2 &= f\left(x_n + \frac{\Delta t}{2}k_1\right), \\
k_3 &= f\left(x_n + \frac{\Delta t}{2}k_2\right), \\
k_4 &= f\left(x_n + \Delta t\,k_3\right),
\end{aligned}
$$
$$
x_{n+1} = x_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4),
$$

Ordnung 4.

### Impliziter Euler

$$
x_{n+1} = x_n + \Delta t\,J\,x_{n+1},
$$

also das lineare System $(1 - \Delta t J)\,x_{n+1} = x_n$, gelöst mit
`np.linalg.solve`. Ordnung 1.

### Zusammenhang mit der exakten Lösung

Für lineare Gleichungen ist jeder Schritt eine Matrix $R(\Delta t J)$:

- Euler: $R = 1 + z$
- Heun: $R = 1 + z + z^2/2$
- RK4: $R = 1 + z + z^2/2 + z^3/6 + z^4/24$
- impliziter Euler: $R = (1 - z)^{-1}$

mit $z = \Delta t J$. Das sind die Taylor- bzw. Padé-Näherungen von $e^z$,
denn die exakte Lösung ist $x(t) = e^{tJ}x(0)$.

---

## 4. Wie man prüft, dass die Ergebnisse stimmen (Aufgabe 1b)

### Analytische Lösung

Für $C(t)$ gilt:

$$
C(t) = 1 - \frac{k_2 e^{-k_1 t} - k_1 e^{-k_2 t}}{k_2 - k_1} \quad (k_1 \ne k_2),
$$
$$
C(t) = 1 - e^{-k_1 t}(1 + k_1 t) \quad (k_1 = k_2).
$$

### Erhaltungssatz

In jedem Zeitschritt muss $A + B + C = 1$ gelten (bis auf Rundung).

### Grenzwerte

- $A, B, C \ge 0$ für alle $t \ge 0$
- $C(t) \to 1$ für $t \to \infty$

### Konvergenz

Halbes $\Delta t$ muss den Fehler um den Faktor $2^p$ verkleinern, wobei
$p$ die Ordnung des Verfahrens ist (1 für Euler/implizit Euler, 2 für Heun,
4 für RK4).

### Ein Schritt gegen die Reihe

Ein einzelner Schritt kann exakt gegen die Taylor-Reihe von $R(\Delta t J)\,x$
geprüft werden, wie in `test_ode.py` mit `test_single_steps_match_series`.

### Vergleich der Verfahren

Bei kleinem $\Delta t$ sollten alle Verfahren übereinstimmen, da sie dann
alle gegen die exakte Lösung konvergieren.

---

## 5. Genauigkeit in Abhängigkeit von $\Delta t$ (Aufgabe 1c)

Gemessen wird der maximale Fehler von $C(t)$ auf $[0, 10]$ für $k_1 = 1$, $k_2 = 10$
(`max_error` in `ode.py`, Plot mit `plot_1c.py` → `accuracy_1c.png`):

$$
e(\Delta t) = \max_n |C_n - C(t_n)|.
$$

### Erwartung

Ein Verfahren der Ordnung $p$ macht pro Schritt einen lokalen Fehler
$\mathcal O(\Delta t^{p+1})$. Über $N = T/\Delta t$ Schritte summiert sich das zum
globalen Fehler $e \approx c\,\Delta t^p$. Im log-log-Plot entsteht eine Gerade:

$$
\log e = p \log \Delta t + \log c,
$$

mit der Steigung $p$. Erwartet werden also 1 (Euler, impliziter Euler), 2 (Heun) und 4 (RK4).
Die Steigung bestimmt `order` per linearer Regression von $\log e$ gegen $\log \Delta t$.

### Messung

$\Delta t = 0{,}1 / 2^k$ mit $k = 0, \dots, 8$; Fit über die kleinsten $\Delta t$
(bei RK4 über $\Delta t = 0{,}025 \dots 0{,}003$):

- expliziter Euler: $p = 1{,}00$
- Heun: $p = 2{,}01$
- RK4: $p = 4{,}09$
- impliziter Euler: $p = 1{,}00$

### Beobachtungen

- Bei $\Delta t = 0{,}1$ liegen expliziter Euler und Heun noch über ihrer Geraden: $k_2 \Delta t = 1$ ist
  kein kleiner Parameter, die Asymptotik $e \sim \Delta t^p$ gilt erst darunter.
- Expliziter und impliziter Euler haben bei kleinem $\Delta t$ fast denselben Fehler
  (etwa $8 \cdot 10^{-5}$ bei $\Delta t \approx 4 \cdot 10^{-4}$), mit entgegengesetztem
  Vorzeichen: Der lokale Fehler ist $\mp\tfrac12 \Delta t^2 J^2 x$ (Taylor von $1+z$ bzw. $(1-z)^{-1}$ gegen $e^z$).
- RK4 erreicht bei $\Delta t \approx 4 \cdot 10^{-4}$ bereits $8 \cdot 10^{-14}$, also fast die
  Rundungsgrenze. Bei noch kleinerem $\Delta t$ würde der Fehler nicht weiter fallen.
- Für gleiche Genauigkeit von $10^{-6}$ braucht RK4 etwa $\Delta t = 0{,}02$, Heun
  etwa $0{,}0012$ und Euler etwa $5 \cdot 10^{-6}$. Ein höherer Aufwand pro Schritt lohnt sich deutlich.

---

## 6. Kritischer Zeitschritt (Aufgabe 1d)

Für die Testgleichung $\dot x = \lambda x$ ist ein Zeitschritt $x_{n+1} = R(\lambda\Delta t)\,x_n$.
Die Lösung ist stabil genau dann, wenn $|R(z)| \le 1$ für $z = \lambda\Delta t$.
Die Eigenwerte der Jacobimatrix $J$ sind $0, -k_1, -k_2$, deshalb gilt
$\Delta t_\text{krit} = c / \max(k_1, k_2)$.

Die Stabilitätsfunktionen $R(z)$ sind:

- Euler: $R(z) = 1 + z$
- Heun: $R(z) = 1 + z + z^2/2$
- RK4: $R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$

Gemessen mit `plot_1d.py` (Bisektion in `critical_dt`) ergeben sich folgende Werte für $c$:

- expliziter Euler: $c = 2$
- Heun: $c = 2$
- RK4: $c = 2{,}785$

Für alle fünf Paare $(k_1, k_2) = (1,1), (1,10), (1,100), (10,1), (50,10)$ ist $c$ gleich.
Beispiel $k_1 = 1, k_2 = 10$: $\Delta t_\text{krit} = 0{,}2$ (Euler, Heun) bzw. $0{,}2785$ (RK4).

Knapp über der Grenze ($1{,}05\,\Delta t_\text{krit}$): die schnelle Mode wächst mit
wechselndem Vorzeichen ($R < -1$), die Lösung oszilliert von Schritt zu Schritt
mit wachsender Amplitude. Max. Fehler bei $0{,}95$ / $1{,}05\,\Delta t_\text{krit}$:

- Euler: $0{,}118$ / $10{,}8$
- Heun: $0{,}0865$ / $13{,}4$
- RK4: $0{,}0821$ / $130$

---

## 7. Eigenwerte und Stabilitätsgebiete (Aufgabe 1e)

Für $k_1 = 1, k_2 = 10$ (aus `plot_1e.py`) sind die Eigenwerte von $J$:

- $\lambda = 0$ mit Eigenvektor $(0, 0, 1)$: Endzustand, alles ist $C$
- $\lambda = -1$ mit $(0{,}667;\ 0{,}074;\ -0{,}741)$: langsamer Zerfall von $A$
- $\lambda = -10$ mit $(0;\ 0{,}707;\ -0{,}707)$: schneller Übergang $B \to C$

Jede Lösung ist eine Summe der Eigenmoden, jede Mode wird pro Schritt mit
$R(\lambda\Delta t)$ multipliziert.

Mit wachsendem $\Delta t$ wandert $z = \lambda\Delta t$ auf der negativen reellen
Achse nach links; zuerst verlässt die Mode $\lambda = -10$ das Stabilitätsgebiet.
Gemessen bei $1{,}05\,\Delta t_\text{krit}$:

- Euler: $|R| = 1{,}100$
- Heun: $|R| = 1{,}105$
- RK4: $|R| = 1{,}231$

Bilder: `stability_1e.png`, `modes_1e.png`.

Impliziter Euler: $R(z) = 1/(1 - z)$, für $\operatorname{Re} z \le 0$ gilt
$|1 - z| \ge 1$, also $|R| \le 1$ in der ganzen linken Halbebene (A-stabil),
daher kein $\Delta t_\text{krit}$.

---

## 8. Steifheit (Aufgabe 1f)

Für $k_2 = 1000$, Ziel max. Fehler $\approx 10^{-2}$ (`plot_1f.py`,
`steps_for_tolerance`):

- Expliziter Euler: $n = 4999$ Schritte, $\Delta t = 0{,}002$ = genau
  $\Delta t_\text{krit} = 2/k_2$; die Stabilität begrenzt $\Delta t$, nicht
  die Genauigkeit, obwohl die schnelle Mode nach $t \approx 5/k_2$ längst
  abgeklungen ist.
- Impliziter Euler: $n = 180$, $\Delta t = 0{,}0556$; hier begrenzt nur
  die Genauigkeit (Auflösung der langsamen Mode $e^{-t}$ und des
  Anfangsübergangs).

Faktor etwa 28 weniger Schritte, jeder Schritt kostet ein lineares 3×3-System.

Definition: steif heißt, die Zeitskalen ($1/k_1$ gegen $1/k_2$) liegen weit
auseinander, und die schnellste erzwingt bei expliziten Verfahren winzige Schritte.
Bild: `stiff_1f.png`.

---

## 9. Robertson-Modell (Aufgabe 2a)

$$
\dot A = -k_1 A + k_3 BC, \qquad
\dot B = k_1 A - k_2 B^2 - k_3 BC, \qquad
\dot C = k_2 B^2
$$

mit $k_1 = 0{,}04$, $k_2 = 3\cdot 10^7$, $k_3 = 10^4$ und $(A, B, C)(0) = (1, 0, 0)$.

### Impliziter Euler mit Newton

Pro Schritt ist $x_{n+1} = y$ die Nullstelle von

$$
g(y) = y - x_n - \Delta t\, f(y).
$$

Newton: $(I - \Delta t\, J_f(y))\,\delta = -g(y)$, $y \leftarrow y + \delta$, Startwert $y = x_n$,
Abbruch bei $\max|\delta| \le 10^{-12}\,(1 + \max|y|)$, sonst nach 50 Iterationen ein Fehler
(`implicit_euler_f`). Die Jacobimatrix $J_f$ steht analytisch in `robertson_jac`.
Für lineares $f = Jx$ ist $g$ linear, Newton ist nach einem Schritt fertig und liefert genau
$(I - \Delta t J)\,x_{n+1} = x_n$ aus Teil 1.

Das Zeitgitter ist logarithmisch, $t = 0, 10^{-6}, \dots, 10^5$ (`integrate_grid`), weil die
Dynamik auf allen Zeitskalen passiert: B baut sich in $t \sim 10^{-3}$ auf, A zerfällt über $t \sim 10^2 \dots 10^5$.

### Ergebnis

`plot_2a.py`, Bild `robertson_2a.png`:

- bei $t = 10^5$ mit 1600 Gitterpunkten: $A = 0{,}01808$, $B = 7{,}36\cdot10^{-8}$, $C = 0{,}98192$
- mit 400 Gitterpunkten: $A = 0{,}01871$, also Unterschied $6\cdot 10^{-4}$, wie bei einem Verfahren erster Ordnung erwartet
- $B$ erreicht höchstens etwa $3{,}6\cdot 10^{-5}$ und bleibt dort bis $t \approx 1$
- $A + B + C = 1$ bleibt bis auf $2\cdot10^{-15}$ erhalten: die Summe der rechten Seiten ist null, und der implizite Euler erhält lineare Invarianten exakt (bis auf die Newton-Toleranz)

### Warum explizite Verfahren versagen

Die Jacobimatrix hat in B-Richtung den Eigenwert etwa $-2k_2 B - k_3 C$. Mit $B \approx 3{,}6\cdot10^{-5}$
ist das etwa $-2000$, der explizite Euler braucht also $\Delta t < 2/2000 = 10^{-3}$.
Gemessen: mit $\Delta t = 10^{-3}$ explodiert er bei $t = 9{,}35$. Für $t$ bis $10^5$ bräuchte er
über $10^8$ Schritte, obwohl sich die Lösung dann nur noch langsam ändert. Wie in Abschnitt 8
begrenzt die Stabilität den Schritt, nicht die Genauigkeit.

### Code aus Teil 1 wiederverwenden?

Nur teilweise. Die Verfahren in `ode.py` brauchen eine feste Matrix $J$. Die expliziten Verfahren gehen
mit $f(x)$ statt $Jx$ direkt über (`euler_f`, `heun_f`, `rk4_f`). Der implizite Euler braucht statt
eines linearen Systems eine Newton-Iteration mit $J_f(y)$, und das gleichmäßige Gitter von `integrate`
reicht für $t$ bis $10^5$ nicht. Für lineares $f$ liefern die neuen Funktionen dasselbe wie die alten
(Test `test_linear_f_matches_matrix_solvers`).

---

## 10. Lorenz-System (Aufgabe 2b)

$$
\dot x = \sigma (y - x), \qquad
\dot y = x(\rho - z) - y, \qquad
\dot z = xy - \beta z
$$

mit $\sigma = 10$, $\rho = 28$, $\beta = 8/3$ und $t \in [0, 40]$.

### Ergebnis

`plot_2b.py`, Bild `lorenz_2b.png`, RK4 von $(1, 1, 1)$ und $(1, 1, 1 + 10^{-8})$:

- Die Bahnen laufen auf dem Attraktor ($x$ zwischen $-17$ und $20$, $z$ bis $48$) lange gemeinsam.
- Der Abstand wächst im Mittel exponentiell, im semilog-Plot eine Gerade. Ein Fit von $\ln d$ über
  $t \approx 10 \dots 27$ ergibt die Rate $0{,}84$ (der Literaturwert des größten Lyapunov-Exponenten ist $0{,}91$).
- Der Abstand überschreitet 1 bei $t = 28{,}88$, mit $\Delta t = 10^{-3}$ und mit $\Delta t = 5\cdot10^{-4}$ gleich.
  Danach sind die beiden Bahnen nicht mehr korreliert.
- Abschätzung: $10^{-8}\, e^{0{,}84\, t} = 1$ ergibt $t \approx 22$, die richtige Größenordnung. Genauer geht es nicht,
  weil der Abstand nicht gleichmäßig wächst, sondern um die Gerade schwankt (Plot unten in `lorenz_2b.png`).

### Bedeutung

Das System ist chaotisch: kleine Störungen wachsen exponentiell. Auch Rundungs- und Diskretisierungsfehler
sind solche Störungen. Zusätzlich gemessen: dieselbe Startbahn mit $\Delta t = 10^{-3}$ und $5\cdot 10^{-4}$
trennt sich ebenfalls bei $t \approx 28{,}9$, expliziter Euler gegen RK4 schon bei $t \approx 5{,}1$,
Heun gegen RK4 bei $t \approx 16{,}9$. Die einzelne Bahn ist also nur bis etwa $t \approx 28$ verlässlich,
der Attraktor (Form, Wertebereich) dagegen bei allen Verfahren derselbe.

---

## 11. Welchem Verfahren vertrauen ohne exakte Lösung? (Aufgabe 2)

- **Konvergenz prüfen:** dieselbe Rechnung mit halbem $\Delta t$ oder feinerem Gitter wiederholen
  (Robertson 400 gegen 1600 Punkte, Lorenz $10^{-3}$ gegen $5\cdot10^{-4}$). Ändert sich das Ergebnis
  kaum, ist der Diskretisierungsfehler klein.
- **Erhaltungsgrößen prüfen:** $A + B + C = 1$ beim Robertson-Modell.
- **Verfahren vergleichen:** höhere gegen niedrigere Ordnung; stimmen sie überein, sind beide genau genug.
- **Stabilität beachten:** bei steifen Problemen dem impliziten Verfahren vertrauen; ein explizites Ergebnis
  mit zu großem $\Delta t$ ist nicht ungenau, sondern unbrauchbar (Blow-up).
- **Bei Chaos:** der einzelnen Bahn nur bis zur Trennzeit vertrauen, danach nur Statistik und Attraktor.
