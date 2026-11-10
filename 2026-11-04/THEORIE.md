# Theorie zu Exercise 4: Symplektische Integratoren für Orbits

Begleittext zu `orbit.py`, den Plot-Skripten und den Tests. Hier steht, woher jede Formel im Code kommt.

## 1a) Kreisbahn des Mondes

### Modell

Die Erde sitzt fest im Ursprung, der Mond bewegt sich in der xy-Ebene. Newtons Gravitationsgesetz für die Kraft auf den Mond (Masse $m$):

$$m\,\ddot{\mathbf r} = -\frac{G M m}{|\mathbf r|^2}\,\frac{\mathbf r}{|\mathbf r|}$$

Der Faktor $\mathbf r/|\mathbf r|$ ist der Einheitsvektor von der Erde zum Mond, das Minus macht die Kraft anziehend. Teilt man durch $m$, fällt die Mondmasse heraus:

$$\ddot{\mathbf r} = -\frac{GM}{|\mathbf r|^3}\,\mathbf r$$

Im Code steht das Produkt als eine Konstante `GM = G * M_EARTH` $= 3{,}986\cdot10^{14}\ \mathrm{m^3/s^2}$, weil in allen Formeln nur das Produkt vorkommt.

### Startwerte für eine Kreisbahn

Auf einer Kreisbahn mit Radius $r$ und konstanter Geschwindigkeit $v$ zeigt die Beschleunigung zum Mittelpunkt und hat den Betrag der Zentripetalbeschleunigung $v^2/r$. Diese Beschleunigung muss genau von der Gravitation kommen:

$$\frac{v^2}{r} = \frac{GM}{r^2} \quad\Longrightarrow\quad v = \sqrt{\frac{GM}{r}}$$

Ist $v$ größer, fliegt der Mond auf einer Ellipse nach außen, ist es kleiner, fällt er nach innen. Nur dieser eine Wert ergibt einen Kreis.

Wir legen den Start auf die x-Achse. Die Geschwindigkeit muss dann senkrecht zum Ortsvektor stehen, also in y-Richtung zeigen:

$$\mathbf r(0) = (r_M,\ 0), \qquad \mathbf v(0) = \Big(0,\ \sqrt{GM/r_M}\Big)$$

Das positive Vorzeichen bedeutet einen Umlauf gegen den Uhrzeigersinn. Mit $r_M = 3{,}844\cdot10^8$ m ergibt sich $v = 1018{,}3$ m/s. Das gibt `circular_orbit(r)` zurück.

### Umlaufzeit

Ein Umlauf ist der Kreisumfang geteilt durch die Geschwindigkeit:

$$T = \frac{2\pi r}{v} = \frac{2\pi r}{\sqrt{GM/r}} = 2\pi\sqrt{\frac{r^3}{GM}}$$

Das ist das dritte Keplersche Gesetz $T^2 \propto r^3$ für den Kreis. Mit den Zahlen des Blatts:

$$T = 2{,}3719\cdot10^6\ \mathrm s = 27{,}453\ \mathrm d$$

Das berechnet `period(r)`.

### Warum nicht 27,32 d?

Der gemessene siderische Monat (ein Umlauf relativ zu den Fixsternen) dauert 27,32 d, unser Modell ergibt 0,13 d mehr. Die wichtigste Ursache ist die Annahme, dass die Erde fest im Ursprung sitzt. In Wirklichkeit kreisen Erde und Mond um ihren gemeinsamen Schwerpunkt. Im Zweikörperproblem bewegt sich der Abstandsvektor $\mathbf r = \mathbf r_\text{Mond} - \mathbf r_\text{Erde}$ nach

$$\ddot{\mathbf r} = -\frac{G(M+m)}{|\mathbf r|^3}\,\mathbf r$$

Statt $GM$ steht also $G(M+m)$, und damit wird

$$T = 2\pi\sqrt{\frac{r^3}{G(M+m)}} = \frac{T_\text{fest}}{\sqrt{1 + m/M}}$$

Mit $m/M = 7{,}342\cdot10^{22}/5{,}972\cdot10^{24} = 0{,}0123$ wird $T = 27{,}29$ d. Damit ist fast die ganze Abweichung erklärt. Den Rest von etwa 0,03 d erklären zwei Effekte: Die echte Bahn ist eine Ellipse (Exzentrizität etwa 0,055), und 384 400 km ist nur die mittlere Entfernung. Außerdem stört die Sonne die Mondbahn. Die Mondmasse steht nicht auf dem Blatt, sie dient nur diesem Vergleich.

### Was die Tests prüfen

- `test_circular_orbit_starts_on_x_axis_with_tangential_speed`: Der Start liegt bei $(r_M, 0)$, und die Geschwindigkeit hat keine x-Komponente, steht also senkrecht auf $\mathbf r$. Außerdem gilt $v^2/r = GM/r^2$, die Kreisbahnbedingung von oben. Der Test prüft damit die Physik selbst und vergleicht nicht nur mit einer abgeschriebenen Zahl.
- `test_period_matches_kepler`: `period` stimmt mit Umfang durch Geschwindigkeit überein, also $2\pi r/v$ mit dem $v$ aus `circular_orbit`. So passen beide Funktionen zusammen. Dazu kommt ein Zahlenwert: $T \approx 27{,}45$ d auf 0,01 d genau, als Schutz gegen einen Fehler in den Konstanten oder den Einheiten.

## 1b) Sechs Verfahren für 10 Umläufe

### Vom System zweiter Ordnung zu erster Ordnung

Alle Standardverfahren lösen $\dot{\mathbf y} = \mathbf f(\mathbf y)$. Wir schreiben Ort und Geschwindigkeit in einen Zustand $\mathbf y = (x, y, v_x, v_y)$:

$$\dot{\mathbf y} = \mathbf f(\mathbf y) = \begin{pmatrix} \mathbf v \\ \mathbf a(\mathbf r) \end{pmatrix}, \qquad \mathbf a(\mathbf r) = -\frac{GM}{|\mathbf r|^3}\,\mathbf r$$

Im Code: `acceleration(r)` und `rhs(y)`. Jeder Schritt hat die Form `step(y, dt) -> y_neu`, `integrate(step, y0, dt, n)` wendet ihn $n$-mal an.

### Die Verfahren

- **Explizites Euler** (Ordnung 1): $\mathbf y_{n+1} = \mathbf y_n + \Delta t\,\mathbf f(\mathbf y_n)$. Die Steigung wird nur am Anfang des Schritts genommen.
- **Heun** (Ordnung 2): erst ein Euler-Schritt als Probe, dann der Mittelwert der Steigungen am Anfang und an der Probe: $\mathbf k_1 = \mathbf f(\mathbf y_n)$, $\mathbf k_2 = \mathbf f(\mathbf y_n + \Delta t\,\mathbf k_1)$, $\mathbf y_{n+1} = \mathbf y_n + \tfrac{\Delta t}{2}(\mathbf k_1 + \mathbf k_2)$.
- **RK4** (Ordnung 4): vier Steigungen, gewichtet $1:2:2:1$, wie in Exercise 3.
- **Implizites Euler** (Ordnung 1): die Steigung am Ende des Schritts, $\mathbf y_{n+1} = \mathbf y_n + \Delta t\,\mathbf f(\mathbf y_{n+1})$. Das ist eine nichtlineare Gleichung für $\mathbf y_{n+1}$, siehe Newton unten.
- **Symplektisches Euler** (Ordnung 1): Ort und Geschwindigkeit werden nacheinander aktualisiert, und der zweite Teilschritt benutzt schon den neuen Wert:
$$\mathbf v_{n+1} = \mathbf v_n + \Delta t\,\mathbf a(\mathbf r_n), \qquad \mathbf r_{n+1} = \mathbf r_n + \Delta t\,\mathbf v_{n+1}$$
Der einzige Unterschied zum expliziten Euler ist $\mathbf v_{n+1}$ statt $\mathbf v_n$ in der Ortsgleichung. Das reicht, damit die Abbildung $(\mathbf r_n, \mathbf v_n) \mapsto (\mathbf r_{n+1}, \mathbf v_{n+1})$ das Phasenraumvolumen erhält (Determinante der Jacobi-Matrix gleich 1), also symplektisch ist.
- **Velocity Verlet** (Ordnung 2): halber Kick, ganzer Drift, halber Kick:
$$\mathbf v_{n+1/2} = \mathbf v_n + \tfrac{\Delta t}{2}\mathbf a(\mathbf r_n), \quad \mathbf r_{n+1} = \mathbf r_n + \Delta t\,\mathbf v_{n+1/2}, \quad \mathbf v_{n+1} = \mathbf v_{n+1/2} + \tfrac{\Delta t}{2}\mathbf a(\mathbf r_{n+1})$$
Symmetrisch in der Zeit, deshalb Ordnung 2 und ebenfalls symplektisch. Pro Schritt nur eine neue Kraftauswertung, weil $\mathbf a(\mathbf r_{n+1})$ im nächsten Schritt wiederverwendet werden kann.

### Newton für das implizite Euler

Gesucht ist die Nullstelle von $\mathbf g(\mathbf z) = \mathbf z - \mathbf y_n - \Delta t\,\mathbf f(\mathbf z)$. Newton linearisiert $\mathbf g$ um die aktuelle Näherung:

$$\big(I - \Delta t\,J(\mathbf z)\big)\,\boldsymbol\delta = -\mathbf g(\mathbf z), \qquad \mathbf z \leftarrow \mathbf z + \boldsymbol\delta$$

Die Jacobi-Matrix von $\mathbf f$ hat Blockform, weil $\dot{\mathbf r} = \mathbf v$ linear ist:

$$J = \begin{pmatrix} 0 & I \\ A & 0 \end{pmatrix}, \qquad A = \frac{\partial \mathbf a}{\partial \mathbf r} = -\frac{GM}{|\mathbf r|^3}\left(I - 3\,\frac{\mathbf r\,\mathbf r^T}{|\mathbf r|^2}\right)$$

$A$ folgt aus der Produktregel für $-GM\,\mathbf r\,|\mathbf r|^{-3}$: Ableiten von $\mathbf r$ gibt $I\,|\mathbf r|^{-3}$, Ableiten von $|\mathbf r|^{-3}$ gibt $-3|\mathbf r|^{-5}\,\mathbf r^T$. Im Code: `jacobian(y)`, der Test vergleicht sie mit zentralen Differenzen.

**Startwert:** der explizite Euler-Schritt $\mathbf z_0 = \mathbf y_n + \Delta t\,\mathbf f(\mathbf y_n)$. Explizites und implizites Euler unterscheiden sich nur um $O(\Delta t^2)$, der Startwert liegt also schon sehr nah an der Lösung, und Newton konvergiert quadratisch. **Ergebnis: 3 Iterationen pro Schritt** (bis die Änderung unter $10^{-14}$ relativ liegt), später auf der schrumpfenden Bahn 4 bis 6.

### Vorhersage: wer spiralt wohin?

Eine Kreisbahn ist lokal eine Schwingung mit $\omega = 2\pi/T$. Für das Testproblem $\dot z = i\omega z$ (reine Rotation, $|z|$ konstant) multipliziert jeder Schritt mit dem Verstärkungsfaktor $R(i\omega\Delta t)$, siehe Stabilitätsfunktion aus Exercise 3. Mit $x = \omega\Delta t = 2\pi \cdot 6\,\mathrm h / 27{,}45\,\mathrm d = 0{,}057$:

- Explizites Euler: $|R|^2 = |1 + ix|^2 = 1 + x^2 > 1$, also **nach außen**, $+0{,}16\,\%$ pro Schritt.
- Heun: $|R|^2 = |1 + ix - x^2/2|^2 = 1 + x^4/4 > 1$, also **nach außen**, aber viel schwächer.
- Implizites Euler: $|R|^2 = 1/(1 + x^2) < 1$, also **nach innen**, $-0{,}16\,\%$ pro Schritt.
- RK4: $|R|^2 = 1 - x^6/72 + x^8/576 < 1$, also **nach innen**, aber kaum sichtbar.
- Symplektisches Euler und Verlet: $|\det| = 1$, die Fläche im Phasenraum bleibt erhalten, **keine Spirale**. Der Radius schwankt, driftet aber nicht.

Beim Kepler-Problem kommt eine Verstärkung dazu: Fällt der Mond nach innen, wird $\omega$ größer und damit auch $x$, und der Fehler pro Schritt wächst. Für das implizite Euler ist das eine Lawine.

### Ergebnis (`plot_1b.py`, `orbits_1b.png`)

1098 Schritte für 10 Umläufe, Endradius relativ zu $r_M$:

- Explizites Euler: 2,757. Die Bahn spiralt weit nach außen.
- Implizites Euler: 0,304 nach 58 Schritten, also vor einem halben Umlauf. Dann hat die implizite Gleichung keine nahe Lösung mehr, Newton konvergiert nicht (auch `scipy.optimize.fsolve` nicht). `integrate` bricht dort ab.
- Heun: 1,006, langsam nach außen.
- Symplektisches Euler: 1,003. Die Bahn ist eine leicht gestauchte Ellipse, die aber stabil bleibt.
- Velocity Verlet: 1,000000.
- RK4: 0,999999, die Drift nach innen sieht man erst in der sechsten Stelle.

Die Vorhersage stimmt für alle sechs Verfahren.

### Warum manche Verfahren stabil bleiben und andere nicht

Jedes Verfahren macht pro Schritt einen kleinen Fehler. Entscheidend ist nicht, wie groß dieser Fehler ist, sondern ob er **immer in dieselbe Richtung** zeigt. Wenn ja, summiert er sich Umlauf für Umlauf zu einer Spirale. Wenn er dagegen mal nach außen und mal nach innen zeigt, hebt er sich über die Zeit auf.

**Geometrisch: Wohin zeigt der Schritt?**

- Explizites Euler fliegt ein Stück entlang der Tangente am Startpunkt. Eine Tangente liegt immer außerhalb des Kreises, also landet jeder Schritt etwas weiter außen, um den Faktor $\sqrt{1 + (\omega\Delta t)^2}$. Bei 1098 Schritten wird daraus eine Spirale nach außen.
- Implizites Euler nimmt die Geschwindigkeit am Ende des Schritts. Diese hat sich schon zur Erde hin gedreht, deshalb landet der Schritt innen, und das jedes Mal.
- Symplektisches Euler gibt zuerst einen Kick, also eine neue Geschwindigkeit, die schon etwas nach innen gedreht ist, und fliegt dann mit ihr. Der Kick allein würde nach innen ziehen, das Fliegen entlang der Tangente nach außen. Beide Fehler sind gleich groß und heben sich auf. Übrig bleibt eine leicht verformte Ellipse, aber keine Drift.

**Rechnerisch: Was macht ein Schritt mit der Fläche im Phasenraum?**

Am harmonischen Oszillator $\dot x = v$, $\dot v = -\omega^2 x$, der lokal einer Kreisbahn entspricht, ist jeder Schritt eine Matrix, die $(x, v)$ auf den nächsten Zustand abbildet ($h = \Delta t$):

$$\text{Euler: } \begin{pmatrix} 1 & h \\ -\omega^2 h & 1 \end{pmatrix}, \qquad \det = 1 + \omega^2 h^2 > 1$$

$$\text{implizites Euler: } \begin{pmatrix} 1 & -h \\ \omega^2 h & 1 \end{pmatrix}^{-1}, \qquad \det = \frac{1}{1 + \omega^2 h^2} < 1$$

$$\text{symplektisches Euler: } \begin{pmatrix} 1 - \omega^2 h^2 & h \\ -\omega^2 h & 1 \end{pmatrix}, \qquad \det = 1$$

Die Matrix des symplektischen Euler entsteht aus $v_{n+1} = v_n - \omega^2 h\,x_n$ und $x_{n+1} = x_n + h\,v_{n+1} = (1 - \omega^2 h^2)\,x_n + h\,v_n$.

Die Determinante gibt an, wie stark ein Schritt eine Fläche im Phasenraum vergrößert oder verkleinert. Die exakte Bewegung erhält die Fläche (Satz von Liouville).

- Bei $\det > 1$ wird der Kreis bei jedem Schritt größer, die Bahn spiralt nach außen.
- Bei $\det < 1$ wird er kleiner, die Bahn spiralt nach innen.
- Bei $\det = 1$ kann keine Spirale entstehen, weil eine Spirale die Fläche ändern würde. Genau das bedeutet **symplektisch**.

**Warum die Ordnung allein nicht reicht**

RK4 hat Ordnung 4 und macht pro Schritt einen winzigen Fehler. Er zeigt aber immer nach innen, denn $|R|^2 = 1 - x^6/72 + \dots$. Über genügend viele Umläufe driftet RK4 deshalb doch. Velocity Verlet hat nur Ordnung 2, aber $\det = 1$: Sein Fehler schwankt und wächst nicht an.

Dahinter steht ein Satz der Rückwärtsfehleranalyse: Ein symplektisches Verfahren erhält eine leicht veränderte „Schatten-Energie“ $\tilde E = E + O(\Delta t^p)$ über sehr lange Zeiten fast exakt. Deshalb bleibt der Energiefehler beschränkt und wächst nicht mit der Zeit.

- **Nicht symplektische Verfahren** (Euler, implizites Euler, Heun, RK4): Der Energiefehler wächst mit der Zeit an, eine höhere Ordnung macht ihn nur langsamer.
- **Symplektische Verfahren** (symplektisches Euler, Verlet): Der Energiefehler bleibt beschränkt und schwankt nur.

Das zeigen die Energieplots in 1c. Es erklärt auch, warum Heun mit Ordnung 2 auf lange Sicht schlechter abschneidet als das symplektische Euler mit Ordnung 1.

### Was die Tests prüfen

- `test_acceleration_…`, `test_rhs_…`: die Bewegungsgleichung und ihre Form als System erster Ordnung.
- `test_jacobian_matches_finite_differences`: die hergeleitete Matrix $J$ gegen zentrale Differenzen.
- `test_newton_solves_implicit_equation_in_few_iterations`: Das Ergebnis erfüllt $\mathbf z = \mathbf y + \Delta t\,\mathbf f(\mathbf z)$ auf $10^{-12}$ genau, mit 1 bis 5 Iterationen.
- `test_integrate_…`: Form der Ausgabe und der Abbruch, wenn das implizite Euler kollabiert.
- `test_all_methods_return_to_start_after_one_orbit_with_small_dt`: Mit 20 000 Schritten pro Umlauf schließt jedes Verfahren den Kreis auf 5 % genau. Das ist eine grobe Konvergenzprüfung. Bei 2000 Schritten liegen die Euler-Verfahren noch 0,19 $r_M$ daneben.
- `test_spiral_direction_after_ten_orbits`: die Vorhersage von oben. „Außen“ und „innen“ werden über das Vorzeichen der Energieänderung geprüft, weil die Drift bei RK4 nur $10^{-7}$ beträgt. Bei den symplektischen Verfahren gilt: Größter und kleinster Radius im letzten Umlauf weichen weniger als 1 % von denen im ersten ab.
- `test_scripts.py`: `plot_1b.py` läuft durch, speichert das Bild und gibt Newton-Iterationen und Endradien aus.

## 1c) Energie und Drehimpuls

### Erhaltung für die exakte Bewegung

Beide Größen sind pro Masseneinheit definiert:

$$E = \tfrac12|\mathbf v|^2 - \frac{GM}{|\mathbf r|}, \qquad \mathbf L = \mathbf r \times \mathbf v$$

**Energie.** Ableiten nach der Zeit, mit $\dot{\mathbf r} = \mathbf v$ und $\frac{d}{dt}|\mathbf r| = \frac{\mathbf r\cdot\dot{\mathbf r}}{|\mathbf r|}$:

$$\dot E = \mathbf v\cdot\dot{\mathbf v} + \frac{GM}{|\mathbf r|^2}\,\frac{\mathbf r\cdot\mathbf v}{|\mathbf r|} = \mathbf v\cdot\Big(-\frac{GM}{|\mathbf r|^3}\mathbf r\Big) + \frac{GM}{|\mathbf r|^3}\,\mathbf r\cdot\mathbf v = 0$$

Die kinetische Energie ändert sich genau um die Arbeit der Gravitation, die potentielle um das Negative davon.

**Drehimpuls.** Mit der Produktregel:

$$\dot{\mathbf L} = \dot{\mathbf r}\times\mathbf v + \mathbf r\times\dot{\mathbf v} = \underbrace{\mathbf v\times\mathbf v}_{0} + \mathbf r\times\Big(-\frac{GM}{|\mathbf r|^3}\mathbf r\Big) = 0$$

Der zweite Term verschwindet, weil die Kraft parallel zu $\mathbf r$ ist, also eine **Zentralkraft**. Das ist der entscheidende Punkt für die numerischen Verfahren. In der Ebene hat $\mathbf L$ nur eine z-Komponente $L = x v_y - y v_x$ (`angular_momentum`).

Für den Kreis gilt $E = -GM/(2r_M)$ und $L = r_M v = \sqrt{GM r_M}$. Das prüft der erste Test.

### Welche Verfahren erhalten L exakt?

Nur das **symplektische Euler** und **Velocity Verlet**. Beide bestehen nur aus zwei Sorten von Teilschritten, und keiner davon ändert $\mathbf L$:

- **Kick** ($\mathbf r$ fest, $\mathbf v \leftarrow \mathbf v + c\,\mathbf a(\mathbf r)$): $\mathbf r\times(\mathbf v + c\,\mathbf a) = \mathbf r\times\mathbf v + c\,\underbrace{\mathbf r\times\mathbf a}_{0}$, weil $\mathbf a \parallel \mathbf r$.
- **Drift** ($\mathbf v$ fest, $\mathbf r \leftarrow \mathbf r + h\,\mathbf v$): $(\mathbf r + h\mathbf v)\times\mathbf v = \mathbf r\times\mathbf v + h\,\underbrace{\mathbf v\times\mathbf v}_{0}$.

Jeder Teilschritt erhält $\mathbf L$ exakt, also auch ihre Hintereinanderausführung. Übrig bleiben nur Rundungsfehler, gemessen $|\Delta L/L| \le 2\cdot10^{-14}$ über 100 Umläufe.

Beim **expliziten Euler** werden Ort und Geschwindigkeit gleichzeitig mit den alten Werten verschoben:

$$\mathbf L_{n+1} = (\mathbf r + h\mathbf v)\times(\mathbf v + h\mathbf a) = \mathbf L_n + h\,\mathbf r\times\mathbf a + h\,\mathbf v\times\mathbf v + h^2\,\mathbf v\times\mathbf a = \mathbf L_n + h^2\,\mathbf v\times\mathbf a$$

$\mathbf v\times\mathbf a \ne 0$, weil die Geschwindigkeit nicht parallel zur Kraft steht, also ändert sich $L$ in jedem Schritt um $O(h^2)$. Beim impliziten Euler gilt dasselbe mit den neuen Werten, also $\mathbf L_n = \mathbf L_{n+1} + h^2\,\mathbf v_{n+1}\times\mathbf a_{n+1}$. Heun und RK4 mischen Steigungen an verschiedenen Orten, auch dort bleibt ein Rest.

### Ergebnis (`plot_1c.py`, `conservation_1c.png`)

100 Umläufe, 10 981 Schritte mit $\Delta t = 6$ h, maximale relative Fehler:

- Explizites Euler: $|\Delta E/E| = 0{,}84$, $|\Delta L/L| = 1{,}4$. Die Energie geht gegen 0, der Mond ist fast entkommen.
- Implizites Euler: nach 58 Schritten abgestürzt, $|\Delta E/E| = 2{,}2$.
- Heun: $|\Delta E/E| = 4{,}8\cdot10^{-2}$, wachsend; $|\Delta L/L| = 2{,}5\cdot10^{-2}$.
- Symplektisches Euler: $|\Delta E/E| = 3{,}3\cdot10^{-3}$, oszilliert einmal pro Umlauf und wächst nicht; $|\Delta L/L| = 4\cdot10^{-15}$.
- Velocity Verlet: $|\Delta E/E| = 2{,}7\cdot10^{-6}$, oszilliert und wächst nicht; $|\Delta L/L| = 2\cdot10^{-14}$.
- RK4: $|\Delta E/E| = 1{,}1\cdot10^{-5}$, wächst linear mit der Zeit; $|\Delta L/L| = 5\cdot10^{-6}$. Nach etwa 20 Umläufen ist RK4 bei der Energie schlechter als Verlet, obwohl es Ordnung 4 hat.

### Warum Heun (Ordnung 2) schlechter endet als das symplektische Euler (Ordnung 1)

Die Ordnung sagt, wie schnell der Fehler **pro Schritt** mit $\Delta t$ kleiner wird. Über lange Zeiten zählt aber, ob sich die Fehler **aufsummieren** oder **aufheben** (siehe 1b, „Warum manche Verfahren stabil bleiben“).

- Heun hat $|R(ix)|^2 = 1 + x^4/4 > 1$. Jeder Schritt pumpt ein wenig Energie hinein, immer mit demselben Vorzeichen. Nach $N$ Schritten ist der Fehler etwa $N \cdot x^4/4$ und wächst linear mit der Zeit: $5{,}6\cdot10^{-3}$ nach 10 Umläufen, $4{,}8\cdot10^{-2}$ nach 100.
- Das symplektische Euler hat pro Schritt den größeren Fehler $O(x)$. Weil es aber eine Schatten-Energie $\tilde E = E + O(\Delta t)$ erhält, schwankt $E$ nur um diesen festen Betrag. Nach 10 und nach 100 Umläufen sind es gleich viel, $3{,}3\cdot10^{-3}$.

Ein wachsender Fehler überholt jeden beschränkten Fehler irgendwann. Bei $\Delta t = 6$ h passiert das nach knapp 6 Umläufen (Kreuzung im Plot). Dazu kommt beim Drehimpuls: Das symplektische Euler hält ihn exakt, Heun verliert $2{,}5\,\%$.

### Was die Tests prüfen

- `test_energy_and_angular_momentum_of_circular_orbit`: $E = -GM/(2r)$ und $L = \sqrt{GM r}$ für den Startwert.
- `test_symplectic_methods_conserve_angular_momentum_to_roundoff`: $|\Delta L/L| < 10^{-12}$ über 100 Umläufe, der Beweis von oben.
- `test_other_methods_change_angular_momentum`: Bei den übrigen vier ist $|\Delta L/L| > 10^{-8}$. Der Test sorgt dafür, dass der vorige Test nicht trivial besteht.
- `test_symplectic_energy_error_stays_bounded`: Der größte Energiefehler in den letzten 10 Umläufen ist höchstens 10 % größer als in den ersten 10.
- `test_nonsymplectic_energy_error_grows`: Bei Heun und RK4 ist er in den letzten 10 Umläufen mehr als 5-mal so groß wie in den ersten.
- `test_heun_ends_worse_than_symplectic_euler`: Der Energiefehler von Heun am Ende ist mehr als 5-mal so groß wie der größte des symplektischen Euler.
- `test_scripts.py`: `plot_1c.py` läuft durch, speichert das Bild und gibt die Fehler aus.

## 1d) Konvergenzordnung bei $t = T/2$

### Messgröße

Nach einem halben Umlauf ist der exakte Mond genau gegenüber, bei $\mathbf r(T/2) = (-r_M, 0)$. Der Fehler ist der Abstand dazu, relativ zu $r_M$:

$$\varepsilon(\Delta t) = \frac{|\mathbf r_{N} - (-r_M, 0)|}{r_M}, \qquad \Delta t = \frac{T}{n},\quad N = \frac n2 \text{ Schritte}$$

Mit $n = 200, 400, \dots, 6400$ fällt der Endzeitpunkt genau auf $T/2$. Im Code: `error_at_half_period(step, n)`.

### Warum eine Gerade im Log-Log-Plot?

Ein Verfahren der Ordnung $p$ hat pro Schritt den lokalen Fehler $O(\Delta t^{p+1})$. Bis $T/2$ sind es $N = T/(2\Delta t)$ Schritte, also ist der globale Fehler

$$\varepsilon \approx C\,\Delta t^p \quad\Longrightarrow\quad \log\varepsilon = \log C + p\,\log\Delta t$$

Die Steigung der Geraden ist die Ordnung $p$. `order(dts, errors)` passt sie mit `np.polyfit` an, wie in Exercise 3.

### Ergebnis (`plot_1d.py`, `convergence_1d.png`)

Steigung und Fehler bei $\Delta t = T/6400$ (etwa 6 min):

- Explizites Euler: Steigung 0,96, Fehler $1{,}1\cdot10^{-2}$
- Implizites Euler: Steigung 1,05, Fehler $1{,}1\cdot10^{-2}$
- Symplektisches Euler: Steigung 0,99, Fehler $2{,}0\cdot10^{-3}$
- Heun: Steigung 2,00, Fehler $4{,}3\cdot10^{-6}$
- Velocity Verlet: Steigung 2,00, Fehler $1{,}1\cdot10^{-6}$
- RK4: Steigung 4,01, Fehler $7{,}5\cdot10^{-14}$

Das sind genau die erwarteten Ordnungen 1, 1, 1, 2, 2 und 4. Bei den beiden Euler-Verfahren weicht die Steigung über alle Punkte etwas von 1 ab, weil bei $\Delta t = T/200$ der Fehler mit 30 bis 40 % schon nicht mehr klein ist. Die Steigung zwischen den letzten beiden Punkten ist 0,99 bzw. 1,01. Bei RK4 ist der Fehler am feinsten Punkt nur noch etwa 1000-mal größer als die Rechengenauigkeit $10^{-16}$; bei noch kleinerem $\Delta t$ würde die Gerade abknicken, weil sich die Rundungsfehler aufsummieren.

Wichtig für den Vergleich mit 1c: **Bei kurzen Zeiten gewinnt die Ordnung.** Nach einem halben Umlauf ist RK4 um Größenordnungen besser als Verlet, und Heun ist besser als das symplektische Euler. Erst über viele Umläufe dreht sich das, weil sich dann die Fehler der nicht symplektischen Verfahren aufsummieren.

### Welches Verfahren für 1000 Jahre Mond?

1000 Jahre sind etwa 13 300 Umläufe (365 250 d / 27,45 d). Dann zählt nicht der Fehler nach einem halben Umlauf, sondern wie der Fehler mit der Zeit wächst:

- Nicht symplektische Verfahren driften in der Energie linear mit der Zeit (1c). RK4 hätte bei $\Delta t = 6$ h nach 1000 Jahren einen Energiefehler von etwa $1{,}4\cdot10^{-3}$, hochgerechnet aus $1{,}1\cdot10^{-5}$ pro 100 Umläufe mal 133, und die Bahn würde langsam nach innen wandern.
- **Velocity Verlet** hält $L$ exakt und den Energiefehler beschränkt ($2{,}7\cdot10^{-6}$ bei 6 h, unabhängig von der Laufzeit). Die Bahn bleibt deshalb auch nach 1000 Jahren eine Kreisbahn mit dem richtigen Radius. Dazu braucht es im Prinzip nur eine Kraftauswertung pro Schritt, weil $\mathbf a(\mathbf r_{n+1})$ im nächsten Schritt wiederverwendet werden kann (in `orbit.py` der Einfachheit halber zwei). RK4 braucht vier.

Meine Wahl ist deshalb **Velocity Verlet**, oder ein symplektisches Verfahren höherer Ordnung wie Yoshida 4. Ordnung, das mehrere Verlet-Schritte kombiniert. Eine Grenze bleibt: Auch bei Verlet wächst der **Phasenfehler**, also wo auf der Bahn der Mond gerade steht, linear mit der Zeit. Wer die Position nach 1000 Jahren genau braucht, muss $\Delta t$ kleiner wählen. Wer die Form der Bahn braucht, kommt mit großem $\Delta t$ aus. Für den echten Mond wären ohnehin die Störungen durch die Sonne wichtiger als der Fehler des Verfahrens.

### Was die Tests prüfen

- `test_error_at_half_period_is_distance_to_opposite_point`: Die Funktion rechnet $n/2$ Schritte und misst den Abstand zu $(-r_M, 0)$.
- `test_order_is_slope_in_log_log`: Für $\varepsilon = 3\,\Delta t^2$ kommt genau 2 heraus.
- `test_convergence_order_at_half_period`: Die gemessene Steigung liegt für jedes Verfahren höchstens 0,1 neben der theoretischen Ordnung 1, 1, 2, 1, 2 bzw. 4.
- `test_scripts.py`: `plot_1d.py` läuft durch, speichert das Bild und gibt alle sechs Steigungen aus.

## 2a) Senkrechter Start

### Modell (`rocket.py`)

Die Rakete startet am Äquator bei $\mathbf r = (R_E, 0)$ und rotiert mit der Erde mit: $\mathbf v = (0, \omega_E R_E) = (0;\ 465\ \mathrm{m/s})$. Solange der Treibstoff brennt, kommt zur Gravitation der Schub dazu:

$$\ddot{\mathbf r} = -\frac{GM}{|\mathbf r|^3}\mathbf r + \frac{F}{m(t)}\,\mathbf e, \qquad m(t) = m_0 - \dot m\,t, \qquad F = \dot m\,v_e$$

Alles wird auf die Startmasse $m_0$ bezogen. Mit $\dot m = F/v_e$ ist

$$\frac{m(t)}{m_0} = 1 - \frac{F}{m_0}\,\frac{t}{v_e}, \qquad \frac{F}{m(t)} = \frac{F/m_0}{1 - (F/m_0)\,t/v_e}$$

Die Beschleunigung durch den Schub wächst also, je leichter die Rakete wird: Am Ende ist sie bei $m_1 = m_0/19$ 19-mal so groß wie am Start. Der Brennschluss ist erreicht, wenn $m(t_b) = m_1$:

$$t_b = \Big(1 - \frac{m_1}{m_0}\Big)\frac{v_e}{F/m_0}$$

Bei $F/m_0 = 10\ \mathrm{m/s^2}$ sind das 417 s, bei $18\ \mathrm{m/s^2}$ 232 s. Im Code: `burn_time`, `rocket_rhs`.

**Schubrichtung.** $\mathbf e = \sin\theta\,\mathbf e_r \pm \cos\theta\,\mathbf e_\varphi$ mit dem Radialvektor $\mathbf e_r = \mathbf r/|\mathbf r|$ und dem Tangentialvektor $\mathbf e_\varphi = (-y, x)/|\mathbf r|$, der in Drehrichtung der Erde (nach Osten) zeigt. $\theta = 90°$ ist senkrecht nach oben, $\theta = 0$ waagerecht (`thrust_direction`).

**Numerik.** RK4 mit Zeitargument, weil die rechte Seite über $m(t)$ von $t$ abhängt. Beim Brennschluss springt die Beschleunigung, und RK4 setzt glatte rechte Seiten voraus. `launch` teilt die Rechnung deshalb in zwei Phasen: Brennen mit $\Delta t \approx 0{,}1$ s und so vielen gleich langen Schritten, dass $t_b$ genau auf dem Gitter liegt, dann Freiflug mit $\Delta t = 1$ s. In jeder Phase ist das Triebwerk fest an oder aus (`burning=True/False`), damit keine RK4-Zwischenstufe genau bei $t_b$ auf der falschen Seite der Unstetigkeit rechnet. Unterschreitet der Radius $R_E$, ist die Rakete aufgeschlagen und die Rechnung endet.

### Ergebnis (`plot_2a.py`, `vertical_2a.png`)

$m_1 = m_0/19$, nach der Raketengleichung also $\Delta v = v_e \ln 19 = 12\,956$ m/s. Werte bei Brennschluss:

- $F/m_0 = 10$: $t_b = 417$ s, Höhe 696 km, $v = 9045$ m/s, $E < 0$. Fällt zurück und schlägt nach 3,9 h auf.
- $F/m_0 = 12$: $t_b = 347$ s, Höhe 698 km, $v = 9714$ m/s, $E < 0$. Gipfel bei etwa 37 000 km nach 4,4 h, dann fällt sie zurück.
- $F/m_0 = 14$: $t_b = 298$ s, Höhe 670 km, $v = 10\,181$ m/s, $E < 0$. Nach 6 h noch steigend, aber gebunden.
- $F/m_0 = 16$: $t_b = 261$ s, Höhe 633 km, $v = 10\,526$ m/s, $E < 0$. Ebenso.
- $F/m_0 = 18$: $t_b = 232$ s, Höhe 595 km, $v = 10\,793$ m/s, $E = +1{,}0\cdot10^6$ J/kg. **Entkommt.**

### Woher die Differenz zur Raketengleichung kommt

Die Raketengleichung folgt aus dem Impulssatz ohne äußere Kräfte: $m\,\dot v = \dot m\,v_e$, integriert $\Delta v = v_e\ln(m_0/m_1)$. Auf der Erde zieht aber während des ganzen Brennens die Gravitation nach unten. Für die Radialgeschwindigkeit gilt deshalb

$$v_r(t_b) \approx v_e\ln\frac{m_0}{m_1} - \int_0^{t_b} g(r(t))\,dt$$

Das ist der **Gravitationsverlust** in radialer Richtung. Er ist kleiner als $g_0\,t_b$, weil $g$ mit der Höhe abnimmt (in 700 km nur noch $8{,}0$ statt $9{,}82\ \mathrm{m/s^2}$): bei $F/m_0 = 10$ ist $\Delta v - v_r = 3{,}9$ km/s ($g_0 t_b = 4093$ m/s), bei 18 nur 2,2 km/s ($g_0 t_b = 2274$ m/s).

Vorsicht beim Vergleich mit 2d: Dort ist der Gravitationsverlust über den Betrag $|\mathbf v|$ definiert, $\int g\sin\gamma\,dt$ (`gravity_loss`). Für den senkrechten Start ergibt das nur 2,6 km/s bei 10 und 1,8 km/s bei 18, weil die Rakete anfangs mit 465 m/s waagerecht fliegt und $\gamma$ erst klein ist. Der Rest steckt dann im Lenkverlust (Schub radial, Geschwindigkeit anfangs waagerecht). Vergleichbar mit 2d ist der **Gesamtverlust** $\Delta v - (|\mathbf v_b| - |\mathbf v_0|)$: 4,4 km/s bei 10, 2,6 km/s bei 18 (`plot_2a.py` gibt alle drei Größen aus). **Je schneller die Rakete brennt, desto weniger Zeit hat die Gravitation**, und deshalb kommt die stärkste Rakete am schnellsten an, obwohl alle denselben Treibstoff verbrennen.

### Welcher Schub reicht zum Entkommen?

Entkommen heißt $E = \tfrac12|\mathbf v|^2 - GM/|\mathbf r| \ge 0$ bei Brennschluss, danach wirkt nur noch die Gravitation, und $E$ bleibt erhalten (1c). Bisektion auf das Vorzeichen von $E(t_b)$ ergibt

$$F/m_0 \ge 17{,}13\ \mathrm{m/s^2}$$

Zum Vergleich: In 600 km Höhe ist die Fluchtgeschwindigkeit $\sqrt{2GM/r} = 10{,}7$ km/s. Die Rakete darf also höchstens etwa $12{,}96 - 10{,}7 \approx 2{,}3$ km/s an die Gravitation verlieren, und das schafft sie erst ab 17,13 m/s².

### Kann ein senkrechter Start einen Orbit erreichen?

Nein. Der Schub zeigt radial, die Gravitation auch. Beide üben kein Drehmoment aus ($\mathbf r\times\mathbf F = 0$), also bleibt der Drehimpuls vom Start erhalten (wie in 1c):

$$L = R_E \cdot \omega_E R_E = 2{,}96\cdot10^9\ \mathrm{m^2/s}$$

Der erdnächste Punkt einer Kegelschnittbahn mit Energie $E$ und Drehimpuls $L$ ist

$$r_p = \frac{L^2}{GM(1 + e)}, \qquad e = \sqrt{1 + \frac{2EL^2}{(GM)^2}}$$

(`periapsis`). Er ist höchstens $L^2/GM = 22$ km vom Erdmittelpunkt entfernt, gemessen 11 km. Jede gebundene Bahn führt also tief durch die Erde: Die Rakete fällt zurück. Für einen Orbit braucht sie $r_p > R_E$, also viel mehr Drehimpuls. Der Schub muss deshalb seitlich zeigen, und genau das macht das Pitch-Programm in 2b.

### Was die Tests prüfen

- `test_launch_state_corotates_with_earth`, `test_thrust_direction_vertical_east_and_west`: Startwert und Richtung $\mathbf e$ für $\theta = 90°$, $0°$ Ost, $0°$ West und $30°$.
- `test_burn_time_ends_when_mass_reaches_m1`, `test_rhs_…`: $m(t_b) = m_1$, auf der Startrampe ist die Beschleunigung $F/m_0 - g_0$, nach Brennschluss nur $-g_0$, und der Schub wächst wie $1/m(t)$.
- `test_launch_hits_burnout_time_on_grid`, `test_launch_stops_when_rocket_hits_the_ground`: $t_b$ liegt auf dem Zeitgitter, und die Rechnung endet beim Aufschlag.
- `test_burnout_speed_is_rocket_equation_minus_gravity_loss`: Für alle fünf Schübe liegt der Verlust zwischen $0{,}8\,g_0 t_b$ und $g_0 t_b$.
- `test_escape_only_with_high_thrust`: Nur $F/m_0 = 18$ entkommt, 10 und 16 nicht.
- `test_periapsis_…`: $r_p$ für einen Kreis, eine Ellipse (aus der Vis-viva-Gleichung) und den senkrechten Start.
- `test_scripts.py`: `plot_2a.py` gibt die Werte aus, findet die Schwelle zwischen 17,0 und 17,3 und sagt „orbit possible: False“.

## 2b) Pitch-Programm

### Modell

$F/m_0 = 15\ \mathrm{m/s^2}$, $m_1 = 0{,}12\,m_0$, also $t_b = 0{,}88 \cdot 4400/15 = 258$ s und $\Delta v = v_e\ln(1/0{,}12) = 9{,}33$ km/s. Der Schubwinkel dreht linear von senkrecht nach waagerecht:

$$\theta(t) = 90° \cdot \max\Big(0,\ 1 - \frac{t}{t_p}\Big)$$

Nach $t_p$ zeigt der Schub waagerecht nach Osten bzw. Westen. Ist $t_p > t_b$, wird die Rakete nie ganz waagerecht.

### Einteilung der Bahnen (`classify`)

Nach Brennschluss wirkt nur noch die Gravitation, die Bahn ist dann ein Kegelschnitt mit festem $E$ und $L$. Der Zustand bei Brennschluss entscheidet also schon alles:

- **crash**: schlägt auf, während das Triebwerk noch brennt (zu früh flach geworden).
- **escape**: $E \ge 0$, Hyperbel- oder Parabelbahn.
- **orbit**: $E < 0$ und Periapsis $r_p > R_E$, die Ellipse verfehlt die Erde.
- **falls back**: $E < 0$ und $r_p < R_E$, suborbital, die Ellipse schneidet die Erde.

### Ergebnis (`plot_2b.py`, `pitch_2b.png`)

Höhe der Periapsis über dem Boden:

- Osten: $t_p = 100$ und 125 s crash. 150 s orbit (47 km), **175 s orbit (98 km)**, 200 s orbit (24 km). 225 bis 300 s falls back ($-317$ bis $-2308$ km).
- Westen: 100 und 125 s crash, 150 bis 300 s falls back. Kein einziger Orbit.

### Welches $t_p$ ist am besten?

**$t_p = 175$ s**, die Periapsis liegt mit 98 km am höchsten über dem Boden. Zwei gegenläufige Effekte bestimmen das:

- **Zu kleines $t_p$:** Die Rakete wird flach, bevor sie hoch genug ist. Sie schiebt dann waagerecht in Bodennähe und schlägt auf (crash).
- **Zu großes $t_p$:** Die Rakete fliegt zu lange steil. Ein großer Teil von $\Delta v$ geht in Höhe und an die Gravitation verloren (wie in 2a), und am Ende fehlt die waagerechte Geschwindigkeit für einen Orbit. Bei 300 s ist die Rakete bei Brennschluss 320 km hoch, aber nur 7,8 km/s schnell und zu steil (falls back).

Dazwischen liegt ein Fenster von etwa 150 bis 200 s. Für einen Orbit muss die Rakete aus der dichtesten Gravitation herauskommen und gleichzeitig fast die ganze Geschwindigkeit waagerecht aufbauen.

### Warum starten Raketen nach Osten?

Die Rakete rotiert schon auf der Startrampe mit der Erde mit, sie hat im Inertialsystem $\omega_E R_E = 465$ m/s nach Osten. Beim Start nach Osten kommt diese Geschwindigkeit geschenkt dazu. Nach Westen muss die Rakete sie erst abbauen. Der Unterschied ist $2 \cdot 465 = 930$ m/s, rund 10 % des gesamten $\Delta v$ von 9,33 km/s. Genau das fehlt nach Westen: Bei $t_p = 175$ s hat die Rakete nach Osten bei Brennschluss 8,6 km/s, nach Westen nur 7,7 km/s, und damit kommt sie bei keinem $t_p$ in einen Orbit. Am Äquator ist der Effekt am größten, deshalb liegen Startplätze möglichst nahe am Äquator (Kourou auf 5° N).

### Was die Tests prüfen

- `test_classify_pitch_program_trajectories`: $t_p = 100$ s Ost crash, 175 s Ost orbit, 250 s Ost falls back, 175 s West falls back.
- `test_classify_escape`: Der senkrechte Start mit 18 m/s² aus 2a wird als escape erkannt.
- `test_apoapsis_…`, `test_periapsis_…`: die Kegelschnitt-Formeln aus $E$ und $L$.
- `test_scripts.py`: alle 18 Fälle, kein Orbit nach Westen, bestes $t_p = 175$ s.

## 2c) Kreisbahnen

### Warum ein einziger Brennvorgang nicht reicht

Nach Brennschluss liegt die Rakete auf einer festen Ellipse, und diese Ellipse geht durch den Brennschlusspunkt. Eine Kreisbahn in der Höhe $h$ verlangt deshalb, dass die Rakete bei Brennschluss **genau in der Höhe $h$** ist, **genau waagerecht** fliegt und **genau** $v_c = \sqrt{GM/(R_E + h)}$ hat (7,79 / 7,62 / 7,35 km/s für 200 / 500 / 1000 km). Das sind drei Bedingungen auf einmal. Mit einem Brennvorgang hat man aber nur $t_p$ und den Brennschluss als Stellgrößen, und der ganze Brennvorgang dauert höchstens 258 s. In dieser Zeit kommt die Rakete mit flacher Bahn nur auf etwa 100 bis 300 km Höhe (2b). Für 500 und 1000 km geht es also gar nicht, und für 200 km nur durch Zufall. So endet die Rakete aus 2b mit $t_p = 175$ s auf einer Ellipse von 98 bis 3396 km Höhe.

### Strategie: brennen, gleiten, nachbrennen

Das Triebwerk darf abgeschaltet und wieder gezündet werden. Wie bei Falcon 9 trennt man das Problem in zwei Teile:

1. **Erster Brennvorgang** mit dem Pitch-Programm $t_p = 175$ s aus 2b, bis die **Apoapsis** auf $R_E + h$ liegt, dann Abschalten. Die Apoapsis folgt direkt aus dem Zustand, $r_a = L^2/(GM(1-e))$ (`apoapsis`).
2. **Freiflug zur Apoapsis.** Dort ist die Rakete in der Höhe $h$ und fliegt waagerecht ($\mathbf r\cdot\mathbf v$ wechselt das Vorzeichen).
3. **Zweiter Brennvorgang** waagerecht nach Osten ($\theta = 0$), bis die Bahn ein Kreis ist, also bis $E = -GM/(2r)$. Schub $F$ wie vorher, aber $F/m$ ist jetzt größer, weil die Rakete leichter ist.

Jeder Übergang ist ein Ereignis $g(\mathbf y) = 0$. `fly_until` fliegt bis zu dem Schritt, in dem $g$ das Vorzeichen wechselt, und halbiert diesen Schritt dann 60-mal. Wichtig ist, dass innerhalb eines Brennvorgangs die Zeit seit dem Zünden weiterläuft (`t0`). Sonst würden Pitch-Winkel und Masse beim Verfeinern neu bei null anfangen. Genau dieser Fehler hat in meinem ersten Versuch das Ergebnis verfälscht.

Die Strategie verlangt, dass die Rakete beim Abschalten noch steigt. Bei $t_p = 150$ und 160 s ist der Scheitel schon überschritten, wenn die Apoapsis die Zielhöhe erreicht. Deshalb bleibt es bei $t_p = 175$ s.

### Ergebnis (`plot_2c.py`, `circular_2c.png`)

- **200 km:** Brennen 251,1 s, Gleiten 900 s, Nachbrennen 1,02 s. Bahn 199,9 bis 200,1 km. **Rest 2,35 % des Treibstoffs** ($0{,}021\,m_0$).
- **500 km:** Brennen 252,4 s, Gleiten 2060 s, Nachbrennen 1,21 s. Bahn 499,8 bis 500,2 km. **Rest 1,74 %** ($0{,}015\,m_0$).
- **1000 km:** Brennen 253,8 s, Gleiten 2554 s, Nachbrennen 2,18 s. Bahn 999,5 bis 1000,5 km. **Rest 0,85 %** ($0{,}007\,m_0$).

Fast der ganze Treibstoff geht in den ersten Brennvorgang, das Nachbrennen dauert nur 1 bis 2 s. Das liegt daran, dass die Rakete mit $t_p = 175$ s schon nahe an der Kreisbahngeschwindigkeit ist und das Gleiten fast einen halben Umlauf dauert. Höhere Bahnen kosten mehr, weil mehr Energie nötig ist: $E = -GM/(2r)$ wächst mit $r$. Für noch höhere Bahnen reicht der Treibstoff dieser Rakete nicht mehr.

### Was die Tests prüfen

- `test_fly_with_start_time_continues_a_burn`: Zwei Brennphasen von je 100 s mit $t_0$ ergeben dasselbe wie eine von 200 s.
- `test_fly_until_stops_exactly_at_event`: Auf einer Kreisbahn wird $x = 0$ nach genau $T/4$ erreicht, auf $10^{-8}$ genau.
- `test_fly_until_returns_none_if_event_not_reached`, `test_fly_coast_…`, `test_fly_stops_on_impact`, `test_fly_burn_matches_launch`, `test_fly_burn_mass_after`: die übrigen Eigenschaften der Flugphasen.
- `test_scripts.py`: Alle drei Bahnen liegen auf 2 km genau bei der Zielhöhe, und es bleibt zwischen 0 und 5 % Treibstoff übrig.

## 2d) Zwei Stufen

### Raketengleichung pro Stufe

Bei einer Stufe ist $m_1 = 0{,}1\,m_0$. Bei zwei Stufen hat die erste Stufe $0{,}8\,m_0$ mit $0{,}72\,m_0$ Treibstoff und $0{,}08\,m_0$ Leermasse, die zweite $0{,}2\,m_0$ mit $0{,}18\,m_0$ Treibstoff. Die erste Stufe brennt von $m_0$ auf $0{,}28\,m_0$. Dann wird ihre leere Hülle abgeworfen, die zweite Stufe startet bei $0{,}2\,m_0$ und brennt auf $0{,}02\,m_0$. Jede Stufe zündet mit $F/m = 15\ \mathrm{m/s^2}$.

$$\Delta v_1 = v_e \ln\frac{1}{0{,}1} = 10{,}13\ \mathrm{km/s}$$

$$\Delta v_2 = v_e\Big(\ln\frac{1}{0{,}28} + \ln\frac{0{,}2}{0{,}02}\Big) = 4{,}4\,(1{,}273 + 2{,}303) = 15{,}73\ \mathrm{km/s}$$

### Warum Stufen helfen

Bei einer Stufe muss der leere Tank der ganzen Rakete ($0{,}1\,m_0$) bis zum Schluss mitbeschleunigt werden. Bei zwei Stufen wird der große leere Tank ($0{,}08\,m_0$) abgeworfen, sobald er leer ist. Danach beschleunigt der Treibstoff nur noch $0{,}02\,m_0$ Leermasse. Das Massenverhältnis $m_0/m_1$ geht logarithmisch ein, und die Verhältnisse der Stufen multiplizieren sich: $3{,}57 \cdot 10 = 35{,}7$ statt 10. Mit dem gleichen Treibstoff von 90 % der Startmasse wird $\Delta v$ damit um 55 % größer.

### Gravitations- und Lenkverlust

Der Betrag der Geschwindigkeit ändert sich wie

$$\frac{d|\mathbf v|}{dt} = \frac{F}{m}\,\mathbf e\cdot\hat{\mathbf v} + \mathbf a_g\cdot\hat{\mathbf v}$$

Der zweite Term ist der **Gravitationsverlust** $g\sin\gamma$, mit dem Bahnneigungswinkel $\gamma$: voll beim senkrechten Flug, null beim waagerechten (`gravity_loss`). Der erste Term ist kleiner als $F/m$, wenn der Schub nicht in Flugrichtung zeigt. Die Differenz heißt **Lenkverlust**. Insgesamt gilt

$$|\mathbf v_b| - |\mathbf v_0| = \Delta v - \text{Gravitationsverlust} - \text{Lenkverlust}$$

Ergebnis (`plot_2d.py`, `staging_2d.png`, Osten, $t_p = 175$ s):

- **Eine Stufe:** $\Delta v = 10\,131$ m/s, Brennschluss nach 264 s in 109 km Höhe, Gewinn 8935 m/s, **Gravitationsverlust 626 m/s**, Lenkverlust 571 m/s, orbit.
- **Zwei Stufen:** $\Delta v = 15\,732$ m/s, Brennschluss nach 475 s in 110 km Höhe, Gewinn 14\,541 m/s, **Gravitationsverlust 620 m/s** (608 m/s in der ersten Stufe, 13 m/s in der zweiten), Lenkverlust 572 m/s, escape.

Die Summen gehen jeweils genau auf. Der Gravitationsverlust entsteht fast ganz in den ersten Minuten, solange die Rakete steil steigt. Die zweite Stufe brennt waagerecht und verliert fast nichts. Insgesamt kostet der Aufstieg mit Pitch-Programm etwa 1,2 km/s (Gravitations- plus Lenkverlust). Der senkrechte Start in 2a verliert nach derselben Definition 2,6 bis 4,4 km/s. Das Pitch-Programm spart also viel, obwohl es einen Lenkverlust erzeugt. Die 1,2 km/s sind rund 12 % des $\Delta v$ der einstufigen Rakete. Mit zwei Stufen reicht das $\Delta v$ bei diesem Pitch-Programm schon zum Verlassen der Erde. Für einen Orbit würde man die zweite Stufe früher abschalten, wie in 2c.

### Was die Tests prüfen

- `test_gravity_loss_is_the_speed_lost_while_coasting`: Ohne Schub ist der Gravitationsverlust genau der Geschwindigkeitsverlust.
- `test_gravity_loss_vanishes_on_circular_orbit`: Auf einer Kreisbahn steht die Gravitation senkrecht auf $\mathbf v$, also ist der Verlust null.
- `test_scripts.py`: $\Delta v$ beider Fälle stimmt mit der Raketengleichung überein, eine Stufe ergibt orbit, zwei Stufen escape, und beide Gravitationsverluste liegen zwischen 300 und 1500 m/s.
