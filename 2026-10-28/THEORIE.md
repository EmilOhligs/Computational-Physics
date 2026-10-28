# Theorie: Gewöhnliche Differentialgleichungen

Exercise 3 · 28.10.2026

Dieses Dokument erklärt die Theorie zu den Teilaufgaben 1a und 1b. Die Verweise
(1a, 1b, …) beziehen sich auf die Angabe in `exercise (1).pdf`, Code in `ode.py`,
Tests in `test_ode.py`.

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
