# Theorie zu Exercise 5: Finite Differenzen für die Wärmeleitungsgleichung

Einlese-Blatt vor der Einheit am 11.11.2026. Hier steht die Theorie hinter jeder Teilaufgabe und was man als Ergebnis erwarten sollte. Die Zahlen unten sind von Hand bzw. mit ein paar Zeilen Python ausgerechnet, nicht aus fertigem Code. Nach der Einheit kommt hier der Bezug zum eigenen Code dazu.

Roter Faden: Teil 1 macht aus einer Differentialgleichung im Raum ein lineares Gleichungssystem $Au = f$. Teil 2 nimmt dieselbe Matrix $A$ und macht aus der zeitabhängigen PDE ein ODE-System $\dot u = -Au$, auf das die Verfahren aus Exercise 3 losgelassen werden. Alles über Stabilität und Steifheit aus Exercise 3 kommt zurück, nur dass die Eigenwerte jetzt vom Gitter kommen.

## 0. Das Gitter

Intervall $[0,1]$, Dirichlet-Ränder $u(0) = u(1) = 0$. Es gibt $N$ innere Punkte

$$x_j = j\,\Delta x,\quad j = 1,\dots,N,\qquad \Delta x = \frac{1}{N+1}$$

Die Randpunkte $x_0 = 0$ und $x_{N+1} = 1$ sind keine Unbekannten, ihre Werte kennt man. Gesucht ist der Vektor $u = (u_1,\dots,u_N)$ mit $u_j \approx u(x_j)$.

Die Werte $N = 3, 7, 15, \dots$ im Blatt sind $N = 2^k - 1$. Dann ist $\Delta x = 2^{-k}$ exakt im Gleitkommaformat darstellbar, und beim Halbieren von $\Delta x$ liegen die alten Gitterpunkte auf dem neuen Gitter.

### Warum gerade $N = 2^k - 1$?

**1. $\Delta x$ wird eine Zweierpotenz.** Aus $N = 2^k - 1$ folgt $N + 1 = 2^k$ und damit

$$\Delta x = \frac{1}{N+1} = 2^{-k}$$

Die Folge $3, 7, 15, 31, \dots$ gehört also zu $\Delta x = \tfrac14, \tfrac18, \tfrac1{16}, \tfrac1{32}, \dots$, jedes Mal halb so groß. Bis $10^6$ geht das bis $k = 20$, also $N = 2^{20} - 1 = 1\,048\,575$.

**2. Zweierpotenzen sind im Gleitkommaformat exakt.** Ein `float64` speichert eine Zahl als $\pm m \cdot 2^e$ mit einer 53-stelligen **binären** Mantisse $m$. Eine Zahl ist genau dann exakt darstellbar, wenn sie ein endlicher Binärbruch ist, also ein Bruch mit einer Zweierpotenz im Nenner, der in 53 Bits passt.

- $2^{-k}$ ist $m = 1$ mit Exponent $-k$, also exakt.
- $1/11$ ist binär periodisch, so wie $1/3 = 0{,}333\ldots$ dezimal. Der Computer muss nach 53 Bits runden. Gemessen ist `1/11` um etwa $2{,}5\cdot10^{-18}$ zu groß. Dasselbe Problem kennt man von `0.1 * 3 == 0.3`, das `False` ergibt.

**3. Deshalb sind auch alle Gitterpunkte exakt.** $x_j = j\cdot 2^{-k}$ mit $j < 2^k$ braucht höchstens $k$ Bits, die Multiplikation ist dann ohne Rundung. Nachgerechnet mit exakten Brüchen (`fractions.Fraction`): Bei $N = 15$ liegen alle 17 Punkte genau auf $j/16$. Bei $N = 10$ liegen 10 von 12 Punkten neben $j/11$, nur $0$ und $1$ stimmen. Auch $1/\Delta x^2 = 4^k$ ist exakt, die Einträge der Matrix $A$ bekommen also keinen Rundungsfehler.

Damit misst der Konvergenzplot in 1b nur den Diskretisierungsfehler und keinen Fehler in der Lage des Gitters. Die Rundungsfehler bei der Rechnung selbst bleiben aber: $u_{j-1} - 2u_j + u_{j+1}$ ist eine Differenz fast gleicher Zahlen, und genau dieser Fehler $\sim \varepsilon/\Delta x^2$ lässt den Fehler bei sehr kleinem $\Delta x$ wieder wachsen (1b). Das exakte Gitter verhindert das nicht.

**4. Die Gitter sind ineinander geschachtelt.** Beim Halbieren wird $k \to k+1$ und $N \to 2N + 1$. Ein alter Punkt liegt dann genau auf einem neuen:

$$x_j = j\cdot 2^{-k} = 2j\cdot 2^{-(k+1)} = x'_{2j}$$

Jeder zweite Punkt des feinen Gitters ist ein Punkt des groben. Bei $N = 3$ sind das $\tfrac14, \tfrac12, \tfrac34$, bei $N = 7$ die Punkte $\tfrac18, \tfrac28, \dots, \tfrac78$, und $\tfrac28, \tfrac48, \tfrac68$ sind genau die alten. Das hilft:

- Lösungen auf zwei Gittern lassen sich direkt Punkt für Punkt vergleichen, ohne Interpolation: `u_fine[1::2]` sind die Werte an den alten Punkten. Python zählt ab 0, $x'_{2j}$ steht deshalb an Index $2j - 1$, also an jedem zweiten Index ab 1.
- Fehlerabschätzungen ohne exakte Lösung (Differenz zweier Gitter) und Richardson-Extrapolation brauchen genau diese gemeinsamen Punkte.
- Im Log-Log-Plot liegen die Punkte im gleichen Abstand, weil sich $\Delta x$ jedes Mal genau halbiert. Die Steigung lässt sich dann sauber ablesen.

Mit $N = 10, 20, 40$ wäre $\Delta x = \tfrac1{11}, \tfrac1{21}, \tfrac1{41}$: Die Gitter hätten außer den Rändern keinen Punkt gemeinsam, und keiner der Abstände wäre exakt darstellbar.

## 1a) Der 3-Punkte-Stern und die Matrix $A$

### Was ein Stern ist

Ein **Differenzenstern** (englisch *stencil*, kurz „Stern“) ist die Rechenvorschrift, mit der man eine Ableitung an einem Gitterpunkt aus den Werten an diesem Punkt und seinen Nachbarn annähert. Er legt fest, **welche Punkte** man benutzt und **mit welchem Gewicht** jeder eingeht. Der 3-Punkte-Stern für $-u''$ ist

$$-u''(x_j) \approx \frac{-1\cdot u_{j-1} + 2\cdot u_j - 1\cdot u_{j+1}}{\Delta x^2}$$

kurz geschrieben als Gewichte $(-1,\ 2,\ -1)$, alles geteilt durch $\Delta x^2$. Man kann ihn sich als Schablone vorstellen, die man über das Gitter schiebt:

```
   u_{j-1}   u_j   u_{j+1}
     -1       2      -1        (alles / Δx²)
```

An jedem Punkt $x_j$ legt man die Schablone an, multipliziert die Werte mit den Gewichten und addiert. Das Ergebnis ist genau eine Zeile der Matrix $A$: Die Gewichte stehen auf der Diagonale und daneben. Deshalb sehen alle Zeilen gleich aus, nur jeweils um eine Spalte verschoben.

Breitere Sterne nehmen mehr Nachbarn mit und sind genauer, zum Beispiel der 5-Punkte-Stern $\frac{1}{12}(1,\ -16,\ 30,\ -16,\ 1)$ mit $u_{j-2}\dots u_{j+2}$ (siehe 1c). Am Rand reicht ein breiter Stern aber über das Intervall hinaus. Der Name kommt aus 2D: Dort sieht der 5-Punkte-Stern für den Laplace-Operator wie ein Kreuz aus, der Punkt in der Mitte mit je einem Nachbarn oben, unten, links und rechts (siehe 1e).

### Herleitung

Taylor um $x_j$ in beide Richtungen:

$$u(x_j \pm \Delta x) = u \pm \Delta x\,u' + \frac{\Delta x^2}{2}u'' \pm \frac{\Delta x^3}{6}u''' + \frac{\Delta x^4}{24}u'''' + \dots$$

Addiert man beide, fallen alle ungeraden Ableitungen weg:

$$\frac{u_{j-1} - 2u_j + u_{j+1}}{\Delta x^2} = u''(x_j) + \frac{\Delta x^2}{12}\,u''''(x_j) + O(\Delta x^4)$$

Der Abbruchfehler (lokaler Diskretisierungsfehler) ist also $O(\Delta x^2)$: Verfahren zweiter Ordnung.

### Die Matrix

Für $-u''$ wird daraus in jeder Zeile $(-u_{j-1} + 2u_j - u_{j+1})/\Delta x^2 = f_j$. Als Matrix:

$$A = \frac{1}{\Delta x^2}\begin{pmatrix} 2 & -1 & & \\ -1 & 2 & -1 & \\ & \ddots & \ddots & \ddots \\ & & -1 & 2 \end{pmatrix} \in \mathbb R^{N\times N}$$

In der ersten und letzten Zeile fehlt jeweils ein Nachbar, weil $u_0 = u_{N+1} = 0$ ist. Bei Rändern $\ne 0$ würde der bekannte Randwert auf die rechte Seite wandern: $f_1 \to f_1 + u_0/\Delta x^2$.

In scipy: `scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N)) / dx**2`, gelöst mit `scipy.sparse.linalg.spsolve` (Format `csr` oder `csc`). Dense geht bis vielleicht $N = 10^4$, bei $N = 10^6$ bräuchte die volle Matrix 8 TB. Sparse hat nur $3N$ Einträge, und das Lösen eines Tridiagonalsystems kostet $O(N)$.

### Wie prüft man $A$, bevor man löst? (Frage aus 1a)

- **Symmetrie:** $A = A^T$, also `abs(A - A.T).max() == 0`.
- **Zeilensummen:** In den inneren Zeilen ist die Summe 0, denn die zweite Ableitung einer Konstanten ist 0. Nur Randzeilen haben Summe $1/\Delta x^2$.
- **Auf bekannte Funktionen anwenden:** $A$ angewendet auf $\sin(\pi x_j)$ muss $\approx \pi^2\sin(\pi x_j)$ geben, mit Fehler $O(\Delta x^2)$. Für ein Polynom zweiten Grades, das an den Rändern verschwindet, etwa $x(1-x)$, ist das Ergebnis exakt $2$ (bis auf Rundung), weil $u'''' = 0$ ist.
- **Eigenwerte:** Die kennt man exakt (siehe 2c). Alle sind positiv, $A$ ist also symmetrisch positiv definit. Für kleines $N$ kann man `np.linalg.eigvalsh(A.toarray())` mit der Formel vergleichen.
- **Klein anschauen:** Für $N = 3$ oder $4$ einfach `A.toarray() * dx**2` ausgeben.

Das ist dieselbe Denkweise wie in Exercise 3: erst Eigenschaften prüfen, die man ohne Lösung kennt, dann rechnen.

### Umsetzung in `fd.py`

- `grid(N)` gibt die inneren Punkte $x_j = j\,\Delta x$ und $\Delta x = 1/(N+1)$ zurück.
- `laplacian(N)` baut $A$ mit `scipy.sparse.diags([-1, 2, -1], [-1, 0, 1])` im Format `csr` und teilt durch $\Delta x^2$.
- `solve(f, N)` wertet $f$ auf dem Gitter aus und löst $Au = f$ mit `spsolve`.

Ergebnis für $f = \pi^2\sin\pi x$: Der maximale Fehler gegen $\sin\pi x$ ist $3{,}22\cdot10^{-3}$ bei $N = 15$, $8{,}04\cdot10^{-4}$ bei $N = 31$ und $2{,}01\cdot10^{-4}$ bei $N = 63$. Pro Halbierung von $\Delta x$ also Faktor 4, und jedes Mal fast genau $\pi^2\Delta x^2/12$ wie in 1b vorhergesagt.

### Was die Tests prüfen (`test_fd.py`)

- `test_grid_*`: $\Delta x = 1/4$ und $x = (0{,}25;\ 0{,}5;\ 0{,}75)$ bei $N = 3$; die Randpunkte 0 und 1 gehören nicht zum Gitter.
- `test_laplacian_is_sparse_square_and_symmetric`: $A$ ist sparse, $N\times N$ und exakt symmetrisch.
- `test_laplacian_n3_is_scaled_tridiagonal_stencil`: $\Delta x^2 A$ ist genau die Matrix aus $2$ und $-1$.
- `test_laplacian_row_sums_*`: Innere Zeilensummen sind 0, Rand-Zeilen $1/\Delta x^2$ (dort fehlt der Nachbar).
- `test_laplacian_is_exact_for_quadratic_with_zero_boundary`: $A\,[x(1-x)] = 2$ ohne Abbruchfehler, weil $u'''' = 0$.
- `test_laplacian_eigenvalues_match_analytic_formula`: Eigenwerte gleich $\frac{4}{\Delta x^2}\sin^2\frac{k\pi\Delta x}{2}$.
- `test_solve_sine_source_gives_exact_discrete_eigenvector_solution`: $u = \frac{\pi^2}{\lambda_1}\sin\pi x_j$ auf 12 Stellen, weil $\sin\pi x_j$ ein Eigenvektor von $A$ ist.
- `test_solve_sine_source_is_close_to_continuous_solution`: Abstand zur exakten Lösung unter $10^{-2}$ bei $N = 15$.

## 1b) Fehler über $\Delta x$

### Gitterkonvergenzstudie: das Gitter immer feiner machen

Für den Plot rechnet man das ganze Problem mehrmals von vorne, jedes Mal auf einem feineren Gitter:

- $N = 3$: 3 Punkte im Abstand $\tfrac14$
- $N = 7$: 7 Punkte im Abstand $\tfrac18$
- $N = 15$: 15 Punkte im Abstand $\tfrac1{16}$
- … bis $N \approx 10^6$ mit Abstand $\approx 10^{-6}$

**Innerhalb eines Laufs sind die Punkte immer gleichmäßig verteilt.** „Nicht linear“ ist nur die Folge der Gitter: $\Delta x$ wird von Lauf zu Lauf halbiert, also geometrisch kleiner und nicht in gleichen Schritten. Bei jeder Halbierung kommt in jede Lücke genau ein neuer Punkt, die alten Punkte bleiben, wo sie sind.

Für jedes Gitter misst man den maximalen Fehler gegen die exakte Lösung. So entsteht pro Gitter ein Punkt im Plot. Warum man $\Delta x$ geometrisch statt linear wählt:

- **Log-log-Plot:** Aus $E \approx C\,\Delta x^p$ folgt $\log E = \log C + p\log\Delta x$, also eine Gerade. Halbiert man $\Delta x$, liegen die Messpunkte auf der log-Achse in gleichen Abständen. Bei linear gewählten $\Delta x$ wären sie am rechten Rand gestaucht, und bei kleinem $\Delta x$ läge kaum ein Punkt.
- **Ordnung direkt ablesen:** Halbiert man $\Delta x$, fällt der Fehler um den Faktor $2^p$, also $p = \log_2\big(E(\Delta x)/E(\Delta x/2)\big)$. Faktor 4 heißt Ordnung 2.
- **Großer Bereich mit wenigen Läufen:** 20 Läufe decken $\Delta x$ von $\tfrac14$ bis $10^{-6}$ ab. Nur so sieht man beides: die Konvergenz mit Steigung 2 und den Wiederanstieg durch Rundungsfehler, sobald das Gitter zu fein ist. Feiner ist also nicht immer besser.

Bei der $\Delta t$-Studie in 2b geht man genauso vor: In jedem Lauf sind alle Zeitschritte gleich lang, aber die verschiedenen $\Delta t$-Werte wählt man geometrisch (Halbieren oder `np.logspace`).

### Erwartung

Der Abbruchfehler ist $O(\Delta x^2)$, und weil das Problem stabil ist ($\|A^{-1}\|$ bleibt beschränkt, kleinster Eigenwert $\approx \pi^2$), wird daraus ein globaler Fehler

$$\max_j |u_j - u(x_j)| \approx C\,\Delta x^2$$

Im log-log-Plot also **Steigung 2**. Für diese rechte Seite kann man die Konstante sogar angeben: $\sin(\pi x_j)$ ist ein Eigenvektor von $A$ (siehe 2c), deshalb ist die diskrete Lösung $u_j = \frac{\pi^2}{\lambda_1}\sin(\pi x_j)$ und der Fehler am Maximum $\approx \pi^2\Delta x^2/12$.

### Warum der Fehler wieder wächst

Mit kleinerem $\Delta x$ sinkt der Diskretisierungsfehler, aber der Rundungsfehler steigt. Die Einträge von $A$ sind $\sim 1/\Delta x^2$, und in $u_{j-1} - 2u_j + u_{j+1}$ ziehen sich fast gleich große Zahlen voneinander ab (Auslöschung, siehe Notiz *Rundungsfehler*). Anders gesagt: die Konditionszahl ist

$$\kappa(A) = \frac{\lambda_N}{\lambda_1} \approx \frac{4}{\pi^2\Delta x^2}$$

und der Rundungsfehler der Lösung skaliert grob wie $\varepsilon\,\kappa \sim \varepsilon/\Delta x^2$ mit $\varepsilon \approx 2{,}2\cdot10^{-16}$. Gesamtfehler etwa

$$E(\Delta x) \approx C\,\Delta x^2 + \frac{\varepsilon}{\Delta x^2}$$

Mit diesem Modell läge das Minimum bei $\Delta x \sim \varepsilon^{1/4} \approx 10^{-4}$, also $N \sim 10^4$, mit einem Fehler um $10^{-8}$. Dasselbe Bild wie beim Differenzenquotienten aus der ersten Einheit, nur mit Exponent 2 statt 1. Die Messung unten zeigt aber: $\varepsilon/\Delta x^2$ ist eine Schranke für den schlimmsten Fall, der echte Rundungsfehler ist viel kleiner.

### Messung (Notebook `exercise_5_fd.ipynb`, 1b)

$N = 2^k - 1$ für $k = 2,\dots,20$, also $N = 3$ bis $1\,048\,575$. Der größte Lauf dauert etwa 0,5 s.

- **Steigung 2,00** für $\Delta x \ge 1{,}2\cdot10^{-4}$ ($N \le 8191$). Der Fehler viertelt sich pro Halbierung von $\Delta x$ (Verhältnis 4,00) und liegt fast genau auf $\pi^2\Delta x^2/12$, etwa $1{,}23\cdot10^{-8}$ bei $N = 8191$.
- **Minimum** $2{,}3\cdot10^{-10}$ bei $\Delta x = 1{,}5\cdot10^{-5}$ ($N = 65535$). Das ist etwa eine Größenordnung feiner und zwei Größenordnungen genauer als die Schätzung $\varepsilon^{1/4}$.
- **Darunter** wächst der Fehler unregelmäßig und steil, bis $6{,}5\cdot10^{-7}$ bei $N \approx 10^6$. Er liegt dabei immer weit unter $\varepsilon/\Delta x^2$, hat aber keine saubere Steigung, weil sich Rundungsfehler statistisch aufsummieren.

Warum besser als die Schranke: Bei $\Delta x = 2^{-k}$ sind die Einträge von $A$ und die Gitterpunkte exakt, gerundet wird nur beim Auswerten von $f$ und beim Lösen. Die Schranke $\varepsilon\,\kappa$ nimmt an, dass sich alle Rundungsfehler in der ungünstigsten Richtung addieren, das passiert praktisch nie.

## 1c) Höhere Ordnung

### Stencils

Nimmt man mehr Nachbarn mit und wählt die Gewichte so, dass mehr Taylor-Terme verschwinden, steigt die Ordnung. Die zentralen Stencils für $u''$:

$$\text{Ordnung 2:}\quad \frac{1}{\Delta x^2}\,(1,\ -2,\ 1)$$

$$\text{Ordnung 4:}\quad \frac{1}{12\,\Delta x^2}\,(-1,\ 16,\ -30,\ 16,\ -1)$$

$$\text{Ordnung 6:}\quad \frac{1}{180\,\Delta x^2}\,(2,\ -27,\ 270,\ -490,\ 270,\ -27,\ 2)$$

Kontrolle: Die Gewichte summieren sich zu 0 (Konstante hat $u'' = 0$). Die Matrix bekommt entsprechend 5 bzw. 7 Diagonalen, `diags` kann das genauso.

Erwartung: Steigung 4 bzw. 6. Der Rundungsboden $\varepsilon/\Delta x^2$ bleibt etwa gleich, aber man erreicht ihn schon bei größerem $\Delta x$. Das Minimum liegt also tiefer und weiter links im Plot, danach steigt der Fehler wieder.

### Problem am Rand (Frage aus 1c)

Der 5-Punkte-Stern braucht an $x_1$ den Wert $u_{-1}$, der außerhalb des Intervalls liegt. Möglichkeiten:

- **Randnahe Zeilen mit dem 3-Punkte-Stern.** Einfach, der Abbruchfehler ist dort nur $O(\Delta x^2)$. Weil das nur zwei Zeilen betrifft, ist der globale Effekt oft kleiner als der lokale Fehler vermuten lässt. Ob die Ordnung 4 hält, sollte man messen, nicht annehmen.
- **Einseitige (unsymmetrische) Stencils** derselben Ordnung an den Randpunkten, mit Gewichten aus Taylor-Entwicklung. Hält die Ordnung, macht $A$ aber unsymmetrisch.
- **Geisterpunkt per Spiegelung:** $u_{-1} = -u_1$. Das ist richtig, wenn die Lösung um den Rand ungerade fortgesetzt werden kann, also $u = u'' = 0$ am Rand. Für $\sin\pi x$ stimmt das, im Allgemeinen nicht.

### Unterschied zwischen `"reduce"` und `"mirror"`

Beide lösen dasselbe Problem: Der 5-Punkte-Stern an $x_1$ braucht $u_{-1}$, also einen Wert links von $x = 0$, außerhalb des Intervalls. Am Beispiel Ordnung 4, Zeile 1, eigentlich

$$\frac{1}{12\,\Delta x^2}\big(u_{-1} - 16u_0 + 30u_1 - 16u_2 + u_3\big), \qquad u_0 = 0$$

**`"reduce"`: den Stern verkleinern.** Man verzichtet auf den fehlenden Punkt und nimmt in dieser Zeile den 3-Punkte-Stern, der ohne $u_{-1}$ auskommt:

$$\frac{1}{\Delta x^2}\big(-u_0 + 2u_1 - u_2\big)$$

- Man braucht keine Annahme über die Lösung außerhalb des Intervalls, das funktioniert immer.
- In dieser Zeile ist die Genauigkeit aber nur Ordnung 2, nicht 4. Der Rand ist die schwächste Stelle.
- $A$ wird unsymmetrisch, weil benachbarte Zeilen verschiedene Sterne haben.

**`"mirror"`: den fehlenden Wert erfinden.** Man behält den vollen 5-Punkte-Stern und ersetzt $u_{-1}$ durch einen Wert, den man aus der Randbedingung erschließt (Geisterpunkt). Bei $u(0) = 0$ setzt man die Lösung **ungerade** über den Rand fort, also wie eine Punktspiegelung am Ursprung:

$$u_{-1} = -u_1$$

Eingesetzt wird Zeile 1 zu $\frac{1}{12\,\Delta x^2}(29u_1 - 16u_2 + u_3)$, weil $u_{-1}$ mit Gewicht $+1$ auf $u_1$ mit Gewicht $-1$ wandert: $30 - 1 = 29$.

- Jede Zeile hat den vollen Stern, die Ordnung bleibt überall 4 (bzw. 6).
- $A$ bleibt symmetrisch.
- Die Annahme ist aber nur richtig, wenn sich die echte Lösung tatsächlich ungerade fortsetzen lässt, d. h. alle geraden Ableitungen am Rand verschwinden ($u = u'' = u'''' = \dots = 0$). Für $\sin\pi x$ stimmt das exakt. Für eine Lösung mit $u''(0) \ne 0$ wäre der Geisterpunkt falsch, und der Fehler am Rand wäre sogar größer als bei `"reduce"`.

**Kurz:** `"reduce"` lässt am Rand Genauigkeit weg und ist dafür immer anwendbar. `"mirror"` behält die volle Genauigkeit, braucht dafür aber Wissen über die Lösung jenseits des Randes. In der Messung kostet `"reduce"` bei Ordnung 6 eine Ordnung (5 statt 6), bei Ordnung 4 fällt es nicht auf.

### Umsetzung in `fd.py`

`laplacian(N, order, boundary)` mit `order` ∈ {2, 4, 6} und `boundary` ∈ {`"reduce"`, `"mirror"`}. Die Gewichte stehen in `STENCILS`. Das Innere baut `scipy.sparse.diags`, die bis zu drei randnahen Zeilen setzt `_edge_row` neu:

- `"reduce"`: der größte zentrale Stern, der in die Zeile passt. Dadurch wird $A$ unsymmetrisch (Zeile 1 hat den 3-Punkte-Stern, Zeile 2 schon den 5-Punkte-Stern).
- `"mirror"`: Der Beitrag des Geisterpunkts $u_{-m} = -u_m$ geht mit umgekehrtem Vorzeichen auf Spalte $m$. Bei Ordnung 4 wird die Diagonale der ersten Zeile so $30 - 1 = 29$ (mal $1/12$). $A$ bleibt symmetrisch.

### Messung (Notebook, 1c)

Raten $\log_2\big(e(\Delta x)/e(\Delta x/2)\big)$ für $N = 7 \dots 255$:

- Ordnung 2: 2,00
- Ordnung 4, `"mirror"`: 4,00
- Ordnung 4, `"reduce"`: 4,66 → 4,14, also von oben gegen 4
- Ordnung 6, `"mirror"`: 6,00
- Ordnung 6, `"reduce"`: **5,00**

Rundungsboden: Ordnung 4 erreicht $6\cdot10^{-12}$ bei $N = 1023$, Ordnung 6 mit Spiegelung $2\cdot10^{-13}$ schon bei $N = 127$. Danach steigen alle Kurven gleich an. Der Rundungsfehler hängt also nicht vom Stern ab.

### Warum Ordnung 6 mit `"reduce"` nur 5 ergibt

Ein Abbruchfehler $\tau_i$ in Zeile $i$ verschiebt die Lösung um $(A^{-1})_{\cdot i}\,\tau_i$. Die Spalte von $A^{-1}$ ist $\Delta x$ mal die Green'sche Funktion $G(x, x_i) = x_i(1-x)$ für $x > x_i$. Nahe am Rand ist $x_i \sim \Delta x$, der Beitrag also etwa $\Delta x^2\,\tau_i$, und zwar überall im Gebiet.

In Zeile 1 steht der 3-Punkte-Stern mit $\tau_1 = \frac{\Delta x^2}{12}u''''(x_1)$. Für $\sin\pi x$ ist $u''''(0) = 0$, also $u''''(x_1) \approx \pi^5\Delta x$ und $\tau_1 \sim \Delta x^3$. Beitrag: $\Delta x^5$.

- Ordnung 4: Der innere Fehler $\Delta x^4$ ist größer als der Randbeitrag $\Delta x^5$, die Ordnung bleibt 4.
- Ordnung 6: Der Randbeitrag $\Delta x^5$ dominiert, man verliert eine Ordnung.

Das zeigt auch der Plot des Fehlers über $x$ bei $N = 63$: Mit `"reduce"` ist der Fehler im ganzen Intervall etwa gleich groß ($2{,}7\cdot10^{-8}$), nicht nur am Rand. Mit Spiegelung ist er 1000-mal kleiner ($2{,}5\cdot10^{-11}$). Für eine Lösung mit $u''''(0) \ne 0$ wäre $\tau_1 \sim \Delta x^2$, und beide höheren Ordnungen würden auf 4 fallen. Das habe ich nicht gemessen.

### Was die neuen Tests prüfen

- innere Zeilen sind genau der 5- bzw. 7-Punkte-Stern
- `"reduce"`: Randzeilen haben den 3-Punkte-Stern, die zweite Zeile bei Ordnung 6 den 5-Punkte-Stern
- `"mirror"`: erste Zeile $[29, -16, 1]/12$, Matrix symmetrisch
- Exaktheit: $x^4$ bzw. $x^6$ werden auf inneren Zeilen exakt differenziert
- gemessene Rate zwischen $N = 31$ und $63$: 4 und 6 mit Spiegelung (±0,2), über 3,8 für Ordnung 4 mit `"reduce"`

## 1d) Neumann-Rand $u'(1) = 0$

Neue Testaufgabe: $f = \frac{\pi^2}{4}\sin\frac{\pi x}{2}$, exakte Lösung $u = \sin\frac{\pi x}{2}$. Probe: $u'' = -\frac{\pi^2}{4}\sin\frac{\pi x}{2}$, also $-u'' = f$; und $u'(1) = \frac{\pi}{2}\cos\frac{\pi}{2} = 0$.

Jetzt ist $u(1)$ unbekannt, $u_{N+1}$ kommt also als zusätzliche Unbekannte dazu und braucht eine eigene Gleichung. Drei Varianten:

- **Einseitig, 1. Ordnung:** $(u_{N+1} - u_N)/\Delta x = 0$. Abbruchfehler $O(\Delta x)$, globaler Fehler **Steigung 1**.
- **Geisterpunkt, 2. Ordnung:** Zentrale Ableitung $(u_{N+2} - u_N)/(2\Delta x) = 0$ ergibt $u_{N+2} = u_N$. Einsetzen in den normalen Stern am Punkt $N+1$: $(-2u_N + 2u_{N+1})/\Delta x^2 = f_{N+1}$. Teilt man die Zeile durch 2, bleibt $A$ symmetrisch (dann rechts $f_{N+1}/2$). **Steigung 2.**
- **Einseitig, 2. Ordnung:** $(3u_{N+1} - 4u_N + u_{N-1})/(2\Delta x) = 0$. Auch Steigung 2, aber $A$ unsymmetrisch.

### Wirkung auf die ganze Lösung (Frage aus 1d)

Bei einer elliptischen Gleichung hängt jeder Punkt von allen anderen ab ($A^{-1}$ ist voll besetzt, die Green'sche Funktion). Ein Fehler in einer einzigen Randzeile bleibt deshalb nicht am Rand, sondern verschiebt die Lösung überall. Mit der Variante 1. Ordnung ist der Fehler im ganzen Intervall $O(\Delta x)$, obwohl alle inneren Zeilen 2. Ordnung sind. Die schwächste Stelle bestimmt die Ordnung. Am besten zeigt man das mit einem Plot des Fehlers über $x$ für beide Varianten.

### Umsetzung in `fd.py`

`solve_neumann(f, N, bc)` mit `bc` ∈ {`"first"`, `"ghost"`}. Das Gitter hat jetzt $N+1$ Unbekannte $x_1, \dots, x_{N+1} = 1$, $\Delta x = 1/(N+1)$ bleibt. Die Matrix ist der 3-Punkte-Stern in der Größe $(N+1)\times(N+1)$, die letzte Zeile ist in beiden Fällen $(-1,\ 1)/\Delta x^2$. Der Unterschied steckt nur in der rechten Seite: 0 bei `"first"`, $f(1)/2$ bei `"ghost"`. Damit bleibt $A$ in beiden Fällen symmetrisch. Die Variante einseitig 2. Ordnung habe ich weggelassen.

### Messung (Notebook, 1d)

- `"first"`: Raten 0,97 → 1,00, also **Steigung 1**. Bei $N = 15$ ist der Fehler am Rand $-0{,}076$.
- `"ghost"`: Raten 2,00, **Steigung 2**. Bei $N = 15$ ist der Fehler etwa $8\cdot10^{-4}$, rund 100-mal kleiner.

### Warum der Fehler bei `"first"` eine Gerade ist

Taylor um $x = 1$:

$$\frac{u_{N+1} - u_N}{\Delta x} = u'(1) - \frac{\Delta x}{2}u''(1) + O(\Delta x^2)$$

Die Zeile „$= 0$“ erzwingt also in Wahrheit $u'(1) = \frac{\Delta x}{2}u''(1) = -\frac{\pi^2}{8}\Delta x$ statt 0. Die Differenz zur echten Lösung erfüllt $-e'' = 0$ mit $e(0) = 0$ und $e'(1) = -\frac{\pi^2}{8}\Delta x$, also

$$e(x) \approx -\frac{\pi^2}{8}\,\Delta x\; x$$

eine Gerade durch den Ursprung. Der Plot bei $N = 15$ liegt genau auf dieser Geraden, der Test prüft sie bei $N = 63$ auf 2 %. Ein falscher Fluss am Rand kippt die ganze Lösung. Das ist die Antwort auf die Frage, wie sich die Randdiskretisierung auf die ganze Lösung auswirkt.

### Was die Tests prüfen

- Das Gitter enthält den rechten Randpunkt $x = 1$.
- `"ghost"` ist exakt für $u = x(2-x)$, weil zentrale Differenz und Stern Polynome 2. Grades exakt behandeln.
- Die Raten liegen bei 1 bzw. 2 (±0,1).
- Bei `"first"` ist der Fehler die Gerade $-\frac{\pi^2}{8}\Delta x\,x$.

## 1e) 2D: Poisson auf dem Einheitsquadrat

$-\Delta u = -(u_{xx} + u_{yy}) = 2\pi^2\sin\pi x\,\sin\pi y$, exakt $u = \sin\pi x\,\sin\pi y$. Gitter $N\times N$ innere Punkte, also $n = N^2$ Unbekannte.

### Matrix über das Kronecker-Produkt

Ordnet man die Unbekannten zu einem Vektor (Index $i$ für $x$, $j$ für $y$), wirkt die 1D-Matrix $A$ in $x$-Richtung auf den ersten und in $y$-Richtung auf den zweiten Index:

$$A_{2D} = A\otimes I + I\otimes A$$

In scipy: `sp.kron(A, I) + sp.kron(I, A)` mit `I = sp.identity(N)`. Das ist der 5-Punkte-Stern $(4u_{ij} - u_{i\pm1,j} - u_{i,j\pm1})/\Delta x^2$. Wichtig: Die rechte Seite muss in derselben Reihenfolge zu einem Vektor werden. Mit `np.meshgrid(x, x, indexing="ij")` und `.ravel()` (Zeilen zuerst) passt das zu `kron(A, I)` für den ersten Index.

Eigenwerte: $\lambda_k + \lambda_l$ aus 1D, Eigenvektoren $\sin(k\pi x_i)\sin(l\pi y_j)$. Die rechte Seite ist wieder genau ein Eigenvektor.

### Skalierung (Frage aus 1e)

- **Fehler:** immer noch $O(\Delta x^2) = O(N^{-2})$, Steigung 2 über $\Delta x$.
- **Laufzeit:** In 1D kostet das Tridiagonalsystem $O(N)$. In 2D ist die Matrix gebändert mit Bandbreite $N$, Gauß ohne Tricks also $O(n\cdot N^2) = O(N^4)$. Der Sparse-Löser (SuperLU in `spsolve`) sortiert die Unbekannten um und kommt damit besser weg, in der Theorie bis etwa $O(n^{1,5}) = O(N^3)$. Dazu kommt der Speicher für das Auffüllen (fill-in) der Faktoren. Was davon bei uns rauskommt, zeigt eine Zeitmessung für mehrere $N$ im log-log-Plot. Für sehr große $N$ wären iterative Löser (CG, Mehrgitter) die nächste Stufe.

## 2. Wärmeleitung in der Zeit: Linienmethode

$u_t = \alpha u_{xx}$ mit $\alpha = 1$, Start $u(x,0) = \sin\pi x$, exakt $u = e^{-\pi^2 t}\sin\pi x$.

## 2a) Linienmethode (method of lines)

Nur der Raum wird diskretisiert, die Zeit bleibt kontinuierlich. Weil $A$ die Matrix von $-\frac{d^2}{dx^2}$ ist, wird aus der PDE ein ODE-System

$$\dot u = -A\,u, \qquad u(0) = \big(\sin\pi x_j\big)_j$$

Das ist genau die lineare Form $\dot x = Jx$ aus Exercise 3, mit $J = -A$. Vorsicht beim Vorzeichen: $A$ ist positiv definit, $J$ hat die negativen Eigenwerte.

Die drei Verfahren mit Stabilitätsfunktion $R(z)$, $z = \lambda\Delta t$ für $\dot y = \lambda y$:

$$\text{expl. Euler:}\quad u^{n+1} = (I - \Delta t A)\,u^n, \qquad R(z) = 1 + z$$

$$\text{impl. Euler:}\quad (I + \Delta t A)\,u^{n+1} = u^n, \qquad R(z) = \frac{1}{1-z}$$

$$\text{Crank–Nicolson:}\quad \Big(I + \tfrac{\Delta t}{2}A\Big)u^{n+1} = \Big(I - \tfrac{\Delta t}{2}A\Big)u^n, \qquad R(z) = \frac{1 + z/2}{1 - z/2}$$

Crank–Nicolson ist die Trapezregel: Mittelwert aus explizitem und implizitem Euler, Ordnung 2.

### Was aus Exercise 3 passt unverändert? (Frage aus 2a)

In `2026-10-28/ode.py` nehmen alle Schritte eine Matrix `J` und einen Zustand `x`:

- `euler_step`, `heun_step`, `rk4_step`: nutzen nur `J @ x`. Das funktioniert auch mit einer sparse-Matrix, also **unverändert** mit `J = -A`.
- `integrate`: ruft nur `step(J, x, dt)` auf, **unverändert**.
- `order`, `stability_function`: unabhängig vom Problem, **unverändert**.
- `implicit_euler_step`: benutzt `np.linalg.solve(np.eye(len(x)) - dt * J, x)`. Mit sparse `J` geht das nicht (bzw. wird dense und langsam). Hier braucht es `spsolve` mit `sp.identity`. Und weil sich die Matrix nie ändert, könnte man sie einmal faktorisieren (`splu`) statt in jedem Schritt neu zu lösen.
- Crank–Nicolson gibt es noch nicht, kommt neu dazu.
- `critical_dt` rechnet mit dem größeren der beiden übergebenen Eigenwerte und läuft deshalb unverändert, wenn man $\lambda_1, \lambda_N$ statt $k_1, k_2$ übergibt (siehe 2c).

## 2b) Genauigkeit

### Zwei Referenzlösungen

Der Startvektor $\sin(\pi x_j)$ ist ein Eigenvektor von $A$ zum kleinsten Eigenwert $\lambda_1$. Deshalb kennt man auch die exakte Lösung des ODE-Systems:

$$u_j^{\text{ODE}}(t) = e^{-\lambda_1 t}\sin(\pi x_j), \qquad \lambda_1 = \frac{4}{\Delta x^2}\sin^2\frac{\pi\Delta x}{2} \approx \pi^2 - \frac{\pi^4\Delta x^2}{12}$$

Damit lässt sich der Fehler sauber trennen:

- **Vergleich mit $u^{\text{ODE}}$:** nur der Zeitfehler.
- **Vergleich mit der PDE-Lösung $e^{-\pi^2 t}\sin\pi x_j$:** Zeitfehler plus Raumfehler.
- **Raumfehler allein:** $|e^{-\lambda_1 t} - e^{-\pi^2 t}|$, Steigung 2 in $\Delta x$.

### Erwartete Steigungen

- In $\Delta t$ (N = 50 fest): expliziter und impliziter Euler **Steigung 1**, Crank–Nicolson **Steigung 2**.
- In $\Delta x$ (kleines $\Delta t$ fest): **Steigung 2** für alle, weil $A$ zweiter Ordnung ist.

### Warum der Fehler aufhört zu fallen (Frage aus 2b)

Gegen die PDE-Lösung kann der Gesamtfehler nicht unter den Raumfehler fallen. Für $N = 50$ bei $t = 0{,}1$:

$$|e^{-\lambda_1 t} - e^{-\pi^2 t}| \approx 1{,}2\cdot10^{-4}$$

Sobald der Zeitfehler darunter liegt, bleibt die Kurve auf diesem Plateau. Umgekehrt bei der Raumstudie: dort bildet der Zeitfehler des festen $\Delta t$ den Boden. Gegen die ODE-Lösung dagegen fällt der Fehler weiter, bis Rundung (viele Schritte, jeder mit $\varepsilon$) ihn stoppt.

### Falle beim expliziten Euler

Für $N = 50$ ist der explizite Euler nur für $\Delta t < 1{,}92\cdot10^{-4}$ stabil (siehe 2c). Bei der $\Delta t$-Studie gibt es für ihn also nur einen schmalen Bereich. Interessant: Der Startwert enthält nur Mode 1, in exakter Arithmetik würde nichts explodieren. Die hohen Moden kommen erst über Rundungsfehler ($\sim10^{-16}$) hinein und wachsen dann pro Schritt um den Faktor $|1 - \lambda_N\Delta t|$. Etwas über der Grenze dauert es deshalb viele Schritte, bis man die Instabilität sieht.

## 2c) Stabilität

### Eigenwerte und Eigenvektoren von $A$

Exakt bekannt (Ansatz $v_j = \sin(k\pi x_j)$ in den Stern einsetzen, Additionstheorem):

$$\lambda_k = \frac{4}{\Delta x^2}\sin^2\frac{k\pi\Delta x}{2}, \qquad v_{k,j} = \sin(k\pi x_j), \qquad k = 1,\dots,N$$

- $\lambda_1 \approx \pi^2$: glatte Grundmode, eine halbe Sinuswelle.
- $\lambda_N \approx 4/\Delta x^2$: höchste Mode, wechselt von Punkt zu Punkt fast das Vorzeichen (Sägezahn, $\approx(-1)^j$ mal Einhüllende).

Die Eigenvektoren sind die diskrete Version der Fourier-Sinusreihe, die Einheit zur DFT lässt grüßen. Beim Plotten von `np.linalg.eigh` darauf achten: Vorzeichen und Normierung der Eigenvektoren sind beliebig.

### Stabilitätsbedingung

Mode $k$ wird pro Schritt mit $R(z_k)$, $z_k = -\lambda_k\Delta t$, multipliziert. Stabil heißt $|R(z_k)| \le 1$ für alle $k$. Alle $z_k$ liegen auf der negativen reellen Achse, zwischen $\approx -\pi^2\Delta t$ und $-4\Delta t/\Delta x^2$. Die Stabilitätsgebiete auf der negativen reellen Achse (aus Exercise 3, Notiz *Stabilitätsgebiet*):

- expliziter Euler: $z \in [-2, 0]$
- Heun: $z \in [-2, 0]$
- RK4: $z \in [-2{,}785, 0]$
- impliziter Euler, Crank–Nicolson: die ganze linke Halbebene (A-stabil)

**Entscheidend ist der größte Eigenwert $\lambda_N$** (Frage aus 2c), denn $z_N$ liegt am weitesten links. Kritischer Zeitschritt:

$$\Delta t_{\text{krit}} = \frac{2}{\lambda_N}\ \ \text{(Euler, Heun)}, \qquad \frac{2{,}785}{\lambda_N}\ \ \text{(RK4)}$$

### Von-Neumann-Analyse

Statt der Matrix nimmt man eine einzelne Fourier-Welle $u_j^n = g^n e^{i\kappa j\Delta x}$ und setzt sie in den expliziten Euler ein. Mit $r = \Delta t/\Delta x^2$:

$$g = 1 - 4r\sin^2\frac{\kappa\Delta x}{2}$$

$|g| \le 1$ für alle $\kappa$ verlangt $4r \le 2$, also

$$\Delta t \le \frac{\Delta x^2}{2}$$

Die Matrix-Rechnung ergibt etwas mehr, weil $\sin^2(N\pi\Delta x/2)$ knapp unter 1 liegt. Vorab gerechnet:

- $N = 10$: $\Delta t_{\text{krit}} = 4{,}22\cdot10^{-3}$, von Neumann $4{,}13\cdot10^{-3}$
- $N = 20$: $1{,}140\cdot10^{-3}$ gegen $1{,}134\cdot10^{-3}$
- $N = 40$: $2{,}979\cdot10^{-4}$ gegen $2{,}974\cdot10^{-4}$
- $N = 80$: $7{,}624\cdot10^{-5}$ gegen $7{,}621\cdot10^{-5}$

Mit feinerem Gitter nähern sich beide an. Und: Halbes $\Delta x$ heißt ein Viertel $\Delta t$. Das ist die harte Grenze expliziter Verfahren für Diffusion.

### Knapp über der Grenze (Frage aus 2c)

$R(z_N) \approx -1 - \delta$: Die höchste Mode wechselt jeden Schritt das Vorzeichen und wird langsam größer. Man sieht ein Zickzack von Punkt zu Punkt, das von Schritt zu Schritt umklappt und exponentiell wächst, während die glatte Lösung brav abklingt. Die anderen Moden sind noch stabil.

### Warum implizit immer stabil ist (Frage aus 2c)

Für reelles $z < 0$:

$$|R_{\text{IE}}(z)| = \frac{1}{1 + |z|} < 1, \qquad |R_{\text{CN}}(z)| = \frac{|1 - |z|/2|}{1 + |z|/2} < 1$$

egal wie groß $|z|$ wird. Beide Verfahren sind A-stabil. Der Preis: In jedem Schritt ein lineares Gleichungssystem lösen. Bei tridiagonalem $A$ ist das $O(N)$, also kaum teurer als ein expliziter Schritt.

## 2d) Steifheit und unglatte Startwerte

Neuer Start: Rechteckpuls $u(x,0) = 1$ für $0{,}25 < x < 0{,}75$, sonst 0. $N = 99$, also $\Delta x = 0{,}01$. Bei $\Delta t = 0{,}01$ ist $r = \Delta t/\Delta x^2 = 100$.

### Warum das ein anderes Problem ist

Der Sprung enthält **alle** Moden, die Sinus-Koeffizienten fallen nur wie $1/k$:

$$b_k = 2\int_{0{,}25}^{0{,}75}\sin(k\pi x)\,dx = \frac{2}{k\pi}\Big(\cos\frac{k\pi}{4} - \cos\frac{3k\pi}{4}\Big)$$

Die exakte PDE-Lösung ist die Reihe $\sum_k b_k e^{-k^2\pi^2 t}\sin(k\pi x)$. Die exakte ODE-Lösung bekommt man genauso über die Eigenzerlegung von $A$: Start in Eigenvektoren zerlegen, jede Komponente mit $e^{-\lambda_k t}$ multiplizieren. Das sind die Referenzen für die Wiederholung von 2b.

Die Eigenwerte reichen von $\lambda_1 \approx 9{,}87$ bis $\lambda_N \approx 4\cdot10^4$, Verhältnis rund 4000: ein steifes System (Notiz *Steife Differentialgleichungen*). Bei $\Delta t = 0{,}01$ ist $z_N = -\lambda_N\Delta t \approx -400$.

### Warum Crank–Nicolson jetzt schlechter ist (Frage aus 2d)

Für die höchste Mode:

$$R_{\text{CN}}(-400) = \frac{1 - 200}{1 + 200} = -\frac{199}{201} \approx -0{,}990, \qquad R_{\text{IE}}(-400) = \frac{1}{401} \approx 0{,}0025, \qquad e^{-400} \approx 0$$

Allgemein gilt $R_{\text{CN}}(z) \to -1$ für $z \to -\infty$, aber $R_{\text{IE}}(z) \to 0$. Der implizite Euler dämpft schnelle Moden fast sofort weg, so wie die echte Lösung (L-stabil). Crank–Nicolson ist zwar stabil, dämpft sie aber kaum und lässt sie jeden Schritt das Vorzeichen wechseln. An den Sprungstellen sieht man deshalb hartnäckige Zickzack-Schwingungen.

Im $|R(z)|$-Plot für $z \in [-1000, 0]$: $e^z$ fällt sofort auf 0, $|R_{\text{IE}}|$ fällt wie $1/|z|$, $|R_{\text{CN}}|$ hat eine Nullstelle bei $z = -2$ und steigt danach wieder gegen 1.

In 2b war CN das beste Verfahren, weil die Lösung nur aus Mode 1 bestand, wo $|z|$ klein ist und die Ordnung 2 zählt. Hier entscheiden die Moden mit großem $|z|$, und dort ist CN qualitativ falsch.

### Verschwinden die schnellen Moden? (Frage aus 2d)

Bei CN schrumpft die höchste Mode pro Schritt nur um den Faktor 0,990. Nach 100 Schritten ($t = 1$) bleibt $(199/201)^{100} \approx 0{,}37$ der Anfangsamplitude. Die glatte Grundmode ist dann schon auf $e^{-\pi^2} \approx 5\cdot10^{-5}$ abgefallen. Die falschen Schwingungen dominieren also irgendwann die ganze Lösung. Sie klingen geometrisch ab, verschwinden also nur asymptotisch. Gängige Abhilfe (zum Ausprobieren): die ersten paar Schritte mit implizitem Euler machen, danach CN.

### Expliziter Euler zum Vergleich (Frage aus 2d)

Für $N = 99$: $\lambda_N = 39\,990$, also $\Delta t_{\text{krit}} = 2/\lambda_N = 5{,}0\cdot10^{-5}$. Bis $t = 0{,}1$ sind das mindestens 2000 Schritte statt 10 mit $\Delta t = 0{,}01$. Halbiert man $\Delta x$, braucht man viermal so viele Schritte, und jeder Schritt kostet doppelt so viel: Aufwand $\propto N^3$. Das ist der Grund, warum man Diffusion implizit rechnet.

## Was nachzulesen hilft

Im Notizbuch: *Euler-Verfahren*, *Runge-Kutta-Verfahren*, *Stabilitätsgebiet*, *Steife Differentialgleichungen*, *Rundungsfehler*, *Diskrete Fourier-Transformation*. Neue Notizen nach der Einheit wahrscheinlich: Finite Differenzen, Linienmethode, Crank–Nicolson, Von-Neumann-Stabilitätsanalyse.
