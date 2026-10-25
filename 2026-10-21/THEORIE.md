# Theorie: Entmagnetisierungsfeld per FFT

Exercise 2 · 21.10.2026

Dieses Dokument erklärt die Theorie hinter allen Teilaufgaben. Die Verweise
(1a, 2b, …) beziehen sich auf die Angabe in `exercise.pdf`. Code in `demag.py`,
Tests in `test_demag.py`, Messungen in `exercise_2_demag.ipynb`.

---

## 1. Das Streufeld ist eine Faltung

Eine Magnetisierung $\vec m(\vec r)$ erzeugt ein Feld $\vec h$. Jedes Volumenelement
wirkt wie ein kleiner Dipol, und die Felder addieren sich:

$$
\vec h(\vec r) = \int K(\vec r - \vec r\,')\, \vec m(\vec r\,')\, d\vec r\,'
$$

Der Kernel $K$ hängt nur vom **Abstand** $\vec r - \vec r\,'$ ab, nicht vom Ort selbst.
Genau das macht das Integral zu einer Faltung, $\vec h = K * \vec m$.

Auf einem Gitter mit $N$ Zellen (Größe 1) wird daraus eine Summe:

$$
\vec h_i = \sum_j K(\vec r_i - \vec r_j)\, \vec m_j
$$

Direkt gerechnet sind das $N$ Summen mit je $N$ Termen, also $O(N^2)$.
Mit dem Faltungssatz geht es in $O(N\log N)$.

---

## 2. Dipolfeld und 1D-Kernel (Aufgabe 1a)

### Feld eines Punktdipols

$$
\vec h(\vec r) = \frac{1}{4\pi r^3}\big(3(\vec m\cdot\hat r)\,\hat r - \vec m\big)
$$

Das Feld fällt wie $1/r^3$. Auf der Achse des Dipols zeigt es in Richtung $\vec m$
(Stärke $2m/4\pi r^3$), seitlich gegen $\vec m$ (Stärke $m/4\pi r^3$).

### Der 1D-Kernel

Liegen alle Dipole und Auswertepunkte auf der x-Achse und zeigt $\vec m$ in $x$,
dann ist $\hat r = \pm\hat x$ und $(\vec m\cdot\hat r)\hat r = m_x\hat x$. Also

$$
h_x = \frac{3 - 1}{4\pi|x|^3}\, m_x = K_{xx}(x)\, m_x,
\qquad K_{xx}(x) = \frac{1}{2\pi|x|^3}.
$$

$K_{xx}$ ist gerade und positiv: Dipole auf einer Linie verstärken sich gegenseitig.

### Selbstwechselwirkung

Bei $x = 0$ divergiert der Punktdipol. Die Zelle ist aber kein Punkt, sondern ein
homogen magnetisierter Würfel. Für ihn gilt im Mittel $\vec h = -N\vec m$ mit dem
Entmagnetisierungsfaktor $N$. Aus Symmetrie ist $N_x = N_y = N_z$, und für jeden
Körper gilt $N_x + N_y + N_z = 1$, also $N = 1/3$. Deshalb:

$$
K(0) = -\tfrac13\, I, \qquad K_{xx}(0) = -\tfrac13 .
$$

Dass genau $-1/3$ richtig ist, folgt auch aus Abschnitt 7: Es ist der Delta-Anteil
der zweiten Ableitung von $1/r$. Mit $K(0) = 0$ würde im Inneren jedes Körpers
genau $-\vec m/3$ fehlen.

---

## 3. Direkte Summe am Segment (Aufgabe 1b)

Ein Segment aus Dipolen $m_x = 1$:

- **In der Mitte** sieht eine Zelle links und rechts unendlich viele Nachbarn:
  $h = -\tfrac13 + 2\sum_{n\ge1}\frac{1}{2\pi n^3} = -\tfrac13 + \frac{\zeta(3)}{\pi} = +0.049$.
- **Am Ende** fehlen die Nachbarn auf einer Seite:
  $h = -\tfrac13 + \frac{\zeta(3)}{2\pi} = -0.142$. Das Feld ist entmagnetisierend.
- **Außerhalb** zeigt das Streufeld in Richtung $\vec m$ und fällt ab.

Anschaulich sitzen die magnetischen Ladungen $\rho = -\partial_x m_x$ an den Enden
des Segments, dort ändert sich das Feld am stärksten. Weil $K \sim 1/x^3$ schnell
abfällt, ist der Übergang auf wenige Zellen beschränkt. Gemessen: $+0.0487$ in der
Mitte, $-0.1421$ am Rand.

Ein echter langer Stab hätte in der Mitte $h \approx 0$. Das kleine Plus ist ein
Fehler der Punktdipol-Näherung zwischen direkten Nachbarn.

---

## 4. Faltungssatz, Zero-Padding und Kernel-Anordnung (Aufgabe 1c)

### Faltungssatz

Die DFT macht aus einer Faltung ein Produkt:

$$
\hat h_k = \hat K_k\, \hat m_k
\quad\Rightarrow\quad
h = \mathcal F^{-1}\big(\mathcal F(K)\cdot\mathcal F(m)\big).
$$

Drei FFTs à $O(N\log N)$ und ein punktweises Produkt à $O(N)$.

### Das Problem: die DFT faltet zyklisch

Die DFT behandelt jedes Array als periodisch. Das Produkt der Spektren entspricht

$$
h_i = \sum_j K_{(i-j) \bmod L}\, m_j .
$$

Bei $L = N$ wird ein negativer Abstand $i - j$ auf $i - j + N$ abgebildet. Der Dipol
ganz links wirkt dann auf die Zelle ganz rechts wie ein direkter Nachbar, als wäre
die Kette zu einem Ring geschlossen. Gemessen: Ein einzelner Dipol am linken Rand
erzeugt am rechten Ende zyklisch $0.159 = K(1)$ statt richtig $6.4\cdot10^{-7} = K(63)$.

### Die Lösung: Zero-Padding

Es gibt $2N - 1$ verschiedene Abstände, von $-(N-1)$ bis $N - 1$. Ist $L \ge 2N - 1$,
bekommt jeder Abstand seinen eigenen Index modulo $L$, und die zyklische Faltung
ist gleich der linearen. Man füllt $m$ mit Nullen auf Länge $L$ auf, rechnet zyklisch
und nimmt vom Ergebnis nur die ersten $N$ Werte.

### Anordnung des Kernels (Wraparound)

Der Kernel muss auf denselben Ring passen:

- Index $0, 1, \dots, N-1$ enthält die Abstände $0, 1, \dots, N-1$,
- die negativen Abstände $-(N-1), \dots, -1$ stehen am **Ende** des Arrays,
- Indizes dazwischen (bei $L > 2N - 1$) treffen nur Nullen von $m$ und sind egal.

Das ist genau die Reihenfolge von `np.fft.fftfreq(L, 1/L)`:
`0, 1, …, L/2-1, -L/2, …, -1`. In 3D gilt dasselbe in jeder Achse einzeln.

---

## 5. Laufzeit und Längen $N \ne 2^m$ (Aufgabe 1d)

- **Direkte Summe:** $O(N^2)$. Gemessen lokal $N^{1.9}$ bei großem $N$; bei kleinem $N$
  drückt der Python-Overhead pro Zeile die Steigung.
- **FFT-Faltung:** $O(N\log N)$, gemessen etwa $6\,\text{ns}\cdot N\log_2 N$.
- Der Vorsprung wächst mit $N$: Faktor 18 bei $N = 64$, 1150 bei $N = 8192$.

### Was passiert für $N \ne 2^m$?

Die FFT braucht keine Zweierpotenz. pocketfft (hinter `np.fft`) zerlegt $L$ in
seine Primfaktoren und macht für jeden Faktor einen Cooley-Tukey-Schritt
(Mixed Radix). Teuer ist nur ein **großer** Primfaktor:

- glatte Länge ($L = 2\cdot 3^4 5^4$): so schnell wie eine Zweierpotenz,
- $L = 2p$ mit Primzahl $p$ (z. B. $N = 65537$): etwa 5-mal langsamer.

Für große Primfaktoren nutzt pocketfft den Bluestein-Algorithmus: Er schreibt die
DFT selbst als Faltung und rechnet diese mit FFTs einer glatten Länge $\ge 2p - 1$.
Das bleibt $O(N\log N)$, kostet aber mehrere FFTs statt einer.

---

## 6. Optimierung (Aufgabe 1e)

Weil $m$ ohnehin gepaddet wird, darf man die Länge frei wählen. Drei Schritte:

1. **Reelle FFT** (`rfft`/`irfft`): Für reelle Daten ist $\hat a_{L-k} = \hat a_k^*$,
   die Hälfte des Spektrums ist überflüssig. Spart etwa 40 %.
2. **Kernel-Spektrum einmal berechnen**: $\hat K$ hängt nur von $L$ ab. In einer
   Simulation ändert sich $m$ jeden Zeitschritt, $K$ nie. Pro Aufruf bleiben 2 statt 3 FFTs.
3. **Glatte Länge**: `scipy.fft.next_fast_len(2N - 1)` statt $2N$. Vermeidet große
   Primfaktoren.

Gemessen: 3.4- bis 4.1-mal schneller bei $N = 2^k$, 22-mal bei Primzahl-$N$.
Weitere Möglichkeiten: mehrere Threads (`workers=-1`), FFTW mit vorab geplantem
Transform, GPU, Pruned FFTs (die Hälfte der Eingabe ist null, die Hälfte der
Ausgabe wird verworfen), `float32` wenn $10^{-7}$ reichen.

---

## 7. Tensorkernel in 3D (Aufgabe 2a)

Das Dipolfeld ist linear in $\vec m$, also eine Matrix mal Vektor:

$$
h_i = \sum_j K_{ij}\, m_j,
\qquad
K_{ij}(\vec r) = \frac{1}{4\pi r^3}\Big(3\frac{r_i r_j}{r^2} - \delta_{ij}\Big),
\qquad K_{ij}(0) = -\tfrac13\delta_{ij}.
$$

- $K$ ist **symmetrisch** ($K_{ij} = K_{ji}$): 6 unabhängige Komponenten
  ($xx, yy, zz, xy, xz, yz$).
- $K$ ist für $r \ne 0$ **spurfrei**: $K_{xx} + K_{yy} + K_{zz} = 0$.

**Anzahl der FFTs:** 6 Kernel-FFTs, einmal vorab. Pro Feldauswertung 3 Vorwärts-FFTs
($m_x, m_y, m_z$), im Fourierraum $\hat h_i = \sum_j \hat K_{ij}\hat m_j$ (9 punktweise
Produkte), dann 3 Rück-FFTs. Also **6 FFTs pro Auswertung**, jede auf $(2n)^3$.

---

## 8. Validierung (Aufgabe 2b)

Zwei Fehlerquellen muss man getrennt prüfen.

### Arithmetik: rechnet die FFT die richtige Summe?

- Vergleich mit der direkten Summe auf kleinen Gittern mit zufälligem $\vec m$.
- Einzelner Dipol: das Feld muss an jeder Zelle `dipole_field` sein.

Gemessen: $10^{-16}$ bis $3\cdot10^{-15}$ ($\le 13\,\varepsilon$), langsam wachsend mit der
Gittergröße, weil mehr Terme summiert werden. Die FFT ist kein Näherungsverfahren.

### Diskretisierung: ist die Summe die richtige Physik?

- **Kugel:** analytisch $\vec h = -\vec m/3$ im Inneren.
- **Fernfeld:** ein kleiner Körper wirkt von weitem wie ein Punktdipol mit dem
  Gesamtmoment.

Gemessen: Im Kern der Kugel $|h_z + 1/3| \approx 10^{-3}$, besser mit wachsendem Radius.
In den Randzellen bis $0.26$, und das wird **nicht** besser: Die Treppenstufen der
gerasterten Kugel sehen aus Sicht einer Randzelle bei jeder Auflösung gleich aus.
Der Fehler hängt also von der Auflösung, vom Abstand zur Oberfläche und von der
Geometrie ab.

Der Volumenmittelwert ist exakt $-1/3$: Die gerasterte Kugel hat kubische Symmetrie,
also sind die Summen über $K_{xx}$, $K_{yy}$, $K_{zz}$ gleich. Weil $K$ spurfrei ist,
ist jede null, und nur der Selbstterm bleibt.

---

## 9. Kugel und Würfel (Aufgabe 2c)

Homogene Magnetisierung hat keine Volumenladungen, nur Flächenladungen
$\sigma = \vec m\cdot\hat n$.

### Warum ist das Feld in der Kugel gleichmäßig?

Eine Kugel mit $\sigma = m\cos\theta$ ist dasselbe wie zwei homogen geladene Kugeln
mit $\pm\rho$, um ein kleines $\vec\delta$ gegeneinander verschoben ($\rho\vec\delta = \vec m$).
Im Inneren einer homogen geladenen Kugel ist das Feld linear, $\vec E = \rho\vec r/3$.
Die Differenz zweier solcher Felder ist konstant:

$$
\vec h = \frac{\rho}{3}(\vec r - \vec\delta) - \frac{\rho}{3}\vec r = -\frac{\vec m}{3}.
$$

Dasselbe Argument gilt für jedes Ellipsoid, aber nur für Ellipsoide.

### Warum nicht im Würfel?

Die Ladungen sitzen auf zwei ebenen, endlichen Platten. Ihr Feld ist wie bei einem
Plattenkondensator mit Randfeld: in der Plattenmitte am stärksten, zu den Kanten
schwächer. Gemessen ($a = 32$): $-0.58$ an den Polflächen, $-1/3$ im Zentrum,
$-0.21$ in der Mitte der Seitenflächen, $-0.13$ an den Kantenmitten.

---

## 10. Magnetische Ladungen (Aufgabe 2d)

$$
\rho = -\nabla\cdot\vec m, \qquad \phi = K_c * \rho, \quad K_c(r) = \frac{1}{4\pi r}, \qquad \vec h = -\nabla\phi .
$$

### Warum äquivalent?

Partielle Integration verschiebt die Ableitung von $\vec m$ auf den Kernel:

$$
\phi = -K_c * (\nabla\cdot\vec m) = -\nabla\cdot(K_c * \vec m),
\qquad
\vec h = \nabla\nabla\cdot(K_c * \vec m) = (\nabla\nabla^T K_c) * \vec m .
$$

Als Distribution gilt

$$
\partial_i\partial_j\frac{1}{4\pi r} = \frac{3\hat r_i\hat r_j - \delta_{ij}}{4\pi r^3} - \frac{\delta_{ij}}{3}\,\delta(\vec r).
$$

Der erste Term ist der Dipol-Kernel, der Delta-Term genau der Selbstterm $-\tfrac13 I$.
Im Kontinuum sind beide Formulierungen also identisch. Sie unterscheiden sich nur
darin, wo diskretisiert wird: beim Dipol-Kernel exakt am Kernel, bei den Ladungen als
finite Differenzen auf $\vec m$ und $\phi$.

### Selbstterm des Coulomb-Kernels

$K_c(0)$ ist das Potential in der Mitte eines Einheitswürfels mit Ladungsdichte 1:

$$
K_c(0) = \frac{1}{4\pi}\int_{\text{Würfel}}\frac{dV}{r} = \frac{3\ln(2+\sqrt3) - \pi/2}{4\pi} \approx 0.1894 .
$$

### Vergleich

- **Genauigkeit innen und außen:** gleich gut ($3\cdot10^{-3}$ innen, $0.015$ außen).
- **An der Oberfläche:** bis $0.31$ Unterschied. Zentrale Differenzen verschmieren die
  Flächenladung über zwei Zellen. Der Dipol-Kernel springt an der Kugel von $-0.39$
  auf $+0.61$ (analytisch $-1/3$ auf $0.608$), die Ladungen nur von $-1/3$ auf $+0.48$.
- **Glattes $\vec m$:** Der Unterschied fällt wie $1/\sigma^2$ (7.1 % bei $\sigma = 3$,
  1.9 % bei $\sigma = 6$), der Fehler 2. Ordnung der zentralen Differenzen.
- **Aufwand:** Ladungen brauchen 2 FFTs pro Auswertung (ein Skalarfeld hin und zurück),
  der Dipol-Kernel 6. Gemessen bei $n = 64$: 0.05 s gegen 0.25 s.

---

## Zusammenfassung

- Das Streufeld ist eine Faltung; mit dem Faltungssatz kostet sie $O(N\log N)$ statt $O(N^2)$.
- Die DFT faltet zyklisch. Zero-Padding auf $L \ge 2N - 1$ und der Kernel in
  Wraparound-Anordnung machen daraus die lineare Faltung.
- Jede Länge mit kleinen Primfaktoren ist schnell; `rfft`, gecachter Kernel und
  `next_fast_len` bringen Faktor 3 bis 20.
- In 3D: 6 Kernel-Komponenten, 6 FFTs pro Auswertung.
- Validierung trennt Arithmetik (gegen direkte Summe, $\varepsilon$) und Physik
  (Kugel $-\vec m/3$, Fernfeld).
- Nur in Ellipsoiden ist das Feld homogen.
- Dipol- und Ladungsformulierung sind im Kontinuum gleich; die Ladungen sind billiger,
  der Dipol-Kernel an Oberflächen schärfer.
