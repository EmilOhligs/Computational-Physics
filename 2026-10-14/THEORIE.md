# Theorie: Diskrete Fourier-Transformation und FFT

Exercise 1 · 14.10.2026

Dieses Dokument erklärt die Theorie hinter allen Teilaufgaben. Die Verweise
(1a, 2b, …) beziehen sich auf die Angabe in `exercise.pdf`.

---

## 1. Ausgangspunkt: Fourier-Reihe

Ein periodisches Signal $a(t)$ mit Periode $T$ lässt sich als Summe von
Schwingungen schreiben:

$$
a(t) = \sum_{m=-\infty}^{\infty} c_m \, e^{2\pi i\, m t / T},
\qquad
c_m = \frac{1}{T}\int_0^T a(t)\, e^{-2\pi i\, m t/T}\, dt
$$

Der Koeffizient $c_m$ gibt an, wie stark die Frequenz $f_m = m/T$ im Signal
vorkommt. Die Fourier-Analyse zerlegt also ein Signal in seine Frequenzen.

Am Computer kennt man das Signal aber nicht als Funktion, sondern nur an
endlich vielen Zeitpunkten. Daraus wird die DFT.

---

## 2. Die diskrete Fourier-Transformation (Aufgabe 1a)

### Abtastung

Das Signal wird im Zeitfenster $[0, T)$ an $N$ gleichmäßig verteilten
Zeitpunkten abgetastet:

$$
t_j = j\,\Delta t, \qquad \Delta t = \frac{T}{N}, \qquad a_j = a(t_j), \qquad j = 0, \dots, N-1
$$

Wichtig: Der Endpunkt $t = T$ gehört **nicht** dazu. Die DFT behandelt das
Signal als periodisch mit Periode $T$, der Punkt $t = T$ wäre also derselbe wie
$t = 0$ und würde doppelt gezählt.

### Definition

$$
\hat a_k = \sum_{j=0}^{N-1} a_j\, e^{-2\pi i\, jk/N}
\qquad\qquad
a_j = \frac{1}{N}\sum_{k=0}^{N-1} \hat a_k\, e^{+2\pi i\, jk/N}
$$

Links die DFT (Zeit → Frequenz), rechts die inverse DFT (Frequenz → Zeit).
Die Konvention ist dieselbe wie in `numpy.fft`: der Faktor $1/N$ steht nur
bei der Rücktransformation.

### Bedeutung der Indizes

Der Index $k$ gehört zur Frequenz

$$
f_k = \frac{k}{T} = \frac{k}{N\,\Delta t}
$$

Für $k > N/2$ ist das aber keine hohe positive Frequenz, sondern die
negative Frequenz $(k - N)/T$, denn $e^{-2\pi i\, jk/N} = e^{-2\pi i\, j(k-N)/N}$.
Die Reihenfolge der Ausgabe ist also

    k = 0, 1, 2, …, N/2 − 1, N/2, N/2 + 1, …, N − 1
        0  positive Frequenzen  Nyquist  negative Frequenzen

`numpy.fft.fftfreq(N, dt)` liefert genau diese Frequenzen, `numpy.fft.fftshift`
sortiert sie von negativ nach positiv.

### Zerlegung in Fourier-Komponenten

Die inverse DFT liest sich als Rezept: Das Signal ist die Summe von $N$
Schwingungen. Die $k$-te Komponente ist

$$
\frac{1}{N}\,\hat a_k\, e^{2\pi i\, jk/N}
$$

mit Amplitude $|\hat a_k|/N$ und Phase $\arg \hat a_k$. Für ein reelles Signal
gehören $k$ und $N-k$ zusammen und ergeben zusammen eine reelle
Kosinusschwingung:

$$
\frac{1}{N}\left(\hat a_k e^{2\pi i jk/N} + \hat a_{N-k} e^{2\pi i j(N-k)/N}\right)
= \frac{2|\hat a_k|}{N} \cos\!\left(\frac{2\pi jk}{N} + \arg \hat a_k\right)
$$

Addiert man die Komponenten nacheinander auf, sieht man, wie das Signal Stück
für Stück entsteht. Das ist die verlangte Visualisierung in 1a.

### Zusammenhang mit der kontinuierlichen Fourier-Transformation

Die Angabe spricht von $\tilde a(\omega)$. Die kontinuierliche Transformation
$\tilde a(\omega) = \int a(t)\, e^{-i\omega t}\, dt$ wird mit der Rechteckregel
über das Fenster näherungsweise

$$
\tilde a(\omega_k) \approx \Delta t \sum_{j} a_j\, e^{-i\omega_k t_j} = \Delta t\, \hat a_k,
\qquad \omega_k = 2\pi f_k
$$

Je nach Fragestellung skaliert man das DFT-Ergebnis also unterschiedlich:

| gesucht | Formel |
|---|---|
| Fourier-Reihen-Koeffizient $c_k$ | $\hat a_k / N$ |
| kontinuierliche Transformation $\tilde a(\omega_k)$ | $\Delta t \cdot \hat a_k$ |
| Amplitude der Kosinusschwingung (reelles Signal, $0<k<N/2$) | $2\lvert\hat a_k\rvert / N$ |

---

## 3. DFT gegen FFT: Rundungsfehler und Laufzeit (Aufgaben 1b, 1c)

### Warum ist die Differenz nicht exakt null? (1b)

DFT und FFT berechnen mathematisch dasselbe, aber auf verschiedenen Wegen.
Jede Gleitkommaoperation rundet auf etwa 16 Stellen
($\varepsilon \approx 2.2\cdot 10^{-16}$), und andere Rechenwege runden anders.

Typische Größenordnung des Unterschieds:

- **naive DFT:** Jedes $\hat a_k$ ist eine Summe aus $N$ Termen. Wie stark
  sich die Rundungsfehler ansammeln, hängt davon ab, **wie** summiert wird:
  Eine einfache Schleife `s += ...` sammelt Fehler $\propto N$ an, die
  paarweise Summation von `np.sum` nur $\propto \log N$. Dazu kommt, dass
  $e^{-2\pi i\, jk/N}$ für große $jk$ ungenauer wird, weil schon das Argument
  mit einem absoluten Fehler $\sim \varepsilon \cdot 2\pi jk/N$ gerundet wird
  (Abhilfe: $jk \bmod N$ verwenden).
- **FFT:** Jeder Wert durchläuft nur $\log_2 N$ Stufen. Der Fehler wächst
  ungefähr mit $\log N$.

**Korrektur nach der Messung (Notebook, 1b):** Ursprünglich stand hier, die
naive DFT sei wegen der $N$ Summanden ungenauer als die FFT. Gemessen ist sie
mit `np.sum` und dem Modulo-Trick bis $N = 4096$ genauso genau (relativer
Fehler $\sim 4\cdot10^{-16}$). Ohne Modulo-Trick wächst ihr Fehler dagegen
$\propto N$ (auf $8\cdot10^{-13}$ bei $N = 4096$).

Ein Unterschied in der Größenordnung $10^{-13}$ bis $10^{-16}$ (relativ zur
Größe der Werte) ist also kein Fehler, sondern zu erwarten. Ein Unterschied von
$10^{-3}$ wäre ein Fehler.

### Laufzeit (1c)

- **DFT:** $N$ Ausgabewerte, jeder eine Summe über $N$ Terme → $\mathcal{O}(N^2)$
- **FFT:** $\mathcal{O}(N \log N)$ (Herleitung in Abschnitt 6)

Für $N = 2^{20} \approx 10^6$ heißt das $10^{12}$ gegen $2\cdot 10^{7}$
Operationen, ein Faktor von etwa 50 000. Im log-log-Plot der Laufzeit gegen $N$
erwartet man Steigung 2 für die DFT und knapp über 1 für die FFT.

---

## 4. Das Dreieckssignal und sein analytisches Spektrum (Aufgabe 1d)

### Das Signal

Aus der Abbildung in der Angabe:

- Amplitude 1, Periode $P = 2$ ms, also Grundfrequenz $f_0 = 500$ Hz
- $a(0) = 1$, Minimum $-1$ bei $t = 1$ ms
- Fenster $T = 8$ ms, also genau 4 Perioden

Innerhalb einer Periode: $a(t) = 1 - 4t/P$ für $0 \le t < P/2$ und
$a(t) = -3 + 4t/P$ für $P/2 \le t < P$.

### Fourier-Reihe

Das Dreieck ist gerade ($a(-t) = a(t)$) und hat Mittelwert 0. Die bekannte
Reihe lautet

$$
a(t) = \frac{8}{\pi^2} \sum_{n = 1, 3, 5, \dots} \frac{1}{n^2} \cos(2\pi n f_0 t)
= \frac{8}{\pi^2}\left(\cos(2\pi f_0 t) + \tfrac19 \cos(6\pi f_0 t) + \tfrac1{25}\cos(10\pi f_0 t) + \dots\right)
$$

Probe bei $t = 0$: $\frac{8}{\pi^2}\sum_{n\ \text{ungerade}} \frac1{n^2} = \frac{8}{\pi^2}\cdot\frac{\pi^2}{8} = 1$. ✓

Es kommen also nur **ungerade Vielfache** der Grundfrequenz vor, mit
Amplitude $\propto 1/n^2$.

### Was die DFT liefern sollte

Das Fenster enthält 4 Perioden, die $n$-te Oberschwingung
($f = n \cdot 500$ Hz) liegt deshalb bei $k = 4n$. Nicht null sind nur

$$
k = 4, 12, 20, 28, \dots \quad (\text{und gespiegelt } N - k)
$$

mit den Fourier-Koeffizienten (komplexe Form, Hälfte der Kosinusamplitude):

$$
c_{\pm 4n} = \frac{4}{\pi^2 n^2} \quad (n \text{ ungerade}), \qquad
\text{erwartet: } \frac{\hat a_k}{N} \approx c_k
$$

### Warum es nicht exakt passt: Aliasing

Abtastung kann Frequenzen nicht unterscheiden, die sich um ein Vielfaches von
$N/T$ unterscheiden: An den Abtastpunkten gilt
$e^{2\pi i (k+N) j/N} = e^{2\pi i k j/N}$. Alle Fourier-Komponenten
$c_{k+lN}$ landen deshalb im selben DFT-Wert. Exakt gilt:

$$
\frac{\hat a_k}{N} = \sum_{l=-\infty}^{\infty} c_{k + lN}
= c_k + \underbrace{c_{k-N} + c_{k+N} + c_{k-2N} + \dots}_{\text{Aliasing-Fehler}}
$$

Das Dreieck hat unendlich viele Oberschwingungen, deshalb ist der Fehler nie
ganz null.

### Warum der Fehler zur Nyquist-Frequenz hin wächst

Die größte darstellbare Frequenz ist die **Nyquist-Frequenz**

$$
f_\text{Nyquist} = \frac{1}{2\Delta t} \qquad (k = N/2)
$$

Der größte Aliasing-Beitrag ist $c_{k-N}$, also die Komponente bei der
negativen Frequenz $k - N$. Für kleines $k$ ist $|k - N| \approx N$ weit weg und
$c_{k-N}$ winzig. Nahe $k = N/2$ ist $|k - N| \approx N/2 \approx k$: Der
Fehlerterm ist dort **genauso groß wie der gesuchte Wert**.

Zahlenbeispiel mit $N = 64$ (vorab nachgerechnet):

| $k$ | $\hat a_k/N$ (DFT) | $c_k$ (analytisch) | relativer Fehler |
|---:|---:|---:|---:|
| 4 | 0.41053 | 0.40528 | 1.3 % |
| 12 | 0.05062 | 0.04503 | 12 % |
| 20 | 0.02260 | 0.01621 | 39 % |
| 28 | 0.01624 | 0.00827 | 96 % |

Mit der Aliasing-Summe stimmen die DFT-Werte dagegen überein. Das ist eine gute
Prüfung für die eigene Implementierung.

### Einfluss von $\Delta t$ und $T$

| Größe | bestimmt | Formel |
|---|---|---|
| Fensterlänge $T$ | Frequenzauflösung (Abstand der Frequenzen) | $\Delta f = 1/T$ |
| Abtastschritt $\Delta t$ | größte darstellbare Frequenz | $f_\text{Nyquist} = 1/(2\Delta t)$ |
| beide | Anzahl Punkte | $N = T/\Delta t$ |

- **Kleineres $\Delta t$** (bei gleichem $T$): mehr Frequenzen oben dazu, das
  Aliasing wird kleiner, weil die Faltung weiter entfernte (kleinere)
  Koeffizienten hereinholt. Die Frequenzen $f_k$ selbst bleiben gleich.
- **Größeres $T$** (bei gleichem $\Delta t$): feineres Frequenzraster. Solange
  $T$ ein Vielfaches der Periode bleibt, ändern sich die Werte von
  $\hat a_k/N$ an den Oberschwingungen nicht, es kommen nur Nullen
  dazwischen. Ist $T$ **kein** Vielfaches der Periode, entsteht Leakage
  (Abschnitt 8).

---

## 5. Eigenschaften der DFT (Aufgaben 2a–c)

Alle Eigenschaften folgen direkt aus der Definition. Sie eignen sich gut als
Tests für die eigene Implementierung.

### 2a: Symmetrie für reelle Signale

$$
\hat a_{N-k} = \sum_j a_j\, e^{-2\pi i\, j(N-k)/N}
= \sum_j a_j\, e^{+2\pi i\, jk/N}
= \left(\sum_j a_j\, e^{-2\pi i\, jk/N}\right)^{\!*} = \hat a_k^{\,*}
$$

Der vorletzte Schritt braucht $a_j = a_j^*$, also ein **reelles** Signal.
Folge: Die negativen Frequenzen enthalten keine neue Information. Für reelle
Signale genügt die Hälfte des Spektrums (`numpy.fft.rfft`).

### 2b: Verschiebungssatz

Verschiebt man das Signal um $m$ Abtastpunkte (periodisch),
$b_j = a_{(j-m) \bmod N}$, dann gilt

$$
\hat b_k = \hat a_k\, e^{-2\pi i\, km/N}
$$

Herleitung: Substitution $j' = j - m$ in der Summe. Eine Verschiebung ändert
also nur die **Phase** jeder Komponente, nicht den Betrag $|\hat a_k|$.
In numpy entspricht die Verschiebung `np.roll(a, m)`.

### Warum ist $\hat a_k$ für das Dreieck rein reell?

Kombiniert man 2a mit der Spiegelung: Ist das Signal **reell und gerade**,
also $a_{N-j} = a_j$, dann gilt auch $\hat a_{N-k} = \hat a_k$ (Substitution
$j \to N - j$). Zusammen mit $\hat a_{N-k} = \hat a_k^*$ folgt
$\hat a_k = \hat a_k^*$, also reell.

Das Dreieck ist gerade, weil es bei $t = 0$ sein Maximum hat:
$a(T - t) = a(-t) = a(t)$. Anschaulich: Es besteht nur aus Kosinus-Termen,
und der Kosinus hat ein reelles Spektrum.

Verschiebt man das Dreieck (2b), ist es nicht mehr gerade und das Spektrum
wird komplex. Das ist die Phase $e^{-2\pi i km/N}$ aus dem Verschiebungssatz.

| Signal | Spektrum |
|---|---|
| reell und gerade (Kosinus, Dreieck mit Maximum bei 0) | reell und gerade |
| reell und ungerade (Sinus, $a_{N-j} = -a_j$) | rein imaginär und ungerade |
| reell, weder noch | komplex mit $\hat a_{N-k} = \hat a_k^*$ |

Das beantwortet auch Aufgabe 3a.

### 2c: Hin- und Rücktransformation, Parseval

**Rücktransformation.** Grundlage ist die Orthogonalität

$$
\sum_{j=0}^{N-1} e^{2\pi i\, j(k - k')/N} = N\, \delta_{kk'}
$$

(geometrische Reihe; für $k \ne k'$ summieren sich die $N$-ten
Einheitswurzeln zu null). Setzt man die DFT in die inverse DFT ein, bleibt
genau $a_j$ übrig. Numerisch erwartet man wieder Abweichungen in der
Größenordnung $10^{-15}$.

**Parseval.** Achtung, in der Angabe fehlt ein Faktor. Mit der verwendeten
Konvention gilt

$$
\sum_{j=0}^{N-1} |a_j|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |\hat a_k|^2
$$

Ohne $1/N$ unterscheiden sich beide Seiten um den Faktor $N$. Nachgerechnet
für das Dreieck mit $N = 64$: links 22, rechts ohne Faktor 1408, mit Faktor
$1408/64 = 22$. ✓

Bedeutung: Die Energie des Signals ist im Zeit- und Frequenzraum gleich. Die
Form ohne Faktor gilt nur für die symmetrisch normierte DFT mit $1/\sqrt N$ auf
beiden Seiten (`norm="ortho"` in numpy).

---

## 6. Die FFT: Cooley-Tukey (Aufgaben 2d, 2e)

### Die Idee

Man teilt die Summe in gerade und ungerade Indizes ($N$ gerade):

$$
\hat a_k
= \sum_{m=0}^{N/2-1} a_{2m}\, e^{-2\pi i\, (2m)k/N}
+ \sum_{m=0}^{N/2-1} a_{2m+1}\, e^{-2\pi i\, (2m+1)k/N}
$$

Mit $e^{-2\pi i\, 2mk/N} = e^{-2\pi i\, mk/(N/2)}$ sind beide Summen selbst
DFTs der Länge $N/2$:

$$
\hat a_k = E_k + w_N^{-k}\, O_k,
\qquad E = \text{DFT}(a_0, a_2, \dots),\quad O = \text{DFT}(a_1, a_3, \dots),
\quad w_N = e^{2\pi i/N}
$$

### Die zweite Hälfte gratis

$E_k$ und $O_k$ sind DFTs der Länge $N/2$ und damit periodisch mit Periode
$N/2$. Der Vorfaktor wechselt das Vorzeichen:
$w_N^{-(k+N/2)} = w_N^{-k}\, e^{-i\pi} = -w_N^{-k}$. Deshalb

$$
\hat a_{k+N/2} = E_k - w_N^{-k}\, O_k, \qquad k = 0, \dots, N/2 - 1
$$

Aus zwei halb so langen DFTs erhält man mit $N/2$ Multiplikationen alle
$N$ Werte. Dieses Paar aus Plus und Minus heißt **Butterfly**.

### Rekursion (2e)

Man wendet dasselbe auf $E$ und $O$ an, bis die Länge 1 erreicht ist. Die DFT
eines einzelnen Werts ist der Wert selbst. Pseudocode:

    fft(a):
        N = len(a)
        if N == 1: return a
        E = fft(a[0::2])
        O = fft(a[1::2])
        w = exp(-2πi k / N) für k = 0 … N/2−1
        return concat(E + w·O, E − w·O)

Voraussetzung: $N$ muss bei jeder Halbierung gerade sein, also eine
**Zweierpotenz** (radix-2). Für andere $N$ braucht man andere Zerlegungen;
numpy kann das, eine einfache Radix-2-Implementierung nicht.

### Aufwand

Es gibt $\log_2 N$ Rekursionsebenen, jede kostet insgesamt $\mathcal{O}(N)$
→ $\mathcal{O}(N \log N)$. In Python wird eine rekursive FFT trotzdem viel
langsamer sein als `numpy.fft` (kompilierter C-Code), aber für große $N$
schneller als die naive DFT.

---

## 7. Abfall des Spektrums und Glattheit (Aufgabe 3b)

Wie schnell $|\hat a_k|$ mit $k$ abfällt, hängt davon ab, wie glatt das
(periodisch fortgesetzte) Signal ist. Grund: Partielle Integration der
Fourier-Formel bringt pro Integration einen Faktor $1/k$ und eine Ableitung
des Signals. Das geht so oft, bis eine Ableitung springt.

| Signal | Eigenschaft | Abfall von $\lvert c_k\rvert$ |
|---|---|---|
| Rechteck, Sägezahn | Sprung im Signal | $\propto 1/k$ |
| Dreieck | stetig, Knick (Sprung in der 1. Ableitung) | $\propto 1/k^2$ |
| Signal mit Sprung in der $p$-ten Ableitung | | $\propto 1/k^{p+1}$ |
| Gauß-Puls, Sinus | unendlich oft glatt | schneller als jede Potenz |
| Rauschen | keine Struktur | kein Abfall (flaches Spektrum) |

Im log-log-Plot von $|\hat a_k|$ gegen $k$ erscheint das als Gerade mit
Steigung $-1$, $-2$, … Für das Dreieck erwartet man $-2$, passend zum
$1/n^2$ der Fourier-Reihe.

Wichtig: Es zählt die **periodische Fortsetzung**. Ein Sinus, der nicht genau
ins Fenster passt, hat am Fensterrand einen Sprung und fällt dann auch nur mit
$1/k$ ab.

---

## 8. Signal passt nicht ins Fenster: Leakage (Aufgabe 3c)

Die DFT nimmt stillschweigend an, dass das Signal mit Periode $T$ weiterläuft.
Ist $T$ kein ganzzahliges Vielfaches der Signalperiode, passen Anfang und Ende
nicht zusammen. Die periodische Fortsetzung hat dann einen **Sprung** am
Fensterrand.

Folgen:

- Die Frequenz liegt zwischen zwei DFT-Frequenzen $f_k$. Statt einer scharfen
  Linie verteilt sich die Energie auf viele Nachbarfrequenzen
  (**spectral leakage**).
- Durch den Sprung fällt das Spektrum nur noch mit $1/k$ ab (Abschnitt 7).

Verbesserungen:

1. **Fenster passend wählen:** $T$ als Vielfaches der Periode. Geht nur, wenn
   man die Periode kennt.
2. **Fensterfunktion:** Signal vor der DFT mit einer Funktion multiplizieren,
   die an den Rändern auf null geht, z. B. Hann-Fenster
   $w_j = \sin^2(\pi j/N)$. Der Sprung verschwindet, die Leakage fällt viel
   schneller ab. Preis: Die Linien werden etwas breiter und die Amplituden
   müssen korrigiert werden.
3. **Längeres Fenster:** feinere Auflösung $\Delta f = 1/T$, die Leakage wird
   relativ zur Linie schmaler.

Zero-Padding (Nullen anhängen) ist **keine** Abhilfe: Es interpoliert nur das
Spektrum auf einem feineren Raster, die Auflösung und die Leakage bleiben
gleich.

---

## Zusammenfassung

| Thema | Kernaussage |
|---|---|
| DFT | $N$ Abtastwerte ↔ $N$ Frequenzen $f_k = k/T$, $k > N/2$ sind negative Frequenzen |
| Skalierung | $\hat a_k/N$ = Fourier-Koeffizient, $\Delta t\,\hat a_k$ ≈ kontinuierliche Transformation |
| DFT gegen FFT | gleiches Ergebnis bis auf Rundung ($\sim 10^{-15}$), Aufwand $N^2$ gegen $N\log N$ |
| $\Delta t$, $T$ | $T$ → Auflösung $1/T$, $\Delta t$ → Nyquist $1/(2\Delta t)$ |
| Aliasing | Frequenzen über Nyquist falten zurück; Fehler am größten nahe Nyquist |
| Symmetrien | reell → $\hat a_{N-k} = \hat a_k^*$; reell + gerade → reelles Spektrum |
| Verschiebung | ändert nur die Phase: Faktor $e^{-2\pi i km/N}$ |
| Parseval | $\sum\lvert a_j\rvert^2 = \frac1N\sum\lvert\hat a_k\rvert^2$ (in der Angabe fehlt $1/N$) |
| FFT | gerade/ungerade Teilung, Butterfly, nur für $N = 2^p$ (radix-2) |
| Glattheit | Sprung → $1/k$, Knick → $1/k^2$, glatt → schneller |
| Leakage | Signal passt nicht ins Fenster → Sprung → verschmiertes Spektrum; Abhilfe Fensterfunktion |
