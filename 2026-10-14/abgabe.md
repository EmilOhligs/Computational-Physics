---
title: "Exercise 1: FFT eines Dreieckssignals"
subtitle: "Computational Physics, WS 2026/27 · Antworten"
author: "Emil Ohligs"
date: "14.10.2026"
lang: de
geometry: "a4paper, margin=2.2cm"
fontsize: 10pt
---

Signal: Dreieck mit Amplitude 1, Periode $P = 2$ ms ($f_0 = 500$ Hz), Fenster $T = 8$ ms, $N = 64$ Abtastpunkte. Code und Abbildungen stehen in `exercise_1_fft.ipynb`.

## 1 DFT

**a) Naive DFT.** Das Spektrum hat nur Linien bei den ungeraden Harmonischen 0.5, 1.5, 2.5 und 3.5 kHz. Es ist reell und symmetrisch, alle anderen $|\hat a_k|$ liegen bei $10^{-16}$. Die Summe der vier Kosinus-Komponenten geht exakt durch alle Abtastpunkte.

**b) Warum ist die Differenz zu numpy nicht exakt null?** Gleitkommazahlen haben nur etwa 16 Stellen, und jede Operation rundet; schon $\pi$ und die Faktoren $e^{-2\pi i jk/N}$ sind gerundet. DFT und FFT rechnen auf verschiedenen Wegen, und Gleitkomma-Addition ist nicht assoziativ. Deshalb unterscheiden sie sich in den letzten Stellen. Gemessen: relativ $2.9\cdot10^{-16} \approx \varepsilon$, das ist reine Rundung.

**c) Laufzeit.** Die naive DFT skaliert wie $N^2$ (gemessene Steigung 1.89), die FFT wie $N\log N$ (etwa 1 ns $\cdot\, N\log_2 N$). Der Vorsprung wächst mit $N$: Faktor 115 bei $N = 64$, $10^4$ bei $N = 4096$. Für $N = 2^{22}$ bräuchte die DFT hochgerechnet etwa 6 Tage, die FFT 0.11 s.

**d) Vergleich mit dem analytischen Ergebnis** $c_m = 4/(\pi^2 n^2)$ für $m = \pm nT/P$, $n$ ungerade. Die Lage der Linien stimmt exakt. Die Höhe weicht ab, von 1.3 % bei 500 Hz bis 96 % bei 3.5 kHz.

- *Einfluss von $\Delta t$ und $T$:* $T$ bestimmt die Frequenzauflösung $\Delta f = 1/T$, also das Raster $f_k = k/T$. Ein längeres Fenster macht das Raster feiner, an den Linien bleibt $\hat a_k/N$ gleich. $\Delta t$ bestimmt die Nyquist-Frequenz $1/(2\Delta t)$ und damit den Aliasing-Fehler. Der fällt beim Dreieck wie $\Delta t^2$ (gemessen: $\Delta t^{2.05}$).
- *Warum wächst der Fehler zur Nyquist-Frequenz hin?* Wegen Aliasing. An den Abtastpunkten sind die Frequenzen $k$ und $k + lN$ nicht zu unterscheiden, deshalb gilt $\hat a_k/N = \sum_l c_{k+lN}$. Der größte Fremdbeitrag kommt von $k - N$. Bei kleinem $k$ ist er vernachlässigbar, nahe Nyquist ist $|k - N| \approx k$ und er wird so groß wie der gesuchte Wert. Die Formel erklärt den gemessenen Fehler vollständig: Für $N = 16 \dots 1024$ liegen alle Werte auf einer Kurve in $f/f_\text{Nyq}$.

## 2 Eigenschaften der DFT

**a) Symmetrie.** $\hat a_{N-k} = \hat a_k^*$ gilt für das Dreieck und ein zufälliges reelles Signal bis auf $3\cdot10^{-16}$. Für ein komplexes Signal gilt sie nicht (Fehler 0.95). Die negativen Frequenzen enthalten für reelle Signale also keine neue Information.

**b) Verschiebungssatz.** $\text{DFT}(\texttt{roll}(a, m)) = \hat a_k e^{-2\pi i km/N}$ gilt für alle getesteten $m$ bis auf $6\cdot10^{-16}$. Die Beträge bleiben gleich, nur die Phase ändert sich.
*Warum ist $\hat a_k$ für das Dreieck reell?* Das Dreieck ist reell und gerade ($a_{N-j} = a_j$, Maximum bei $t = 0$). Aus „gerade“ folgt $\hat a_{N-k} = \hat a_k$, aus „reell“ folgt $\hat a_{N-k} = \hat a_k^*$, zusammen also $\hat a_k = \hat a_k^*$. Anschaulich: Das Dreieck besteht nur aus Kosinus-Termen. Verschiebt man es, ist es nicht mehr gerade, und das Spektrum wird komplex.

**c) Rücktransformation und Parseval.** $\mathcal F^{-1}\mathcal F(a) = a$ und $\mathcal F\mathcal F^{-1}(a) = a$ gelten bis $N = 2048$ auf $\le 4\cdot10^{-16}$. Parseval gilt mit dieser Konvention nur mit dem Faktor $1/N$: $\sum_j |a_j|^2 = \frac1N \sum_k |\hat a_k|^2$, für das Dreieck $22 = 1408/64$. In der Angabe fehlt dieser Faktor. Ohne ihn gilt die Formel nur mit `norm="ortho"`.

**d) Cooley-Tukey.** $\hat a_k = E_k + w_N^{-k}O_k$ und $\hat a_{k+N/2} = E_k - w_N^{-k}O_k$ stimmen bis auf $3.5\cdot10^{-16}$ mit der DFT überein. Die zweite Hälfte kostet nichts extra, weil $E$ und $O$ mit $N/2$ periodisch sind und $w_N^{-(k+N/2)} = -w_N^{-k}$ gilt. Ein Schritt halbiert den Aufwand.

**e) Rekursive FFT.** Die rekursive Radix-2-FFT stimmt mit `np.fft.fft` auf wenige $\varepsilon$ überein und skaliert wie $N\log N$. Bei $N = 4096$ ist sie 21-mal schneller als die naive DFT, beide sind in Python geschrieben. Gegenüber numpy ist sie 200- bis 500-mal langsamer, wegen des Python-Overheads von etwa $2N$ Funktionsaufrufen. Eine iterative Variante mit denselben Butterflies ist 55-mal schneller als die rekursive.

## 3 Andere Signale

Untersucht wurden Rechteck, Sägezahn, Sinus, Gauß-Puls (zentriert und verschoben) und Rauschen.

**a) Welche Spektren sind reell, imaginär, komplex?**

| Signal | Symmetrie | Spektrum |
|---|---|---|
| Dreieck, Rechteck, Gauß-Puls | gerade | reell |
| Sägezahn, Sinus | ungerade | rein imaginär |
| verschobener Gauß, Rauschen | keine | komplex |

Grund: Ein reelles Signal hat immer $\hat a_{N-k} = \hat a_k^*$. Ist es zusätzlich gerade, ist das Spektrum reell (nur Kosinus), ist es ungerade, ist es imaginär (nur Sinus). Die Symmetrie muss dabei auf dem Abtastgitter gelten: An Sprungstellen braucht man den Mittelwert. Ohne ihn bekommt das Spektrum des Sägezahns einen Realteil von 20 %.

**b) Abfall und Glattheit.** Springt die $p$-te Ableitung des periodisch fortgesetzten Signals, fällt $|\hat a_k| \propto k^{-(p+1)}$ ab. Grund: Jede partielle Integration im Fourier-Integral bringt einen Faktor $1/k$, bis eine Ableitung springt. Gemessen (fein abgetastet, um Aliasing zu vermeiden):

- Rechteck und Sägezahn (Sprung): Steigung $-1.01$
- Dreieck (Knick): $-1.99$
- Parabelwelle (Sprung in der 2. Ableitung): $-3.00$

Glatte Signale fallen schneller als jede Potenz: Der Gauß-Puls fällt Gauß-förmig bis zur Rundungsgrenze. Rauschen hat ein flaches Spektrum (Steigung $\approx 0$).

**c) Signal passt nicht ins Fenster.** Die DFT setzt das Fenster periodisch fort. Ist $T$ kein Vielfaches der Periode, hat die Fortsetzung am Rand einen Sprung. Für einen Sinus mit 8.8 Perioden im Fenster heißt das:

- Die Energie verteilt sich auf viele Nachbarlinien (Leakage, 2.8 % außerhalb von $\pm 2$ Bins).
- Die Seitenlinien fallen nur wie $1/|k - k_0|$ ab.
- Die Spitze ist 7 % zu niedrig.

Verbesserungen:

1. $T$ als Vielfaches der Periode wählen, wenn man sie kennt.
2. Ein Hann-Fenster $w_j = \sin^2(\pi j/N)$ verwenden. Die Leakage sinkt um den Faktor 256, der Abfall wird $1/|k-k_0|^3$, die Amplitude stimmt nach Korrektur. Der Preis: Die Linie wird doppelt so breit.
3. Ein längeres Fenster nehmen. Dann wird die Leakage relativ zur Auflösung schmaler.

Zero-Padding hilft nicht, es interpoliert nur das Spektrum.
