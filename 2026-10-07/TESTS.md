# Erklärung der Tests

Datei: `test_monte_carlo_pi.py` · 17 Testfälle · Laufzeit unter 1 s

```bash
../.venv/bin/python -m pytest -v
```

## Grundidee

Der Schätzer liefert bei jedem Aufruf eine andere Zahl. Die Frage „stimmt das
Ergebnis?“ lässt sich deshalb nicht mit `ergebnis == 3.14159` beantworten,
sondern nur mit „liegt das Ergebnis dort, wo es bei einem korrekten Schätzer
liegen muss?“.

Wo das ist, kann man ausrechnen. Jeder Punkt ist ein Treffer mit
Wahrscheinlichkeit p = π/4. Die Trefferzahl ist binomialverteilt mit Varianz
N·p·(1−p). Mit π̂ = 4·Treffer/N folgt:

    Erwartungswert:      E[π̂] = π
    Standardabweichung:  σ(N) = sqrt(π(4−π)/N) ≈ 1.64/√N

Diese Formel steht in der Testdatei als Hilfsfunktion `sigma_theory(n)` und
liefert alle Toleranzen der statistischen Tests.

Zwei Regeln gelten für alle Tests:

- **Feste Seeds.** Jeder Zufallstest ruft `estimate_pi(n, seed=...)` mit festem
  Seed auf. Die Tests liefern also bei jedem Durchlauf exakt dasselbe und
  schlagen nie „zufällig“ fehl.
- **Kein Zufall, wo keiner nötig ist.** Geometrie und Formel stecken in eigenen
  Funktionen (`is_inside`, `pi_from_hits`) und werden exakt getestet.

## Teil 1: Tests ohne Zufall

### `test_is_inside_known_points`

Prüft die Geometrie mit fünf Punkten, deren Lage man im Kopf nachrechnen kann:

| Punkt | x² + y² | innen? |
|---|---|---|
| (0, 0) | 0 | ja |
| (0.5, 0.5) | 0.5 | ja |
| (0.6, 0.8) | 1.0 | ja (genau am Rand) |
| (1, 1) | 2 | nein |
| (0.9, 0.9) | 1.62 | nein |

Der Punkt (0.6, 0.8) legt fest, dass der Rand als innen zählt (`<=`).
(0.9, 0.9) ist wichtig, weil beide Koordinaten kleiner als 1 sind, der Punkt
aber trotzdem außerhalb liegt – ein Test auf `x < 1 and y < 1` würde hier
auffallen.

### `test_pi_from_hits_matches_figure_on_exercise_sheet`

Die Abbildung auf dem Angabeblatt gibt ein Zahlenbeispiel vor: N = 1500,
1188 Treffer, π ≈ 3.168. Der Test prüft, dass `pi_from_hits(1188, 1500)` genau
das ergibt. Damit ist die Formel 4·Treffer/N gegen eine unabhängige Quelle
geprüft und nicht nur gegen meine eigene Rechnung.

### `test_pi_from_hits_extreme_cases`

Kein Treffer → 0, alle Treffer → 4. Das sind die Grenzen des möglichen
Wertebereichs.

## Teil 2: Schnittstelle

### `test_same_seed_gives_same_result`

Zweimal `estimate_pi(1000, seed=42)` muss dieselbe Zahl liefern. Ohne das wären
alle folgenden Tests nicht reproduzierbar.

### `test_different_seeds_give_different_results`

Seed 1 und Seed 2 müssen verschiedene Zahlen liefern. Fängt den Fall ab, dass
der Seed gar nicht verwendet wird oder die Funktion eine Konstante zurückgibt.

### `test_result_is_multiple_of_4_over_n`

Rechnet aus dem Ergebnis die Trefferzahl zurück (π̂·N/4) und prüft, dass sie
ganzzahlig ist und zwischen 0 und N liegt. Das Ergebnis muss also wirklich aus
einer Zählung stammen. Fängt z. B. eine falsche Normierung (Division durch
N+1) ab.

### `test_invalid_n_raises` (2 Fälle: N = 0, N = −5)

Bei N ≤ 0 muss ein `ValueError` kommen. Ohne diese Prüfung gäbe N = 0 eine
Division durch null (`nan`) statt einer klaren Fehlermeldung.

## Teil 3: Statistische Korrektheit

### `test_estimate_within_5_sigma_of_pi` (5 Fälle: N = 10², 10³, 10⁴, 10⁵, 10⁶)

Der zentrale Test: |π̂ − π| < 5·σ(N).

Warum 5σ? Für großes N ist π̂ näherungsweise normalverteilt. Ein korrekter
Schätzer liegt nur mit Wahrscheinlichkeit ≈ 6·10⁻⁷ außerhalb von 5σ. Die
Schranke ist also weit genug, um einen korrekten Schätzer praktisch nie
abzulehnen.

Gleichzeitig wird sie mit wachsendem N immer schärfer:

| N | erlaubter Fehler 5σ(N) |
|---:|---:|
| 10² | 0.82 |
| 10³ | 0.26 |
| 10⁴ | 0.082 |
| 10⁵ | 0.026 |
| 10⁶ | 0.0082 |

Der Fall N = 10² allein wäre fast wertlos (alles zwischen 2.3 und 4 besteht).
Der Fall N = 10⁶ verlangt dagegen π auf 0.26 % genau.

### `test_estimator_is_unbiased`

2000 unabhängige Schätzungen mit je N = 1000, davon der Mittelwert. Der
Mittelwert aus M Läufen streut nur noch mit σ(N)/√M ≈ 0.0012. Der Test
verlangt, dass er höchstens 5 dieser Standardfehler (≈ 0.0058) von π abweicht.

Unterschied zum vorigen Test: Dort wird ein einzelner Lauf geprüft, hier der
Mittelwert vieler Läufe. Ein systematischer Fehler (Bias) mittelt sich nicht
weg und fällt so auf, auch wenn jeder einzelne Lauf noch plausibel aussieht.

### `test_spread_matches_theory` (2 Fälle: N = 10², 10⁴)

400 Läufe, daraus der RMS-Fehler sqrt(Mittelwert((π̂ − π)²)). Er muss auf 15 %
mit σ(N) übereinstimmen.

Die 15 % sind nicht geraten: Ein RMS aus M = 400 Stichproben hat selbst eine
relative Unsicherheit von etwa 1/√(2M) = 3.5 %. 15 % sind also gut 4σ.

Dieser Test prüft etwas, das die vorigen nicht sehen: ob die **Streuung**
stimmt. Ein Schätzer könnte im Mittel π treffen und trotzdem falsch streuen,
z. B. wenn x und y nicht unabhängig sind oder wenn intern weniger Punkte
verwendet werden als angegeben.

### `test_error_scales_like_one_over_sqrt_n`

RMS-Fehler bei N = 10² und bei N = 10⁴, je 400 Läufe. Bei 1/√N-Verhalten muss
das Verhältnis 10 sein (100-mal mehr Punkte → 10-mal kleinerer Fehler).
Toleranz 20 %.

Die Seeds sind `n + s`, damit die Läufe für die beiden N nicht dieselben
Seeds verwenden und unabhängig sind.

Das ist die Aussage, die später auch der Plot zeigt – hier als automatischer
Test statt als Bild, das man anschauen muss.

## Test der Tests

Grüne Tests sind nur dann etwas wert, wenn sie bei falschem Code rot werden.
Das habe ich geprüft, indem ich in einer Kopie von `monte_carlo_pi.py`
absichtlich Fehler eingebaut habe:

| Eingebauter Fehler | rot | grün | erkannt? |
|---|---:|---:|---|
| `x + y <= 1` (Dreieck statt Viertelkreis) | 10 | 7 | ja |
| Faktor 3.15 statt 4 | 11 | 6 | ja |
| `y = x` (Koordinaten nicht unabhängig) | 8 | 9 | ja |
| `x² + y² <= 1.002` (Radius minimal zu groß) | 0 | 17 | **nein** |

## Was die Tests nicht abdecken

- **Kleine systematische Fehler unter ca. 0.25 %.** Der letzte Fall oben
  verschiebt π̂ um etwa 0.006. Die schärfste Schranke ist 5σ(10⁶) ≈ 0.008, der
  Fehler rutscht also durch. Für eine schärfere Prüfung bräuchte man größeres N
  (z. B. N = 10⁸ → Schranke 0.0008), was die Tests langsamer macht.
- **Die Qualität des Zufallsgenerators.** Die Tests setzen voraus, dass numpy
  gleichverteilte, unabhängige Zahlen liefert.
- **Das Skript `main()`**, also Schleife über N, Fit der Steigung und Plot.
  Das habe ich nur durch Ausführen und Ansehen des Plots geprüft.
- **Fixe Seeds bedeuten fixe Stichprobe.** Die Tests prüfen den Schätzer für
  genau diese Seeds. Dass sie bei anderen Seeds ebenfalls bestehen würden, ist
  durch die 5σ-Schranken sehr wahrscheinlich, aber nicht getestet.
