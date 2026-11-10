---
title: Kepler-Problem
aliases: Zweikörperproblem, Kreisbahn, Periapsis, Apoapsis, Vis-viva
area: physmod
summary: Bewegung in der Zentralkraft $\ddot{\mathbf r} = -GM\mathbf r/|\mathbf r|^3$; Energie $E$ und Drehimpuls $L$ sind erhalten und legen den Kegelschnitt fest, mit Periapsis $r_p = L^2/(GM(1+e))$.
links: symplektische-integratoren, raketengleichung
sources: 2026-11-04/THEORIE.md, 2026-11-04/orbit.py, 2026-11-04/rocket.py
---
## Idee und Definition

Newtons Gravitationsgesetz geteilt durch die Masse des umlaufenden Körpers:

$$\ddot{\mathbf r} = -\frac{GM}{|\mathbf r|^3}\,\mathbf r$$

Nur das Produkt $GM$ kommt vor (Erde: $3{,}986\cdot10^{14}\ \mathrm{m^3/s^2}$).

**Kreisbahn:** Die Gravitation liefert genau die Zentripetalbeschleunigung, $v^2/r = GM/r^2$, also $v = \sqrt{GM/r}$ und

$$T = 2\pi\sqrt{\frac{r^3}{GM}}$$

(drittes Keplersches Gesetz). Mond mit $r_M = 384\,400$ km: $v = 1018$ m/s, $T = 27{,}45$ d. Der siderische Monat ist 27,32 d, weil Erde und Mond um den gemeinsamen Schwerpunkt kreisen: Im Zweikörperproblem steht $G(M+m)$ statt $GM$, das gibt 27,29 d.

## Algorithmus und Aufwand

**Erhaltungsgrößen** pro Masseneinheit:

$$E = \tfrac12|\mathbf v|^2 - \frac{GM}{|\mathbf r|}, \qquad \mathbf L = \mathbf r\times\mathbf v$$

$\dot E = 0$, weil die Arbeit der Gravitation genau die potentielle Energie abbaut; $\dot{\mathbf L} = \mathbf r\times\mathbf a = 0$, weil die Kraft parallel zu $\mathbf r$ ist (Zentralkraft).

**Bahn aus einem Zustand:** $E$ und $L$ bestimmen den Kegelschnitt mit Exzentrizität $e = \sqrt{1 + 2EL^2/(GM)^2}$:

$$r_p = \frac{L^2}{GM(1+e)}, \qquad r_a = \frac{L^2}{GM(1-e)}\ (e < 1)$$

$E < 0$ Ellipse, $E \ge 0$ Flucht. Für die Kreisbahn gilt $E = -GM/(2r)$ und $L = \sqrt{GMr}$. Vis-viva: $v^2 = GM(2/r - 1/a)$ mit großer Halbachse $a = -GM/(2E)$.

## Numerische Fallstricke

- Fehler eines Verfahrens zeigen sich als Spirale: Explizites Euler (nach außen) und implizites Euler (nach innen) zerstören $E$ und $L$; [[Symplektische Integratoren]] nicht.
- Der implizite Euler mit $\Delta t = 6$ h stürzt den Mond nach 58 Schritten fast ins Zentrum: Fällt er nach innen, wächst $\omega\Delta t$ und damit die Dämpfung pro Schritt, eine Lawine.
- Kreisbahnen eignen sich als Test, weil die exakte Lösung bekannt ist: nach $T/2$ liegt der Mond bei $(-r_M, 0)$.

## In Python

```python
def periapsis(y):
    r, v = y[:2], y[2:]
    E = 0.5 * v @ v - GM / np.linalg.norm(r)
    L = r[0] * v[1] - r[1] * v[0]
    e = np.sqrt(max(0.0, 1 + 2 * E * L**2 / GM**2))
    return L**2 / (GM * (1 + e))
```
