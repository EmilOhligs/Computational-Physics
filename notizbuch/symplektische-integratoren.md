---
title: Symplektische Integratoren
aliases: symplektisches Euler, Velocity Verlet, Leapfrog, Kick-Drift-Kick
area: dgl
summary: Verfahren für $\ddot{\mathbf r} = \mathbf a(\mathbf r)$, die die Fläche im Phasenraum erhalten ($\det = 1$); ihr Energiefehler bleibt für beliebig lange Zeiten beschränkt, bei Zentralkräften erhalten sie den Drehimpuls exakt.
links: euler-verfahren, runge-kutta-verfahren, stabilitaetsgebiet, kepler-problem
sources: 2026-11-04/THEORIE.md, 2026-11-04/orbit.py, 2026-11-04/plot_1b.py, 2026-11-04/plot_1c.py
---
## Idee und Definition

Statt Ort und Geschwindigkeit gleichzeitig mit den alten Werten zu verschieben, wechseln sich **Kick** ($\mathbf r$ fest, $\mathbf v$ ändert sich) und **Drift** ($\mathbf v$ fest, $\mathbf r$ ändert sich) ab.

**Symplektisches Euler** (Ordnung 1):

$$\mathbf v_{n+1} = \mathbf v_n + \Delta t\,\mathbf a(\mathbf r_n), \qquad \mathbf r_{n+1} = \mathbf r_n + \Delta t\,\mathbf v_{n+1}$$

Der einzige Unterschied zum expliziten [[Euler-Verfahren]] ist $\mathbf v_{n+1}$ statt $\mathbf v_n$ in der Ortsgleichung.

**Velocity Verlet** (Ordnung 2, zeitsymmetrisch): halber Kick, ganzer Drift, halber Kick:

$$\mathbf v_{n+1/2} = \mathbf v_n + \tfrac{\Delta t}{2}\mathbf a(\mathbf r_n), \quad \mathbf r_{n+1} = \mathbf r_n + \Delta t\,\mathbf v_{n+1/2}, \quad \mathbf v_{n+1} = \mathbf v_{n+1/2} + \tfrac{\Delta t}{2}\mathbf a(\mathbf r_{n+1})$$

**Symplektisch** heißt: Die Abbildung $(\mathbf r_n, \mathbf v_n) \mapsto (\mathbf r_{n+1}, \mathbf v_{n+1})$ erhält die Fläche im Phasenraum, wie die exakte Bewegung (Satz von Liouville). Beim harmonischen Oszillator sieht man das an der Schrittmatrix: Euler hat $\det = 1 + \omega^2\Delta t^2$, also spiralt er nach außen; der implizite Euler hat $\det = 1/(1 + \omega^2\Delta t^2)$, er spiralt nach innen; das symplektische Euler hat $\det = 1$, eine Spirale ist unmöglich.

## Algorithmus und Aufwand

- Eine Kraftauswertung pro Schritt (im Prinzip: Verlet kann $\mathbf a(\mathbf r_{n+1})$ im nächsten Schritt wiederverwenden; `orbit.py` rechnet der Einfachheit halber zweimal), RK4 braucht vier.
- **Drehimpuls exakt erhalten** bei Zentralkräften: Ein Kick ändert $\mathbf v$ parallel zu $\mathbf r$, also $\mathbf r\times\Delta\mathbf v = 0$; ein Drift ändert $\mathbf r$ parallel zu $\mathbf v$. Gemessen beim Mond über 100 Umläufe $|\Delta L/L| \le 2\cdot10^{-14}$, reine Rundung.
- **Energiefehler beschränkt:** Rückwärtsfehleranalyse zeigt, dass eine Schatten-Energie $\tilde E = E + O(\Delta t^p)$ über sehr lange Zeiten fast exakt erhalten bleibt. Beim Mond mit $\Delta t = 6$ h: symplektisches Euler $3{,}3\cdot10^{-3}$, Verlet $2{,}7\cdot10^{-6}$, nach 10 und nach 100 Umläufen gleich.

## Numerische Fallstricke

- Die Ordnung allein entscheidet nicht über die Langzeitgüte: Heun (Ordnung 2) ist beim Mond nach knapp 6 Umläufen schlechter als das symplektische Euler (Ordnung 1), RK4 (Ordnung 4) nach etwa 20 Umläufen schlechter als Verlet. Nicht symplektische Fehler summieren sich mit gleichem Vorzeichen.
- Auf kurzen Zeiten gewinnt die Ordnung: Fehler nach einem halben Umlauf bei $T/6400$ RK4 $7{,}5\cdot10^{-14}$, Verlet $1{,}1\cdot10^{-6}$.
- Auch symplektische Verfahren haben einen **Phasenfehler**, der linear wächst: Die Bahnform stimmt, die Position auf der Bahn driftet.
- Die Bahn des symplektischen Euler ist bei großem $\Delta t$ eine leicht verformte Ellipse (Radius bis 1,003 $r_M$), die aber stabil bleibt.
- Schub, der von der Zeit oder der Geschwindigkeit abhängt, zerstört die Struktur; für die Rakete wurde deshalb RK4 verwendet.

## In Python

```python
def velocity_verlet(y, dt):
    v_half = y[2:] + dt / 2 * acceleration(y[:2])
    r = y[:2] + dt * v_half
    return np.concatenate([r, v_half + dt / 2 * acceleration(r)])
```

Für 1000 Jahre Mondbahn ist Velocity Verlet die Wahl: $L$ exakt, $E$ beschränkt, billig.
