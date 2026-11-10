---
title: Raketengleichung
aliases: Ziolkowski, Gravitationsverlust, Stufenrakete, Pitch-Programm, Delta-v
area: physmod
summary: Ohne äußere Kräfte gewinnt eine Rakete $\Delta v = v_e\ln(m_0/m_1)$; im Schwerefeld geht davon der Gravitationsverlust $\int g\sin\gamma\,dt$ ab, und Stufen vergrößern $\Delta v$, weil sich die Massenverhältnisse multiplizieren.
links: kepler-problem, runge-kutta-verfahren
sources: 2026-11-04/THEORIE.md, 2026-11-04/rocket.py, 2026-11-04/plot_2a.py, 2026-11-04/plot_2b.py, 2026-11-04/plot_2c.py, 2026-11-04/plot_2d.py
---
## Idee und Definition

Impulssatz für die Rakete mit Ausströmgeschwindigkeit $v_e$: $m\,\dot v = \dot m\,v_e$, integriert

$$\Delta v = v_e \ln\frac{m_0}{m_1}$$

Im Schwerefeld mit konstantem Schub $F = \dot m v_e$:

$$\ddot{\mathbf r} = -\frac{GM}{|\mathbf r|^3}\mathbf r + \frac{F/m_0}{1 - (F/m_0)\,t/v_e}\,\mathbf e, \qquad t_b = \Big(1 - \frac{m_1}{m_0}\Big)\frac{v_e}{F/m_0}$$

Die Beschleunigung durch den Schub wächst, je leichter die Rakete wird.

## Algorithmus und Aufwand

- **Verluste:** $\frac{d|\mathbf v|}{dt} = \frac{F}{m}\mathbf e\cdot\hat{\mathbf v} + \mathbf a_g\cdot\hat{\mathbf v}$. Der zweite Term ist der Gravitationsverlust $g\sin\gamma$ (voll beim senkrechten Flug), die Lücke im ersten der Lenkverlust (Schub nicht in Flugrichtung).
- **Senkrechter Start** ($m_1 = m_0/19$, $v_e = 4{,}4$ km/s, $\Delta v = 12{,}96$ km/s): radialer Gravitationsverlust $\Delta v - v_r$ 3,9 km/s bei $F/m_0 = 10$, 2,2 km/s bei 18; Gesamtverlust $\Delta v - (|v_b| - |v_0|)$ 4,4 bzw. 2,6 km/s. Entkommen ab $F/m_0 = 17{,}13\ \mathrm{m/s^2}$. Ein Orbit ist unmöglich: Kein Drehmoment, $L = R_E^2\omega_E$ bleibt winzig, die Periapsis liegt höchstens 22 km vom Erdmittelpunkt.
- **Pitch-Programm** $\theta(t) = 90°\max(0, 1 - t/t_p)$ mit $F/m_0 = 15$, $m_1 = 0{,}12$: Orbit nur nach Osten mit $t_p = 150$ bis 200 s, am besten 175 s. Die Erdrotation schenkt nach Osten 465 m/s, das macht 930 m/s Unterschied zu Westen.
- **Kreisbahn:** brennen, bis die Apoapsis auf Zielhöhe liegt, zur Apoapsis gleiten, dort waagerecht nachbrennen, bis $E = -GM/(2r)$. Rest an Treibstoff bei 200 / 500 / 1000 km: 2,35 / 1,74 / 0,85 %.
- **Stufen:** $\Delta v = v_e\sum_k \ln(m_{0,k}/m_{1,k})$. Eine Stufe mit 90 % Treibstoff: 10,13 km/s; zwei Stufen (80 % + 20 %, je 90 % Treibstoff): 15,73 km/s, weil der leere erste Tank abgeworfen wird. Gravitationsverlust jeweils etwa 620 m/s, Lenkverlust etwa 570 m/s, zusammen 1,2 km/s statt 2,6 bis 4,4 km/s beim senkrechten Start.

## Numerische Fallstricke

- Beim Brennschluss springt die Beschleunigung. RK4 setzt eine glatte rechte Seite voraus, deshalb muss $t_b$ auf dem Zeitgitter liegen und jede Phase mit festem Triebwerkszustand rechnen.
- Ereignisse wie „Apoapsis erreicht“ oder „Bahn kreisförmig“ findet man, indem man bis zum Schritt mit Vorzeichenwechsel fliegt und dann nur diesen Schritt halbiert. Innerhalb eines Brennvorgangs muss dabei die Zeit seit dem Zünden weiterlaufen, sonst stimmen Pitch-Winkel und Masse nicht.
- Eine einzige Brennphase endet auf einer Ellipse durch den Brennschlusspunkt; für eine Kreisbahn in beliebiger Höhe braucht es Abschalten und erneutes Zünden.

## In Python

```python
def burn_time(thrust, m1):
    return (1 - m1) * V_EXHAUST / thrust

dv_two_stages = V_EXHAUST * (np.log(1 / 0.28) + np.log(0.2 / 0.02))
```
