---
title: Absorbierende Randbedingungen
aliases: komplexes absorbierendes Potential, CAP, optisches Potential, V - iW
area: physmod
summary: Ein imaginäres Potential $-iW(x)$ in einer Randschicht entzieht der Welle Wahrscheinlichkeit, $\frac{d}{dt}\|\psi\|^2 = -2\int W|\psi|^2$, und macht das Rechengebiet nach außen offen.
links: crank-nicolson-verfahren, schroedinger-gleichung-finite-differenzen, randbehandlung-finite-differenzen
sources: 2026-11-18/theory.md, 2026-11-18/schroedinger.py, 2026-11-18/figures.py
---
## Idee und Definition

$\psi = 0$ am Rand des Gitters ist eine unendlich hohe Wand: Jedes Paket wird vollständig reflektiert und läuft zurück ins Gebiet. Für Streuprobleme (Tunneln, Doppelspalt) verfälscht das die Messung.

Mit $V \to V - iW(x)$, $W \ge 0$, gilt

$$\frac{d}{dt}\int|\psi|^2\,dx = -2\int W(x)\,|\psi|^2\,dx \le 0$$

Wo $W > 0$ ist, verschwindet Wahrscheinlichkeit; die Welle wird geschluckt statt reflektiert.

## Algorithmus und Aufwand

- Quadratische Rampe in einer Randschicht der Breite $w$: $W = W_0\,((w - d)/w)^2$ für Randabstand $d < w$, sonst 0.
- Kostet nichts extra: $H$ wird komplex, Crank–Nicolson bleibt gleich.
- In 2D: $W(x, y) = W_x(x) + W_y(y)$ an allen vier Rändern.
- Messung trotz Absorber: transmittierte Wahrscheinlichkeit = was rechts ist + $2\int dt\int W|\psi|^2$ im rechten Absorber.
- Paket $k_0 = 2$, Gebiet $[-30, 30]$, Schicht $w = 10$, $W_0 = 2$: übrig bleiben $5\cdot10^{-5}$.

## Numerische Fallstricke

- Zu steiles $W$ reflektiert selbst (Sprung im Potential), zu schwaches lässt die Welle bis zur Wand und zurück laufen. Die Schicht sollte mehrere Wellenlängen breit sein.
- Langsame Anteile (kleines $k$) werden schlechter absorbiert als schnelle.
- Die Norm ist mit Absorber keine Erhaltungsgröße mehr; Normtests nur ohne $W$.

## In Python

```python
d = np.minimum(x - a, b - x)
W = np.where(d < width, strength * ((width - d) / width) ** 2, 0.0)
H = hamiltonian(V - 1j * W, dx)
```
