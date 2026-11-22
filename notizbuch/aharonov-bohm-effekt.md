---
title: Aharonov-Bohm-Effekt
aliases: Peierls-Substitution, Vektorpotential, Mach-Zehnder-Interferometer
area: physmod
summary: Zwei Wege um einen eingeschlossenen Fluss $\Phi$ bekommen die Phasendifferenz $q\Phi$, auch wenn $\mathbf B = 0$ auf beiden Wegen ist; im Mach-Zehnder-Interferometer gilt $P_{\rm oben} = \cos^2(q\Phi/2)$.
links: schroedinger-gleichung-finite-differenzen, crank-nicolson-verfahren, absorbierende-randbedingungen
sources: 2026-11-18/theory.md, 2026-11-18/schroedinger.py, 2026-11-18/figures.py
---
## Idee und Definition

Mit Magnetfeld ist $H = \frac12(-i\nabla - q\mathbf A)^2 + V$. Ein Weg sammelt die Phase $q\int\mathbf A\cdot d\mathbf l$; zwei Wege unterscheiden sich um

$$\Delta\varphi = q\oint\mathbf A\cdot d\mathbf l = q\int\!\!\int\mathbf B\cdot d\mathbf S = q\Phi$$

Die Phase ist eichinvariant und hängt nur vom eingeschlossenen Fluss ab. Klassisch zählt nur die Lorentzkraft am eigenen Ort, also nichts, wenn dort $\mathbf B = 0$ ist.

$\mathbf A$ kann außerhalb eines beschränkten Gebiets nicht verschwinden: Jeder Kreis um die Spule hat $\oint\mathbf A\cdot d\mathbf l = \Phi \ne 0$.

Unterschied zum skalaren Potential: $V\psi$ wirkt lokal. Eine Potentialscheibe zwischen den Armen eines Mach-Zehnder-Interferometers ändert nichts (oben 0,93 bis 0,94), eine Spule an derselben Stelle vertauscht die Ausgänge.

## Algorithmus und Aufwand

- **Peierls-Substitution:** Hopping zwischen Nachbarn $p$ und $p + \mathbf e$ ist $-\frac{1}{2\Delta x^2}e^{-iqA_e\Delta x}$ ($A_e$ an der Kantenmitte), Gegenrichtung konjugiert. So bleibt $H$ hermitesch, und eine ebene Welle hat die Energie von $(k - qA)^2/2$.
- Test: Konstantes $\mathbf A$ ist eine reine Eichung, das Spektrum muss exakt gleich bleiben.
- Spule in Coulomb-Eichung: $\mathbf A = \frac{B_0r}{2}\mathbf e_\theta$ innen, $\frac{B_0r_0^2}{2r}\mathbf e_\theta$ außen, $\Phi = \pi B_0r_0^2$.
- Mach-Zehnder aus waagrechten Barrieren (gitterparallel), Paket $k_0 = 8$ unter 45°. Der Strahlteiler wirkt nur auf $E_\perp = k_y^2/2 = 16$; das 2D-Problem zerfällt exakt, die Höhe für $T = \frac12$ kommt aus einer 1D-Bisektion: $V_{\rm BS} = 20{,}87$ (Dicke 0,24).
- Ergebnis: ohne Feld oben 0,929 / unten 0,045; $B_0 = 1$ ($\Phi = \pi$): oben 0,032 / unten 0,938; Periode $B_0 = 2$.

## Numerische Fallstricke

- Ohne Feld geht fast alles in einen Ausgang, weil $r/t$ bei einer symmetrischen Barriere rein imaginär ist: $|t^2 + r^2| = ||t|^2 - |r|^2| = 0$.
- Eine Potentialscheibe im Arm gibt keinen vollen Kontrast: Das Paket ($\sigma \approx 1{,}2$) ist breiter als die Scheibe (Radius 1), jeder Teil sieht eine andere Phase. Bestes Ergebnis bei $V = -15$: unten 0,56, oben 0,33.
- Barrieren gitterparallel legen und ihre Dicke als ganze Zahl von Gitterpunkten wählen; schräge Wände werden treppig.

## In Python

```python
theta = A_x(x_mid, y) * dx            # Phase auf jeder x-Kante
hop = -0.5 / dx**2 * np.exp(-1j * theta)
H = sp.coo_matrix((hop, (p, p + ny)), shape=(n, n))
H = H + H.conj().T + sp.diags(2 / dx**2 + V.ravel())
```
