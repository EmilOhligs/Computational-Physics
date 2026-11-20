---
title: Randbehandlung bei finiten Differenzen
aliases: Geisterpunkt, Ghost Point, Neumann-Randbedingung, Dirichlet-Randbedingung, Spiegelung am Rand
area: dgl
summary: Am Rand fehlen dem Differenzenstern Nachbarn; wie man sie ersetzt, entscheidet über die Ordnung der ganzen Lösung, weil sich bei elliptischen Gleichungen ein Randfehler überall auswirkt.
links: finite-differenzen-methode
sources: 2026-11-11/THEORIE.md, 2026-11-11/fd.py, 2026-11-11/exercise_5_fd.ipynb
---
## Idee und Definition

**Dirichlet** ($u$ am Rand gegeben): Der bekannte Randwert fällt aus der Matrix heraus und wandert auf die rechte Seite, $f_1 \to f_1 + u_0/\Delta x^2$. Bei $u_0 = 0$ fehlt in der ersten Zeile einfach ein Nachbar.

**Neumann** ($u'$ am Rand gegeben, hier $u'(1) = 0$): $u_{N+1}$ wird eine zusätzliche Unbekannte und braucht eine eigene Gleichung.

- **Einseitig, 1. Ordnung:** $(u_{N+1} - u_N)/\Delta x = 0$.
- **Geisterpunkt, 2. Ordnung:** zentrale Ableitung $(u_{N+2} - u_N)/(2\Delta x) = 0$, also $u_{N+2} = u_N$. In den normalen Stern eingesetzt: $(-2u_N + 2u_{N+1})/\Delta x^2 = f_{N+1}$. Zeile durch 2 teilen, dann bleibt $A$ symmetrisch (rechts $f_{N+1}/2$).

**Breite Sterne am Dirichlet-Rand** (Ordnung 4 braucht $u_{-1}$):

- **Reduzieren:** in randnahen Zeilen den kleineren Stern nehmen. Immer anwendbar, aber dort nur Ordnung 2, und $A$ wird unsymmetrisch.
- **Spiegeln:** ungerade fortsetzen, $u_{-1} = -u_1$. Erste Zeile bei Ordnung 4: $\frac{1}{12}(29,\ -16,\ 1)$. Volle Ordnung, $A$ symmetrisch, aber nur richtig, wenn $u = u'' = u'''' = \dots = 0$ am Rand.

## Algorithmus und Aufwand

Gemessen mit $-u'' = \frac{\pi^2}{4}\sin\frac{\pi x}{2}$, exakt $u = \sin\frac{\pi x}{2}$:

- einseitig 1. Ordnung: **Steigung 1**, Fehler bei $N = 15$ am Rand $-0{,}076$
- Geisterpunkt: **Steigung 2**, Fehler bei $N = 15$ etwa $8\cdot10^{-4}$, rund 100-mal kleiner

Gemessen mit $f = \pi^2\sin\pi x$ (Ordnungsraten):

- Ordnung 4: Spiegeln 4,00, Reduzieren 4,66 → 4,14 (von oben gegen 4)
- Ordnung 6: Spiegeln 6,00, Reduzieren nur **5,00**

## Numerische Fallstricke

- **Der Randfehler bleibt nicht am Rand.** $A^{-1}$ ist voll besetzt (Green'sche Funktion), jede Zeile beeinflusst die ganze Lösung. Bei der einseitigen Neumann-Variante erzwingt die Zeile in Wahrheit $u'(1) = \frac{\Delta x}{2}u''(1) = -\frac{\pi^2}{8}\Delta x$. Der Fehler ist die Gerade $e(x) \approx -\frac{\pi^2}{8}\Delta x\,x$ im ganzen Intervall. Die schwächste Zeile bestimmt die Ordnung.
- Warum Reduzieren bei Ordnung 6 eine Ordnung kostet: In Zeile 1 steht der 3-Punkte-Stern mit $\tau_1 = \frac{\Delta x^2}{12}u''''(x_1) \sim \Delta x^3$ (weil $u''''(0) = 0$ für $\sin\pi x$). Über $A^{-1}$ wirkt das als $\Delta x^2\,\tau_1 \sim \Delta x^5$ überall. Bei Ordnung 4 ist der innere Fehler $\Delta x^4$ größer, man merkt nichts.
- Spiegeln ist falsch, sobald $u''(0) \ne 0$. Dann wäre der Fehler am Rand größer als beim Reduzieren.
- Siehe [[Finite-Differenzen-Methode]] für Stern und Matrix.

## In Python

```python
# Neumann u'(1) = 0 with ghost point, N+1 unknowns x_1..x_{N+1} = 1
A = scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N + 1, N + 1), format="lil")
A[N, N - 1], A[N, N] = -1, 1      # last row halved: stays symmetric
b = f(x); b[N] /= 2
u = scipy.sparse.linalg.spsolve(A.tocsr() / dx**2, b)
```
