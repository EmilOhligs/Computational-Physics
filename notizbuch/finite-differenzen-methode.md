---
title: Finite-Differenzen-Methode
aliases: FDM, Differenzenstern, Stencil, 3-Punkte-Stern, 5-Punkte-Stern
area: dgl
summary: Ableitungen werden auf einem regulären Gitter durch Differenzenquotienten ersetzt; aus $-u'' = f$ wird ein dünnbesetztes lineares System $Au = f$ mit Fehler $O(\Delta x^2)$ beim 3-Punkte-Stern.
links: randbehandlung-finite-differenzen, linienmethode, rundungsfehler, von-neumann-stabilitaetsanalyse
sources: 2026-11-11/THEORIE.md, 2026-11-11/fd.py, 2026-11-11/exercise_5_fd.ipynb
---
## Idee und Definition

Gitter $x_j = j\,\Delta x$ mit $N$ inneren Punkten, $\Delta x = 1/(N+1)$. Ein **Differenzenstern** legt fest, welche Nachbarn mit welchem Gewicht in die Ableitung eingehen. Aus der Taylor-Entwicklung in beide Richtungen (ungerade Terme fallen weg):

$$\frac{u_{j-1} - 2u_j + u_{j+1}}{\Delta x^2} = u''(x_j) + \frac{\Delta x^2}{12}\,u''''(x_j) + O(\Delta x^4)$$

Für $-u'' = f$ mit $u(0) = u(1) = 0$ wird jede Gitterpunkt-Gleichung eine Zeile von

$$A = \frac{1}{\Delta x^2}\begin{pmatrix} 2 & -1 & & \\ -1 & 2 & \ddots & \\ & \ddots & \ddots & -1 \\ & & -1 & 2 \end{pmatrix}$$

Die Eigenwerte sind exakt bekannt: $\lambda_k = \frac{4}{\Delta x^2}\sin^2\frac{k\pi\Delta x}{2}$, Eigenvektoren $\sin(k\pi x_j)$. $A$ ist symmetrisch positiv definit.

Höhere Ordnung durch breitere Sterne (Gewichte für $-u''$, ohne $1/\Delta x^2$):

- Ordnung 2: $(-1,\ 2,\ -1)$
- Ordnung 4: $\frac{1}{12}(1,\ -16,\ 30,\ -16,\ 1)$
- Ordnung 6: $\frac{1}{180}(-2,\ 27,\ -270,\ 490,\ -270,\ 27,\ -2)$

## Algorithmus und Aufwand

- $A$ als sparse-Matrix bauen (`scipy.sparse.diags`), mit `spsolve` lösen. Tridiagonal kostet $O(N)$, dense wäre bei $N = 10^6$ nicht speicherbar.
- **$A$ prüfen, bevor man löst:** Symmetrie, Zeilensummen innen 0, $A\,x(1-x) = 2$ exakt (weil $u'''' = 0$), Eigenwerte gegen die Formel.
- Gemessen ($f = \pi^2\sin\pi x$): **Steigung 2,00**, der Fehler liegt genau auf $\pi^2\Delta x^2/12$. Ordnung 4 und 6 (mit Spiegelung am Rand) ergeben 4,00 und 6,00.
- **2D** auf dem Einheitsquadrat: $A_{2D} = A\otimes I + I\otimes A$ (`scipy.sparse.kron`), das ist der 5-Punkte-Stern $(4u_{ij} - u_{i\pm1,j} - u_{i,j\pm1})/\Delta x^2$. Eigenwerte $\lambda_k + \lambda_l$.
- 2D gemessen: Fehler weiter Steigung 2, aber die Laufzeit wächst mit $\approx N^{2{,}8}$, also fast $n^{1{,}5}$ bei $n = N^2$ Unbekannten. $N = 511$: 1,3 s, $N = 1023$ (eine Million Unbekannte): 10 s. Grund: Bandbreite $N$, beim Faktorisieren füllt sich das Band auf (fill-in).

## Numerische Fallstricke

- **Rundungsminimum:** Der Fehler fällt nicht beliebig. In 1D ist das Minimum $2{,}3\cdot10^{-10}$ bei $\Delta x = 1{,}5\cdot10^{-5}$ ($N = 65535$), darunter wächst er wieder auf $6\cdot10^{-7}$ bei $N \approx 10^6$. Ursache: Einträge $\sim 1/\Delta x^2$, Auslöschung im Stern, Kondition $\kappa(A) \approx 4/(\pi^2\Delta x^2)$, siehe [[Rundungsfehler]].
- Höhere Ordnung erreicht den Rundungsboden schon bei gröberem Gitter: Ordnung 6 hat $2\cdot10^{-13}$ bei $N = 127$.
- Breite Sterne ragen am Rand über das Gebiet hinaus, siehe [[Randbehandlung bei finiten Differenzen]].
- In 2D muss die rechte Seite in derselben Reihenfolge zum Vektor werden wie die Unbekannten: `meshgrid(x, y, indexing="ij")` und `.ravel()` passen zu `kron(A, I)` für den ersten Index. Test mit unsymmetrischer Quelle $\sin\pi x\,\sin 2\pi y$ findet vertauschte Achsen.
- Für zeitabhängige Probleme wird $A$ zum Ausgangspunkt der [[Linienmethode]].

## In Python

```python
def laplacian(N):
    dx = 1 / (N + 1)
    return scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format="csr") / dx**2

def laplacian_2d(N):
    A, I = laplacian(N), scipy.sparse.identity(N)
    return (scipy.sparse.kron(A, I) + scipy.sparse.kron(I, A)).tocsr()

x = np.arange(1, N + 1) / (N + 1)
u = scipy.sparse.linalg.spsolve(laplacian(N), f(x))
```
