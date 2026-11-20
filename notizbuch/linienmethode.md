---
title: Linienmethode
aliases: Method of Lines, MOL, Crank-Nicolson, Semidiskretisierung, Wärmeleitungsgleichung
area: dgl
summary: Nur der Raum wird diskretisiert; aus $u_t = u_{xx}$ wird das ODE-System $\dot u = -Au$, das man mit jedem ODE-Löser (Euler, Crank-Nicolson, RK4) in der Zeit integriert.
links: finite-differenzen-methode, von-neumann-stabilitaetsanalyse, euler-verfahren, stabilitaetsgebiet, steife-differentialgleichungen
sources: 2026-11-11/THEORIE.md, 2026-11-11/fd.py, 2026-11-11/exercise_5_fd.ipynb, 2026-10-28/ode.py
---
## Idee und Definition

Wärmeleitung $u_t = \alpha u_{xx}$ ($\alpha = 1$), $u(0) = u(1) = 0$. Mit der Matrix $A$ von $-\frac{d^2}{dx^2}$ aus der [[Finite-Differenzen-Methode]]:

$$\dot u = -A\,u$$

Das ist genau die lineare Form $\dot x = Jx$ mit $J = -A$. Die drei Standardverfahren:

$$\text{expl. Euler (FTCS):}\quad u^{n+1} = (I - \Delta t A)\,u^n, \qquad R(z) = 1 + z$$

$$\text{impl. Euler (BTCS):}\quad (I + \Delta t A)\,u^{n+1} = u^n, \qquad R(z) = \frac{1}{1-z}$$

$$\text{Crank-Nicolson:}\quad \Big(I + \tfrac{\Delta t}{2}A\Big)u^{n+1} = \Big(I - \tfrac{\Delta t}{2}A\Big)u^n, \qquad R(z) = \frac{1 + z/2}{1 - z/2}$$

Crank-Nicolson ist die Trapezregel, Ordnung 2 in der Zeit. Gesamtfehler: $O(\Delta t^p) + O(\Delta x^2)$.

## Algorithmus und Aufwand

- Die Schritte aus Exercise 3 (`euler_step`, `heun_step`, `rk4_step`, `integrate`) laufen **unverändert** mit sparse $J$, weil sie nur `J @ x` brauchen. Nur der implizite Schritt braucht `spsolve` statt `np.linalg.solve`.
- Crank-Nicolson = halber expliziter Schritt, dann halber impliziter Schritt.
- Ein impliziter Schritt kostet bei tridiagonalem $A$ nur $O(N)$, kaum mehr als ein expliziter.
- **Zwei Referenzlösungen** bei Start $\sin\pi x$ (Eigenvektor von $A$): PDE exakt $e^{-\pi^2 t}\sin\pi x_j$, ODE-System exakt $e^{-\lambda_1 t}\sin\pi x_j$. Gegen die ODE-Lösung misst man nur den Zeitfehler, ihr Abstand zur PDE-Lösung ist der reine Raumfehler.
- Gemessen ($N = 50$, $t = 0{,}1$): Steigungen in $\Delta t$ 1,00 / 1,00 / 2,00 (expl. Euler, impl. Euler, CN). In $\Delta x$ bei $\Delta t = 10^{-4}$: CN Steigung 2 bis $4{,}6\cdot10^{-6}$.

## Numerische Fallstricke

- **Fehlerboden:** Gegen die PDE-Lösung fällt der Fehler nie unter den Fehler der anderen Diskretisierung. Bei $N = 50$ bleibt er beim Raumfehler $1{,}16\cdot10^{-4}$ stehen. Umgekehrt bleibt der implizite Euler in der $\Delta x$-Studie bei $1{,}9\cdot10^{-4}$, seinem Zeitfehler bei $\Delta t = 10^{-4}$.
- **Fehlerauslöschung:** Wegen $\lambda_1 < \pi^2$ klingt die ODE-Lösung zu langsam ab, expl. Euler und CN dämpfen zu stark. Die Fehler heben sich teilweise auf: Der explizite Euler liegt *unter* dem Raumfehler, CN hat bei $\Delta t \approx 6\cdot10^{-3}$ einen tiefen Knick. Ein kleiner Fehler beweist also nicht, dass alles stimmt.
- **Verzögerte Instabilität:** Der explizite Euler ist nur für $\Delta t < 2/\lambda_N = 1{,}92\cdot10^{-4}$ stabil. Startet man mit reinem $\sin\pi x$, kommen die hohen Moden erst über Rundung ($10^{-16}$) hinein. Bei $\Delta t = 0{,}05$ (2 Schritte) sieht man nichts, bei $\Delta t = 3{,}9\cdot10^{-4}$ (256 Schritte) steht der Fehler bei $10^{107}$. Grund: [[Von-Neumann-Stabilitätsanalyse]].
- Bei unglatten Startwerten ist CN trotz Ordnung 2 schlechter als der implizite Euler, siehe [[Steife Differentialgleichungen]].

## In Python

```python
J = -laplacian(N)                     # u' = J u, sparse

def implicit_euler_step(J, x, dt):
    I = scipy.sparse.identity(len(x), format="csc")
    return scipy.sparse.linalg.spsolve(I - dt * J, x)

def crank_nicolson_step(J, x, dt):    # (I - dt/2 J) x_new = (I + dt/2 J) x
    return implicit_euler_step(J, x + dt / 2 * (J @ x), dt / 2)

t, U = integrate(crank_nicolson_step, J, np.sin(np.pi * x), 1e-4, 0.1)
```
