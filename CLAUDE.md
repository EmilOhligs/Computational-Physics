# Computational Physics (VU, Uni Wien, WS 2026/27)

Emils Übungsprojekte. Jede Einheit liegt in einem Ordner mit Datum (`2026-10-14/` …), meist mit `exercise.pdf`, Notebook/Code, `THEORIE.md`, Tests.
Sprache: Erklärungen und Notizen auf Deutsch, Code und Bezeichner auf Englisch.

## Notizbuch (Wissensdatenbank in Sheldon)

Der Ordner `notizbuch/` ist Emils Lern-Wissensdatenbank. Wenn Emil mit einer Woche fertig ist, gibt er in Sheldon den Import frei; dann wird jede Datei ein Eintrag im Sheldon-Notizbuch (Fach „Computational Physics“, mit Themenkarte und Begriff des Tages). Notizen deshalb bis zum Wochenende fertig und sauber halten.

**Wann:** Nach jeder Einheit (oder wenn Emil sagt „ins Notizbuch“) die zentralen Methoden/Begriffe als Notizen anlegen oder bestehende ergänzen. Ein Thema = eine Datei. Lieber 3–5 gute Notizen pro Einheit als viele kleine. Vorher `ls notizbuch/` prüfen und bestehende Dateien erweitern statt doppeln.

**Datei:** `notizbuch/<slug>.md`, Slug = Kleinbuchstaben, Bindestriche, keine Umlaute (z. B. `runge-kutta-verfahren.md`). Der Slug ist die feste ID – nicht umbenennen.

**Format:**

```markdown
---
title: Runge-Kutta-Verfahren
aliases: RK4, Runge-Kutta
area: dgl
summary: Ein bis zwei Sätze, Formeln mit $…$ erlaubt.
links: euler-verfahren, schrittweitensteuerung
sources: 2026-10-21/THEORIE.md, 2026-10-21/rk4.py
---
## Idee und Definition
… Text mit Formeln …

## Algorithmus und Aufwand
…

## Numerische Fallstricke
…

## In Python
…
```

- `area` genau einer von: `numerik` (Numerik & Fehler), `fourier` (Fourier & Signale), `linalg` (Lineare Algebra), `dgl` (Differentialgleichungen), `stochastik` (Monte Carlo & Stochastik), `optimierung` (Optimierung & Nullstellen), `physmod` (Physikalische Modelle).
- `aliases`, `links`, `sources`: kommagetrennt. `links` = Slugs anderer Notizen; im Text verweist `[[Titel]]` auf eine andere Notiz.
- **Formeln immer in LaTeX:** inline `$…$`, abgesetzt `$$…$$` auf eigenen Zeilen. Sheldon setzt sie mit MathJax.
- Markdown-Umfang: `##`-Überschriften, Absätze, `- `-Listen, `**fett**`, `` `code` ``, Codeblöcke mit ```. Keine Tabellen, keine Bilder.
- Inhalt aus dem, was in der Einheit wirklich gemacht wurde (THEORIE.md, Code, Messergebnisse) – konkrete Zahlen aus den eigenen Messungen sind gut. Nichts erfinden.
