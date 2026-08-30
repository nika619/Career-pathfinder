# Reskin Layer — "Cartographer's Observatory"

This build includes an **optional presentational reskin** on top of the base dark-editorial design system. It is fully isolated in `backend/static/reskin.css`, loaded *after* the base stylesheet, and changes **only appearance — never structure or behaviour.** All 54 tests and every interaction are identical with or without it.

## What it changes
- **Editorial display serif** (Fraunces, optical sizing) for headlines — the signature move toward editorial sophistication.
- **Warmer "brass" ink palette** so the gold accent reads as candlelight rather than a highlighter.
- **Hairline letterhead rules** across the top of cards, and refined monospace kickers with a small brass tick.
- **Tabular instrument numerals** for every metric (readiness, mastery, rubric points).
- **Brass-gradient primary button**, finer ghost buttons, pills, and chips.
- Retuned atmospheric glows and a brass scrollbar.

## How to turn the reskin OFF (revert to the base look)
Remove these two lines from `backend/static/index.html`:
1. the **Fraunces** font `<link>` (marked `<!-- RESKIN: editorial display serif -->`)
2. the **reskin.css** `<link>` (marked `<!-- RESKIN: presentational overlay ... -->`)

Optionally delete `backend/static/reskin.css`. Nothing else references it.

> A companion build with the reskin already removed (original visual design, same refinements and optimizations) is provided separately as `career-pathfinder-no-reskin.zip`.
