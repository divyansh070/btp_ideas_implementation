# BTP: Cross-chemistry battery RUL — Version 1 (Track A → Track B)

## Source of truth
- docs/*Design_Guide*.pdf — Part 1 (Version 1) is what we build. Ignore Part 2 (Version 2) unless asked.
- docs/*PBT*What_the_Paper*.pdf — facts about PBT (arXiv:2512.16334 v4).
- docs/*.pptx — slide summary (read with python-pptx).
If code and the design guide disagree, STOP and ask me. Never silently pick one.

## Current phase
Track A (benchmark PBT) + shared data pipeline. Do NOT start Track B until I say the week-6 gate passed.

## Logging rule (mandatory, every change)
Maintain document.md at repo root. Top of file: "## Decisions" (dated, one line each) and "## Known issues / open questions".
Below that, append one entry per change, never edit old entries (append corrections instead):

### YYYY-MM-DD HH:MM — <short title>
- Changed: <files / functions>
- Why: <reason, link to design-guide section>
- Verified by: <exact command> → <result / numbers>
- Assumptions / risks: <anything guessed>
- Commit: <hash>

Also log every failed attempt and every error + fix. That's the main reason the file exists.

## Non-negotiable rules (from the design guide)
- Splits are cell-level: all windows of a cell go to exactly one split.
- SNL = three domains (LFP / NMC / NCA). Split by cell ID; no cell tagged twice.
- Fail loudly on missing cycle_life; never fall back to last observed cycle.
- PBT filters: drop cells <100 cycles; drop cells whose SOH never reaches 82.5%; linear-extrapolate life for cells reaching 82.5% but not 80%.
- Scaler fit on source-train + target-adaptation windows only; clip to [-0.5, 1.5]; blind-test cells never enter the fit.
- Validation gate passes before ANY training (counts, zero EOL fallbacks, RUL–capacity corr > 0.5, feature ranges, std > 0.01).
- Metrics per chemistry, never pooled R². Model/hparam selection on validation only, never blind test.
- LayerNorm, not BatchNorm, in prediction heads.
- Using true EOL at test time is a leak: flag it in document.md whenever it happens.
- Fix and log seeds.

## Layout
docs/ configs/ src/ scripts/ results/ document.md
data/ and third_party/PBT are gitignored (local only).

## Working style
- Small commits, one logical change each; commit message mentions the document.md entry title. Push after each commit.
- Ask before: downloads > 1 GB, deleting anything, installing system packages, changing the evaluation protocol.
- Never commit data, weights, or tokens (HF token goes in .env).
