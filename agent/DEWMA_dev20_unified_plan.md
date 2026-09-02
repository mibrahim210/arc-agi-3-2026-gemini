# DEWMA Dev20 Unified Plan

## Purpose

This document is the unified working plan for improving DEWMA after the local 25-game run archived at:

- `traces_archive/2026-08-31_13-37-17`

It merges:

- the formal diagnostic bundle in `diagnostics/trace_clustering/2026-08-31_13-37-17`
- the additional forensic feedback produced by the Gemini agent

This file is meant to be the single operational roadmap for the current development cycle.

## Evaluation Sets

### Baseline Protect Set

These five games must be preserved unless a change produces a clearly stronger portfolio gain:

- `tn36`
- `vc33`
- `lp85`
- `r11l`
- `lf52`

What they protect:

- `tn36`: simple productive conversion path
- `vc33`: control-band to interior conversion
- `lp85`: component-delete style productive path
- `r11l`: oscillation breakout plus bounded continuation
- `lf52`: early local-structure persistence

### Holdout Set

These games should not drive tuning directly. They are checkpoint tests for generalization:

- `dc22`
- `sp80`
- `sb26`
- `ls20`
- `su15`

What they stress:

- `dc22`: high-meaningingful finish-corridor nonconversion
- `sp80`: heavy interior-phase nonconversion
- `sb26`: oscillation / local-structure conversion failure
- `ls20`: weak mechanism/search underconversion
- `su15`: local continuation conflict plus death risk

### Dev20 Set

The active 20-game development set is:

- `tn36`, `vc33`, `lp85`, `r11l`, `lf52`
- `ar25`, `bp35`, `cd82`, `cn04`, `ft09`
- `g50t`, `ka59`, `m0r0`, `re86`, `s5i5`
- `sc25`, `sk48`, `tr87`, `tu93`, `wa30`

## Current Portfolio Status

From the August 31, 2026 run:

- aggregate scorecard score: `0.17819664591688916`
- full 25-game progress: `5 / 25`
- preserved baseline games: `tn36`, `vc33`, `lp85`, `r11l`, `lf52`

Core diagnosis:

- DEWMA is not mainly failing because a major reasoning layer is absent.
- DEWMA is currently **conversion-bound** and **action-economy-bound**.
- The biggest waste is now inside active continuation layers, especially `finish_corridor_active` and `interior_application_phase_active`, rather than only generic fallback.

## Dev20 Cluster Map

### 1. Productive Partial Solvers

Games:

- `tn36`
- `vc33`
- `lp85`
- `r11l`
- `lf52`

Role:

- regression control
- preserve these while improving adjacent unsolved games

Rule:

- do not patch globally from this cluster
- use this cluster as the safety gate for every meaningful controller change

### 2. Finish-Corridor Nonconversion

Games:

- `re86`
- `tr87`
- `tu93`

Shared profile:

- high or very high meaningful change
- repeated continuation pressure
- very high `finish_corridor_active`
- no level conversion

Interpretation:

- the agent is often on a physically active branch
- continuation is being over-trusted or over-extended
- the corridor is not narrowing into a decisive terminal maneuver

Unified recommendation:

- tighten finish-corridor entry and persistence only on non-progressing games
- require evidence of convergence, not only repeated state change
- decay corridor priority earlier when meaningful change continues but branch geometry does not narrow

Gemini-specific insight to preserve:

- `tr87` and `re86` may benefit from directional continuation logic when straight-line motion saturates
- treat this as a subcase of finish-corridor nonconversion, not a totally separate architecture

### 3. Interior-Phase Nonconversion

Games:

- `ar25`
- `cn04`
- `m0r0`
- `sc25`

Shared profile:

- `control_band_saturation_detected` and `interior_application_phase_active` are real
- the control-band phase switch is happening
- but interior action selection remains too diffuse

Interpretation:

- detection is not the problem
- selection inside the interior canvas is the problem

Unified recommendation:

- audit which interior-biased candidate actually wins after phase switch
- favor boundary, contour, and active component targets over broad canvas sweeping
- require live payoff inside interior mode to keep it active

Gemini-specific insight to preserve:

- “contour-centric interior biasing” is a useful implementation framing
- interior candidates near connected-component boundaries should outrank empty interior sweeps

### 4. Local-Structure or Corridor Nonconversion

Games:

- `s5i5`

Shared profile:

- local continuation layers activate
- but unlike `lf52`, the preserved pocket does not convert

Interpretation:

- local anchoring is not enough by itself
- the agent still needs a way to tell productive local novelty from recycled micro-change

Unified recommendation:

- distinguish productive anchor reuse from pocket recycling
- keep `lf52` as the hard regression guard for any local-window change

Gemini-specific insight to preserve:

- breakout hysteresis is helpful here if it prevents immediate reversion into the same micro-pocket

### 5. Mechanism Belief or Search Underconversion

Games:

- `bp35`
- `g50t`
- `ka59`
- `sk48`
- `wa30`

Shared profile:

- weak or collapsed mechanism priors
- little useful deep search
- low conversion despite available action budget

Interpretation:

- the agent often settles too early on the wrong explanatory family
- deep search is present but too low-volume to rescue these games

Unified recommendation:

- improve diagnostic visibility of deep-search candidate generation versus selected actions
- add belief diversification pressure when one family dominates early with zero progress
- increase discriminating-probe pressure before a family fully collapses

Gemini-specific insight to preserve:

- `sk48` looks like a multi-object motion / tile-sliding misclassification case
- motion-preserving multi-component shifts should boost motion/translation-like beliefs over static recolor/delete beliefs

### 6. Oscillation Detection Without Conversion

Games:

- `cd82`
- `ft09`

Shared profile:

- oscillation signals fire
- breakout machinery exists
- but the agent returns to the loop or fails to exploit the breakout

Interpretation:

- breakout detection is not enough
- post-breakout branch commitment is still too weak

Unified recommendation:

- add a short breakout hysteresis window
- temporarily penalize the reciprocal loop action after suppression
- force 2 to 3 steps of real branch exploration before old-loop actions recover

Gemini-specific insight to preserve:

- the “6-step breakout anti-reversion penalty” is a good concrete starting point

## Priority Roadmap

### Priority 1: Finish-Corridor Tightening

Primary targets:

- `re86`
- `tr87`
- `tu93`

Why first:

- this is the highest-leverage unsolved dev cluster
- these games are active and close enough to conversion that better continuation discipline could pay off quickly

Success signal:

- lower `finish_corridor_active` counts on these games
- preserved baseline on all five baseline games
- ideally first new dev-game progress or solve from `re86` or `tr87`

### Priority 2: Interior-Phase Selection Quality

Primary targets:

- `ar25`
- `cn04`
- `m0r0`
- `sc25`

Why second:

- the control-band mechanism already works in `vc33`
- these games likely need better target choice, not a new architectural family

Success signal:

- lower interior-phase churn
- earlier first progress after phase switch
- no `vc33` regression

### Priority 3: Oscillation Breakout Hysteresis

Primary targets:

- `cd82`
- `ft09`
- secondary relevance to `s5i5`

Why third:

- it is narrowly scoped
- it is relatively baseline-safe if applied only after explicit oscillation detection

Success signal:

- fewer immediate returns to suppressed anchors
- more meaningful progress after breakout
- no `r11l` regression

### Priority 4: Mechanism/Search Diversification

Primary targets:

- `bp35`
- `g50t`
- `ka59`
- `sk48`
- `wa30`

Why fourth:

- this is important, but it is also easier to spend compute without improving action economy
- should be pursued after higher-conversion clusters are tightened

Success signal:

- more `deep_search_used`
- more diverse top-mechanism beliefs before collapse
- improved meaningful progress without higher action burn

## Concrete Experiment Sequence

### Experiment A

Goal:

- tighten finish-corridor only for non-progressing high-meaningful games

Run order:

1. baseline 5
2. `re86`, `tr87`, `tu93`
3. full Dev20
4. holdout 5

Promotion rule:

- keep all baseline games
- improve at least one of `re86`, `tr87`, `tu93`

### Experiment B

Goal:

- improve interior target ranking after control-band to interior transition

Run order:

1. baseline 5
2. `ar25`, `cn04`, `m0r0`, `sc25`
3. full Dev20
4. holdout 5

Promotion rule:

- keep `vc33`
- show earlier or stronger interior conversion on at least one target game

### Experiment C

Goal:

- add breakout anti-reversion hysteresis

Run order:

1. baseline 5
2. `cd82`, `ft09`, `s5i5`
3. full Dev20
4. holdout 5

Promotion rule:

- keep `r11l`
- reduce loop recurrence on at least one target game

### Experiment D

Goal:

- improve mechanism diversification and deep-search visibility

Run order:

1. baseline 5
2. `bp35`, `g50t`, `ka59`, `sk48`, `wa30`
3. full Dev20
4. holdout 5

Promotion rule:

- keep baseline 5
- increase deep-search usage or progress quality without obvious latency-only inflation

## Promotion Policy

Reject a change if:

- it loses 2 or more baseline games
- it worsens the holdout set broadly without a strong dev gain

Treat as warning if:

- it loses 1 baseline game
- it improves only the dev set but not the holdout after repeated checks

Promote a change if:

- all 5 baseline games survive
- at least 1 target dev cluster improves
- no major regression appears in the other dev clusters
- holdout does not show systematic collapse

## Practical Reading Of Gemini Feedback

What to keep from Gemini’s feedback:

- directional saturation and orthogonal turning is a useful sub-hypothesis for `tr87` and `re86`
- contour-centric interior targeting is a good concrete framing for the interior cluster
- breakout hysteresis is a good concrete framing for oscillation games
- multi-object motion / tile-sliding priors are a good concrete framing for `sk48`

What to avoid:

- treating each game as requiring a bespoke game-specific heuristic
- patching all four ideas at once without cluster isolation
- changing baseline-critical mechanisms before cluster-specific tests

## Current Best Single Next Step

If only one change should be attempted next, it should be:

- **tighten finish-corridor activation and continuation only on non-progressing, high-meaningful, high-corridor games**

Why:

- this is the highest-priority experiment from the formal diagnostic
- it overlaps with Gemini’s directional-saturation observations
- it targets the most action-expensive active nonconverting cluster in the Dev20 set

## How To Use This Document

Before each patch cycle:

1. choose one priority track only
2. run the baseline 5 first
3. run the target cluster second
4. run full Dev20 third
5. run holdout 5 last
6. update this file only when the active roadmap or promotion rules genuinely change

If only one roadmap file is maintained for this phase, maintain this one.
