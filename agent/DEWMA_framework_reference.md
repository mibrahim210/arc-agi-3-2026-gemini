# DEWMA-ARC Unified Framework Reference

## Purpose

This file is the single source of truth for the current **DEWMA** framework as implemented in [my_agent.py](/C:/Users/mdibr/Desktop/Work_Space_Professional/Kaggle%20Competition/arc-arg-3/GeminiModule/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py).

It should be updated whenever the implementation changes meaningfully so the theory, runtime behavior, and trace interpretation stay aligned.

## 1. What DEWMA Means Here

**DEWMA** stands for **Developmental Epistemic World-Model Agent**.

In this codebase, DEWMA is a generic ARC-AGI-3 interactive agent that:

- treats the raw grid as authoritative
- learns from test-time interaction rather than game-id branching
- maintains competing goals, mechanism beliefs, and executable programs
- switches between exploration and exploitation based on evidence
- uses bounded deterministic planning first and optional LLM abduction second
- is hardened for Kaggle runtime, concurrency, and trace forensics

This is still a **generic solver with strong priors**, not a memorized public-task lookup system.

## 2. Core Design Principles

### 2.1 Raw Grid Authority

The current grid is the final source of truth. Entities, components, goals, and induced programs are helper abstractions that can be revised whenever the raw transition evidence disagrees.

### 2.2 Developmental Learning

The agent starts each game without puzzle-specific rules and gradually learns:

- which actions no-op
- which actions are dangerous
- which regions are repeatedly unproductive
- which goal families correlate with progress
- which program families replay consistently
- which mechanism families fit observed transitions

### 2.3 Mechanism-First Reasoning

The agent now explicitly tracks soft beliefs over mechanism families such as:

- `movement_control`
- `targeted_recolor`
- `component_delete`
- `drag_or_push`
- `line_or_beam`
- `flood_or_fill`
- `gravity_or_fall`
- `copy_or_stamp`
- `count_or_trigger`
- `topology_switch`

These beliefs are updated from both deterministic transition evidence and LLM abduction hints, with positive and negative evidence, TTL-based decay, and per-step shift telemetry.

### 2.4 World Models Over Single Heuristics

The policy is composed from several interacting predictive layers:

- replay graph
- causal world model
- spatial action hash
- hypothesis memory
- executable program library
- goal alignment verifier
- bounded counterfactual search
- promising-state deep search
- finish-corridor continuation bias

### 2.5 Runtime-Adaptive Cognition

The agent uses runtime tiers and local safety gates so more expensive cognition is only used when it is likely to pay off and still fits the remaining budget.

## 3. End-to-End Decision Lifecycle

For each settled frame update, the effective lifecycle is:

1. perceive the latest grid sequence and build a `Scene`
2. compare pre/post states and derive an `Event`
3. update memory, goals, programs, mechanism beliefs, and diagnostics
4. generate candidate actions
5. predict candidate outcomes using replay, programs, or heuristics
6. verify safety and alignment
7. choose the best admissible action from the metacognitive ladder
8. execute and observe the next transition

## 4. Major Implementation Components

### 4.1 Perception and Representation

The perception system extracts:

- background and palette statistics
- connected components
- bounding boxes and centroids
- border contact
- temporal diffs and animation traces
- field-vs-object mode

The agent distinguishes between:

- **object-like scenes**, where controllability and entity motion matter
- **field-like scenes**, where dense-grid probing and raw-state reasoning matter more

### 4.2 Trace Memory

`TraceMemory` stores episodic transition history and now tracks:

- visits and replay opportunities
- no-op streaks
- same-family streaks
- recent state loops
- spatial visits by coordinate and action
- LLM family payoff summaries
- mechanism beliefs and shift events
- winning pattern transfer state
- post-breakthrough exploitation windows
- early local-structure persistence windows and anti-diversion telemetry
- control-band saturation and setup-budget state
- oscillation suppression pockets and breakout seeds
- meaningful-progress vs micro-churn telemetry
- finish-corridor activation and exit reasons

### 4.3 Goal Hypothesis Manager

Goals are explicit, competing, and revisable. The manager maintains:

- goal confidence and contradictions
- progress estimates
- puzzle-family priors
- cross-level transfer priors
- stabilization diagnostics

Tracked convergence-style diagnostics include:

- `transitions_until_stabilized`
- `top_goal_changes_this_level`
- `prior_adjustments_this_level`
- `puzzle_family_switch_counter`
- `stabilized_before_progress`
- `churn_score`

### 4.4 Executable Program Library

The current program layer can induce, simulate, verify, and replay program families including:

- `exact_replay`
- `local_patch_replace`
- `translation`
- `rot90`
- `flip_h`
- `flip_v`
- `color_map`
- `component_delete`
- `component_recolor`
- `conditional_recolor`
- `cellular_rule`
- `line_connect`
- `drag_component`
- `gravity`
- `flood_fill`
- `copy_pattern`
- `count_and_fill`

Programs remain provisional until they accumulate replay support and verification quality.

### 4.5 Goal Alignment Verifier

Every candidate action is filtered through legality and safety checks. The verifier can reject or heavily penalize:

- illegal actions
- incomplete coordinates
- out-of-bounds coordinates
- coordinate fatigue
- persistent deterministic no-ops
- semantic mismatches
- high-death contexts
- dead signatures

This layer is still the final guardrail over all deterministic and LLM-backed proposals.

### 4.6 Path Planner and Control Inference

The deterministic navigation layer uses:

- learned passability when enabled
- action dynamics summaries
- control inference
- bounded path search
- target cooldowns and cycle control

This bridges low-level probing and higher-level planning.

### 4.7 Counterfactual Planning

The bounded counterfactual planner searches short verified futures using executable programs and goal alignment. It remains a core deterministic planner and is used before generic fallback.

In the current implementation it is also shaped by anti-churn conversion logic:

- micro-change churn should not automatically renew counterfactual persistence
- productive breakout seeds can bias nearby continuation
- small-cycle suppression can temporarily block recently oscillating pockets
- finish-corridor mode can temporarily prefer same-neighborhood continuation over lateral branch drift

### 4.8 Optional Local Reasoner

The LLM layer acts primarily as a **bounded abductive assistant**, not as a free-running policy.

Current behavior:

- prefers local Ollama when available
- falls back to local `llama_cpp` when possible
- otherwise disables itself and preserves deterministic behavior

The reasoner can use a constrained tool interface:

- `inspect`
- `python`

It is asked to emit structured JSON with fields such as:

- `mechanism_family`
- `mechanism_confidence`
- `priority_program_families`
- `invariants_to_preserve`
- `recommended_probe_type`
- `suggested_programs`

All model outputs are still filtered by legality, semantic gating, alignment, dead-signature checks, probe budgets, and cooldown/backoff rules.

### 4.9 Puzzle-Family Classification

The code still includes a deterministic `classify_puzzle_family()` helper that initializes soft priors such as:

- `pathfinding`
- `color_matching`
- `field_diffusion`
- `geometric_transformation`

These remain soft priors rather than hard routes.

### 4.10 Promising-State Deep Search

The current implementation now includes a dedicated **promising-state deep search** layer.

It activates only when the controller sees strong signals such as:

- prior level progress
- active post-breakthrough window
- high mechanism confidence
- active follow-through
- live priority program families
- repeated productive transitions

It is runtime-tier bounded and searches more deeply only on promising states, not globally on every step.

### 4.11 Post-Breakthrough Level Exploitation

The current code now records a compact winning pattern whenever a step causes `level_delta > 0` or `win`, including:

- action name
- program kind when available
- top mechanism family
- top mechanism confidence
- coordinates when applicable
- source of the winning action
- step number within the level

On the next level transition, this can activate a **post-breakthrough exploitation window** that:

- seeds a transferred winning family/program prior
- biases ranking toward semantically similar actions
- extends exploitation more strongly after early breakthroughs
- can abort early on contradiction

This is a temporary high-confidence exploitation overlay, not a permanent lock-in.

### 4.11B Early Local-Structure Persistence

The current code also contains a generic **early local-structure persistence** layer for puzzles whose first productive evidence appears in a small interior neighborhood.

Within roughly the first 60 steps of Level 0, repeated nearby non-noop `ACTION6` interactions can:

- open a bounded `early_local_structure_window`
- preserve nearby local continuation around a learned anchor
- extend the window while nearby actions keep producing real changes
- suppress top-band diversion when a control band is not yet strongly confirmed
- close on repeated no-ops, large jumps, or level transition

This mechanism is intended to preserve productive interior exploration without breaking true control-band puzzles.

### 4.12 Control-Band and Interior-Application Phasing

The current code includes a generic phase transition for puzzles that appear to require interaction with a control strip or border band before applying effects inside the main canvas.

When repeated actions saturate a top/bottom/left/right band or a thin horizontal/vertical slice without level progress, the agent can:

- detect `control_band_saturation`
- decrement a temporary `control_band_setup_budget`
- activate an `interior_application_phase`
- strongly suppress repeated band actions during that phase
- bias toward low-visit interior component targets

This mechanism is generic and based on geometry plus progress evidence, not on game IDs.

### 4.13 Oscillation Breakout and Finish-Corridor Conversion

The current implementation contains a multi-stage generic anti-loop conversion layer.

First, it detects and suppresses:

- exact two-anchor alternation such as `A-B-A-B`
- longer two-anchor repetition such as `A-B-A-B-A-B`
- three-anchor small cycles such as `A-B-C-A-B-C`
- tight local pocket churn

Then, if a breakout from that suppressed pocket produces real progress, it can:

- store a short-lived productive breakout seed
- keep a local directional hint when one is available
- distinguish `meaningful_progress_detected` from `micro_change_churn_detected`
- activate a temporary `finish_corridor_active` mode after coherent repeated progress

Finish-corridor mode is the current bridge between “escaping a loop” and “actually converting the productive branch into a level clear.”

In the current implementation, finish-corridor is also one of the dominant generic conversion layers in traces, so it must be analyzed as both a strength and a possible source of non-converting action burn.

## 5. Current Decision Ladder

The effective metacognitive order in the current implementation is:

1. exact replay progress / replay graph plan
2. verified queued plan continuation
3. instant reflex fast-path for very strong goal alignment
4. bounded macro replay for productive recent programs
5. LLM follow-through continuation window
6. evidence-guided novelty exploration when strongly stuck
7. forced structural probe on family-collapse boredom
8. bounded counterfactual planning with anti-churn gating
9. deterministic path planning
10. replay/hash/hypothesis arbitration
11. promising-state bounded deep search
12. finish-corridor and productive-branch continuation bias
13. milestone-gated local reasoner
14. physical discriminating probes
15. alignment-constrained fallback
16. fail-closed reset / legal fallback

This ladder is intentionally biased toward cheap verified behavior first, then bounded planning, then model assistance, then safe fallback.

## 6. Runtime Tiers

The runtime tiers remain:

- `A9`: richest reasoning stack
- `A8`: reduced but still planning-heavy
- `A7`: budget-conscious reasoning
- `A5`: aggressive survival / fallback mode

The deep-search layer is also tier-bounded. It only runs where the tier and remaining time justify the cost.

## 7. Runtime Safety and LLM Budgeting

The implementation includes explicit safety controls around model use:

- decision-level model budget
- per-level consultation budget
- per-decision tool-round limits
- projected-latency skip gates
- minimum remaining-time gates
- warmup staggering
- lock backoff for parallel contention
- cooldown gates
- repeated-failure backoff
- deterministic fallback when unavailable or unsafe

This is part of the architecture, not just infrastructure hardening.

## 8. Anti-Churn and Exploitation Control

The current code contains several anti-collapse layers:

- coordinate fatigue rejection
- persistent no-op hard gating
- same-family boredom probes
- fallback loop tracking
- counterfactual fallback tracking
- semantic gating for mismatched action families
- post-breakthrough contradiction aborts
- caps on post-breakthrough `stuck_mode_exploration`
- caps on post-breakthrough alignment fallback churn
- early local-structure persistence with temporary top-band suppression
- control-band setup budgeting
- interior-application phase forcing
- two-anchor oscillation suppression
- small-cycle suppression
- productive breakout seed continuation
- meaningful-progress vs micro-churn separation
- finish-corridor continuation with explicit exit reasons

This is especially important because many ARC-AGI-3 failures are not instant mistakes but long low-value loops.

## 9. Logging and Diagnostics

### 9.1 Main Action Traces

The main traces are written to `DEWMA_TRACE_DIR` (default `./traces`) using names like:

- `trace_<game_id>_<timestamp>_<flush_index>.jsonl`

These traces now include:

- transition and action records
- decision stage and final action source
- runtime tier
- goal and mechanism telemetry
- LLM consultation and rejection counters
- macro replay telemetry
- deep-search telemetry
- post-breakthrough exploitation telemetry
- per-family payoff summaries

Important fields in the current implementation include:

- `mode`
- `top_mechanism_family`
- `top_mechanism_confidence`
- `competing_mechanism_families`
- `mechanism_family_scores`
- `priority_program_families`
- `invariants_to_preserve`
- `mechanism_shift_event`
- `mechanism_aligned_action`
- `mechanism_discriminating_probe_used`
- `promising_state_detected`
- `promising_state_reasons`
- `deep_search_used`
- `deep_search_depth`
- `deep_search_width`
- `deep_search_nodes_evaluated`
- `deep_search_best_score`
- `deep_search_time_ms`
- `deep_search_selected_family`
- `deep_search_aborted_reason`
- `post_breakthrough_window_active`
- `post_levelup_exploit_steps_remaining`
- `transferred_winning_family`
- `transferred_winning_program_kind`
- `transferred_winning_action_name`
- `transferred_winning_coords`
- `post_breakthrough_aborted_reason`
- `post_breakthrough_bias_used`
- `control_band_saturation_detected`
- `control_band_orientation`
- `control_band_setup_budget_remaining`
- `interior_application_phase_active`
- `interior_exploitation_window_active`
- `control_band_bias_applied`
- `interior_transition_bias_applied`
- `two_anchor_oscillation_detected`
- `two_anchor_oscillation_breakout_applied`
- `two_anchor_suppressed_anchors`
- `small_cycle_oscillation_detected`
- `small_cycle_suppression_applied`
- `productive_breakout_branch_seed_active`
- `productive_breakout_branch_seed_coord`
- `productive_breakout_direction`
- `directional_breakout_bias_applied`
- `meaningful_progress_detected`
- `micro_change_churn_detected`
- `finish_corridor_active`
- `finish_corridor_steps_remaining`
- `finish_corridor_anchor`
- `finish_corridor_family`
- `finish_corridor_bias_applied`
- `finish_corridor_exit_reason`

### 9.2 LLM Forensics

LLM forensic logs are written under the same trace directory using names like:

- `llm_forensics_<game_id>.jsonl`

They capture:

- prompt/response pairs
- response type
- structured proposal contents
- parse outcomes
- rejection reasons
- model-side failures

These logs are intended to remain append-only during Kaggle runs.

## 10. What Is Truly General vs What Is Biased

### General

- no game-id branching
- no public-task answer lookup
- learning from transitions
- explicit goal competition
- explicit mechanism competition
- verification before execution
- bounded planning instead of memorized scripts

### Biased but Still Framework-Consistent

- entity/control priors
- geometric and spatial priors
- finite program vocabulary
- action-economy heuristics
- mechanism-family scoring rules
- transferred winning-pattern exploitation

So the best description of the current code is:

**a generic interactive reasoning framework with strong structured priors, bounded post-breakthrough exploitation, and explicit anti-churn conversion layers.**

## 11. Current Strengths

- strong anti-memorization posture
- explicit developmental learning loop
- multiple interacting world-model layers
- mechanism-first reasoning is now real, not just prompt language
- deep search is selective rather than globally expensive
- Kaggle-safe runtime hardening is strong
- traces are rich enough for real forensic debugging
- post-breakthrough exploitation now exists as a first-class policy layer
- control-band to interior phasing is now an explicit generic mechanism
- oscillation breakout can now be converted into short-lived finish-corridor commitment
- early local-structure persistence can restore baseline interior solves without game-specific branching

### Current Calibration Snapshot

The current repository calibration point is the local 25-game run archived at:

- `traces_archive/2026-08-31_13-37-17`

Observed result:

- aggregate scorecard score: `0.17819664591688916`
- progressed games: `tn36`, `vc33`, `r11l`, `lf52`, `lp85`

What this run currently validates:

- `vc33` still converts through control-band setup into interior application
- `r11l` still converts through oscillation breakout, bounded continuation, and finish behavior
- `lf52` baseline recovery depends on the early local-structure persistence layer
- post-breakthrough and deep-search logic are present but remain lower-volume than the broader bounded conversion stack
- many unsolved games still spend much more time in conversion-heavy bounded continuation than in explicit deep-search or post-breakthrough follow-through

## 12. Current Weaknesses

- the program vocabulary is still limited relative to full ARC novelty
- mechanism beliefs can still overcommit on hard hidden tasks
- the agent can still spend many actions in fallback churn
- LLM proposals remain filtered heavily and may contribute unevenly
- post-breakthrough bias attribution is still easier to trace than to optimize
- meaningful-progress calibration is still delicate and can over- or under-hold a local branch
- finish-corridor continuation is still short-horizon and may miss longer hidden-task solution chains
- several unsolved games still show very high `finish_corridor_active`, `interior_application_phase_active`, or `micro_change_churn_detected` without converting to level progress
- trace archives often shard a single episode across many JSONL files, so diagnostics must aggregate by game and step rather than treating each file as an independent run
- hidden-game generalization remains the open problem, not notebook plumbing

## 13. How To Update This File

Update this file whenever any of the following change in `agent/my_agent.py`:

- supported program families
- decision ladder ordering
- mechanism-belief logic
- local reasoner backend or schema
- runtime tier behavior
- deep-search policy
- post-breakthrough transfer/exploitation behavior
- control-band / oscillation / finish-corridor conversion behavior
- trace schema
- forensic log behavior

If only one DEWMA document is maintained, maintain this one.
