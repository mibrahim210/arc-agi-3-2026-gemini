# DEWMA-ARC Unified Framework Reference

## Purpose

This file is the single source of truth for the current **DEWMA** framework as implemented in [`agent/my_agent.py`](C:/Users/mdibr/Desktop/Work_Space_Professional/Kaggle%20Competition/arc-arg-3/GeminiModule/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py).

It replaces the need to keep multiple theory notes in sync. When `my_agent.py` changes meaningfully, update this file so the conceptual framework, runtime behavior, and implementation details stay aligned.

---

## 1. What DEWMA Means Here

**DEWMA** stands for **Developmental Epistemic World-Model Agent**.

The current agent is a game-agnostic interactive reasoning system for ARC-AGI-3 that:

- learns during test-time interaction rather than relying on per-game hardcoding
- treats the raw grid as ground truth
- builds and revises internal causal models from action outcomes
- uses explicit hypothesis competition rather than single-rule commitment
- shifts between exploration and exploitation based on confidence and runtime budget
- degrades gracefully under strict Kaggle runtime constraints

This is **not** a memorized public-game solver. It is a structured reasoning scaffold with strong hand-designed inductive priors over geometry, causality, controllability, symmetry, replay, and local transformations.

---

## 2. Core Design Principles

### 2.1 Raw Grid Authority

The 2D integer grid is always authoritative. Derived abstractions such as components, entities, goals, and induced programs are confidence-weighted helpers. They are never allowed to silently override raw-grid evidence.

### 2.2 Developmental Learning

The agent begins without game-specific rules. It incrementally learns from transitions:

- which actions are no-ops
- which actions are dangerous
- which local patterns predict effects
- which goals correlate with level progress
- which induced programs replay correctly

### 2.3 Epistemic Action Selection

When the environment is uncertain, the agent prefers actions that increase information:

- untried signatures
- frontier exploration
- discriminating probes
- safe probing around recent changes

When confidence grows, the agent shifts toward:

- replayed progress
- verified program execution
- bounded counterfactual planning
- deterministic navigation

### 2.4 Internal World Models

The agent uses multiple internal predictive layers rather than one monolithic model:

- exact replay graph
- spatial action hash
- hypothesis memory
- executable induced programs
- goal alignment verification
- bounded counterfactual planning

### 2.5 Runtime-Adaptive Cognition

The agent is designed for long offline Kaggle runs under wall-clock pressure. It dynamically changes reasoning depth by runtime tier and by explicit local reasoner safety gates.

---

## 3. End-to-End Decision Lifecycle

For each settled environment update, the effective pipeline is:

1. Perceive the grid and temporal sequence.
2. Build a `Scene` representation.
3. Compare pre/post action states and extract an `Event`.
4. Update memory, hypotheses, goals, programs, alignment, and diagnostics.
5. Generate candidate actions.
6. Predict outcomes using replay, induced programs, or heuristics.
7. Verify alignment and safety.
8. Choose the best safe action under the metacognitive ladder.
9. Execute and observe the next transition.

---

## 4. Major Implementation Components

### 4.1 Perception and Representation

The perception stack extracts:

- background color
- palette statistics
- connected components
- bounding boxes
- centroids
- border contact
- temporal diffs
- field-vs-object mode

The system distinguishes between:

- **object-like scenes**, where connected entities and controllability matter
- **field-like scenes**, where raw spatial probing is safer than brittle object assumptions

### 4.2 Temporal Entity Tracking

Tracked entities are used to infer:

- persistence
- motion
- controllability
- likely affordances
- possible player-like groups

This is a strong inductive bias, but it is not game-specific branching.

### 4.3 Trace Memory

`TraceMemory` stores episodic transitions and supports:

- replay of known effects
- probe counting
- loop detection
- no-op streaks
- recent-state recurrence checks
- same-family streak detection
- action-family entropy estimation
- longest action-family streak measurement
- spatial visit counting for coordinate-based probing

### 4.4 Causal / Hypothesis Memory

`HypothesisMemory` stores contextual action signatures and observed outcomes such as:

- progress
- no-op
- death
- effect hash

These signatures are used to estimate:

- hypothesis confidence
- expected information gain
- known no-op risk
- goal bonus

### 4.5 Fast Spatial Action Hash

This is a local transformation cache over small patches. It captures recurring mechanics like:

- local flips
- local recolors
- repeated click effects

It supports fast approximate prediction without full global reasoning.

### 4.6 Goal Hypothesis Manager

Goals are explicit, competing, and revisable.

Examples of goal families include:

- collect a color
- touch or reach a target color
- move a controllable entity toward a target
- reduce obstacles
- preserve valuable structures
- trigger topology change

Goals are strengthened or weakened from actual transition evidence and level progress.

The goal layer also tracks convergence-oriented diagnostics such as:

- top-goal identity changes within a level
- transitions until the top goal stabilizes
- whether stabilization occurred before progress
- puzzle-family switch count
- prior-adjustment count from abductive hints

### 4.7 Executable Program Library

The current world-model program layer induces, verifies, and reuses explicit transformation programs.

Supported program families currently include:

- `exact_replay`
- `translation`
- `color_map`
- `component_delete`
- `component_recolor`
- `rot90`
- `flip_h`
- `flip_v`
- `line_connect`
- `drag_component`
- `gravity`
- `flood_fill`
- `copy_pattern`
- `conditional_recolor`
- `count_and_fill`

Programs are only trusted for generalized planning once they accumulate sufficient replay support and verification quality.

### 4.8 Goal Alignment Verifier

Every potentially executed action is filtered through an alignment and safety gate.

It rejects or penalizes actions associated with:

- illegal or incomplete actions
- out-of-bounds coordinates
- destructive change without justification
- predicted fatal loss
- repeated no-op-like behavior
- violation of learned progress invariants

### 4.9 Path Planner and Passability Model

The deterministic navigation layer uses:

- learned passability
- candidate target entities
- bounded search over passable regions
- path cooldowns and cycle detection

This is a fallback and bridge layer between raw probing and abstract planning.

### 4.10 Counterfactual Planner

The bounded counterfactual planner uses replay-verified induced programs to simulate a short beam of future possibilities and select an action that improves predicted goal alignment.

It is explicitly bounded in:

- depth
- beam size
- candidate count

and it is throttled when looping behavior appears.

### 4.11 Optional Local Reasoner

The local text reasoner acts as an **abductive hypothesis generator**, producing structural program priors, action candidates, and puzzle-family classifications from observed transitions rather than serving as an unconstrained direct policy.

Current backend strategy:

- prefer **Ollama** when available
- fall back to **llama_cpp** with local GGUF weights
- otherwise disable the model and continue with deterministic logic

Current default offline model path points to a **Qwen 2.5 Coder 7B GGUF** dataset path, while `llama_cpp` remains a fallback backend.

The reasoner has access to a constrained tool interface:

- `inspect`
- `python`

through a safe REPL-like interface over grid summaries and recent transitions.

Its proposals are still subject to:

- parsing checks
- legality checks
- coordinate checks
- program ingestion and verification
- dead-signature filtering
- probe-budget filtering
- alignment verification

### 4.12 Puzzle-Family Classification and Goal Priors

The agent includes a deterministic soft `classify_puzzle_family()` pass over grid density, color diversity, and connected-component structure to initialize high-level priors such as:

- `pathfinding`
- `color_matching`
- `field_diffusion`
- `geometric_transformation`

These priors are then updated by LLM abductive proposals through `GoalHypothesisManager.adjust_priors()`, which boosts compatible goal families without hard-routing into game-specific logic.

### 4.13 Boredom and Anti-Collapse

To prevent action-family collapse, `MetacognitiveController` monitors `same_family_streak`.

When the streak exceeds a threshold, it triggers a **Forced Structural Probe** that:

- filters out the current dominant action family
- scores candidates by underexplored action usage
- prefers less-visited coordinate regions through `TraceMemory.spatial_visits`
- gives extra exploratory preference to novel `ACTION6` interactions when legal

This is intended to broaden exploration without breaking the alignment gate.

### 4.14 Convergence and Trace Diagnostics

Telemetry logs capture rich, Kaggle-safe metrics including:

- action-family metrics:
  - `action_family_entropy`
  - `longest_family_streak`
  - `same_family_streak`
- convergence metrics:
  - `transitions_until_stabilized`
  - `top_goal_changes_this_level`
  - `prior_adjustments_this_level`
  - `stabilized_before_progress`
  - `puzzle_family_switch_counter`
  - `churn_score`
- LLM impact metrics:
  - `llm_abductions_parsed`
  - `llm_abductions_selected`
  - `llm_action_progress`
  - `llm_program_progress`
  - `llm_abduction_prior_shift_progress`
- counterfactual metrics:
  - `counterfactual_streak`

---

## 5. Runtime Tiers

The implementation uses runtime tiers:

- `A9`: full reasoning stack
- `A8`: reduced but still hypothesis-rich reasoning
- `A7`: simpler, more budget-conscious reasoning
- `A5`: aggressive fallback / survival mode

Tier selection depends on remaining time and projected budget pressure.

This runtime control is part of DEWMA, not just an engineering afterthought.

---

## 6. Current Decision Ladder

The metacognitive controller currently follows this high-level priority order:

1. exact replay progress or replay graph plan
2. verified multi-step plan queue
3. forced structural probe during action-family collapse
4. bounded counterfactual planning with verified programs
5. deterministic path planning
6. replay/hash/hypothesis arbitration
7. milestone-gated local reasoner
8. physical information-seeking probes
9. alignment-constrained fallback

This order is important: cheap, safe, and verified mechanisms outrank expensive speculative ones.

---

## 7. Runtime Safety and LLM Budgeting

The current DEWMA implementation includes explicit local-reasoner safety controls:

- decision-level model budget
- per-level model budget
- per-decision max rounds
- per-decision max wall-clock time
- minimum remaining runtime threshold before model use
- short Ollama lock timeout to avoid parallel thread stalls
- deterministic fallback when the model is unavailable or unsafe to call

This is an implementation of **runtime-aware cognition**, which is a core DEWMA behavior under Kaggle constraints.

---

## 8. Logging and Diagnostics

Two major diagnostic channels exist:

### 8.1 Main Action Traces

Written under:

- local: `traces/trace_*.jsonl`

These capture:

- state/action transitions
- no-op and death signals
- progress markers
- decision stages
- representation confidence
- runtime tier
- convergence counters
- LLM contribution counters

### 8.2 LLM Forensics

Written under:

- local: `traces/llm_forensics/<run_id>.jsonl`
- Kaggle rerun: `/kaggle/working/llm_forensics/<run_id>.jsonl`

These capture:

- raw model responses
- parsed response type
- tool usage
- proposal program kinds and params
- verification scores
- rejection reasons
- skip reasons like low remaining time or lock unavailability

Kaggle forensic logs are intentionally append-only and must not be deleted during a run.

---

## 9. What Is Truly General vs What Is Biased

### General

- no game-ID branching
- no public-game solution lookup
- causal learning from transitions
- explicit hypothesis competition
- replay-first reasoning
- bounded planning with verification
- runtime-adaptive control

### Biased but Still Framework-Consistent

- entity-centric controllability assumptions
- geometric coordinate priors
- a bounded transformation vocabulary
- path planning assumptions
- alignment and preservation heuristics

So the current system is best described as:

**a game-agnostic reasoning framework with strong abstract priors**, not a universal unbiased solver.

---

## 10. Current Strengths

- Strong anti-memorization posture
- Explicit developmental learning loop
- Multiple interacting world-model layers
- Verified-program planning rather than blind speculation
- Better anti-collapse probing through spatial visit tracking
- Good runtime-awareness for Kaggle conditions
- Separate forensic logging for policy and model behavior

---

## 11. Current Weaknesses

- The program vocabulary is still limited relative to the full novelty space of ARC-AGI-3
- The agent can still overcommit to the wrong abstraction family
- Action-family collapse is mitigated, not eliminated
- LLM proposals are often syntactically valid but semantically unverifiable
- Tool-use by the local reasoner may still be underexploited

---

## 12. How To Update This File

When `agent/my_agent.py` changes meaningfully, update this file if any of the following shift:

- supported induced program kinds
- local reasoner backend
- decision ladder ordering
- runtime tier logic
- model budget / timeout / skip rules
- trace or forensic log paths
- major goal or alignment semantics
- convergence metrics
- LLM contribution metrics
- overall DEWMA philosophical framing

If only one documentation file is maintained going forward, maintain this one.
