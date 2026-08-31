# DEWMA Architecture-Conformance and Trace-Clustering Diagnostic Prompt v4

## Purpose

Use this prompt when reviewing the current DEWMA implementation in `agent/my_agent.py`, the architecture reference in `DEWMA_framework_reference.md`, and every trace in `traces_archive/`.

This version is aligned with the updated architecture and implementation as of August 31, 2026. It combines a reproducible **25-game forensic clustering workflow** with an **architecture-to-code-to-trace conformance audit**. The goal is to determine which games fail for the same operational reason, whether the intended controller actually wins arbitration in those games, and which change has the best expected payoff across the evaluation set.

## Mandatory Inputs and Scope

Analyze the repository directly. Do not ask me to paste trace rows that already exist on disk.

- Main implementation: `agent/my_agent.py`
- Architecture source of truth: `DEWMA_framework_reference.md`
- Trace-run input: a specific timestamped subfolder such as `traces_archive/2026-08-30_14-35-20`
- Expected evaluation set: 25 distinct game IDs
- Possible auxiliary evidence in the same folder:
  - `trace_<game_id>_*.jsonl`
  - `llm_forensics_<game_id>.jsonl`
  - run summaries or scorecards

Recursively discover the files. Derive the game ID primarily from each JSON record's `game_id`; use the filename only as a fallback. Report missing, malformed, empty, or unassigned files. If fewer or more than 25 distinct games are found, continue the analysis but flag the discrepancy prominently.

Analyze only the trace-run folder explicitly provided by the user. Do not silently combine sibling folders under `traces_archive/`. Accept both Windows-style input such as `traces_archive\2026-08-30_14-35-20` and POSIX-style input such as `traces_archive/2026-08-30_14-35-20`; normalize the path before processing.

Derive `run_id` from the final normalized trace-folder name. Sanitize it to filesystem-safe characters `[A-Za-z0-9._-]`, replacing other characters with `_`. For the example above:

- `run_id = 2026-08-30_14-35-20`
- run output directory: `diagnostics/trace_clustering/2026-08-30_14-35-20/`
- primary report: `diagnostic_2026-08-30_14-35-20.md`

Never overwrite or merge another run's diagnostic artifacts. If the same `run_id` already exists, compare its stored input fingerprint with the current trace folder. Reuse/update it only when it represents the same input run; otherwise append a deterministic short fingerprint to the output directory and report filename.

## Source-of-Truth and Version Rules

The architecture document describes intent; `my_agent.py` defines executable behavior; traces show observed behavior. Treat disagreements as findings, not as permission to silently choose one source.

1. Read the architecture and code before interpreting trace fields.
2. Compute a code fingerprint and trace-schema fingerprint so the report states exactly what was analyzed.
3. Determine whether the archive was produced by this attached code version. Use trace-field coverage and configuration/runtime evidence; if exact provenance is unavailable, label compatibility as `confirmed`, `likely`, `mixed`, or `unknown`.
4. If a documented field is absent from older traces, mark it `not_in_trace_version`; do not infer `false`.
5. If a field exists in code but is never emitted, or is emitted but never changes, identify whether that is expected inactivity, an instrumentation defect, or unreachable logic.
6. If architecture ordering differs from actual arbitration order in code, report both and use code behavior for causal diagnosis.
7. Separate historical/pre-update traces from post-update traces when both exist. Do not cluster incompatible generations together without a version indicator or sensitivity analysis.

---

## Updated Architecture Under Test

The diagnostic must explicitly test these current layers:

- raw-grid authority with object/field representation
- trace memory, competing goals, and executable program induction
- goal-alignment verification and deterministic path/control inference
- bounded counterfactual planning
- optional local LLM abduction with Ollama-first and `llama_cpp` fallback
- promising-state deep search
- post-breakthrough winning-pattern transfer and regrounding
- structured productive-branch persistence and finish-bound branch control
- control-band saturation detection and interior-application phasing
- two-anchor and small-cycle oscillation suppression
- productive breakout seed and directional continuation
- meaningful-progress versus micro-change calibration
- finish-corridor conversion
- runtime-tier and finalization safety

Do not reduce the updated architecture to the earlier generic “fallback churn” hypothesis. Determine which conversion layer fails first and whether later layers ever receive a valid opportunity to help.

## Non-Negotiable Data-Hygiene Rules

Before computing metrics:

1. Validate every JSONL row independently and count parse failures by file.
2. Distinguish trace shards from separate episodes. Do not blindly concatenate files and inflate counts.
3. Detect overlapping or cumulative shards using `(game_id, level, step, timestamp, before_key, after_key, action)` and filename time ranges.
4. Deduplicate only exact or demonstrably repeated events. Preserve genuine retries, resets, and repeated actions.
5. Keep both raw-row counts and deduplicated physical-decision counts.
6. Never sum cumulative counters such as `model_decisions_used`, `stuck_mode_activations`, `reasoner_decision_attempts`, or rejection counters across rows. Use terminal values per episode, or differences when a counter resets.
7. Treat missing fields as unknown, not `false` or zero. State which metrics are unavailable for which games.
8. Separate physical actions from repeated logging/animation frames. Explain the event identity rule used.
9. Segment episodes and levels using explicit reset/level transitions where available; otherwise infer boundaries conservatively from step, level, timestamp, and state-key discontinuities.
10. Keep LLM forensic records linked to, but separate from, physical trace events unless a reliable step/timestamp join exists.

The clustering is invalid if it is based on duplicated cumulative shards or cumulative counters incorrectly summed across rows.

---

## Required Workflow: Inventory → Features → Clusters → Fixes

### Phase 1 — Corpus Inventory

Produce an inventory with one row per discovered game:

| Game | Trace files | LLM files | Episodes | Levels reached | Unique actions | Last step | Terminal state | Data-quality warning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |

Also report files that could not be assigned to a game.

### Phase 1B — Architecture-to-Code-to-Trace Conformance

Build this matrix before clustering:

| Intended layer | Architecture claim | Code symbol/line | Trace field(s) | Reachable? | Activated in archive? | Won control? | Produced progress? | Finding |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

At minimum include every stage in the documented decision ladder and the new conversion layers. Verify the actual precedence in the attached code, especially:

- post-breakthrough priority versus structured persistence and finish-bound continuation
- the baseline-safe structured-persistence gate
- persistence decay after missing evidence/convergence
- stuck-mode guards during continuation/persistence
- control-band penalties versus interior boosts
- oscillation suppression in candidate generation and counterfactual selection
- finish-corridor activation, ranking bias, and exit logic
- promising-state deep search placement relative to path planning, physical probes, and fallback

For every layer distinguish:

- `available_in_code`
- `eligible`
- `activated`
- `candidate_generated`
- `candidate_admissible`
- `ranking_bias_applied`
- `selected_or_kept_control`
- `effective_transition`
- `meaningful_progress`
- `level_progress`

This conversion funnel is required. A boolean feature saying a mechanism was “used” is insufficient when the code separates detection, bias, selection, and outcome.

### Phase 2 — Per-Game Forensic Feature Matrix

Build a machine-readable and human-readable feature table. At minimum calculate, where available:

- outcome: levels completed/reached, progress-event count, first-progress step, last-progress step, terminal state
- budget: unique physical actions, maximum level step, near-cap flag, deterministic and model latency, remaining wall time
- effects: no-op rate, death rate, changed-state/effective-action rate, repeated-effect-signature concentration
- decision ladder: top decision stages, stage shares, fallback share, counterfactual share, stuck-mode share, macro-replay share
- mechanism belief: dominant `top_mechanism_family`, its share, mean confidence, family entropy, family switches, collapse duration
- search: promising-state rate, deep-search attempt/use rate, search-exhausted rate, nodes evaluated, useful-search rate
- LLM: consultations/calls, success and parse rates, surviving proposals, final-action-from-LLM count, suppression reasons, rejection reasons, repeated-family behavior
- conversion: progress after deep search, progress after LLM action, progress after macro replay, progress after counterfactual search
- continuation: post-breakthrough window steps, bias/control usage, regrounding, relocalization, transfer, later progress
- branch control: productive-branch commitment, structured persistence, persistence-gate pass rate, evidence-decay exits, finish-bound control, collision recovery, preemption/block reasons, and post-breakthrough-versus-persistence precedence
- phase conversion: control-band saturation, inferred orientation, setup-budget exhaustion, band suppression, interior-phase activation, interior target/bias usage, and progress after switching to the interior
- oscillation conversion: two-anchor and small-cycle detection, suppression, breakout application, productive breakout seed activation, directional bias, recurrence after suppression, and later progress
- progress quality: meaningful-progress and micro-change rates, false-positive continuation windows, branch convergence, and conversion to level progress
- finish corridor: activation, family/anchor stability, bias application, duration, exit-reason distribution, level-clear conversion, no-op exits, fatal exits, and window-expiry exits
- exploration diversity: action entropy, coordinate/anchor diversity where present, family diversity, longest identical-stage/action/family streak

Use rates with explicit denominators. Include both counts and rates when a small count could be misleading.

### Phase 3 — Evidence-Based Clustering

Cluster the 25 games by **observed failure mode**, not by game name and not merely by final score. Use a hybrid method:

1. Normalize numeric features robustly.
2. Encode categorical stage/family distributions.
3. Compare at least two reasonable cluster counts or methods.
4. Select the smallest stable, interpretable partition; prefer 4–7 primary clusters unless evidence strongly supports another number.
5. Explain why each game belongs to its cluster using at least three discriminating trace features.
6. Give every assignment a confidence: high, medium, or low.
7. Allow a primary cluster plus a secondary tag for mixed failures.
8. Put sparse/ambiguous traces into an explicit `insufficient_evidence` group rather than forcing a diagnosis.

The cluster labels should describe causal operational failures. Candidate labels include, but are not limited to:

- `alignment_fallback_lock_in`
- `mechanism_belief_collapse`
- `promising_but_search_exhausted`
- `reasoner_gated_or_nonconverting`
- `unsafe_probe_or_death_cycle`
- `no_op_coordinate_churn`
- `counterfactual_nonconversion`
- `post_breakthrough_transfer_failure`
- `structured_persistence_gate_failure`
- `control_band_setup_lock_in`
- `interior_phase_nonconversion`
- `oscillation_detection_without_breakout`
- `breakout_without_meaningful_progress`
- `micro_change_false_commitment`
- `finish_corridor_nonconversion`
- `representation_or_control_detection_failure`
- `productive_partial_solver`
- `insufficient_evidence`

Do not use these labels mechanically. Rename, merge, or split them based on the actual distribution.

### Phase 4 — Counterfactual Debugging per Cluster

For every primary cluster, answer:

- What is the earliest reliable warning signal?
- Which decision stage takes control afterward?
- What useful subsystem is losing arbitration or failing internally?
- Is the bottleneck perception, mechanism inference, goal inference, program induction, action ranking, safety rejection, continuation, or budget allocation?
- What trace evidence would falsify this diagnosis?
- What is the smallest code/configuration experiment that isolates the cause?
- What improvement should appear in traces if the experiment works?
- What safety or regression risk could the change introduce?

Do not prescribe a global threshold change when only one cluster needs it. Prefer cluster-conditional policies based on signals available online before the game is solved.

### Phase 5 — Cross-Cluster Prioritization

Rank fixes using:

`priority = affected_games × confidence × expected_conversion_gain ÷ implementation_and_regression_risk`

The exact arithmetic can be qualitative, but all four terms must be discussed. Identify:

- one broad fix with the highest portfolio payoff
- one fix for the two progressing games' continuation problem
- one instrumentation fix required before further tuning
- any game cluster that should not yet drive architectural changes

---

## Historical Evidence from the Earlier Supplied Samples

Use this only as a hypothesis seed. These samples may predate the updated control-band, oscillation, persistence-decay, and finish-corridor logic. First establish trace/code compatibility, then recompute from the full archive:

- `tu93` and `vc33` are both zero-progress samples with heavy `alignment_constrained_fallback` control and strong `targeted_recolor` belief concentration.
- Both frequently mark states as promising while `deep_search_used` remains false and searches report exhaustion. This suggests a semantic mismatch between “promising,” “search attempted,” and “usable action found.”
- `tu93` shows substantial death exposure; distinguish hazardous exploration from ordinary fallback churn.
- `vc33` shows a secondary `drag_or_push` belief regime; test whether this is meaningful diversification or merely another nonconverting collapse.
- The LLM forensic samples propose translation-like hypotheses even when trace-side mechanism beliefs concentrate elsewhere. Audit whether this is useful disagreement, poor grounding, or proposals being filtered before arbitration.

- If the new trace fields are absent, do not use these samples to judge the new mechanisms. Classify them as a historical baseline and use them for before/after comparison only.

Do not generalize these two games to all 25 without the full feature matrix.

## Core Position

The current DEWMA agent does **not** appear primarily compute-bound, and it does **not** appear primarily trapped by immediate early exits.

The stronger working hypothesis is:

> DEWMA is currently **churn-bound and weak-conversion-bound**. It often spends a large fraction of its physical action budget in fallback-heavy or low-yield exploration stages, and when it does achieve early progress, it still struggles to convert that progress into additional level completions.

This means the most important diagnostic question is not:

- "Why doesn't the agent run longer?"

It is:

- "Why are DEWMA's stronger continuation mechanisms not converting promising states into more completions before the action budget is exhausted?"

---

## 1. Performance Context

I am running the DEWMA-ARC agent on Kaggle ARC-AGI-3.

Known context:

- local 25-game evaluation is substantially slower than Kaggle evaluation
- Kaggle evaluation finishes far below the 9-hour wall-clock limit
- traces show many games still consume large physical action budgets
- recent runs often end near the per-game action cap rather than via immediate hard early termination

Important interpretation:

- spare wall-clock budget may exist
- but the primary bottleneck may still be **action-budget waste**, not raw compute scarcity

So the key question is whether DEWMA is:

- under-activating high-value reasoning
- over-spending steps in churn-heavy fallback stages
- failing to exploit early breakthroughs

---

## 2. Refined Hypothesis

My working hypothesis is:

> DEWMA is not mainly failing because it lacks more algorithms. It already contains counterfactual planning, promising-state deep search, mechanism beliefs, and bounded LLM abduction. The main issue is that these stronger layers are either activating too narrowly, activating too late, or failing to dominate the action-selection ladder strongly enough after promising signals appear. As a result, the agent still burns too many actions in `stuck_mode_exploration`, `counterfactual_fallback`, and `alignment_constrained_fallback`, especially after first progress or level transitions.

Do you agree? If yes, identify the exact activation or arbitration bottlenecks. If no, identify the real bottleneck with code-grounded evidence.

---

## 3. Architecture-Aware Analysis Instructions

Analyze the code through the following DEWMA-specific lenses.

### A. Decision Ladder Dominance Analysis

Trace the real decision ladder in `my_agent.py`.

Questions:

- Which stages dominate actual action selection in traces?
- Which high-value stages are present in code but rarely win arbitration?
- Does `fail_closed_reset` or legal fallback actually dominate episodes, or is the real mass in fallback-heavy live action stages such as:
  - `stuck_mode_exploration`
  - `counterfactual_fallback`
  - `alignment_constrained_fallback`
- For each major stage, answer:
  - how often it is reached
  - how often it wins
  - whether it leads to progress or just churn

### B. Action-Budget Waste vs. Wall-Clock Waste

The Kaggle run finishes well under the 9-hour limit, but traces often show many games consuming hundreds of physical steps.

Questions:

- Is DEWMA under-using wall-clock compute, over-using physical actions, or both?
- Are games ending because of:
  - `WIN`
  - `GAME_OVER`
  - near-max-actions exhaustion
  - finalization reserve
  - hard resets
- What percentage of games reach roughly 200, 300, or 400 steps?
- Does the code currently spend too little compute on promising states relative to how many actions it burns there?

### C. Promising-State Deep Search Activation Audit

The agent already contains `promising_state_deep_search`.

Questions:

- How often does `promising_state_detected = true` occur?
- How often does `deep_search_used = true` occur?
- How often does deep search produce progress?
- Is deep search activating at the right moments:
  - after first progress
  - during post-breakthrough windows
  - after level transitions
- Or is it still activating too rarely or too weakly to matter?

### D. Post-Breakthrough Conversion Audit

The agent now contains:

- winning pattern capture
- post-breakthrough transfer
- regrounding
- optional level-up relocalization
- post-breakthrough anti-churn protections

Questions:

- When a level is solved early, does the post-breakthrough window actually improve continuation?
- Does `post_breakthrough_window_active` lead to:
  - additional progress
  - stronger verified program usage
  - better deep-search behavior
  - or just delayed churn
- Does `post_breakthrough_bias_used` actually occur on chosen actions?
- Are `transferred_winning_family`, `transferred_winning_program_kind`, and `transferred_winning_coords` influencing the winning action path often enough?

### E. Regrounding and Level-Transition Audit

The current code includes:

- component re-grounding
- bounded LLM relocalization fallback

Questions:

- How often does regrounding succeed?
- How often is `regrounded_winning_coords` actually used downstream?
- Does level-up relocalization fire only when needed, or too often / too rarely?
- Are level transitions still causing the agent to fall back into blind coordinate probing?

### F. Mechanism Belief Collapse Audit

DEWMA now maintains explicit mechanism-family beliefs.

Questions:

- Does the agent still collapse too heavily onto a small subset such as:
  - `targeted_recolor`
  - `line_or_beam`
- Are the new direct detectors for:
  - `drag_or_push`
  - `gravity_or_fall`
  - `component_delete`
  - `flood_or_fill`
  materially changing top-family selection?
- When a mechanism family is wrong, how quickly does the system switch?

### G. Anti-Churn Mechanism Audit

The code includes:

- coordinate fatigue
- persistent no-op gating
- same-family boredom probes
- post-breakthrough churn caps
- exploitation-mode no-op blacklisting

Questions:

- Are these mechanisms reducing churn in practice?
- Or are they just redistributing churn from one fallback stage to another?
- Do blacklists meaningfully reduce repeated coordinate no-ops after first progress?
- Is the agent still spending too much time in `stuck_mode_exploration` even after anti-churn updates?

### H. LLM Budgeting and Role Audit

The LLM layer is now only one part of the reasoning stack.

Questions:

- Is the model mainly helping:
  - generic stagnation
  - post-level-up relocalization
  - continuation after promising states
- Are skip gates and backoff rules too conservative for continuation mode?
- Is the deterministic stack too dominant even when the LLM or deep-search layer has better continuation priors?

---

## 4. Specific Diagnostic Questions

Please answer these concretely from the code and traces.

1. Which decision stages dominate total action count across the run?
2. Which stages dominate specifically **after first progress**?
3. How often does `promising_state_deep_search` activate, and how often does it help?
4. How often does `post_breakthrough_window_active` occur, and does it convert into later-level progress?
5. How often does `post_breakthrough_bias_used` occur on chosen actions?
6. How often do regrounding and level-up relocalization succeed?
7. Are games primarily ending from:
   - near action-cap exhaustion
   - game over
   - explicit done logic
   - finalization reserve
8. Is DEWMA under-using wall-clock compute on promising states even while over-using physical actions globally?
9. Which mechanism families dominate beliefs, and do the new detectors diversify them?
10. What is the single biggest reason the current code fails to convert early progress into more completions?

---

## 5. Prescriptive Requests

If you agree that DEWMA is currently churn-bound / weak-conversion-bound, provide:

1. The exact code paths where stronger continuation logic is losing to churn-heavy fallback.
2. The exact conditions that should be relaxed or reprioritized so:
   - deep search fires earlier on promising continuation states
   - post-breakthrough bias wins ranking more often
   - post-level-up re-grounding has more downstream impact
   - fallback churn shrinks after first progress
3. A recommended compute-allocation rule for DEWMA:
   - spend little extra compute on hopeless states
   - spend much more compute on promising states and post-breakthrough states
4. A recommended continuation-mode policy that is more aggressive after first progress without removing safety.
5. Three trace fields or summaries that would best confirm the fix worked.

If you disagree, explain:

1. why the current behavior is expected
2. why the bottleneck is not churn / weak conversion
3. what the true bottleneck is instead

---

## 6. Constraints

- Do not suggest removing safety mechanisms entirely.
- Do not suggest generic new algorithms without first auditing the current DEWMA stack.
- Do not assume the main problem is early termination unless the traces prove it.
- Focus on:
  - activation conditions
  - ranking dominance
  - continuation after first progress
  - action-economy waste
  - selective compute use

The agent must still remain Kaggle-safe under multi-game evaluation.

---

## 7. Key Insight

The most useful question for the current DEWMA agent is not:

- "What algorithm is missing?"

It is:

- "Why are the algorithms already present not winning control often enough when the state becomes promising?"

That is the right diagnostic frame for the current architecture.

---

## 8. Required Deliverables

Return all of the following. Do not stop at a narrative summary.

### Mandatory Consolidated Report

Write the complete human-readable analysis to the run-specific path:

`diagnostics/trace_clustering/<run_id>/diagnostic_<run_id>.md`

This file is the primary deliverable and must be created even when some traces are missing, malformed, or from an older code generation. Do not leave the full analysis only in the chat response, terminal output, CSV files, or JSON files.

`diagnostic_<run_id>.md` must be self-contained and include:

- analyzed inputs, code fingerprint, trace-schema fingerprint, and compatibility verdict
- data-quality and deduplication summary
- executive findings and the verdict on the churn/weak-conversion hypothesis
- the complete 25-game assignment table
- cluster definitions, members, confidence, and representative evidence
- architecture-to-code-to-trace conformance findings
- global and per-cluster decision-stage and mechanism-family findings
- deep-search, LLM, post-breakthrough, persistence, control-band, oscillation, meaningful-progress, and finish-corridor conversion audits
- progressing-game continuation analysis
- root causes, falsification criteria, and uncertainties
- ranked experiments with success and regression thresholds
- the single recommended next change or, when evidence is insufficient, the exact rerun/instrumentation recommendation
- links using repository-relative paths to all supporting CSV, JSON, and Markdown artifacts

Use clear headings, tables, counts, percentages with denominators, and specific trace file/step references. Clearly distinguish measured facts, code-derived interpretations, hypotheses, and recommendations. If the analysis is incomplete, add a prominent `Limitations and Missing Evidence` section rather than omitting the report.

### A. Executive Finding

In no more than ten bullets, state:

- how many games and usable events were analyzed
- how many primary clusters were selected
- the dominant portfolio-wide failure
- whether the original churn/weak-conversion hypothesis is supported, partially supported, or rejected
- the three highest-priority experiments

### B. Complete 25-Game Assignment Table

Include every game exactly once:

| Game | Outcome | Primary cluster | Secondary tag | Assignment confidence | Three strongest evidence points | First intervention |
| --- | --- | --- | --- | --- | --- | --- |

After the table, list any expected game that is missing and any unexpected game found.

### C. Cluster Cards

For each cluster provide:

- member games
- cluster size
- centroid/typical feature profile
- strongest discriminating features versus other clusters
- representative trace sequence with file and step references
- root-cause hypothesis
- falsification test
- smallest diagnostic patch or ablation
- expected before/after trace signature
- regression risk

### D. System-Level Decision-Ladder Analysis

Give global and per-cluster stage distributions. Explicitly show where control passes from a potentially useful stage into churn. Identify code symbols and approximate line ranges in `agent/my_agent.py`; do not invent symbol names when code evidence is absent.

### E. LLM Disagreement Audit

Join `llm_forensics_<game>.jsonl` to trace behavior when possible. For each analyzed game report:

- model calls and valid parses
- distinct hypothesis-family count
- proposal survival/promotion/final-action counts
- dominant trace belief versus dominant LLM proposal
- whether disagreement was productive, ignored, rejected, or ungrounded
- top rejection/suppression reason

Quote only short decisive fragments; use file and step references for the full evidence.

### F. Ranked Experiment Plan

Provide 5–10 experiments in this format:

| Rank | Cluster(s) | Exact change/ablation | Why it isolates the cause | Success metric | Stop/regression condition |
| ---: | --- | --- | --- | --- | --- |

Each experiment must change one causal factor where practical. Include a control configuration and fixed seeds/game list so results are comparable.

### G. Artifacts to Create

Create the reusable analyzer at:

- `diagnostics/trace_clustering/analyze_trace_clusters.py`

Create these run-specific outputs under `diagnostics/trace_clustering/<run_id>/`:

- `diagnostic_<run_id>.md` — primary consolidated report containing all analysis and key points
- `game_features.csv` — one row per game
- `game_cluster_assignments.csv` — complete membership and confidence
- `cluster_report.md` — evidence-backed report
- `architecture_conformance.md` — documented ladder versus code precedence versus observed trace funnel
- `trace_data_quality.json` — malformed rows, deduplication, missing fields, episode boundaries, and coverage
- `trace_schema_coverage.csv` — per-field coverage and variability by game/trace generation
- `cluster_model.json` — feature names, preprocessing, method, parameters, centroids/medoids, stability information, code fingerprint, and schema fingerprint

The script must run from the repository root with:

```bash
python diagnostics/trace_clustering/analyze_trace_clusters.py \
  --trace-dir "traces_archive/2026-08-30_14-35-20" \
  --output-root diagnostics/trace_clustering \
  --expected-games 25
```

The analyzer must derive `run_id`, create the run-specific output directory, and print the exact primary report path. It may also support an optional `--run-id` override, but must validate and sanitize it. The default must always come from the trace folder name.

Use the Python standard library where sufficient. If optional clustering packages are unavailable, implement a deterministic fallback or emit the feature matrix plus a transparent rule-based grouping. Never silently skip clustering.

### H. Verification

Run the analyzer and report:

- whether `diagnostic_<run_id>.md` was created in the correct run-specific directory and contains every mandatory section
- distinct games discovered
- assigned games
- unassigned games
- duplicate events removed
- malformed rows
- cluster sizes
- whether rerunning with the same inputs produces identical assignments

Also perform a leave-one-game-out or bootstrap stability check where the sample size permits. Flag assignments that are unstable.

### I. Updated Conversion-Layer Audit

Report these funnels globally and per cluster:

1. `control_band_saturation_detected` → setup budget reduced → `interior_application_phase_active` → `control_band_bias_applied`/`interior_transition_bias_applied` → interior action selected → meaningful progress → level progress
2. `two_anchor_oscillation_detected` or `small_cycle_oscillation_detected` → suppression applied → breakout action selected → productive seed active → directional bias applied → recurrence avoided → meaningful progress
3. meaningful transition → `productive_branch_commitment_used` → structured persistence eligible → gate passed → kept control → convergence rose → level progress
4. post-breakthrough window active → transferred pattern regrounded/relocalized → priority preserved → persistence correctly outranked or retained → effective continuation → next-level progress
5. coherent progress → `finish_corridor_active` → bias applied → branch/family remained stable → corridor exit → level clear

For each funnel give the absolute count and conditional conversion rate at every edge. Name the first edge with the largest loss. Where a required selection field is missing, say what instrumentation should be added rather than pretending bias application proves action selection.

### J. Before/After Compatibility Plan

If only pre-update traces are present, do not claim to validate the updated architecture. Instead:

- complete the historical clustering
- establish a frozen baseline feature table
- identify the smallest representative game set covering every historical cluster
- specify the exact post-update rerun command/configuration
- define before/after acceptance thresholds for action economy, meaningful-progress conversion, recurrence, deaths, and levels completed

If both generations are present, match games and compare paired metrics. Avoid interpreting improvements caused only by different action caps, runtime tiers, LLM availability, seeds, or trace logging density.

## 9. Analytical Guardrails

- Correlation is not causation: a stage dominating late steps may be a consequence of failure rather than its origin.
- `promising_state_detected=true` is not proof that the state is objectively promising; audit the reasons and downstream yield.
- `deep_search_used=false` with nonzero nodes/time may mean search was attempted but produced no selectable action. Keep `attempted`, `found_candidate`, and `selected` separate.
- A low no-op rate does not imply useful exploration. State-changing actions can still be destructive or cyclic.
- Repeated deaths can be informative probes or waste. Measure whether each death eliminates a hypothesis or changes later policy.
- High mechanism confidence is valuable only if it predicts effects or progress. Measure calibration and conversion.
- The two progressing games (`lf52`, `lp85` in the supplied summary) should be analyzed both as successes and as failures to continue.
- Detection is not intervention: `control_band_saturation_detected` or oscillation detection alone is not evidence that the corrective candidate won control.
- Bias is not selection: `control_band_bias_applied`, `interior_transition_bias_applied`, `directional_breakout_bias_applied`, or `finish_corridor_bias_applied` only proves scoring changed unless the chosen action is attributable.
- Micro-change is not meaningful progress. Test whether continuation logic is being activated by effects that never improve goal completion or level outcome.
- A finish-corridor timeout may indicate an insufficient window, a wrong family, a wrong anchor, or false corridor activation. Separate these causes before lengthening the window.
- Control-band geometry can be legitimate puzzle content rather than a control strip. Measure false phase switches and preserve a contradiction escape.
- Do not claim an LLM bottleneck merely from low call count; determine whether calls were needed, usable, promoted, and causally connected to actions.
- Do not recommend more wall-clock compute unless a specific search/reasoning path can consume it without increasing physical-action waste.

## 10. Final Question

After completing the evidence and artifacts, answer this directly:

> If I can implement only one change before the next 25-game run, what exact change should I make, which cluster signals should activate it online, and what measurable trace result would justify keeping it?

Also state whether that recommendation changes depending on whether the archive is pre-update or post-update. If the updated conversion layers have not yet been exercised, the first recommendation should be the minimal paired rerun/instrumentation needed to evaluate them—not another speculative controller patch.
