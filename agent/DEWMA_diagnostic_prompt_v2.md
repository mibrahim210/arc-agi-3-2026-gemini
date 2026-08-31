# DEWMA Deep Diagnostic Prompt v2

## Purpose

Use this prompt when reviewing the current DEWMA implementation in [`agent/my_agent.py`](/C:/Users/mdibr/Desktop/Work_Space_Professional/Kaggle%20Competition/arc-arg-3/GeminiModule/ARC-AGI-3-Kaggle-Starter/agent/my_agent.py).

This version is aligned with the architecture and trace evidence as of August 30, 2026. It is intentionally more precise than a generic "use more search" or "it exits too early" diagnosis.

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
