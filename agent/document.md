# DEWMA-ARC v4.7.0 Baseline Architecture

## Overview
This document outlines the architecture for the **DEWMA** (Developmental Epistemic World-Model Agent) tailored for the ARC-AGI-3 competition. The system is designed to solve abstract reasoning tasks without internet access by relying on an offline polyglot pipeline. It synthesizes deterministic spatial heuristics, causal memory graphs, and a localized generative AI reasoner into a single robust decision-making loop.

---

## The DEWMA Concept in Practice
The architecture directly implements the core philosophies of the DEWMA paradigm:

*   **Developmental (Iterative Learning):** The agent does not start with hardcoded rules for winning a level. Instead, it observes the environment and iteratively builds its understanding. The `GoalHypothesisManager` maintains multiple competing hypotheses (e.g., "collect color X" or "reach object Y") and updates their confidence based on what actions cause positive level deltas.
*   **Epistemic (Information-Seeking):** When the agent encounters a novel state where its internal models yield low confidence, it executes safe "physical probes." The `CandidateGenerator` scores exploratory actions based on Expected Information Gain (EIG), prioritizing moves that resolve uncertainty in the `HypothesisMemory`.
*   **World-Model (Internal Simulation):** Before executing an action in the real environment, the agent simulates it internally. This multi-tiered world model utilizes the `CausalWorldModel` for exact graph replays, the `FastSpatialActionHash` for local patch predictions, and the `ExecutableProgramLibrary` to induce and run verifiable Python-like rules (translations, color maps) over the grid.

---

## Core System Architecture

### 1. Perception & Representation
*   **`PerceptionSystem`:** Ingests the raw 2D integer grid and extracts abstract features. It calculates entropy and determines if the grid should be treated as discrete objects or a dense texture field.
*   **`TemporalEntityTracker`:** Groups connected components into entities and tracks them across frames. It calculates bounding box intersection-over-union (IoU), spatial centroids, and identifies which entities respond directly to user input (controllability).

### 2. Memory & Mechanics (The World Model)
*   **`TraceMemory`:** An episodic buffer that logs every exact state, action, and resulting transition.
*   **`CausalWorldModel`:** A directed graph of verified state transitions. If the agent returns to a previously solved state, it retrieves the exact path to progress, completely bypassing expensive recalculations.
*   **`FastSpatialActionHash`:** A highly efficient $3 \times 3$ and $5 \times 5$ patch-matching table. It memorizes the local geometric consequences of actions (e.g., "clicking a blue pixel turns the $3 \times 3$ patch red") without needing to understand the global grid context.
*   **`DeadSignatureMemory`:** A local cache that remembers highly abstract action signatures that resulted in no-ops or deaths, instantly pruning them from future consideration to preserve the tight action budget.

### 3. Program Induction & Planning
*   **`ExecutableProgramLibrary`:** The core simulation engine. It observes transitions and induces declarative programs (e.g., `translation`, `color_map`, `component_delete`). It verifies these programs against recent memory replays. Once verified, these programs simulate future grid states with high accuracy.
*   **`CounterfactualPlanner`:** A bounded beam-search planner. It uses the induced `WorldProgram` models to simulate multiple steps ahead, searching for a sequence of actions that maximize goal progression before committing to a physical move.
*   **`PathPlanner`:** A deterministic BFS fallback planner that generates physical paths to high-value targets when abstract reasoning fails.

### 4. Alignment & Safety
*   **`GoalAlignmentVerifier`:** The safety gatekeeper. Before any physical action is executed, its predicted outcome is evaluated here. It rejects actions that trigger "predicted no-ops", "unjustified global change", or the "disappearance of a controlled entity", ensuring the agent does not fatally corrupt the environment state.

### 5. Milestone-Gated Generative Reasoner
*   **`OptionalLocalReasoner`:** An offline text-model adapter built on `llama-cpp-python` to execute the local Gemma 4 model directly from `.gguf` weights. 
*   **Execution Strategy:** To preserve the 9-hour competition compute budget, inference is heavily gated. The model is only invoked at major milestones (e.g., level start, severe stagnation, or massive topology changes). It is given access to a `SafeRepl` to query grid statistics (counts, bounding boxes) and propose high-level macro-actions.

---

## The Decision Ladder (Metacognitive Controller)

The `MetacognitiveController` orchestrates action selection through a strict priority cascade, ensuring the fastest and safest methods are evaluated first:

1.  **Replay Progress:** If the exact current state is in the `CausalWorldModel`, execute the known safe path.
2.  **Plan Queue:** Pop the next action from an already verified multi-step plan.
3.  **Counterfactual Search:** Run a beam search over induced `WorldProgram` models to find a new multi-step plan.
4.  **Path Planning:** Attempt deterministic pathfinding if a controllable entity and a clear target exist.
5.  **Mental Arbitration:** Evaluate heuristics (spatial hashes, hypothesis memory) and pass them through the `GoalAlignmentVerifier`.
6.  **Gemma 4 Milestone LLM:** If stuck, invoke the offline generative model to propose a high-level action via the REPL.
7.  **Physical Probe:** If all internal models yield low confidence, execute the safest unknown action to gather new information.
8.  **Fail-Closed Fallback:** Pick the least destructive legal action available or issue a `RESET`.

---

## Execution Environment Constraints
*   **Strict Offline Execution:** All internet access is disabled. Model weights must be mounted via Kaggle datasets, and dependencies (like `llama-cpp-python`) must be installed from local `.whl` files at runtime.
*   **No Multi-Frame Collapse:** Environment responses containing animations are analyzed frame-by-frame via `FrameSequence` to detect transient mechanics and topological shifts, rather than just reading the final settled state.
*   **Dynamic Tier Downgrading:** The `RuntimeBudget` continuously monitors the wall-clock time. If the agent approaches the 9-hour limit, it gracefully degrades from full counterfactual planning (Tier A9) down to fast heuristics (Tier A5) to ensure successful submission.