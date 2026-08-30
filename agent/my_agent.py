"""DEWMA-ARC v4.1 runtime/alignment hardened: Qwen2.5-Coder model integrated ARC-AGI-3 agent.

Drop-in replacement for ``agent/my_agent.py`` in the official
ARC-AGI-3 Kaggle Starter.

Design goals
------------
* Raw integer grid remains authoritative.
* Derived objects are confidence-gated and never shadow raw-grid reasoning.
* Stored traces and local simulation are consulted before physical probing.
* A fast spatial action hash handles repeated local mechanics.
* Contextual dead signatures evict repeatedly ineffective affordances.
* Expensive local-model inference is optional and milestone-gated.
* Every returned action is checked against the current legal action set.
* Multi-frame responses are interpreted as temporal event sequences, not collapsed.
* Competing goal hypotheses remain explicit and are revised from progress evidence.
* Declarative executable world-model programs are induced, replay-verified, and persisted.
* Counterfactual search applies verified programs to unseen states before acting.
* A goal-alignment gate rejects predicted no-ops, harmful destruction, and goal regressions.
* Lightweight JSONL-ready diagnostics, boundary-aware local hashes, and runtime tier degradation are merged from the compact baseline.

This is a generic research baseline, not a guaranteed winning solver. ARC-AGI-3
contains hidden, out-of-distribution environments, so empirical ablation and
iteration remain necessary.
"""

from __future__ import annotations
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
import ast
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# =====================================================================
# OFFLINE DEPENDENCY BOOTSTRAP FOR KAGGLE
# =====================================================================
# We use standard urllib.request to talk to the local Ollama server.
# No offline wheels required.

# Compatible with both the official starter layout and local vendored runs.
try:
    _MODULE_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover - notebook execution fallback
    _MODULE_ROOT = Path.cwd()
_FRAMEWORK_DIR = _MODULE_ROOT / "vendor" / "ARC-AGI-3-Agents"
if _FRAMEWORK_DIR.exists() and str(_FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_DIR))

# MUST IMPORT AFTER PATH INJECTION


# One monotonic clock is shared by every agent instance in this Python process.
# This prevents a later-created swarm agent from receiving a fresh wall-time budget.
_PROCESS_STARTED_AT = time.monotonic()
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_REGISTERED_GAMES: set[str] = set()
_RUNTIME_COMPLETED_GAMES: set[str] = set()
_RUNTIME_GAME_DURATIONS_SEC: deque[float] = deque(maxlen=256)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, percentile)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class Config:
    max_actions: int = _env_int("DEWMA_MAX_ACTIONS", 600)
    enable_entities: bool = _env_bool("DEWMA_ENTITIES", True)
    enable_spatial_hash: bool = _env_bool("DEWMA_SPATIAL_HASH", True)
    enable_dead_signatures: bool = _env_bool("DEWMA_DEAD_SIGNATURES", True)
    enable_path_planner: bool = _env_bool("DEWMA_PATH_PLANNER", True)
    enable_model: bool = _env_bool("DEWMA_USE_MODEL", True)

    entity_confidence_threshold: float = _env_float(
        "DEWMA_ENTITY_CONFIDENCE", 0.55)
    no_op_dead_threshold: int = _env_int("DEWMA_DEAD_THRESHOLD", 4)
    max_physical_probes_per_level: int = _env_int("DEWMA_PROBE_BUDGET", 16)
    severe_stagnation_steps: int = _env_int("DEWMA_STAGNATION_RESET", 18)
    loop_window: int = _env_int("DEWMA_LOOP_WINDOW", 12)
    plan_queue_limit: int = _env_int("DEWMA_PLAN_QUEUE", 6)

    # Kaggle Offline GGUF path fallback
    model_path: str = os.getenv(
        "DEWMA_MODEL_PATH", "/kaggle/input/qwen25-coder-7b-gguf/qwen2.5-coder-7b-instruct-q4_k_m.gguf").strip()
    model_call_budget: int = _env_int("DEWMA_MODEL_CALL_BUDGET", 12)
    model_cooldown_steps: int = _env_int("DEWMA_MODEL_COOLDOWN", 3)
    model_max_new_tokens: int = _env_int("DEWMA_MODEL_MAX_NEW_TOKENS", 256)
    model_tool_rounds: int = _env_int("DEWMA_MODEL_TOOL_ROUNDS", 2)

    patch_sizes: tuple[int, ...] = (3, 5)
    spatial_hash_min_support: int = _env_int("DEWMA_HASH_MIN_SUPPORT", 2)
    spatial_hash_max_conflict: float = _env_float(
        "DEWMA_HASH_MAX_CONFLICT", 0.25)
    max_complex_candidates: int = _env_int("DEWMA_COMPLEX_CANDIDATES", 64)
    max_track_components: int = _env_int("DEWMA_MAX_TRACK_COMPONENTS", 192)

    enable_control_inference: bool = _env_bool("DEWMA_CONTROL_INFERENCE", True)
    control_min_support: int = _env_int("DEWMA_CONTROL_MIN_SUPPORT", 2)
    control_min_confidence: float = _env_float(
        "DEWMA_CONTROL_MIN_CONFIDENCE", 0.60)
    control_min_margin: float = _env_float("DEWMA_CONTROL_MIN_MARGIN", 0.15)
    control_min_persistence: float = _env_float(
        "DEWMA_CONTROL_MIN_PERSISTENCE", 0.45)
    control_max_candidates: int = _env_int("DEWMA_CONTROL_MAX_CANDIDATES", 64)

    enable_control_groups: bool = _env_bool("DEWMA_CONTROL_GROUPS", True)
    control_group_min_members: int = _env_int(
        "DEWMA_CONTROL_GROUP_MIN_MEMBERS", 2)
    control_group_max_members: int = _env_int(
        "DEWMA_CONTROL_GROUP_MAX_MEMBERS", 12)
    control_group_min_support: int = _env_int(
        "DEWMA_CONTROL_GROUP_MIN_SUPPORT", 2)
    control_group_min_confidence: float = _env_float(
        "DEWMA_CONTROL_GROUP_MIN_CONFIDENCE", 0.60)
    control_group_min_coherence: float = _env_float(
        "DEWMA_CONTROL_GROUP_MIN_COHERENCE", 0.80)
    control_group_min_spatial_coherence: float = _env_float(
        "DEWMA_CONTROL_GROUP_MIN_SPATIAL_COHERENCE", 0.10)
    control_group_max_gap: int = _env_int("DEWMA_CONTROL_GROUP_MAX_GAP", 6)
    control_group_membership_overlap: float = _env_float(
        "DEWMA_CONTROL_GROUP_MEMBERSHIP_OVERLAP", 0.60)
    control_applicability_min_observations: int = _env_int(
        "DEWMA_CONTROL_APPLICABILITY_MIN_OBSERVATIONS", 3)
    control_complex_action_ratio: float = _env_float(
        "DEWMA_CONTROL_COMPLEX_ACTION_RATIO", 0.75)


    enable_learned_passability: bool = _env_bool(
        "DEWMA_LEARNED_PASSABILITY", True)
    passability_min_support: int = _env_int("DEWMA_PASSABILITY_MIN_SUPPORT", 1)
    passability_min_confidence: float = _env_float(
        "DEWMA_PASSABILITY_MIN_CONFIDENCE", 0.60)
    passability_failure_penalty: float = _env_float(
        "DEWMA_PASSABILITY_FAILURE_PENALTY", 1.25)
    path_max_anchor_states: int = _env_int(
        "DEWMA_PATH_MAX_ANCHOR_STATES", 4096)
    path_max_target_goal_anchors: int = _env_int(
        "DEWMA_PATH_MAX_TARGET_GOAL_ANCHORS", 512)


    enable_goal_inference: bool = _env_bool("DEWMA_GOALS", True)
    enable_programs: bool = _env_bool("DEWMA_PROGRAMS", True)
    enable_counterfactual_planner: bool = _env_bool(
        "DEWMA_COUNTERFACTUAL", True)
    enable_alignment_gate: bool = _env_bool("DEWMA_ALIGNMENT", True)
    max_goal_hypotheses: int = _env_int("DEWMA_MAX_GOALS", 14)

    program_min_support: int = _env_int("DEWMA_PROGRAM_MIN_SUPPORT", 2)
    program_min_confidence: float = _env_float(
        "DEWMA_PROGRAM_MIN_CONFIDENCE", 0.70)
    program_min_cell_accuracy: float = _env_float(
        "DEWMA_PROGRAM_MIN_CELL_ACCURACY", 0.85)

    counterfactual_depth: int = _env_int("DEWMA_CF_DEPTH", 2)
    counterfactual_beam: int = _env_int("DEWMA_CF_BEAM", 4)
    counterfactual_candidate_limit: int = _env_int("DEWMA_CF_CANDIDATES", 8)
    alignment_min_score: float = _env_float("DEWMA_ALIGNMENT_MIN", -0.20)
    animation_history_limit: int = _env_int("DEWMA_ANIMATION_HISTORY", 12)
    max_programs: int = _env_int("DEWMA_MAX_PROGRAMS", 320)

    enable_runtime_diagnostics: bool = _env_bool(
        "DEWMA_RUNTIME_DIAGNOSTICS", True)
    a9_rank_candidates: int = _env_int("DEWMA_A9_RANK_CANDIDATES", 16)
    a9_path_planner_budget_sec: float = _env_float(
        "DEWMA_A9_PATH_PLANNER_BUDGET_SEC", 0.45)
    fallback_path_planner_budget_sec: float = _env_float(
        "DEWMA_FALLBACK_PATH_PLANNER_BUDGET_SEC", 0.35)
    no_progress_consecutive_threshold: int = _env_int(
        "DEWMA_NO_PROGRESS_CONSECUTIVE", 4)
    no_progress_cooldown_steps: int = _env_int("DEWMA_NO_PROGRESS_COOLDOWN", 6)
    no_progress_exhaustion_threshold: int = _env_int(
        "DEWMA_NO_PROGRESS_EXHAUSTION", 10)

    path_cycle_window: int = _env_int("DEWMA_PATH_CYCLE_WINDOW", 8)
    path_no_progress_limit: int = _env_int("DEWMA_PATH_NO_PROGRESS_LIMIT", 4)
    path_target_cooldown_steps: int = _env_int(
        "DEWMA_PATH_TARGET_COOLDOWN", 16)
    enable_frontier_exploration: bool = _env_bool(
        "DEWMA_FRONTIER_EXPLORATION", True)
    frontier_trigger_noop_streak: int = _env_int(
        "DEWMA_FRONTIER_TRIGGER_NOOPS", 2)
    program_replay_window: int = _env_int("DEWMA_PROGRAM_REPLAY_WINDOW", 16)
    program_verify_limit: int = _env_int("DEWMA_PROGRAM_VERIFY_LIMIT", 64)
    min_time_for_llm_sec: int = _env_int("DEWMA_MIN_TIME_FOR_LLM", 45)

    llm_level_warmup_steps: int = _env_int("DEWMA_LLM_LEVEL_WARMUP_STEPS", 12)
    llm_recent_transitions_limit: int = _env_int("DEWMA_LLM_RECENT_TRANSITIONS_LIMIT", 4)
    llm_goals_limit: int = _env_int("DEWMA_LLM_GOALS_LIMIT", 3)
    llm_programs_limit: int = _env_int("DEWMA_LLM_PROGRAMS_LIMIT", 3)
    llm_action_semantics_limit: int = _env_int("DEWMA_LLM_ACTION_SEMANTICS_LIMIT", 4)


    trace_enabled: bool = _env_bool("DEWMA_TRACE_ENABLED", True)
    trace_to_disk: bool = _env_bool("DEWMA_TRACE_TO_DISK", True)
    trace_dir: str = os.getenv("DEWMA_TRACE_DIR", "./traces")
    trace_max_records: int = _env_int("DEWMA_TRACE_MAX_RECORDS", 2048)
    session_limit_sec: int = _env_int(
        "DEWMA_SESSION_LIMIT_SEC", int(8.75 * 3600))
    finalization_reserve_sec: int = _env_int(
        "DEWMA_FINALIZATION_RESERVE_SEC", 300)
    runtime_a7_ratio: float = _env_float("DEWMA_RUNTIME_A7_RATIO", 0.35)
    runtime_a5_ratio: float = _env_float("DEWMA_RUNTIME_A5_RATIO", 0.20)
    runtime_required_margin_ratio: float = _env_float(
        "DEWMA_RUNTIME_REQUIRED_MARGIN", 0.20)
    runtime_target_margin_ratio: float = _env_float(
        "DEWMA_RUNTIME_TARGET_MARGIN", 0.27)
    runtime_expected_games: int = _env_int("DEWMA_EXPECTED_GAMES", 110)
    runtime_effective_game_parallelism: int = _env_int(
        "DEWMA_EFFECTIVE_GAME_PARALLELISM", 110)
    runtime_game_p95_fallback_sec: float = _env_float(
        "DEWMA_GAME_P95_FALLBACK_SEC", 90.0)
    runtime_projection_min_samples: int = _env_int(
        "DEWMA_RUNTIME_MIN_SAMPLES", 3)
    force_runtime_tier: str = os.getenv("DEWMA_FORCE_TIER", "").strip().upper()
    model_call_budget_per_level: int = _env_int(
        "DEWMA_MODEL_CALLS_PER_LEVEL", 6)

    alignment_min_progress_events: int = _env_int(
        "DEWMA_ALIGNMENT_MIN_PROGRESS_EVENTS", 3)
    alignment_preservation_support: int = _env_int(
        "DEWMA_ALIGNMENT_PRESERVE_SUPPORT", 3)
    alignment_preservation_ratio: float = _env_float(
        "DEWMA_ALIGNMENT_PRESERVE_RATIO", 0.80)
    alignment_fatal_min_support: int = _env_int(
        "DEWMA_ALIGNMENT_FATAL_SUPPORT", 2)
    allow_transferred_program_one_step: bool = _env_bool(
        "DEWMA_ALLOW_TRANSFERRED_ONE_STEP", False)


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    tier: str
    use_hypotheses: bool
    use_goals: bool
    use_programs: bool
    use_alignment: bool
    use_counterfactual: bool
    use_model: bool
    use_path_planner: bool
    learn_hypotheses: bool
    learn_goals: bool
    learn_programs: bool
    learn_alignment: bool


def _runtime_profile(tier: str, config: Config) -> RuntimeProfile:
    normalized = tier.upper() if tier else "A9"
    if normalized not in {"A5", "A7", "A8", "A9"}:
        normalized = "A9"

    if normalized == "A5":
        return RuntimeProfile(
            normalized, False, False, False, False, False, False, False,
            False, False, False, False,
        )

    if normalized == "A7":
        return RuntimeProfile(
            normalized, False, False, False, False, False,
            config.enable_model, config.enable_path_planner,
            False, False, False, False,
        )

    # A8 adds the slower conditional-hypothesis layer and virtual-first
    # discrimination, but not explicit goals or executable programs.
    if normalized == "A8":
        return RuntimeProfile(
            normalized, True, False, False, False, False,
            config.enable_model, config.enable_path_planner,
            True, False, False, False,
        )

    return RuntimeProfile(
        "A9",
        True,
        config.enable_goal_inference,
        config.enable_programs,
        config.enable_alignment_gate,
        config.enable_counterfactual_planner and config.enable_programs and config.enable_goal_inference,
        config.enable_model,
        config.enable_path_planner,
        True,
        config.enable_goal_inference,
        config.enable_programs,
        config.enable_alignment_gate,
    )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


Point = tuple[int, int]  # x, y


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    data: tuple[tuple[str, int], ...] = ()
    source: str = "policy"
    predicted_effect: str | None = None
    score: float = 0.0
    program_id: str | None = None
    predicted_state_key: str | None = None
    goal_ids: tuple[str, ...] = ()

    @property
    def data_dict(self) -> dict[str, int]:
        return dict(self.data)

    @property
    def key(self) -> tuple[str, tuple[tuple[str, int], ...]]:
        return self.name, self.data


@dataclass(frozen=True, slots=True)
class Component:
    local_id: int
    color: int
    cells: tuple[Point, ...]
    bbox: tuple[int, int, int, int]  # min_x, min_y, max_x, max_y
    centroid: tuple[float, float]
    area: int
    shape_key: str
    touches_border: bool
    compactness: float


@dataclass(slots=True)
class TrackedEntity:
    entity_id: str
    color: int
    cells: tuple[Point, ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    shape_key: str
    age: int = 1
    missing_frames: int = 0
    persistence_confidence: float = 0.5
    controllability: float = 0.0
    autonomous_motion: float = 0.0
    last_move: tuple[int, int] = (0, 0)
    affordances: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class Scene:
    grid: np.ndarray
    previous_grid: np.ndarray | None
    delta_mask: np.ndarray
    changed_cells: tuple[Point, ...]
    exact_key: str
    abstract_key: str
    background: int
    color_counts: tuple[tuple[int, int], ...]
    components: tuple[Component, ...]
    entities: tuple[TrackedEntity, ...]
    controlled_entity_id: str | None
    entity_confidence: float
    field_mode: bool
    change_ratio: float
    delta_entropy: float
    level: int

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])


@dataclass(frozen=True, slots=True)
class FrameSequence:
    """All visual frames returned for one environment action."""

    grids: tuple[np.ndarray, ...]

    @property
    def settled(self) -> np.ndarray:
        if self.grids:
            return self.grids[-1]
        return np.zeros((64, 64), dtype=np.int16)


@dataclass(frozen=True, slots=True)
class AnimationStep:
    """One observable sub-transition inside an action response animation."""

    index: int
    changed_count: int
    changed_bbox: tuple[int, int, int, int] | None
    appeared_colors: tuple[int, ...]
    disappeared_colors: tuple[int, ...]
    motion_vectors: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class AnimationTrace:
    frame_count: int
    step_changed_counts: tuple[int, ...]
    cumulative_changed_count: int
    persistent_changed_count: int
    transient_changed_count: int
    oscillating_cell_count: int
    changed_bboxes: tuple[tuple[int, int, int, int] | None, ...]
    color_count_trajectories: tuple[tuple[int, tuple[int, ...]], ...]
    motion_vectors: tuple[tuple[int, int, int], ...]  # color, dx, dy
    settled_stable: bool
    temporal_signature: str
    steps: tuple[AnimationStep, ...] = ()


@dataclass(frozen=True, slots=True)
class Event:
    changed_count: int
    changed_bbox: tuple[int, int, int, int] | None
    no_op: bool
    level_delta: int
    game_over: bool
    win: bool
    entity_moves: tuple[tuple[str, int, int], ...]
    appeared_colors: tuple[int, ...]
    disappeared_colors: tuple[int, ...]
    topology_change: bool
    effect_signature: str
    subframe_count: int = 1
    cumulative_changed_count: int = 0
    transient_changed_count: int = 0
    oscillating_cell_count: int = 0
    animation_vectors: tuple[tuple[int, int, int], ...] = ()
    temporal_signature: str = ""
    settled_no_op: bool = False
    animation_steps: tuple[AnimationStep, ...] = ()


@dataclass(frozen=True, slots=True)
class Transition:
    before_exact: str
    before_abstract: str
    action: ActionSpec
    after_exact: str
    after_abstract: str
    event: Event
    level: int
    step_index: int
    before_grid: np.ndarray | None = None
    after_grid: np.ndarray | None = None
    response_frames: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticEntityDelta:
    entity_id: str
    dx: int
    dy: int
    survived: bool = True
    appeared: bool = False
    disappeared: bool = False
    confidence: float = 1.0


@dataclass(slots=True)
class DiagnosticTraceRecord:
    timestamp: float
    game_id: str
    level: int
    step: int
    before_key: str
    action: dict[str, Any]
    after_key: str
    decision_stage: str
    expected_effect: str | None
    observed_effect: str
    no_op: bool
    death: bool
    progress: bool
    physical_probe: bool
    model_called: bool
    local_patch_hash: str | None
    active_goal_ids: tuple[str, ...]
    program_ids: tuple[str, ...]
    representation_mode: str
    representation_confidence: float
    entity_deltas: list[dict[str, Any]] = field(default_factory=list)
    model_latency_sec: float = 0.0
    deterministic_latency_sec: float = 0.0
    remaining_wall_time_sec: float | None = None
    runtime_tier: str = "A9"
    model_decisions_used: int = 0
    model_decisions_used_this_level: int = 0
    reasoner_decision_attempts: int = 0
    reasoner_decision_skips_by_reason: dict[str, int] = field(default_factory=dict)
    reasoner_decision_successes: int = 0
    reasoner_consultations_this_level: int = 0
    fallback_loop_streak: int = 0
    counterfactual_fallback_streak: int = 0
    alignment_fallback_streak: int = 0
    steps_since_progress: int = 0
    progress_events_this_level: int = 0
    progress_density: float = 0.0
    recent_lock_skip_count: int = 0
    reasoner_lock_backoff_steps: int = 0
    reasoner_suppressed: bool = False
    reasoner_suppression_reason: str = ""
    llm_illegal_action_rejections: int = 0
    llm_repetitive_kind_rejections: int = 0
    llm_empty_abduction_count: int = 0
    llm_parsed_abduction_count: int = 0
    llm_diversity_pruned_count: int = 0
    llm_impossible_hypothesis_rejections: int = 0
    llm_unsupported_family_rejections: int = 0
    llm_family_repeat_rejections: int = 0
    llm_schema_valid_but_unexecutable: int = 0
    llm_filtered_for_action_mismatch: int = 0
    reasoner_consulted_this_step: bool = False
    reasoner_parsed_abduction_this_step: bool = False
    reasoner_surviving_proposals_this_step: int = 0
    final_action_from_llm: bool = False
    final_action_source: str = ""
    stagnation_override_used: bool = False
    competing_hypothesis_count: int = 0
    stuck_mode_activations: int = 0
    llm_rejected_by_alignment: int = 0
    llm_rejected_by_semantic_gate: int = 0
    llm_rejected_by_dead_signature: int = 0
    llm_rejected_by_probe_budget: int = 0
    llm_rejected_by_coordinate_fatigue: int = 0
    llm_escape_hatch_used: bool = False
    llm_semantic_override_attempts: int = 0
    llm_semantic_override_used: bool = False
    llm_semantic_override_rejections: int = 0
    llm_breakout_used_this_step: bool = False
    llm_override_eligible: bool = False
    llm_override_block_reason: str = ""
    llm_severe_stagnation_signal: str = ""
    llm_follow_through_active: bool = False
    llm_follow_through_family: str = ""
    llm_promoted_programs_count: int = 0
    churn_stagnation_detected: bool = False
    follow_through_led_to_progress: bool = False
    early_breakout_acceleration_used: bool = False
    level_budget_pressure_signal: float = 0.0
    level_steps_elapsed: int = 0
    level_budget_target: int = 0
    level_budget_pressure: float = 0.0
    level_budget_escalation_used: bool = False
    macro_replay_active: bool = False
    macro_replay_steps_remaining: int = 0
    macro_replay_family: str = ""
    macro_replay_program_id: str = ""
    macro_replay_aborted_reason: str = ""
    level_transition_transfer_used: bool = False
    transferred_mechanism_family: str = ""
    early_classified_mechanism: str = ""
    instant_reflex_used: bool = False
    counterfactual_pruned_count: int = 0
    subgoal_distance_reward: float = 0.0
    negative_hypothesis_eliminations: int = 0
    mode: str = "discovery_mode"
    top_mechanism_family: str = "movement_control"
    top_mechanism_confidence: float = 0.1
    competing_mechanism_families: list[str] = field(default_factory=list)
    mechanism_family_scores: dict[str, float] = field(default_factory=dict)
    priority_program_families: list[str] = field(default_factory=list)
    invariants_to_preserve: list[str] = field(default_factory=list)
    mechanism_shift_event: bool = False
    mechanism_aligned_action: bool = False
    mechanism_discriminating_probe_used: bool = False
    llm_top_productive_family: str = ""
    llm_top_unproductive_family: str = ""
    llm_family_payoff_summary: dict[str, Any] = field(default_factory=dict)
    action_semantics_summary: dict[str, Any] = field(default_factory=dict)
    promising_state_detected: bool = False
    promising_state_reasons: list[str] = field(default_factory=list)
    deep_search_used: bool = False
    deep_search_depth: int = 0
    deep_search_width: int = 0
    deep_search_nodes_evaluated: int = 0
    deep_search_best_score: float = 0.0
    deep_search_time_ms: float = 0.0
    deep_search_selected_family: str = ""
    deep_search_aborted_reason: str = ""
    post_breakthrough_window_active: bool = False
    post_levelup_exploit_steps_remaining: int = 0
    transferred_winning_family: str = ""
    transferred_winning_program_kind: str = ""
    transferred_winning_action_name: str = ""
    transferred_winning_coords: Any = None
    post_breakthrough_aborted_reason: str = ""
    post_breakthrough_bias_used: bool = False
    post_breakthrough_attempts: int = 0
    post_breakthrough_effective_attempts: int = 0
    post_breakthrough_noop_attempts: int = 0
    post_breakthrough_failed_attempts: int = 0
    post_breakthrough_continuation_bias: float = 0.0
    post_breakthrough_local_search_used: bool = False
    post_breakthrough_abort_reason: str = ""
    counterfactual_streak_renewed: bool = False
    mechanism_collapse_breaker_used: bool = False
    mechanism_family_penalized: str = ""
    mechanism_family_promoted: str = ""
    component_delete_component_locked: bool = False
    line_beam_structured_candidates_used: bool = False
    counterfactual_completion_bias: float = 0.0
    terminal_condition_bonus: float = 0.0
    mechanism_completion_bonus: float = 0.0
    productive_search_convergence_pressure: float = 0.0
    line_beam_closure_bias_used: bool = False
    component_delete_payoff_bias_used: bool = False
    atomic_drag_drop_paired: bool = False
    sequential_component_sweep_active: bool = False
    orthogonal_collision_steer_used: bool = False
    cell_cycle_persistence_used: bool = False
    near_terminal_finish_mode_active: bool = False
    near_terminal_finish_steps_remaining: int = 0
    near_terminal_finish_family: str = ""
    near_terminal_finish_trigger_reason: str = ""
    near_terminal_finish_exit_reason: str = ""
    productive_branch_commitment_used: bool = False
    near_terminal_finish_gate_family_consensus: bool = False
    near_terminal_finish_gate_branch_consensus: bool = False
    near_terminal_finish_gate_completion_stability: bool = False
    near_terminal_finish_gate_allowed: bool = False
    finish_branch_continuation_family: str = ""
    finish_branch_continuation_steps: int = 0
    finish_branch_continuation_kept_control: bool = False
    finish_branch_continuation_break_reason: str = ""
    finish_family_collision_suppressed: bool = False
    finish_family_collision_suppressed_family: str = ""
    finish_family_collision_suppressed_until_step: int = 0
    finish_mode_preempted_counterfactual: bool = False
    finish_mode_preempt_block_reason: str = ""
    productive_branch_signature: str = ""
    productive_branch_source: str = ""
    productive_branch_action_name: str = ""
    productive_branch_anchor: Any = None
    productive_branch_family_hint: str = ""
    productive_branch_program_kind: str = ""
    productive_branch_streak: int = 0
    productive_branch_last_effective_step: int = 0
    productive_branch_family_wobble_tolerated: bool = False
    productive_branch_family_wobble_reason: str = ""
    structured_branch_persistence_used: bool = False
    structured_branch_persistence_steps: int = 0
    structured_branch_persistence_steps_remaining: int = 0
    structured_branch_persistence_kept_control: bool = False
    structured_branch_persistence_break_reason: str = ""
    finish_bound_branch_signature: str = ""
    finish_bound_branch_source: str = ""
    finish_bound_branch_anchor: Any = None
    finish_bound_branch_action_name: str = ""
    finish_bound_branch_kept_control: bool = False
    post_breakthrough_priority_preserved: bool = False
    post_breakthrough_preempt_block_reason: str = ""
    post_breakthrough_local_branch_reused: bool = False
    post_breakthrough_local_branch_break_reason: str = ""
    productive_branch_collision_recovery_used: bool = False
    productive_branch_collision_recovery_variant: str = ""
    productive_branch_collision_recovery_preserved_family: bool = False
    productive_window_extended: bool = False
    productive_window_extension_reason: str = ""
    productive_branch_preempt_blocked: bool = False
    productive_branch_preempt_block_reason: str = ""
    collision_soft_retry_used: bool = False
    collision_soft_retry_variant: str = ""
    regrounded_winning_coords: Any = None
    regrounding_delta: Any = None
    regrounding_confidence: float = 0.0
    regrounding_used: bool = False
    regrounding_failed_reason: str = ""
    levelup_relocalization_llm_attempted: bool = False
    levelup_relocalization_llm_used: bool = False
    levelup_relocalization_llm_confidence: float = 0.0
    exploitation_noop_blacklist_checks: int = 0
    exploitation_noop_blacklist_hits: int = 0
    mechanism_detector_triggered: str = ""


@dataclass(slots=True)
class Candidate:
    spec: ActionSpec
    signature: str
    is_probe: bool
    score: float
    rationale: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GoalHypothesis:
    goal_id: str
    kind: str
    params: dict[str, Any]
    confidence: float
    progress_estimate: float = 0.0
    support: float = 0.0
    contradictions: float = 0.0
    status: str = "active"
    source: str = "scene_prior"
    evidence: deque[str] = field(default_factory=lambda: deque(maxlen=16))


@dataclass(frozen=True, slots=True)
class ProgramPrediction:
    grid: np.ndarray
    confidence: float
    program_id: str
    kind: str
    expected_effect: str
    uncertainty: float = 0.0


@dataclass(frozen=True, slots=True)
class AlignmentDecision:
    allowed: bool
    score: float
    goal_delta: float
    risk: float
    goal_ids: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AbductionProposal:
    action: ActionSpec | None
    program_spec: Mapping[str, Any] | None = None
    goal_spec: Mapping[str, Any] | None = None
    puzzle_family: str | None = None


# ---------------------------------------------------------------------------
# Grid extraction and deterministic REPL-like inspection tools
# ---------------------------------------------------------------------------


def _stable_hash_bytes(payload: bytes, digest_size: int = 12) -> str:
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def _coerce_2d_grid(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int16)
    if arr.size == 0:
        return np.zeros((64, 64), dtype=np.int16)
    while arr.ndim > 2:
        arr = arr[-1]
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        side = int(round(math.sqrt(arr.size)))
        arr = arr.reshape(side, side) if side * \
            side == arr.size else arr.reshape(1, -1)
    return np.ascontiguousarray(arr, dtype=np.int16)


def _extract_frame_sequence(frame_data: FrameData) -> FrameSequence:
    """Preserve every subframe returned by one action.

    Official ARC-AGI-3 responses may contain an animation sequence.  The old
    baseline silently selected only the final layer; DEWMA v2 retains every
    frame for event and causal analysis while still using the settled frame as
    the authoritative state for the next decision.
    """
    raw = getattr(frame_data, "frame", None)
    if raw is None:
        return FrameSequence((np.zeros((64, 64), dtype=np.int16),))
    try:
        arr = np.asarray(raw, dtype=np.int16)
        if arr.size == 0:
            return FrameSequence((np.zeros((64, 64), dtype=np.int16),))
        if arr.ndim <= 2:
            return FrameSequence((_coerce_2d_grid(arr),))
        # The final two dimensions are spatial; any leading dimensions are
        # flattened into chronological frames.
        reshaped = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
        return FrameSequence(tuple(np.ascontiguousarray(x, dtype=np.int16) for x in reshaped))
    except Exception:
        # Ragged JSON fallback.
        grids: list[np.ndarray] = []
        if isinstance(raw, Sequence) and raw:
            for item in raw:
                try:
                    candidate = _coerce_2d_grid(item)
                    if candidate.ndim == 2:
                        grids.append(candidate)
                except Exception:
                    continue
        return FrameSequence(tuple(grids) or (np.zeros((64, 64), dtype=np.int16),))


def _extract_latest_grid(frame_data: FrameData) -> np.ndarray:
    """Compatibility helper returning the settled frame only."""
    return _extract_frame_sequence(frame_data).settled


def _color_centroid(grid: np.ndarray, color: int) -> tuple[float, float] | None:
    ys, xs = np.where(grid == color)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _analyze_animation(previous: np.ndarray | None, sequence: FrameSequence) -> AnimationTrace:
    frames = sequence.grids or (sequence.settled,)
    baseline = frames[0] if previous is None or previous.shape != frames[0].shape else previous
    timeline = (baseline,) + tuple(frames)
    step_masks: list[np.ndarray] = []
    changed_bboxes: list[tuple[int, int, int, int] | None] = []
    steps: list[AnimationStep] = []
    change_tally = np.zeros_like(frames[-1], dtype=np.int16)
    union = np.zeros_like(frames[-1], dtype=bool)
    for index, (left, right) in enumerate(zip(timeline, timeline[1:]), start=1):
        if left.shape != right.shape:
            mask = np.ones_like(right, dtype=bool)
        else:
            mask = left != right
        step_masks.append(mask)
        union |= mask
        change_tally += mask.astype(np.int16)
        ys, xs = np.where(mask)
        changed = [(int(x), int(y)) for y, x in zip(ys, xs)]
        bbox = _bbox(changed)
        changed_bboxes.append(bbox)

        left_colors = {int(v) for v in np.unique(left)}
        right_colors = {int(v) for v in np.unique(right)}
        step_motions: list[tuple[int, int, int]] = []
        for color in sorted(left_colors & right_colors):
            first = _color_centroid(left, color)
            last = _color_centroid(right, color)
            if first is None or last is None:
                continue
            dx = int(round(last[0] - first[0]))
            dy = int(round(last[1] - first[1]))
            if dx or dy:
                step_motions.append((color, dx, dy))
        steps.append(
            AnimationStep(
                index=index,
                changed_count=int(np.count_nonzero(mask)),
                changed_bbox=bbox,
                appeared_colors=tuple(sorted(right_colors - left_colors)),
                disappeared_colors=tuple(sorted(left_colors - right_colors)),
                motion_vectors=tuple(step_motions),
            )
        )

    final_mask = np.ones_like(
        frames[-1], dtype=bool) if baseline.shape != frames[-1].shape else baseline != frames[-1]
    transient = union & ~final_mask
    oscillating = change_tally >= 2

    colors = sorted({int(v) for frame in timeline for v in np.unique(frame)})
    trajectories: list[tuple[int, tuple[int, ...]]] = []
    motions: list[tuple[int, int, int]] = []
    for color in colors:
        counts = tuple(int(np.count_nonzero(frame == color))
                       for frame in timeline)
        trajectories.append((color, counts))
        first = _color_centroid(timeline[0], color)
        last = _color_centroid(timeline[-1], color)
        if first is not None and last is not None:
            dx = int(round(last[0] - first[0]))
            dy = int(round(last[1] - first[1]))
            if dx or dy:
                motions.append((color, dx, dy))

    payload = {
        "n": len(frames),
        "steps": [
            {
                "changed": step.changed_count,
                "bbox": step.changed_bbox,
                "appear": step.appeared_colors,
                "disappear": step.disappeared_colors,
                "motion": step.motion_vectors,
            }
            for step in steps
        ],
        "persistent": int(np.count_nonzero(final_mask)),
        "transient": int(np.count_nonzero(transient)),
        "osc": int(np.count_nonzero(oscillating)),
        "motions": motions,
    }
    return AnimationTrace(
        frame_count=len(frames),
        step_changed_counts=tuple(int(np.count_nonzero(m))
                                  for m in step_masks),
        cumulative_changed_count=int(np.count_nonzero(union)),
        persistent_changed_count=int(np.count_nonzero(final_mask)),
        transient_changed_count=int(np.count_nonzero(transient)),
        oscillating_cell_count=int(np.count_nonzero(oscillating)),
        changed_bboxes=tuple(changed_bboxes),
        color_count_trajectories=tuple(trajectories),
        motion_vectors=tuple(motions),
        settled_stable=(len(frames) < 2 or np.array_equal(
            frames[-1], frames[-2])),
        temporal_signature=_stable_hash_bytes(
            json.dumps(payload, sort_keys=True).encode(), 10),
        steps=tuple(steps),
    )


class RuntimeBudget:
    """Process-global, projection-aware runtime coordinator.

    The active limit is configurable because competition limits can change.
    Every agent in this Python process shares ``_PROCESS_STARTED_AT`` and the
    completed-game duration history. The projection is intentionally
    conservative and can be calibrated through environment variables.
    """

    def __init__(self, config: Config, game_id: str = "unknown") -> None:
        self.config = config
        self.started_at = _PROCESS_STARTED_AT
        self.game_started_at = time.monotonic()
        self.game_id = game_id or f"agent-{id(self)}"
        self._completion_marked = False
        with _RUNTIME_LOCK:
            _RUNTIME_REGISTERED_GAMES.add(self.game_id)

    def elapsed_sec(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def remaining_sec(self) -> float | None:
        if self.config.session_limit_sec <= 0:
            return None
        return max(0.0, self.config.session_limit_sec - self.elapsed_sec())

    def mark_game_complete(self) -> None:
        if self._completion_marked:
            return
        duration = max(0.0, time.monotonic() - self.game_started_at)
        with _RUNTIME_LOCK:
            if self.game_id not in _RUNTIME_COMPLETED_GAMES:
                _RUNTIME_COMPLETED_GAMES.add(self.game_id)
                _RUNTIME_GAME_DURATIONS_SEC.append(duration)
        self._completion_marked = True

    def completed_games(self) -> int:
        with _RUNTIME_LOCK:
            return len(_RUNTIME_COMPLETED_GAMES)

    def measured_game_p95_sec(self) -> float:
        with _RUNTIME_LOCK:
            samples = list(_RUNTIME_GAME_DURATIONS_SEC)
        if len(samples) < max(1, self.config.runtime_projection_min_samples):
            return max(1.0, self.config.runtime_game_p95_fallback_sec)
        measured = _percentile(samples, 0.95)
        return max(1.0, measured or self.config.runtime_game_p95_fallback_sec)

    def projected_remaining_sec(self) -> float | None:
        if self.config.session_limit_sec <= 0:
            return None
        remaining_games = max(
            0, self.config.runtime_expected_games - self.completed_games())
        parallelism = max(1, self.config.runtime_effective_game_parallelism)
        return remaining_games * self.measured_game_p95_sec() / parallelism

    def projected_safety_margin_ratio(self) -> float | None:
        remaining = self.remaining_sec()
        projected = self.projected_remaining_sec()
        if remaining is None or projected is None or self.config.session_limit_sec <= 0:
            return None
        margin = remaining - self.config.finalization_reserve_sec - projected
        return margin / max(1.0, float(self.config.session_limit_sec))

    def usable_ratio(self) -> float | None:
        remaining = self.remaining_sec()
        if remaining is None or self.config.session_limit_sec <= 0:
            return None
        usable = max(0.0, remaining - self.config.finalization_reserve_sec)
        denominator = max(1.0, self.config.session_limit_sec -
                          self.config.finalization_reserve_sec)
        return usable / denominator

    def in_finalization_reserve(self) -> bool:
        remaining = self.remaining_sec()
        return remaining is not None and remaining <= self.config.finalization_reserve_sec

    def tier(self) -> str:
        forced = self.config.force_runtime_tier
        if forced in {"A5", "A7", "A8", "A9"}:
            return forced
        ratio = self.usable_ratio()
        projected_margin = self.projected_safety_margin_ratio()
        if ratio is None:
            return "A9"
        if self.in_finalization_reserve():
            return "A5"
        if ratio <= self.config.runtime_a5_ratio or (projected_margin is not None and projected_margin <= 0.0):
            return "A5"
        if (
            ratio <= self.config.runtime_a7_ratio
            or (projected_margin is not None and projected_margin < self.config.runtime_required_margin_ratio)
        ):
            return "A7"
        return "A9"


class DiagnosticTraceLogger:
    """Bounded in-memory diagnostics with optional JSONL persistence.

    The logger is deliberately outside the action policy. It implements the
    proposal's Binet-style acquisition diagnostics without influencing actions.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.records: deque[DiagnosticTraceRecord] = deque(
            maxlen=config.trace_max_records)
        self._flush_index = 0

    def record(self, record: DiagnosticTraceRecord) -> None:
        if self.config.trace_enabled:
            self.records.append(record)

    def flush(self, game_id: str = "unknown") -> str | None:
        if not (self.config.trace_enabled and self.config.trace_to_disk and self.records):
            return None
        directory = Path(self.config.trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_game_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", game_id or "unknown")
        path = directory / \
            f"trace_{safe_game_id}_{int(time.time())}_{self._flush_index}.jsonl"
        self._flush_index += 1
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record),
                             separators=(",", ":"), default=str) + "\n")
        return str(path)


def _boundary_aware_patch_hash(grid: np.ndarray, x: int, y: int, radius: int = 1) -> str:
    """Compact local signature adapted from the user's lightweight baseline."""

    size = 2 * radius + 1
    patch = np.full((size, size), -1, dtype=np.int16)
    h, w = grid.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    px0, py0 = x0 - (x - radius), y0 - (y - radius)
    patch[py0: py0 + (y1 - y0), px0: px0 + (x1 - x0)] = grid[y0:y1, x0:x1]
    return hashlib.blake2b(patch.tobytes(), digest_size=10).hexdigest()


class GridInspector:
    """Read-only numpy-backed inspection API used for zero-action reasoning."""

    def __init__(
        self,
        grid: np.ndarray,
        previous: np.ndarray | None = None,
        frames: Sequence[np.ndarray] = (),
    ) -> None:
        self._grid = np.ascontiguousarray(grid)
        self._previous = None if previous is None else np.ascontiguousarray(
            previous)
        self._frames = tuple(np.ascontiguousarray(x) for x in frames)

    def grid(self) -> np.ndarray:
        return self._grid.copy()

    def shape(self) -> tuple[int, int]:
        return int(self._grid.shape[0]), int(self._grid.shape[1])

    def find_color(self, value: int) -> list[Point]:
        ys, xs = np.where(self._grid == int(value))
        return [(int(x), int(y)) for y, x in zip(ys, xs)]

    def count(self, value: int) -> int:
        return int(np.count_nonzero(self._grid == int(value)))

    def patch(self, x: int, y: int, radius: int = 1) -> np.ndarray:
        x0 = max(0, int(x) - radius)
        x1 = min(self._grid.shape[1], int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(self._grid.shape[0], int(y) + radius + 1)
        return self._grid[y0:y1, x0:x1].copy()

    def diff_last_frame(self) -> np.ndarray:
        if self._previous is None or self._previous.shape != self._grid.shape:
            return np.zeros_like(self._grid)
        return self._grid.astype(np.int32) - self._previous.astype(np.int32)

    def changed_cells(self) -> list[Point]:
        if self._previous is None or self._previous.shape != self._grid.shape:
            return []
        ys, xs = np.where(self._grid != self._previous)
        return [(int(x), int(y)) for y, x in zip(ys, xs)]

    def color_histogram(self) -> dict[int, int]:
        values, counts = np.unique(self._grid, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def animation_length(self) -> int:
        return len(self._frames)

    def animation_changed_counts(self) -> list[int]:
        if not self._frames:
            return []
        left = self._previous if self._previous is not None else self._frames[0]
        counts: list[int] = []
        for frame in self._frames:
            counts.append(int(np.count_nonzero(frame != left))
                          if frame.shape == left.shape else int(frame.size))
            left = frame
        return counts

    def transient_cells(self) -> list[Point]:
        if not self._frames or self._previous is None or self._previous.shape != self._grid.shape:
            return []
        union = np.zeros_like(self._grid, dtype=bool)
        left = self._previous
        for frame in self._frames:
            union |= frame != left
            left = frame
        transient = union & (self._grid == self._previous)
        ys, xs = np.where(transient)
        return [(int(x), int(y)) for y, x in zip(ys, xs)]

    def connected_components(self, color: int | None = None) -> list[dict[str, Any]]:
        components = []
        h, w = self._grid.shape
        seen = set()
        for y in range(h):
            for x in range(w):
                c = int(self._grid[y, x])
                if (x, y) in seen: continue
                if color is not None and c != int(color): continue
                if color is None and c == 0: continue
                
                stack = [(x, y)]
                comp_points = []
                while stack:
                    cx, cy = stack.pop()
                    if (cx, cy) in seen: continue
                    if int(self._grid[cy, cx]) != c: continue
                    seen.add((cx, cy))
                    comp_points.append((cx, cy))
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            stack.append((nx, ny))
                
                xs = [p[0] for p in comp_points]
                ys = [p[1] for p in comp_points]
                components.append({
                    "color": c,
                    "size": len(comp_points),
                    "centroid": (sum(xs)//len(xs), sum(ys)//len(ys)),
                    "bbox": (min(xs), min(ys), max(xs), max(ys))
                })
        return components

    def delta_summary(self) -> str:
        if self._previous is None or self._previous.shape != self._grid.shape:
            return "No previous frame or shape mismatch."
        
        diff = self._grid != self._previous
        changed_count = int(np.count_nonzero(diff))
        if changed_count == 0:
            return "No changes."
            
        prev_hist = {int(v): int(c) for v, c in zip(*np.unique(self._previous, return_counts=True))}
        curr_hist = self.color_histogram()
        
        color_changes = []
        for c in sorted(set(prev_hist.keys()) | set(curr_hist.keys())):
            delta = curr_hist.get(c, 0) - prev_hist.get(c, 0)
            if delta != 0:
                color_changes.append(f"Color {c}: {'+' if delta>0 else ''}{delta}")
                
        ys, xs = np.where(diff)
        bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
        
        return f"{changed_count} cells changed. Bounding box of changes: {bbox}. Net color shifts: {', '.join(color_changes)}"

    def symmetry_hints(self) -> dict[str, bool]:
        h, w = self._grid.shape
        hints = {}
        hints["horizontal"] = bool(np.array_equal(self._grid, np.fliplr(self._grid)))
        hints["vertical"] = bool(np.array_equal(self._grid, np.flipud(self._grid)))
        if h == w:
            hints["diagonal_main"] = bool(np.array_equal(self._grid, self._grid.T))
        return hints

    def mechanism_snapshot(self, recent_transitions: list[dict[str, Any]] | None = None, controlled_entity_shift: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._previous is None or self._previous.shape != self._grid.shape:
            return {"error": "no_previous_frame"}
            
        diff = self._grid != self._previous
        changed_count = int(np.count_nonzero(diff))
        if changed_count == 0:
            return {"status": "no_changes"}
            
        prev_hist = {int(v): int(c) for v, c in zip(*np.unique(self._previous, return_counts=True))}
        curr_hist = self.color_histogram()
        
        pure_translation = (prev_hist == curr_hist)
        
        color_deltas = {}
        dominant_shift_color = -1
        max_shift = 0
        for c in set(prev_hist.keys()) | set(curr_hist.keys()):
            delta = curr_hist.get(c, 0) - prev_hist.get(c, 0)
            if delta != 0:
                color_deltas[c] = delta
                if abs(delta) > abs(max_shift):
                    max_shift = delta
                    dominant_shift_color = c

        if max_shift > 0:
            dominant_color_shift = f"spawn_color_{dominant_shift_color}"
        elif max_shift < 0:
            dominant_color_shift = f"delete_color_{dominant_shift_color}"
        else:
            dominant_color_shift = "none"
        
        prev_inspector = GridInspector(self._previous)
        prev_comps = len(prev_inspector.connected_components())
        curr_comps = len(self.connected_components())
        topology_changed = (prev_comps != curr_comps)
        
        component_change_summary = {
            "prev_component_count": prev_comps,
            "curr_component_count": curr_comps,
            "component_delta": curr_comps - prev_comps,
            "dominant_changed_color": int(dominant_shift_color),
        }
        
        def _sym_score(g: np.ndarray) -> int:
            score = 0
            if np.array_equal(g, np.fliplr(g)): score += 1
            if np.array_equal(g, np.flipud(g)): score += 1
            if g.shape[0] == g.shape[1] and np.array_equal(g, g.T): score += 1
            return score
            
        prev_sym = _sym_score(self._previous)
        curr_sym = _sym_score(self._grid)
        if curr_sym > prev_sym: sym_status = "increased"
        elif curr_sym < prev_sym: sym_status = "decreased"
        else: sym_status = "unchanged"
        
        ys, xs = np.where(diff)
        bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
        bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        grid_area = self._grid.size
        change_ratio = round(changed_count / grid_area, 3)
        bbox_ratio = bbox_area / grid_area
        
        if bbox_ratio < 0.1:
            locality_hint = "highly_local"
        elif bbox_ratio < 0.4:
            locality_hint = "regional"
        elif bbox_ratio > 0.8 and change_ratio > 0.5:
            locality_hint = "global"
        elif bbox_ratio > 0.8 and change_ratio < 0.2:
            locality_hint = "diffuse"
        else:
            locality_hint = "unknown"
            
        if bbox_area <= 9 or bbox_area < grid_area * 0.15:
            scope = "local"
        elif bbox_area >= grid_area * 0.8:
            scope = "global"
        else:
            scope = "regional"
            
        prev_changed_colors = np.unique(self._previous[diff])
        curr_changed_colors = np.unique(self._grid[diff])
        is_flood_like = (len(prev_changed_colors) == 1 and len(curr_changed_colors) == 1 and changed_count > 4)
        if is_flood_like:
            scope = "flood-like"
            
        if pure_translation:
            motion_hint = "translation_like"
        elif not pure_translation and is_flood_like:
            motion_hint = "recolor_like"
        elif max_shift != 0 and abs(sum(color_deltas.values())) == abs(max_shift):
            motion_hint = "spawn_delete_like"
        else:
            motion_hint = "mixed"
            
        recent_transition_pattern = "unknown"
        if recent_transitions:
            actions = [str(t.get("action", "")) for t in recent_transitions]
            if len(set(actions)) == 1 and actions[0].startswith("ACTION"):
                if actions[0] in {"ACTION6", "ACTION7", "ACTION8", "ACTION9"}:
                    recent_transition_pattern = "movement_repeated"
                elif actions[0] in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}:
                    recent_transition_pattern = "recolor_or_delete_repeated"
                else:
                    recent_transition_pattern = "same_action_repeated"
            elif all("ACTION" in a for a in actions):
                recent_transition_pattern = "mixed"
            
        return {
            "pure_translation": pure_translation,
            "topology_change": topology_changed,
            "topology_changed": topology_changed,
            "symmetry": sym_status,
            "effect_scope": scope,
            "changed_cell_count": changed_count,
            "color_deltas": color_deltas,
            "dominant_color_shift": dominant_color_shift,
            "changed_bbox": bbox,
            "change_ratio": change_ratio,
            "locality_hint": locality_hint,
            "motion_hint": motion_hint,
            "component_change_summary": component_change_summary,
            "controlled_entity_shift": controlled_entity_shift or {"status": "unknown"},
            "recent_transition_pattern": recent_transition_pattern,
        }

    def summary(self) -> dict[str, Any]:
        hist = self.color_histogram()
        changed = self.changed_cells()
        return {
            "shape": self.shape(),
            "colors": hist,
            "changed_count": len(changed),
            "changed_bbox": _bbox(changed),
            "delta_summary": self.delta_summary(),
            "animation_frames": self.animation_length(),
            "animation_step_changes": self.animation_changed_counts(),
            "transient_count": len(self.transient_cells()),
        }


class SafeRepl:
    """Tiny expression-only REPL for future local-model adapters.

    It intentionally supports expressions, not statements. Only public methods on
    ``state`` and a small safe builtin set are available. The default deterministic
    agent does not need to execute generated code, but the interface makes the
    proposal's REPL inspection path concrete without exposing arbitrary file or OS
    access.
    """

    _ALLOWED_NODES = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.keyword,
        ast.Subscript,
        ast.Slice,
        ast.UnaryOp,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
    )

    def __init__(self, state: GridInspector) -> None:
        self.state = state

    def run(self, expression: str) -> Any:
        tree = ast.parse(expression, mode="eval")
        safe_names = {
            "state",
            "len",
            "min",
            "max",
            "sum",
            "sorted",
            "list",
            "tuple",
            "dict",
            "int",
            "float",
            "abs",
        }
        safe_state_methods = {
            "summary",
            "shape",
            "find_color",
            "count",
            "patch",
            "diff_last_frame",
            "changed_cells",
            "color_histogram",
            "animation_length",
            "animation_changed_counts",
            "transient_cells",
            "connected_components",
            "delta_summary",
            "symmetry_hints",
            "mechanism_snapshot",
        }
        for node in ast.walk(tree):
            if not isinstance(node, self._ALLOWED_NODES):
                raise ValueError(
                    f"REPL node is not allowed: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in safe_names:
                raise ValueError(f"Name is not allowed: {node.id}")
            if isinstance(node, ast.Attribute):
                # Only direct calls to an explicitly listed GridInspector method
                # are allowed. This blocks ndarray methods such as tofile(), dump(),
                # resize(), and chained attribute traversal.
                if not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "state"
                    and node.attr in safe_state_methods
                ):
                    raise ValueError(f"Attribute is not allowed: {node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in safe_names - {"state"}:
                    raise ValueError(
                        f"Callable is not allowed: {node.func.id}")
        scope = {
            "state": self.state,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "int": int,
            "float": float,
            "abs": abs,
        }
        return eval(compile(tree, "<dewma-repl>", "eval"), {"__builtins__": {}}, scope)

    def exec_python_script(self, code: str, recent_transitions: list) -> str:
        import sys
        import io
        import traceback
        
        # Convert transitions to simple dicts for the LLM script
        transitions_data = []
        for t in recent_transitions:
            transitions_data.append({
                "action": t.action.name,
                "data": t.action.data_dict,
                "before_grid": t.before_grid.tolist() if getattr(t, "before_grid", None) is not None else None,
                "after_grid": t.after_grid.tolist() if getattr(t, "after_grid", None) is not None else None
            })
            
        namespace = {"recent_transitions": transitions_data}
        
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec(code, namespace)
            output = redirected_output.getvalue()
            if len(output) > 2400:
                output = output[:2400] + "...<truncated>"
            return output if output else "Execution succeeded but printed nothing."
        except Exception:
            return f"Error executing python code:\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Perception: raw grid, components, temporal entities, confidence arbitration
# ---------------------------------------------------------------------------


def _bbox(points: Sequence[Point]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _shape_key(cells: Sequence[Point]) -> str:
    if not cells:
        return "empty"
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    normalized = sorted((x - min_x, y - min_y) for x, y in cells)
    payload = ";".join(f"{x},{y}" for x, y in normalized).encode()
    return _stable_hash_bytes(payload, 8)


def _connected_components(grid: np.ndarray, background: int) -> tuple[Component, ...]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[Component] = []
    next_id = 0
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if visited[y, x] or color == background:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            cells: list[Point] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and not visited[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            box = _bbox(cells)
            assert box is not None
            x0, y0, x1, y1 = box
            box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            centroid = (
                float(sum(px for px, _ in cells) / len(cells)),
                float(sum(py for _, py in cells) / len(cells)),
            )
            components.append(
                Component(
                    local_id=next_id,
                    color=color,
                    cells=tuple(sorted(cells)),
                    bbox=box,
                    centroid=centroid,
                    area=len(cells),
                    shape_key=_shape_key(cells),
                    touches_border=(x0 == 0 or y0 == 0 or x1 ==
                                    w - 1 or y1 == h - 1),
                    compactness=float(len(cells) / box_area),
                )
            )
            next_id += 1
    return tuple(components)


def _position_score(a: tuple[float, float], b: tuple[float, float], diag: float) -> float:
    distance = math.dist(a, b)
    return max(0.0, 1.0 - distance / max(1.0, diag))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    return inter / max(1, area_a + area_b - inter)


class TemporalEntityTracker:
    def __init__(self) -> None:
        self.entities: dict[str, TrackedEntity] = {}
        self.next_id = 0
        self.controlled_entity_id: str | None = None
        self.last_match_confidence = 0.0

    def reset(self) -> None:
        self.entities.clear()
        self.next_id = 0
        self.controlled_entity_id = None
        self.last_match_confidence = 0.0

    def _new_id(self) -> str:
        value = f"e{self.next_id}"
        self.next_id += 1
        return value

    def _match_score(
        self, entity: TrackedEntity, comp: Component, width: int, height: int
    ) -> float:
        diag = math.hypot(width, height)
        position = _position_score(entity.centroid, comp.centroid, diag)
        shape = 1.0 if entity.shape_key == comp.shape_key else _bbox_iou(
            entity.bbox, comp.bbox)
        color = 1.0 if entity.color == comp.color else 0.2
        size = min(entity.area, comp.area) / \
            max(1, max(entity.area, comp.area))
        overlap = _bbox_iou(entity.bbox, comp.bbox)
        return 0.30 * position + 0.25 * shape + 0.15 * color + 0.10 * size + 0.20 * overlap

    def update(
        self,
        components: Sequence[Component],
        width: int,
        height: int,
        pending_action: ActionSpec | None,
    ) -> tuple[tuple[TrackedEntity, ...], tuple[tuple[str, int, int], ...], float]:
        previous = list(self.entities.values())
        possible: list[tuple[float, str, int]] = []
        for entity in previous:
            for comp in components:
                score = self._match_score(entity, comp, width, height)
                possible.append((score, entity.entity_id, comp.local_id))
        possible.sort(reverse=True)

        used_entities: set[str] = set()
        used_components: set[int] = set()
        matches: dict[int, tuple[str, float]] = {}
        for score, entity_id, comp_id in possible:
            if score < 0.32 or entity_id in used_entities or comp_id in used_components:
                continue
            used_entities.add(entity_id)
            used_components.add(comp_id)
            matches[comp_id] = (entity_id, score)

        current: dict[str, TrackedEntity] = {}
        moves: list[tuple[str, int, int]] = []
        confidence_values: list[float] = []
        for comp in components:
            match = matches.get(comp.local_id)
            if match is None:
                entity = TrackedEntity(
                    entity_id=self._new_id(),
                    color=comp.color,
                    cells=comp.cells,
                    bbox=comp.bbox,
                    centroid=comp.centroid,
                    area=comp.area,
                    shape_key=comp.shape_key,
                    persistence_confidence=0.35,
                )
            else:
                entity_id, score = match
                old = self.entities[entity_id]
                dx = int(round(comp.centroid[0] - old.centroid[0]))
                dy = int(round(comp.centroid[1] - old.centroid[1]))
                if dx or dy:
                    moves.append((entity_id, dx, dy))
                entity = TrackedEntity(
                    entity_id=entity_id,
                    color=comp.color,
                    cells=comp.cells,
                    bbox=comp.bbox,
                    centroid=comp.centroid,
                    area=comp.area,
                    shape_key=comp.shape_key,
                    age=old.age + 1,
                    missing_frames=0,
                    persistence_confidence=min(
                        1.0, 0.65 * old.persistence_confidence + 0.35 * score),
                    controllability=old.controllability,
                    autonomous_motion=old.autonomous_motion,
                    last_move=(dx, dy),
                    affordances=set(old.affordances),
                )
                confidence_values.append(score)
            current[entity.entity_id] = entity

        # Keep unmatched entities for one frame to tolerate occlusion, but do not
        # expose them as current visual objects.
        for old in previous:
            if old.entity_id in used_entities:
                continue
            if old.missing_frames < 1:
                old.missing_frames += 1
                old.persistence_confidence *= 0.65

        # Infer controllability from action-correlated movement. No fixed color is assumed.
        if pending_action is not None and not pending_action.data:
            moved_ids = {entity_id for entity_id, dx, dy in moves if dx or dy}
            if len(moved_ids) == 1:
                moved_id = next(iter(moved_ids))
                for entity in current.values():
                    if entity.entity_id == moved_id:
                        entity.controllability = min(
                            1.0, entity.controllability + 0.22)
                    elif entity.last_move != (0, 0):
                        entity.autonomous_motion = min(
                            1.0, entity.autonomous_motion + 0.10)
            elif len(moved_ids) > 1:
                for moved_id in moved_ids:
                    current[moved_id].autonomous_motion = min(
                        1.0, current[moved_id].autonomous_motion + 0.05
                    )

        if current:
            best = max(current.values(), key=lambda e: e.controllability)
            if best.controllability >= 0.42:
                self.controlled_entity_id = best.entity_id

        # Basic affordance hypotheses. They remain hypotheses until transitions support them.
        for entity in current.values():
            if entity.controllability >= 0.42:
                entity.affordances.add("controllable")
            if entity.area <= 9:
                entity.affordances.add("possibly_collectible_or_button")
            if entity.bbox[0] == 0 or entity.bbox[1] == 0 or entity.bbox[2] == width - 1 or entity.bbox[3] == height - 1:
                entity.affordances.add("boundary_related")

        self.entities = current
        if components:
            matched_fraction = len(matches) / max(1, len(components))
            mean_score = sum(confidence_values) / \
                max(1, len(confidence_values))
            self.last_match_confidence = 0.55 * matched_fraction + 0.45 * mean_score
        else:
            self.last_match_confidence = 1.0
        return tuple(current.values()), tuple(moves), self.last_match_confidence


class PerceptionSystem:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.tracker = TemporalEntityTracker()
        self.previous_scene: Scene | None = None
        self.latest_entity_moves: tuple[tuple[str, int, int], ...] = ()

    def reset(self) -> None:
        self.tracker.reset()
        self.previous_scene = None
        self.latest_entity_moves = ()

    def perceive(
        self,
        grid: np.ndarray,
        level: int,
        pending_action: ActionSpec | None,
    ) -> Scene:
        previous_grid = None if self.previous_scene is None else self.previous_scene.grid
        if previous_grid is not None and previous_grid.shape == grid.shape:
            delta_mask = grid != previous_grid
        else:
            delta_mask = np.zeros_like(grid, dtype=bool)
        ys, xs = np.where(delta_mask)
        changed_cells = tuple((int(x), int(y)) for y, x in zip(ys, xs))
        change_ratio = float(np.count_nonzero(delta_mask) / max(1, grid.size))

        values, counts = np.unique(grid, return_counts=True)
        color_counts = tuple(sorted((int(v), int(c))
                             for v, c in zip(values, counts)))
        background = int(values[int(np.argmax(counts))]) if len(values) else 0
        components = _connected_components(
            grid, background) if self.config.enable_entities else ()
        component_overload = len(components) > self.config.max_track_components

        # Dense cellular/texture fields can create thousands of tiny connected
        # components. Quadratic entity matching would waste the notebook budget
        # and produce meaningless identities, so temporal tracking is suspended
        # immediately while the raw grid and frame differences remain intact.
        if component_overload:
            self.tracker.reset()
            entities: tuple[TrackedEntity, ...] = ()
            moves: tuple[tuple[str, int, int], ...] = ()
            match_confidence = 0.0
        else:
            entities, moves, match_confidence = self.tracker.update(
                components, int(grid.shape[1]), int(
                    grid.shape[0]), pending_action
            )
        self.latest_entity_moves = moves

        # High-entropy widespread change is a strong signal that object-centric
        # parsing is unreliable (cellular automata, fluids, overlays, etc.).
        if changed_cells:
            changed_values = grid[delta_mask]
            _, changed_counts = np.unique(changed_values, return_counts=True)
            probabilities = changed_counts / max(1, changed_counts.sum())
            entropy = float(-np.sum(probabilities *
                            np.log2(probabilities + 1e-12)))
            max_entropy = math.log2(max(2, len(changed_counts)))
            normalized_entropy = min(1.0, entropy / max(1e-9, max_entropy))
        else:
            normalized_entropy = 0.0

        count_stability = 1.0
        if self.previous_scene is not None:
            prev_count = len(self.previous_scene.components)
            count_stability = 1.0 - \
                min(1.0, abs(len(components) - prev_count) /
                    max(1, prev_count + len(components)))

        change_penalty = min(1.0, change_ratio * 5.0)
        entity_confidence = max(
            0.0,
            min(
                1.0,
                0.50 * match_confidence
                + 0.20 * count_stability
                + 0.15 * (1.0 - normalized_entropy)
                + 0.15 * (1.0 - change_penalty),
            ),
        )
        if self.previous_scene is None:
            entity_confidence = 0.62 if components else 0.2
        if component_overload:
            entity_confidence = 0.0
        field_mode = (
            component_overload
            or not self.config.enable_entities
            or entity_confidence < self.config.entity_confidence_threshold
            or (change_ratio > 0.18 and normalized_entropy > 0.55)
        )

        exact_key = _stable_hash_bytes(grid.tobytes())
        abstract_payload = {
            "shape": tuple(int(v) for v in grid.shape),
            "counts": sorted(c for _, c in color_counts),
            "components": sorted(
                (c.area, c.shape_key, c.touches_border) for c in components
            )[:96],
            "field": field_mode,
        }
        abstract_key = _stable_hash_bytes(
            json.dumps(abstract_payload, sort_keys=True).encode(), 10
        )

        scene = Scene(
            grid=np.ascontiguousarray(grid),
            previous_grid=previous_grid,
            delta_mask=np.ascontiguousarray(delta_mask),
            changed_cells=changed_cells,
            exact_key=exact_key,
            abstract_key=abstract_key,
            background=background,
            color_counts=color_counts,
            components=tuple(components),
            entities=tuple(entities),
            controlled_entity_id=self.tracker.controlled_entity_id,
            entity_confidence=entity_confidence,
            field_mode=field_mode,
            change_ratio=change_ratio,
            delta_entropy=normalized_entropy,
            level=level,
        )
        self.previous_scene = scene
        return scene


# ---------------------------------------------------------------------------
# Event extraction and memories
# ---------------------------------------------------------------------------


def _changed_bbox(changed: Sequence[Point]) -> tuple[int, int, int, int] | None:
    return _bbox(changed)


def _event_from_scenes(
    before: Scene,
    after: Scene,
    entity_moves: Sequence[tuple[str, int, int]],
    level_delta: int,
    state: GameState,
    animation: AnimationTrace | None = None,
) -> Event:
    before_colors = {v for v, c in before.color_counts if c > 0}
    after_colors = {v for v, c in after.color_counts if c > 0}
    changed_count = len(after.changed_cells)
    animation = animation or AnimationTrace(
        frame_count=1,
        step_changed_counts=(changed_count,),
        cumulative_changed_count=changed_count,
        persistent_changed_count=changed_count,
        transient_changed_count=0,
        oscillating_cell_count=0,
        changed_bboxes=(_changed_bbox(after.changed_cells),),
        color_count_trajectories=(),
        motion_vectors=(),
        settled_stable=True,
        temporal_signature="",
    )
    topology_change = (
        abs(len(after.components) - len(before.components)) >= 1
        or before.background != after.background
        or changed_count > max(8, int(after.grid.size * 0.04))
        or animation.transient_changed_count > max(8, int(after.grid.size * 0.02))
    )
    state_name = getattr(state, "name", str(state))
    signature_payload = {
        "n": min(changed_count, 9999),
        "bbox": _changed_bbox(after.changed_cells),
        "ld": level_delta,
        "moves": sorted((dx, dy) for _, dx, dy in entity_moves),
        "appear": sorted(after_colors - before_colors),
        "disappear": sorted(before_colors - after_colors),
        "topology": topology_change,
        "state": state_name,
        "animation": animation.temporal_signature,
    }
    effect_signature = _stable_hash_bytes(
        json.dumps(signature_payload, sort_keys=True).encode(), 8
    )
    settled_no_op = changed_count == 0 and level_delta == 0
    # A transient animation is evidence: do not label it a no-op merely because
    # the final grid returned to the starting configuration.
    no_op = (
        animation.cumulative_changed_count == 0
        and changed_count == 0
        and level_delta == 0
        and "GAME_OVER" not in state_name
        and "WIN" not in state_name
    )
    return Event(
        changed_count=changed_count,
        changed_bbox=_changed_bbox(after.changed_cells),
        no_op=no_op,
        level_delta=level_delta,
        game_over=("GAME_OVER" in state_name),
        win=(state_name == "WIN"),
        entity_moves=tuple(entity_moves),
        appeared_colors=tuple(sorted(after_colors - before_colors)),
        disappeared_colors=tuple(sorted(before_colors - after_colors)),
        topology_change=topology_change,
        effect_signature=effect_signature,
        subframe_count=animation.frame_count,
        cumulative_changed_count=animation.cumulative_changed_count,
        transient_changed_count=animation.transient_changed_count,
        oscillating_cell_count=animation.oscillating_cell_count,
        animation_vectors=animation.motion_vectors,
        temporal_signature=animation.temporal_signature,
        settled_no_op=settled_no_op,
        animation_steps=animation.steps,
    )


class TraceMemory:
    def __init__(self, maxlen: int = 512) -> None:
        self.transitions: deque[Transition] = deque(maxlen=maxlen)
        self.visits: Counter[str] = Counter()
        self.actions_by_state: dict[str, Counter[tuple[str,
                                                       tuple[tuple[str, int], ...]]]] = defaultdict(Counter)
        self.global_action_outcomes: dict[str,
                                          Counter[str]] = defaultdict(Counter)
        self.progress_actions: dict[str, list[ActionSpec]] = defaultdict(list)
        self.no_op_streak = 0
        self.same_family_streak = 0
        self.probes_this_level = 0
        self.total_mental_eliminations = 0
        self.spatial_visits: dict[tuple[int, int], int] = defaultdict(int)
        self.spatial_visits_by_action: dict[tuple[str, int, int], int] = defaultdict(int)
        self.escape_hatch_used = False
        self.llm_family_payoff: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.llm_follow_through_window: int = 0
        self.llm_follow_through_family: str = ""
        self.llm_follow_through_coords: tuple[int, int] | None = None
        self.llm_recent_committed_family: str = ""
        self.llm_family_cooldown: dict[str, int] = defaultdict(int)
        self.llm_promoted_program_ids: set[str] = set()
        self.llm_follow_through_recent_steps: int = 0
        self.follow_through_led_to_progress: bool = False
        self.level_steps: int = 0
        self.current_level_index: int = 0
        self.level_progress_events: int = 0
        self.macro_replay_queue: deque[ActionSpec] = deque()
        self.macro_replay_family: str = ""
        self.macro_replay_program_id: str = ""
        self.macro_replay_aborted_reason: str = ""
        self.transferred_mechanism_family: str = ""
        self.level_transition_transfer_used: bool = False
        self.early_classified_mechanism: str = ""
        self.counterfactual_pruned_count: int = 0
        self.negative_hypothesis_eliminations: int = 0
        self.mode: str = "discovery_mode"
        self.mechanism_scores: dict[str, float] = {
            "movement_control": 0.2, "targeted_recolor": 0.1, "component_delete": 0.05,
            "drag_or_push": 0.05, "line_or_beam": 0.05, "flood_or_fill": 0.05,
            "gravity_or_fall": 0.05, "copy_or_stamp": 0.05, "count_or_trigger": 0.05,
            "topology_switch": 0.05
        }
        self.top_mechanism_family: str = "movement_control"
        self.top_mechanism_confidence: float = 0.2
        self.competing_mechanism_families: list[str] = ["movement_control", "targeted_recolor"]
        self.priority_program_families: list[str] = ["translation", "conditional_recolor"]
        self.invariants_to_preserve: list[str] = []
        self.mechanism_shift_event: bool = False
        self.recommended_probe_type: str = ""
        self.recommended_probe_ttl: int = 0
        self.llm_last_mechanism_confidence: float = 0.5
        self.priority_program_ttl: int = 0
        self.last_winning_action_name: str = ""
        self.last_winning_program_kind: str = ""
        self.last_winning_family: str = ""
        self.last_winning_coords: tuple[int, int] | None = None
        self.last_winning_component_color: int | None = None
        self.last_winning_component_area: int = 0
        self.last_winning_component_aspect: float = 1.0
        self.last_winning_component_shape_hash: str = ""
        self.last_winning_source: str = ""
        self.last_winning_mechanism_family: str = ""
        self.last_winning_mechanism_confidence: float = 0.0
        self.last_winning_step_in_level: int = 0
        self.post_breakthrough_window_active: bool = False
        self.post_breakthrough_window_steps_remaining: int = 0
        self.transferred_winning_family: str = ""
        self.transferred_winning_program_kind: str = ""
        self.transferred_winning_action_name: str = ""
        self.transferred_winning_coords: tuple[int, int] | None = None
        self.post_breakthrough_aborted_reason: str = ""
        self.post_breakthrough_bias_used: bool = False
        self.post_breakthrough_attempts: int = 0
        self.post_breakthrough_effective_attempts: int = 0
        self.post_breakthrough_noop_attempts: int = 0
        self.post_breakthrough_failed_attempts: int = 0
        self.post_breakthrough_continuation_bias: float = 0.0
        self.post_breakthrough_local_search_used: bool = False
        self.counterfactual_streak_renewed: bool = False
        self.mechanism_collapse_breaker_used: bool = False
        self.mechanism_family_penalized: str = ""
        self.mechanism_family_promoted: str = ""
        self.component_delete_component_locked: bool = False
        self.line_beam_structured_candidates_used: bool = False
        self.counterfactual_completion_bias: float = 0.0
        self.terminal_condition_bonus: float = 0.0
        self.mechanism_completion_bonus: float = 0.0
        self.productive_search_convergence_pressure: float = 0.0
        self.line_beam_closure_bias_used: bool = False
        self.component_delete_payoff_bias_used: bool = False
        self.atomic_drag_drop_paired: bool = False
        self.sequential_component_sweep_active: bool = False
        self.orthogonal_collision_steer_used: bool = False
        self.cell_cycle_persistence_used: bool = False
        self.active_sweep_component_id: int | None = None
        self.active_sweep_clicks_remaining: int = 0
        self.visited_sweep_component_anchors: set[tuple[int, int]] = set()
        self.recommended_orthogonal_turn: str = ""
        self.last_changed_coord: tuple[int, int] | None = None
        self.coord_cycle_clicks_remaining: int = 0
        self.near_terminal_finish_mode_active: bool = False
        self.near_terminal_finish_steps_remaining: int = 0
        self.near_terminal_finish_family: str = ""
        self.near_terminal_finish_source: str = ""
        self.near_terminal_finish_trigger_reason: str = ""
        self.near_terminal_finish_exit_reason: str = ""
        self.productive_branch_commitment_used: bool = False
        self.consecutive_completion_signals: int = 0
        self.finish_mode_noop_streak: int = 0
        self.near_terminal_finish_gate_family_consensus: bool = False
        self.near_terminal_finish_gate_branch_consensus: bool = False
        self.near_terminal_finish_gate_completion_stability: bool = False
        self.near_terminal_finish_gate_allowed: bool = False
        self.finish_branch_continuation_family: str = ""
        self.finish_branch_continuation_steps: int = 0
        self.finish_branch_continuation_kept_control: bool = False
        self.finish_branch_continuation_break_reason: str = ""
        self.finish_family_collision_suppressed: bool = False
        self.finish_family_collision_suppressed_family: str = ""
        self.finish_family_collision_suppressed_until_step: int = 0
        self.finish_mode_preempted_counterfactual: bool = False
        self.finish_mode_preempt_block_reason: str = ""
        self.productive_branch_signature: str = ""
        self.productive_branch_source: str = ""
        self.productive_branch_action_name: str = ""
        self.productive_branch_anchor: tuple[int, int] | None = None
        self.productive_branch_family_hint: str = ""
        self.productive_branch_program_kind: str = ""
        self.productive_branch_streak: int = 0
        self.productive_branch_last_effective_step: int = 0
        self.productive_branch_family_wobble_tolerated: bool = False
        self.productive_branch_family_wobble_reason: str = ""
        self.structured_branch_persistence_used: bool = False
        self.structured_branch_persistence_steps: int = 0
        self.structured_branch_persistence_steps_remaining: int = 0
        self.structured_branch_persistence_kept_control: bool = False
        self.structured_branch_persistence_break_reason: str = ""
        self.finish_bound_branch_signature: str = ""
        self.finish_bound_branch_source: str = ""
        self.finish_bound_branch_anchor: tuple[int, int] | None = None
        self.finish_bound_branch_action_name: str = ""
        self.finish_bound_branch_kept_control: bool = False
        self.post_breakthrough_priority_preserved: bool = False
        self.post_breakthrough_preempt_block_reason: str = ""
        self.post_breakthrough_local_branch_reused: bool = False
        self.post_breakthrough_local_branch_break_reason: str = ""
        self.productive_branch_collision_recovery_used: bool = False
        self.productive_branch_collision_recovery_variant: str = ""
        self.productive_branch_collision_recovery_preserved_family: bool = False
        self.productive_window_extended: bool = False
        self.productive_window_extension_reason: str = ""
        self.productive_branch_preempt_blocked: bool = False
        self.productive_branch_preempt_block_reason: str = ""
        self.collision_soft_retry_used: bool = False
        self.collision_soft_retry_variant: str = ""
        self.collision_soft_retry_count: int = 0
        self.finish_family_collision_counts: dict[str, int] = {}
        self.recent_completion_scores: list[float] = []
        self.recent_mechanism_families: list[str] = []
        self.recent_action_sources: list[str] = []
        self.family_stagnation_steps: dict[str, int] = {}
        self.regrounded_winning_coords: tuple[int, int] | None = None
        self.regrounding_delta: tuple[int, int] | None = None
        self.regrounding_confidence: float = 0.0
        self.regrounding_used: bool = False
        self.regrounding_failed_reason: str = ""
        self.levelup_relocalization_llm_attempted: bool = False
        self.levelup_relocalization_llm_used: bool = False
        self.levelup_relocalization_llm_confidence: float = 0.0
        self.exploitation_noop_blacklist: set[tuple[str, int, int]] = set()
        self.exploitation_noop_neighborhood_blacklist: set[tuple[int, int]] = set()
        self.exploitation_noop_blacklist_checks: int = 0
        self.exploitation_noop_blacklist_hits: int = 0
        self.mechanism_detector_triggered: str = ""

    def reset_level(self) -> None:
        self.transitions.clear()
        self.visits.clear()
        self.actions_by_state.clear()
        self.progress_actions.clear()
        self.no_op_streak = 0
        self.same_family_streak = 0
        self.probes_this_level = 0
        self.spatial_visits.clear()
        self.spatial_visits_by_action.clear()
        self.escape_hatch_used = False
        self.llm_follow_through_window = 0
        self.llm_follow_through_family = ""
        self.llm_follow_through_coords = None
        self.llm_recent_committed_family = ""
        self.llm_family_cooldown.clear()
        self.llm_promoted_program_ids.clear()
        self.llm_follow_through_recent_steps = 0
        self.follow_through_led_to_progress = False
        self.level_steps = 0
        self.current_level_index += 1
        self.level_progress_events = 0
        self.macro_replay_queue.clear()
        self.macro_replay_family = ""
        self.macro_replay_program_id = ""
        self.macro_replay_aborted_reason = ""
        # Seed post-breakthrough exploitation prior from stored winning pattern
        if self.last_winning_family or self.last_winning_action_name or self.last_winning_program_kind:
            self.post_breakthrough_window_active = True
            # Fast early breakthrough (<= 50 steps) grants a longer 16-step window, otherwise 10 steps
            self.post_breakthrough_window_steps_remaining = 16 if (0 < self.last_winning_step_in_level <= 50) else 10
            self.transferred_winning_family = self.last_winning_family or self.last_winning_mechanism_family or self.top_mechanism_family
            self.transferred_winning_program_kind = self.last_winning_program_kind
            self.transferred_winning_action_name = self.last_winning_action_name
            self.transferred_winning_coords = self.last_winning_coords
            self.transferred_mechanism_family = self.transferred_winning_family
            self.level_transition_transfer_used = True
            self.post_breakthrough_aborted_reason = ""
        elif self.top_mechanism_confidence >= 0.35 and self.top_mechanism_family:
            self.transferred_mechanism_family = self.top_mechanism_family
            self.level_transition_transfer_used = True
            self.post_breakthrough_window_active = False
            self.post_breakthrough_window_steps_remaining = 0
            self.post_breakthrough_aborted_reason = ""
        else:
            self.transferred_mechanism_family = "undetermined"
            self.level_transition_transfer_used = False
            self.post_breakthrough_window_active = False
            self.post_breakthrough_window_steps_remaining = 0
            self.post_breakthrough_aborted_reason = ""

        self.early_classified_mechanism = self.transferred_mechanism_family if self.transferred_mechanism_family != "undetermined" else ""
        self.counterfactual_pruned_count = 0
        self.negative_hypothesis_eliminations = 0
        self.mode = "exploitation_mode" if self.post_breakthrough_window_active else "discovery_mode"
        self.mechanism_scores = {
            "movement_control": 0.35 if self.transferred_mechanism_family == "movement_control" else 0.08,
            "targeted_recolor": 0.35 if self.transferred_mechanism_family == "targeted_recolor" else 0.08,
            "component_delete": 0.35 if self.transferred_mechanism_family == "component_delete" else 0.05,
            "drag_or_push": 0.35 if self.transferred_mechanism_family == "drag_or_push" else 0.05,
            "line_or_beam": 0.35 if self.transferred_mechanism_family == "line_or_beam" else 0.05,
            "flood_or_fill": 0.35 if self.transferred_mechanism_family == "flood_or_fill" else 0.05,
            "gravity_or_fall": 0.35 if self.transferred_mechanism_family == "gravity_or_fall" else 0.05,
            "copy_or_stamp": 0.35 if self.transferred_mechanism_family == "copy_or_stamp" else 0.05,
            "count_or_trigger": 0.35 if self.transferred_mechanism_family == "count_or_trigger" else 0.05,
            "topology_switch": 0.35 if self.transferred_mechanism_family == "topology_switch" else 0.05,
        }
        self.top_mechanism_family = max(self.mechanism_scores, key=self.mechanism_scores.get)
        self.top_mechanism_confidence = 0.40 if self.post_breakthrough_window_active else 0.20
        self.competing_mechanism_families = [self.top_mechanism_family]
        if self.transferred_winning_program_kind:
            self.priority_program_families = [self.transferred_winning_program_kind]
            self.priority_program_ttl = self.post_breakthrough_window_steps_remaining
        else:
            self.priority_program_families = []
            self.priority_program_ttl = 0
        self.invariants_to_preserve.clear()
        self.visited_sweep_component_anchors.clear()
        self.mechanism_shift_event = False
        self.recommended_probe_type = ""
        self.recommended_probe_ttl = 0
        self.llm_last_mechanism_confidence = 0.5

    def reset_attempt(self) -> None:
        # Preserve learned transitions, state-action counts, global outcomes, and
        # the level-wide probe budget across death/restart attempts.
        self.no_op_streak = 0

    def record(self, transition: Transition, was_probe: bool) -> None:
        self.transitions.append(transition)
        self.visits[transition.after_exact] += 1
        
        if transition.action.data:
            data_dict = dict(transition.action.data)
            x, y = data_dict.get("x"), data_dict.get("y")
            if x is not None and y is not None:
                self.spatial_visits[(x, y)] += 1
                self.spatial_visits_by_action[(transition.action.name, x, y)] += 1
        
        # Track same family streak to detect boredom/collapse
        if len(self.transitions) > 1 and transition.event.level_delta == 0:
            last = self.transitions[-2]
            if last.action.name == transition.action.name:
                self.same_family_streak += 1
            else:
                self.same_family_streak = 0
        else:
            self.same_family_streak = 0
        self.actions_by_state[transition.before_exact][transition.action.key] += 1
        outcome = "progress" if transition.event.level_delta > 0 else (
            "death" if transition.event.game_over else (
                "noop" if transition.event.no_op else "change"
            )
        )
        self.global_action_outcomes[transition.action.name][outcome] += 1
        if transition.event.level_delta > 0 or transition.event.win:
            self.progress_actions[transition.before_exact].append(
                transition.action)
        self.no_op_streak = self.no_op_streak + 1 if transition.event.no_op else 0
        if was_probe:
            self.probes_this_level += 1

    def repeated_effect_signature(self, window: int = 8) -> tuple[str, int] | None:
        if len(self.transitions) < 2:
            return None
        recent = list(self.transitions)[-window:]
        counts: Counter[str] = Counter(
            t.event.effect_signature for t in recent if t.event.effect_signature
        )
        if not counts:
            return None
        signature, support = counts.most_common(1)[0]
        return (signature, support) if support >= 3 else None

    def recent_death_count(self, window: int = 10) -> int:
        recent = list(self.transitions)[-window:]
        return sum(1 for t in recent if t.event.game_over)

    def recent_action_pattern(self, window: int = 8) -> str:
        recent = list(self.transitions)[-window:]
        if not recent:
            return "none"
        actions = [t.action.name for t in recent]
        if len(set(actions)) == 1:
            return f"single_action:{actions[0]}"
        signatures = [t.event.effect_signature for t in recent if t.event.effect_signature]
        if signatures and len(set(signatures)) == 1:
            return "repeated_effect_signature"
        if all(t.event.no_op for t in recent):
            return "all_noop"
        return "mixed"

    def tried_count(self, state_key: str, spec: ActionSpec) -> int:
        return self.actions_by_state[state_key][spec.key]

    def action_success_rate(self, action_name: str) -> float:
        outcomes = self.global_action_outcomes[action_name]
        total = sum(outcomes.values())
        if total == 0:
            return 0.5
        return (outcomes["progress"] * 2.0 + outcomes["change"] * 0.8) / (2.0 * total)

    def action_noop_rate(self, action_name: str) -> float:
        outcomes = self.global_action_outcomes[action_name]
        total = sum(outcomes.values())
        return outcomes["noop"] / total if total else 0.0

    def recent_state_loop(self, window: int = 12) -> bool:
        if len(self.transitions) < 4:
            return False
        recent = [t.after_exact for t in list(self.transitions)[-window:]]
        return len(set(recent)) <= max(2, len(recent) // 3)

    def longest_family_streak(self) -> int:
        if not self.transitions:
            return 0
        max_streak = 0
        current_streak = 0
        last_family = None
        for t in self.transitions:
            if t.action.name == last_family:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 1
                last_family = t.action.name
        return max(max_streak, current_streak)
        
    def action_family_entropy(self) -> float:
        import math
        if not self.transitions:
            return 0.0
        counts = {}
        for t in self.transitions:
            counts[t.action.name] = counts.get(t.action.name, 0) + 1
        total = sum(counts.values())
        return -sum((c/total) * math.log2(c/total) for c in counts.values())

    def find_replay_progress(self, exact_key: str) -> ActionSpec | None:
        candidates = self.progress_actions.get(exact_key)
        if candidates:
            self.total_mental_eliminations += 1
            return candidates[-1]
        return None

    def last(self) -> Transition | None:
        return self.transitions[-1] if self.transitions else None


class DeadSignatureMemory:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.noops: Counter[str] = Counter()
        self.effects: Counter[str] = Counter()
        self.deaths: Counter[str] = Counter()
        self.level_epoch = 0

    def reset_level(self) -> None:
        self.noops.clear()
        self.effects.clear()
        self.deaths.clear()
        self.level_epoch += 1

    def record(self, signature: str, event: Event) -> None:
        if event.game_over:
            self.deaths[signature] += 1
            return
        ineffective_animation = (
            event.settled_no_op
            and event.transient_changed_count <= 4
            and not event.topology_change
            and event.level_delta == 0
        )
        if event.no_op or ineffective_animation:
            self.noops[signature] += 1
        else:
            self.effects[signature] += 1
            # A previously dead class can revive after topology/phase change.
            if event.topology_change:
                self.noops[signature] = max(0, self.noops[signature] - 1)
                self.deaths[signature] = max(0, self.deaths[signature] - 1)

    def is_dead(self, signature: str) -> bool:
        noop_dead = self.noops[signature] >= self.threshold and self.effects[signature] == 0
        death_dead = self.deaths[signature] >= max(1, self.effects[signature])
        return noop_dead or death_dead

    def invalidate_on_phase_change(self) -> None:
        self.noops = Counter({k: max(0, v - 1) for k, v in self.noops.items()})
        self.deaths = Counter({k: max(0, v - 1)
                              for k, v in self.deaths.items()})


@dataclass(slots=True)
class SpatialOutcome:
    counts: Counter[bytes] = field(default_factory=Counter)
    examples: int = 0


class FastSpatialActionHash:
    """Stores (local before patch, action) -> local delta patch distributions."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.table: dict[tuple[int, bytes, str], SpatialOutcome] = {}

    @staticmethod
    def _patch_at(grid: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
        radius = size // 2
        padded = np.pad(grid, radius, mode="constant", constant_values=-1)
        px, py = x + radius, y + radius
        return np.ascontiguousarray(
            padded[py - radius: py + radius + 1, px - radius: px + radius + 1],
            dtype=np.int16,
        )

    @staticmethod
    def _anchor(scene: Scene, action: ActionSpec, event: Event | None = None) -> Point:
        data = action.data_dict
        if "x" in data and "y" in data:
            return int(data["x"]), int(data["y"])
        if scene.controlled_entity_id is not None:
            for entity in scene.entities:
                if entity.entity_id == scene.controlled_entity_id:
                    return int(round(entity.centroid[0])), int(round(entity.centroid[1]))
        if event is not None and event.changed_bbox is not None:
            x0, y0, x1, y1 = event.changed_bbox
            return (x0 + x1) // 2, (y0 + y1) // 2
        return scene.width // 2, scene.height // 2

    def record(self, before: Scene, after: Scene, action: ActionSpec, event: Event) -> None:
        if before.grid.shape != after.grid.shape:
            return
        x, y = self._anchor(before, action, event)
        for size in self.config.patch_sizes:
            before_patch = self._patch_at(before.grid, x, y, size)
            after_patch = self._patch_at(after.grid, x, y, size)
            delta = after_patch.astype(np.int32) - \
                before_patch.astype(np.int32)
            key = (size, before_patch.tobytes(), action.name)
            outcome = self.table.setdefault(key, SpatialOutcome())
            outcome.counts[delta.tobytes()] += 1
            outcome.examples += 1

    def predict(self, scene: Scene, action: ActionSpec) -> tuple[bool, float, str | None]:
        x, y = self._anchor(scene, action)
        best: tuple[bool, float, str | None] = (False, 0.0, None)
        for size in self.config.patch_sizes:
            patch = self._patch_at(scene.grid, x, y, size)
            outcome = self.table.get((size, patch.tobytes(), action.name))
            if outcome is None or outcome.examples < self.config.spatial_hash_min_support:
                continue
            predicted_bytes, support = outcome.counts.most_common(1)[0]
            conflict = 1.0 - support / max(1, outcome.examples)
            confidence = (support / max(1, outcome.examples)) * \
                min(1.0, outcome.examples / 4)
            if conflict <= self.config.spatial_hash_max_conflict and confidence > best[1]:
                arr = np.frombuffer(predicted_bytes, dtype=np.int32)
                nonzero = int(np.count_nonzero(arr))
                effect = f"patch{size}:delta_cells={nonzero}"
                best = (True, confidence, effect)
        return best


class CausalWorldModel:
    """Replay-verified graph over exact observed states."""

    def __init__(self) -> None:
        self.edges: dict[str, dict[tuple[str, tuple[tuple[str, int], ...]], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.edge_specs: dict[tuple[str, tuple[str,
                                               tuple[tuple[str, int], ...]]], ActionSpec] = {}
        self.progress_states: set[str] = set()

    def reset_level(self) -> None:
        self.edges.clear()
        self.edge_specs.clear()
        self.progress_states.clear()

    def record(self, transition: Transition) -> None:
        key = transition.action.key
        self.edges[transition.before_exact][key][transition.after_exact] += 1
        self.edge_specs[(transition.before_exact, key)] = transition.action
        if transition.event.level_delta > 0 or transition.event.win:
            self.progress_states.add(transition.before_exact)

    def known_action(self, exact_state: str) -> ActionSpec | None:
        if exact_state in self.progress_states:
            options = self.edges.get(exact_state, {})
            for key in options:
                spec = self.edge_specs[(exact_state, key)]
                return spec
        return None

    def plan_to_progress(self, exact_state: str, max_depth: int = 32) -> ActionSpec | None:
        """Return the first action on a replay-verified path to progress.

        This is a zero-environment-action search over observed transitions. Only
        the most-supported successor of each state-action edge is traversed, and
        self-loops are ignored. It is especially useful after a death when the
        initial state recurs and the level model has been preserved.
        """
        direct = self.known_action(exact_state)
        if direct is not None:
            return direct
        queue: deque[tuple[str, ActionSpec | None, int]] = deque(
            [(exact_state, None, 0)]
        )
        seen = {exact_state}
        while queue:
            state, first_action, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for key, outcomes in self.edges.get(state, {}).items():
                if not outcomes:
                    continue
                successor, _ = outcomes.most_common(1)[0]
                if successor == state or successor in seen:
                    continue
                spec = self.edge_specs.get((state, key))
                if spec is None:
                    continue
                initial = first_action or spec
                if successor in self.progress_states:
                    return initial
                seen.add(successor)
                queue.append((successor, initial, depth + 1))
        return None

    def replay_predict(self, exact_state: str, spec: ActionSpec) -> tuple[str | None, float]:
        outcomes = self.edges.get(exact_state, {}).get(spec.key)
        if not outcomes:
            return None, 0.0
        state, support = outcomes.most_common(1)[0]
        return state, support / max(1, sum(outcomes.values()))


class HypothesisMemory:
    """Competing predictive hypotheses over contextual action signatures.

    The memory deliberately separates fast predictive association from the
    replay-verified causal graph. A high-confidence association may guide
    execution, but only observed exact-state transitions are treated as causal
    evidence for simulation.
    """

    def __init__(self) -> None:
        self.local_effects: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_effects: dict[str, Counter[str]] = defaultdict(Counter)
        self.local_progress: Counter[str] = Counter()
        self.global_progress: Counter[str] = Counter()

    def reset_level(self) -> None:
        self.local_effects.clear()
        self.local_progress.clear()

    @staticmethod
    def _label(event: Event) -> str:
        if event.win or event.level_delta > 0:
            return "progress"
        if event.game_over:
            return "death"
        if event.no_op:
            return "noop"
        return f"effect:{event.effect_signature}"

    def record(self, signature: str, event: Event) -> None:
        label = self._label(event)
        self.local_effects[signature][label] += 1
        self.global_effects[signature][label] += 1
        if label == "progress":
            self.local_progress[signature] += 1
            self.global_progress[signature] += 1

    def distribution(self, signature: str) -> Counter[str]:
        local = self.local_effects.get(signature)
        if local and sum(local.values()) >= 1:
            return local
        return self.global_effects.get(signature, Counter())

    def predict(self, signature: str) -> tuple[bool, float, str | None]:
        outcomes = self.distribution(signature)
        total = sum(outcomes.values())
        if total < 2:
            return False, 0.0, None
        label, support = outcomes.most_common(1)[0]
        confidence = support / total
        return confidence >= 0.75, confidence, label

    def information_value(self, signature: str) -> float:
        outcomes = self.distribution(signature)
        total = sum(outcomes.values())
        if total == 0:
            return 1.0
        probabilities = np.asarray(
            list(outcomes.values()), dtype=np.float64) / total
        entropy = float(-np.sum(probabilities *
                        np.log2(probabilities + 1e-12)))
        max_entropy = math.log2(max(2, len(probabilities)))
        normalized = entropy / max(1e-9, max_entropy)
        # Repeated confident outcomes have little information value; conflicting
        # hypotheses remain valuable only when the choice can affect a plan.
        return max(0.05, min(1.0, normalized + 1.0 / (1.0 + total)))

    def goal_bonus(self, signature: str) -> float:
        local = self.local_progress[signature]
        global_count = self.global_progress[signature]
        return min(2.0, 1.2 * local + 0.35 * global_count)


# ---------------------------------------------------------------------------
# Explicit competing goals, executable programs, and goal alignment
# ---------------------------------------------------------------------------


def _grid_color_counts(grid: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _symmetry_score(grid: np.ndarray) -> float:
    if grid.size == 0:
        return 0.0
    horizontal = float(np.mean(grid == np.fliplr(grid)))
    vertical = float(np.mean(grid == np.flipud(grid)))
    return max(horizontal, vertical)

def classify_puzzle_family(grid: np.ndarray) -> str:
    """Soft classifier for initializing goal priors."""
    bg = int(np.bincount(grid.ravel()).argmax())
    mask = grid != bg
    density = float(np.mean(mask))
    unique_colors = len(np.unique(grid))
    
    if density < 0.15:
        return "pathfinding"
    if unique_colors > 6 and density > 0.6:
        return "color_matching"
    from scipy.ndimage import label
    labeled, num_features = label(mask)
    if num_features > 10 and density < 0.5:
        return "field_diffusion"
    return "geometric_transformation"


class GoalHypothesisManager:
    """Maintains several candidate objectives instead of collapsing early.

    Goals are deliberately generic and grounded in measurable grid properties.
    A level completion strengthens hypotheses whose predicted progress indicators
    were present immediately before the completion event.  Successful goal
    schemas are transferred to later levels with reduced confidence.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.hypotheses: dict[str, GoalHypothesis] = {}
        self.transfer_priors: Counter[tuple[str,
                                            tuple[tuple[str, Any], ...]]] = Counter()
        self.level_start_grid = None
        self.level_start_components = 0
        self._counter = 0
        self._value_cache: dict[str, float] = {}
        self.transitions_processed = 0
        self.top_goal_id = None
        self.top_goal_changes_this_level = 0
        self.prior_adjustments_this_level = 0
        self.stabilized_before_progress = False
        self.top_puzzle_family: str | None = None
        self.puzzle_family_switch_counter = 0

    def reset_level(self, scene: Scene | None = None) -> None:
        self.hypotheses.clear()
        self.level_start_grid = None if scene is None else scene.grid.copy()
        self.level_start_components = 0 if scene is None else len(
            scene.components)
        self.transitions_processed = 0
        self.top_goal_id = None
        self.top_goal_changes_this_level = 0
        self.prior_adjustments_this_level = 0
        self.stabilized_before_progress = False
        self.top_puzzle_family = None
        self.puzzle_family_switch_counter = 0
        if scene is not None:
            self.seed(scene)

    @staticmethod
    def _param_key(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted((str(k), v) for k, v in params.items() if isinstance(v, (str, int, float, bool))))

    def _add(
        self,
        kind: str,
        params: Mapping[str, Any],
        confidence: float,
        source: str,
    ) -> GoalHypothesis:
        signature = f"{kind}:{self._param_key(params)}"
        for goal in self.hypotheses.values():
            if f"{goal.kind}:{self._param_key(goal.params)}" == signature:
                goal.confidence = max(goal.confidence, confidence)
                return goal
        self._counter += 1
        goal = GoalHypothesis(
            goal_id=f"G{self._counter:03d}",
            kind=kind,
            params=dict(params),
            confidence=max(0.02, min(0.95, confidence)),
            source=source,
        )
        prior = self.transfer_priors[(kind, self._param_key(params))]
        if prior:
            goal.confidence = min(0.9, goal.confidence + 0.08 * prior)
            goal.source = "cross_level_transfer"
        self.hypotheses[goal.goal_id] = goal
        self._trim()
        return goal
        
    def adjust_priors(self, puzzle_family: str) -> None:
        """Dynamically reweight active goals based on an abducted puzzle family."""
        family = puzzle_family.lower().strip()
        boost_occurred = False
        for goal in self.active(limit=10):
            boost = 0.0
            if "pathfinding" in family and goal.kind in {"reach_or_touch_color", "move_controllable_to_target"}:
                boost = 0.4
            elif "geometric" in family and goal.kind in {"preserve_object_relationships", "cause_topology_change"}:
                boost = 0.35
            elif "field_diffusion" in family and goal.kind in {"collect_color", "cause_topology_change"}:
                boost = 0.4
            elif "color_matching" in family and goal.kind == "collect_color":
                boost = 0.35
            elif "object_interaction" in family and goal.kind in {"destroy_components", "move_controllable_to_target"}:
                boost = 0.3
                
            if boost > 0.0:
                goal.confidence = min(0.95, goal.confidence + boost)
                goal.source = f"llm_abduction_{family}"
                self._value_cache.clear()
                boost_occurred = True
        if boost_occurred:
            self.prior_adjustments_this_level += 1

    def _trim(self) -> None:
        if len(self.hypotheses) > self.config.max_goal_hypotheses:
            ranked = sorted(
                self.hypotheses.values(),
                key=lambda g: (g.status == "active", g.confidence,
                               g.support - g.contradictions),
                reverse=True,
            )[: self.config.max_goal_hypotheses]
            self.hypotheses = {g.goal_id: g for g in ranked}
            
        active_goals = [g for g in self.hypotheses.values() if g.status == "active"]
        if active_goals:
            current_top = max(active_goals, key=lambda g: g.confidence)
            if current_top.goal_id != self.top_goal_id:
                if self.top_goal_id is not None:
                    self.top_goal_changes_this_level += 1
                self.top_goal_id = current_top.goal_id
                self.transitions_until_stabilized = self.transitions_processed
                
            new_top_family = str(current_top.source).replace("llm_abduction_", "") if str(current_top.source).startswith("llm_abduction") else current_top.kind
            if new_top_family != self.top_puzzle_family:
                if self.top_puzzle_family is not None:
                    self.puzzle_family_switch_counter += 1
                self.top_puzzle_family = new_top_family

    def seed(self, scene: Scene) -> None:
        if self.level_start_grid is None:
            self.level_start_grid = scene.grid.copy()
            self.level_start_components = len(scene.components)
        
        family = classify_puzzle_family(scene.grid)
        pathfinding_boost = 0.15 if family == "pathfinding" else 0.0
        
        non_background = [(c, n)
                          for c, n in scene.color_counts if c != scene.background]
        non_background.sort(key=lambda item: item[1])
        for index, (color, count) in enumerate(non_background):
            self._add(
                "collect_color",
                {"color": color, "baseline_count": count},
                (0.20 if count <= 12 else 0.10) + (0.10 if family == "color_matching" else 0.0),
                "rare_quantity_prior",
            )
            self._add(
                "reach_or_touch_color",
                {"color": color},
                (0.24 if count <= 9 else 0.12) + pathfinding_boost,
                "affordance_prior",
            )
            if index < 3 and count <= 16:
                self._add(
                    "activate_color",
                    {"color": color, "baseline_count": count},
                    0.20,
                    "small_object_affordance",
                )
        # Ordered composite hypotheses model the proposal's ``X then Y`` goals
        # while remaining generic and measurable. They compete with simpler
        # goals until evidence from progress events separates them.
        if len(non_background) >= 2:
            first_color, first_count = non_background[0]
            target_color, _ = non_background[1]
            self._add(
                "collect_then_reach",
                {"source_color": first_color, "baseline_count": first_count,
                    "target_color": target_color},
                0.19,
                "ordered_goal_prior",
            )
            self._add(
                "activate_then_reach",
                {"source_color": first_color, "baseline_count": first_count,
                    "target_color": target_color},
                0.18,
                "ordered_goal_prior",
            )

        # Controllable Avatar & Navigation Hypotheses
        if scene.controlled_entity_id is not None:
            for color, count in non_background[:3]:
                self._add(
                    "move_controllable_to_target",
                    {"target_color": color},
                    0.28,
                    "avatar_navigation_prior",
                )
                self._add(
                    "clear_obstacle_color",
                    {"obstacle_color": color, "baseline_count": count},
                    0.20,
                    "obstacle_clearance_prior",
                )

        # Axis Alignment / Spatial Completion
        if len(scene.components) >= 2:
            self._add(
                "align_component_axis",
                {"baseline_components": len(scene.components)},
                0.22,
                "spatial_alignment_prior",
            )

        self._add(
            "cause_topology_change",
            {
                "baseline_components": len(scene.components),
                "baseline_nonbackground": int(np.count_nonzero(scene.grid != scene.background)),
            },
            0.16,
            "environment_structure_prior",
        )
        self._add(
            "increase_pattern_order",
            {"baseline_symmetry": round(_symmetry_score(scene.grid), 3)},
            0.10,
            "pattern_prior",
        )
        self._add("discover_progress_mechanism", {}, 0.18, "epistemic_prior")

    def _controlled_position(self, scene: Scene, grid: np.ndarray) -> tuple[float, float] | None:
        controlled = next(
            (e for e in scene.entities if e.entity_id == scene.controlled_entity_id), None)
        if controlled is not None:
            coords = _color_centroid(grid, controlled.color)
            return coords or controlled.centroid
        return None

    def _value_one(self, goal: GoalHypothesis, scene: Scene, grid: np.ndarray) -> float:
        counts = _grid_color_counts(grid)
        kind = goal.kind
        if kind == "collect_color":
            color = int(goal.params.get("color", -1))
            baseline = max(1, int(goal.params.get(
                "baseline_count", counts.get(color, 1))))
            return max(0.0, min(1.0, 1.0 - counts.get(color, 0) / baseline))
        if kind in {"reach_or_touch_color", "move_controllable_to_target"}:
            color = int(goal.params.get("color", -1) if "color" in goal.params else goal.params.get("target_color", -1))
            target_ys, target_xs = np.where(grid == color)
            pos = self._controlled_position(scene, grid)
            if pos is None or len(target_xs) == 0:
                return 0.0
            distances = np.abs(target_xs - pos[0]) + np.abs(target_ys - pos[1])
            distance = float(np.min(distances))
            scale = max(1.0, scene.width + scene.height)
            return max(0.0, 1.0 - distance / scale)
        if kind == "clear_obstacle_color":
            obs_color = int(goal.params.get("obstacle_color", -1))
            baseline = max(1, int(goal.params.get("baseline_count", counts.get(obs_color, 1))))
            curr = counts.get(obs_color, 0)
            return max(0.0, min(1.0, (baseline - curr) / baseline))
        if kind == "align_component_axis":
            if len(scene.components) < 2:
                return 0.0
            aligned_pairs = 0
            for i in range(len(scene.components)):
                for j in range(i + 1, len(scene.components)):
                    c1, c2 = scene.components[i].centroid, scene.components[j].centroid
                    if abs(c1[0] - c2[0]) < 1.5 or abs(c1[1] - c2[1]) < 1.5:
                        aligned_pairs += 1
            max_pairs = max(1, len(scene.components) * (len(scene.components) - 1) // 2)
            return min(1.0, aligned_pairs / max_pairs)
        if kind == "activate_color":
            color = int(goal.params.get("color", -1))
            baseline = max(1, int(goal.params.get(
                "baseline_count", counts.get(color, 1))))
            current = counts.get(color, 0)
            return min(1.0, abs(current - baseline) / baseline)
        if kind in {"collect_then_reach", "activate_then_reach"}:
            source_color = int(goal.params.get("source_color", -1))
            target_color = int(goal.params.get("target_color", -1))
            baseline = max(1, int(goal.params.get(
                "baseline_count", counts.get(source_color, 1))))
            if kind == "collect_then_reach":
                first_phase = max(
                    0.0, min(1.0, 1.0 - counts.get(source_color, 0) / baseline))
            else:
                first_phase = min(
                    1.0, abs(counts.get(source_color, 0) - baseline) / baseline)
            target_ys, target_xs = np.where(grid == target_color)
            pos = self._controlled_position(scene, grid)
            reach = 0.0
            if pos is not None and len(target_xs):
                distance = float(
                    np.min(np.abs(target_xs - pos[0]) + np.abs(target_ys - pos[1])))
                reach = max(0.0, 1.0 - distance /
                            max(1.0, scene.width + scene.height))
            # Before the prerequisite is mostly satisfied, target movement is a
            # weak signal; afterwards it becomes the dominant second phase.
            return (0.68 * first_phase + 0.08 * reach) if first_phase < 0.75 else (0.58 + 0.42 * reach)
        if kind == "cause_topology_change":
            baseline = int(goal.params.get(
                "baseline_components", self.level_start_components))
            background = max(counts, key=counts.get) if counts else 0
            # Exact connected-component reconstruction is useful on compact
            # object-like scenes but too costly and brittle for dense field
            # dynamics. In field mode use a raw occupancy-change proxy instead.
            if scene.field_mode or grid.size > 2048:
                baseline_occupied = max(1, int(goal.params.get(
                    "baseline_nonbackground", np.count_nonzero(scene.grid != scene.background))))
                occupied = int(np.count_nonzero(grid != background))
                return min(1.0, abs(occupied - baseline_occupied) / baseline_occupied)
            comp_count = len(_connected_components(grid, background))
            return min(1.0, abs(comp_count - baseline) / max(1, baseline))
        if kind == "increase_pattern_order":
            baseline = float(goal.params.get("baseline_symmetry", 0.0))
            return max(0.0, min(1.0, _symmetry_score(grid) - baseline + 0.5))
        if kind == "discover_progress_mechanism":
            return 0.0
        return 0.0

    def state_value(self, scene: Scene, grid: np.ndarray | None = None) -> float:
        grid = scene.grid if grid is None else grid
        active = self.active(limit=6)
        if not active:
            return 0.0
        goal_signature = tuple(
            (g.goal_id, round(g.confidence, 3), g.status) for g in active)
        cache_key = _stable_hash_bytes(
            grid.tobytes() + repr(goal_signature).encode(), 12)
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            return cached
        weight_sum = sum(max(0.01, g.confidence) for g in active)
        value = sum(max(0.01, g.confidence) * self._value_one(g, scene, grid)
                    for g in active) / weight_sum
        if len(self._value_cache) > 512:
            self._value_cache.clear()
        self._value_cache[cache_key] = value
        return value

    def prediction_delta(self, scene: Scene, predicted_grid: np.ndarray) -> float:
        return self.state_value(scene, predicted_grid) - self.state_value(scene, scene.grid)

    def discrimination_value(self, action_signature: str, hypotheses: HypothesisMemory) -> float:
        base_eig = hypotheses.information_value(action_signature)
        active_goals = self.active(limit=4)
        if len(active_goals) >= 2:
            conf_spread = max(g.confidence for g in active_goals) - min(g.confidence for g in active_goals)
            return base_eig + 0.35 * conf_spread
        return base_eig

    def update(self, transition: Transition, before: Scene, after: Scene) -> None:
        self._value_cache.clear()
        self.transitions_processed += 1
        if not self.hypotheses:
            self.seed(before)
        event = transition.event
        before_counts = _grid_color_counts(before.grid)
        after_counts = _grid_color_counts(after.grid)
        changed_colors = {
            color for color in set(before_counts) | set(after_counts)
            if before_counts.get(color, 0) != after_counts.get(color, 0)
        }
        for color in sorted(changed_colors):
            if color == before.background:
                continue
            if after_counts.get(color, 0) < before_counts.get(color, 0):
                self._add(
                    "collect_color",
                    {"color": color,
                        "baseline_count": before_counts.get(color, 1)},
                    0.22,
                    "observed_quantity_reduction",
                )
            if event.topology_change:
                self._add(
                    "activate_color",
                    {"color": color,
                        "baseline_count": before_counts.get(color, 1)},
                    0.24,
                    "topology_correlated_color",
                )

        progress_features: set[str] = set()
        if event.topology_change:
            progress_features.add("topology")
        if event.disappeared_colors:
            progress_features.add("quantity")
        if event.entity_moves or event.animation_vectors:
            progress_features.add("movement")
        if event.changed_count > before.grid.size * 0.20:
            progress_features.add("global_transform")

        for goal in self.hypotheses.values():
            predicted_delta = self._value_one(
                goal, before, after.grid) - self._value_one(goal, before, before.grid)
            goal.progress_estimate = 0.75 * goal.progress_estimate + 0.25 * predicted_delta
            aligned = predicted_delta > 0.02
            if event.level_delta > 0 or event.win:
                reward = 1.8 if aligned else 0.35
                if goal.kind == "cause_topology_change" and "topology" in progress_features:
                    reward += 0.8
                if goal.kind == "collect_color" and "quantity" in progress_features:
                    reward += 0.6
                if goal.kind in {"reach_or_touch_color", "move_controllable_to_target"} and "movement" in progress_features:
                    reward += 0.35
                goal.support += reward
                goal.confidence = min(0.98, goal.confidence + 0.12 * reward)
                goal.evidence.append(
                    f"progress@{transition.step_index}:{event.effect_signature}")
            elif event.game_over:
                if aligned:
                    goal.contradictions += 0.8
                    goal.confidence = max(0.02, goal.confidence - 0.12)
                    goal.evidence.append(
                        f"death_counterexample@{transition.step_index}")
            elif event.no_op and goal.progress_estimate > 0.2:
                goal.contradictions += 0.25
                goal.confidence = max(0.02, goal.confidence - 0.04)
            elif aligned:
                # Sub-goal intermediate progress reward
                goal.support += 0.28
                goal.confidence = min(0.96, goal.confidence + 0.035)
                goal.evidence.append(f"subgoal_progress@{transition.step_index}")

            if goal.contradictions >= goal.support + 2.0:
                goal.status = "rejected"
            elif goal.confidence < 0.05:
                goal.status = "dormant"

        if event.level_delta > 0 or event.win:
            if not self.stabilized_before_progress and self.transitions_processed > 0 and self.transitions_until_stabilized < self.transitions_processed:
                self.stabilized_before_progress = True
            for goal in self.active(limit=3):
                self.transfer_priors[(
                    goal.kind, self._param_key(goal.params))] += 1
        self._trim()

    def active(self, limit: int = 5) -> list[GoalHypothesis]:
        active = [g for g in self.hypotheses.values() if g.status == "active"]
        active.sort(
            key=lambda g: (g.confidence, g.support -
                           g.contradictions, g.progress_estimate),
            reverse=True,
        )
        # Preserve competition: never collapse to a single objective before
        # direct progress evidence exists.
        return active[: max(1, limit)]

    def summary(self, limit: int = 6) -> list[dict[str, Any]]:
        return [
            {
                "id": g.goal_id,
                "kind": g.kind,
                "params": g.params,
                "confidence": round(g.confidence, 3),
                "progress": round(g.progress_estimate, 3),
                "status": g.status,
            }
            for g in self.active(limit)
        ]


@dataclass(slots=True)
class WorldProgram:
    program_id: str
    kind: str
    action_name: str
    payload: dict[str, Any]
    source: str = "induced"
    support: int = 0
    conflicts: int = 0
    exact_matches: int = 0
    cell_accuracy_sum: float = 0.0
    eligible_replays: int = 0
    current_level_support: int = 0
    current_level_conflicts: int = 0
    status: str = "candidate"
    evidence: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    verified_replay_ids: set[str] = field(default_factory=set)

    @property
    def replay_accuracy(self) -> float:
        return self.exact_matches / self.eligible_replays if self.eligible_replays else 0.0

    @property
    def cell_accuracy(self) -> float:
        return self.cell_accuracy_sum / self.eligible_replays if self.eligible_replays else 0.0

    @property
    def confidence(self) -> float:
        if self.kind == "exact_replay" and self.support:
            return 1.0
        evidence = self.support + self.conflicts
        consistency = self.support / evidence if evidence else 0.0
        replay = max(self.replay_accuracy, self.cell_accuracy)
        return max(0.0, min(1.0, 0.45 * consistency + 0.55 * replay))

    def _anchor(self, grid: np.ndarray, action: ActionSpec, scene: Scene | None) -> Point | None:
        data = action.data_dict
        if "x" in data and "y" in data:
            return int(data["x"]), int(data["y"])
        if scene is not None and scene.controlled_entity_id is not None:
            entity = next(
                (e for e in scene.entities if e.entity_id == scene.controlled_entity_id), None)
            if entity is not None:
                return int(round(entity.centroid[0])), int(round(entity.centroid[1]))
        controlled_color = self.payload.get("controlled_color")
        if controlled_color is not None:
            centroid = _color_centroid(grid, int(controlled_color))
            if centroid is not None:
                return int(round(centroid[0])), int(round(centroid[1]))
        return None

    @staticmethod
    def _extract_patch(grid: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
        radius = size // 2
        padded = np.pad(grid, radius, mode="constant", constant_values=-1)
        px, py = x + radius, y + radius
        return np.ascontiguousarray(padded[py-radius:py+radius+1, px-radius:px+radius+1], dtype=np.int16)

    @staticmethod
    def _write_patch(grid: np.ndarray, x: int, y: int, patch: np.ndarray) -> np.ndarray:
        result = grid.copy()
        radius = patch.shape[0] // 2
        for py in range(patch.shape[0]):
            for px in range(patch.shape[1]):
                gx, gy = x + px - radius, y + py - radius
                value = int(patch[py, px])
                if 0 <= gx < result.shape[1] and 0 <= gy < result.shape[0] and value >= 0:
                    result[gy, gx] = value
        return result

    def applies(self, grid: np.ndarray, action: ActionSpec, scene: Scene | None = None) -> bool:
        if action.name != self.action_name:
            return False
        if self.kind == "exact_replay":
            return _stable_hash_bytes(np.ascontiguousarray(grid, dtype=np.int16).tobytes()) == self.payload.get("before_key")
        if self.kind == "local_patch_replace":
            anchor = self._anchor(grid, action, scene)
            if anchor is None:
                return False
            expected = self.payload["before_patch"]
            current = self._extract_patch(
                grid, anchor[0], anchor[1], int(self.payload["size"]))
            return current.shape == expected.shape and np.array_equal(current, expected)
        if self.kind == "translation":
            color = self.payload.get("selector_color")
            return color is not None and np.any(grid == int(color))
        if self.kind == "color_map":
            return any(np.any(grid == int(src)) for src in self.payload.get("mapping", {}))
        if self.kind in {"component_delete", "component_recolor"}:
            anchor = self._anchor(grid, action, scene)
            if anchor is None:
                return False
            x, y = anchor
            return 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]
        if self.kind == "cellular_rule":
            return bool(self.payload.get("rules"))
        return False

    def simulate(self, grid: np.ndarray, action: ActionSpec, scene: Scene | None = None) -> np.ndarray | None:
        if not self.applies(grid, action, scene):
            return None
        if self.kind == "exact_replay":
            return np.ascontiguousarray(self.payload["after_grid"].copy(), dtype=np.int16)
        if self.kind == "local_patch_replace":
            anchor = self._anchor(grid, action, scene)
            assert anchor is not None
            return self._write_patch(grid, anchor[0], anchor[1], self.payload["after_patch"])
        if self.kind == "translation":
            dx, dy = int(self.payload["dx"]), int(self.payload["dy"])
            color = int(self.payload["selector_color"])
            mask = grid == color
            result = grid.copy()
            result[mask] = int(self.payload.get("background", 0))
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < result.shape[1] and 0 <= ny < result.shape[0]:
                    result[ny, nx] = color
            return result
        if self.kind == "color_map":
            result = grid.copy()
            original = grid.copy()
            for src, dst in self.payload.get("mapping", {}).items():
                result[original == int(src)] = int(dst)
            return result
        if self.kind == "rot90":
            return np.rot90(grid, k=int(self.payload.get("k", 1)))
        if self.kind == "flip_h":
            return np.fliplr(grid)
        if self.kind == "flip_v":
            return np.flipud(grid)
        if self.kind == "line_connect":
            p1 = self.payload.get("p1")
            p2 = self.payload.get("p2")
            if p1 is None or p2 is None:
                anchor = self._anchor(grid, action, scene)
                if anchor is None:
                    return None
                x1, y1 = anchor
                direction = str(self.payload.get("direction", "horizontal"))
                background = int(self.payload.get("background", 0))
                if direction == "vertical":
                    top = y1
                    while top > 0 and int(grid[top - 1, x1]) != background:
                        top -= 1
                    bottom = y1
                    while bottom + 1 < grid.shape[0] and int(grid[bottom + 1, x1]) != background:
                        bottom += 1
                    p1, p2 = (x1, top), (x1, bottom)
                else:
                    left = x1
                    while left > 0 and int(grid[y1, left - 1]) != background:
                        left -= 1
                    right = x1
                    while right + 1 < grid.shape[1] and int(grid[y1, right + 1]) != background:
                        right += 1
                    p1, p2 = (left, y1), (right, y1)
            x1, y1 = p1
            x2, y2 = p2
            color = int(self.payload.get("color", 1))
            result = grid.copy()
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            cx, cy = x1, y1
            while True:
                if 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]:
                    result[cy, cx] = color
                if cx == x2 and cy == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx += sx
                if e2 < dx:
                    err += dx
                    cy += sy
            return result
        if self.kind == "drag_component":
            p1 = self.payload.get("p1")
            p2 = self.payload.get("p2")
            if p1 is not None and p2 is not None:
                x1, y1 = p1
                x2, y2 = p2
                dx, dy = x2 - x1, y2 - y1
            else:
                anchor = self._anchor(grid, action, scene)
                if anchor is None:
                    return None
                x1, y1 = anchor
                dx = int(self.payload.get("dx", 0))
                dy = int(self.payload.get("dy", 0))
            if not (0 <= x1 < grid.shape[1] and 0 <= y1 < grid.shape[0]):
                return None
            color = int(grid[y1, x1])
            result = grid.copy()
            stack = [(x1, y1)]
            seen: set[Point] = set()
            comp_pts: list[Point] = []
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in seen or not (0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]):
                    continue
                if int(grid[cy, cx]) != color:
                    continue
                seen.add((cx, cy))
                comp_pts.append((cx, cy))
                stack.extend(((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)))
            bg = int(self.payload.get("background", 0))
            for cx, cy in comp_pts:
                result[cy, cx] = bg
            for cx, cy in comp_pts:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    result[ny, nx] = color
            return result
        if self.kind == "gravity":
            dx = int(self.payload.get("dx", 0))
            dy = int(self.payload.get("dy", 0))
            background = int(self.payload.get("background", 0))
            if (dx, dy) not in {(0, 1), (0, -1), (1, 0), (-1, 0)}:
                return None
            result = np.full_like(grid, background)
            if dx == 0:
                for x in range(grid.shape[1]):
                    values = [int(grid[y, x]) for y in range(grid.shape[0]) if int(grid[y, x]) != background]
                    if dy > 0:
                        start = grid.shape[0] - len(values)
                        for idx, value in enumerate(values):
                            result[start + idx, x] = value
                    else:
                        for idx, value in enumerate(values):
                            result[idx, x] = value
            else:
                for y in range(grid.shape[0]):
                    values = [int(grid[y, x]) for x in range(grid.shape[1]) if int(grid[y, x]) != background]
                    if dx > 0:
                        start = grid.shape[1] - len(values)
                        for idx, value in enumerate(values):
                            result[y, start + idx] = value
                    else:
                        for idx, value in enumerate(values):
                            result[y, idx] = value
            return result
        if self.kind == "flood_fill":
            anchor = self._anchor(grid, action, scene)
            if anchor is None:
                anchor = (int(self.payload.get("x", -1)), int(self.payload.get("y", -1)))
            x, y = anchor
            if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
                return None
            source_color = int(grid[y, x])
            target_color = int(self.payload.get("target_color", source_color))
            if source_color == target_color:
                return grid.copy()
            result = grid.copy()
            stack = [(x, y)]
            seen: set[Point] = set()
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in seen or not (0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]):
                    continue
                if int(grid[cy, cx]) != source_color:
                    continue
                seen.add((cx, cy))
                result[cy, cx] = target_color
                stack.extend(((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)))
            return result
        if self.kind == "copy_pattern":
            anchor = self._anchor(grid, action, scene)
            if anchor is None:
                return None
            offsets = self.payload.get("offsets", ())
            colors = self.payload.get("colors", ())
            if not offsets or len(offsets) != len(colors):
                return None
            result = grid.copy()
            ax, ay = anchor
            for (dx, dy), color in zip(offsets, colors):
                x, y = ax + int(dx), ay + int(dy)
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    result[y, x] = int(color)
            return result
        if self.kind == "conditional_recolor":
            source_color = self.payload.get("source_color")
            target_color = self.payload.get("target_color")
            neighbor_color = self.payload.get("neighbor_color")
            if source_color is None or target_color is None or neighbor_color is None:
                return None
            result = grid.copy()
            h, w = grid.shape
            for y in range(h):
                for x in range(w):
                    if int(grid[y, x]) != int(source_color):
                        continue
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and int(grid[ny, nx]) == int(neighbor_color):
                            result[y, x] = int(target_color)
                            break
            return result
        if self.kind == "count_and_fill":
            anchor = self._anchor(grid, action, scene)
            if anchor is None:
                return None
            offsets = self.payload.get("offsets", ())
            target_color = self.payload.get("target_color")
            if not offsets or target_color is None:
                return None
            result = grid.copy()
            ax, ay = anchor
            for dx, dy in offsets:
                x, y = ax + int(dx), ay + int(dy)
                if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                    result[y, x] = int(target_color)
            return result
        if self.kind in {"component_delete", "component_recolor"}:
            anchor = self._anchor(grid, action, scene)
            assert anchor is not None
            x, y = anchor
            color = int(grid[y, x])
            background = int(self.payload.get("background", 0))
            replacement = background if self.kind == "component_delete" else int(
                self.payload["target_color"])
            result = grid.copy()
            stack = [(x, y)]
            seen: set[Point] = set()
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in seen or not (0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]):
                    continue
                if int(grid[cy, cx]) != color:
                    continue
                seen.add((cx, cy))
                result[cy, cx] = replacement
                stack.extend(((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)))
            return result
        if self.kind == "cellular_rule":
            rules: Mapping[bytes, int] = self.payload.get("rules", {})
            padded = np.pad(grid, 1, mode="constant", constant_values=-1)
            result = grid.copy()
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    patch = np.ascontiguousarray(
                        padded[y:y+3, x:x+3], dtype=np.int16)
                    if patch.tobytes() in rules:
                        result[y, x] = int(rules[patch.tobytes()])
            return result
        return None

    def to_source(self) -> str:
        """Human-readable persistent program representation for trace review."""
        if self.kind == "translation":
            return f"def step(grid, action): move_color(grid, {self.payload.get('selector_color')}, dx={self.payload.get('dx')}, dy={self.payload.get('dy')})"
        if self.kind == "color_map":
            return f"def step(grid, action): recolor(grid, mapping={self.payload.get('mapping')!r})"
        if self.kind == "local_patch_replace":
            return f"def step(grid, action): replace_local_patch(size={self.payload.get('size')})"
        if self.kind == "component_delete":
            return "def step(grid, action): delete_clicked_component(grid, action.x, action.y)"
        if self.kind == "component_recolor":
            return f"def step(grid, action): recolor_clicked_component(grid, action.x, action.y, {self.payload.get('target_color')})"
        if self.kind == "cellular_rule":
            return f"def step(grid, action): apply_local_rule_table(grid, rules={len(self.payload.get('rules', {}))})"
        if self.kind == "line_connect":
            return f"def step(grid, action): draw_line(grid, p1={self.payload.get('p1')}, p2={self.payload.get('p2')}, color={self.payload.get('color')})"
        if self.kind == "drag_component":
            return f"def step(grid, action): drag_component(grid, dx={self.payload.get('dx')}, dy={self.payload.get('dy')})"
        if self.kind == "gravity":
            return f"def step(grid, action): apply_gravity(grid, dx={self.payload.get('dx')}, dy={self.payload.get('dy')})"
        if self.kind == "flood_fill":
            return f"def step(grid, action): flood_fill(grid, x={self.payload.get('x')}, y={self.payload.get('y')}, color={self.payload.get('target_color')})"
        if self.kind == "copy_pattern":
            return f"def step(grid, action): copy_pattern(grid, cells={len(self.payload.get('offsets', ()))})"
        if self.kind == "conditional_recolor":
            return f"def step(grid, action): recolor_when_adjacent(grid, src={self.payload.get('source_color')}, nbr={self.payload.get('neighbor_color')}, dst={self.payload.get('target_color')})"
        if self.kind == "count_and_fill":
            return f"def step(grid, action): count_and_fill(grid, count={self.payload.get('count')}, cells={len(self.payload.get('offsets', ()))})"
        return "def step(grid, action): return exact_recorded_successor(grid, action)"


class ExecutableProgramLibrary:
    """Induces and verifies declarative executable world-model programs."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.programs: dict[str, WorldProgram] = {}
        self.signature_index: dict[str, str] = {}
        self.recent_transitions: deque[Transition] = deque(maxlen=128)
        self.generated_sources: dict[str, str] = {}
        self._counter = 0
        # Per-program-version memoization prevents repeated simulation of the
        # same grid/action during one metacognitive decision. The cache is
        # cleared whenever evidence changes a program's status or confidence.
        self._prediction_cache: dict[tuple[Any, ...],
                                     ProgramPrediction | None] = {}
        self._simulation_cache: dict[tuple[Any, ...], np.ndarray | None] = {}

    def reset_level(self) -> None:
        # Preserve abstract programs only as transferred hypotheses. They must
        # obtain fresh current-level replay support before controlling planning.
        retained: dict[str, WorldProgram] = {}
        for pid, program in self.programs.items():
            if program.kind == "exact_replay":
                continue
            program.support = max(0, program.support - 1)
            program.current_level_support = 0
            program.current_level_conflicts = 0
            if program.status != "rejected":
                program.status = "transferred_provisional"
            retained[pid] = program
        self.programs = retained
        self.signature_index = {
            self._signature(p.kind, p.action_name, p.payload): pid
            for pid, p in retained.items()
        }
        self.recent_transitions.clear()
        self._prediction_cache.clear()
        self._simulation_cache.clear()

    @staticmethod
    def _payload_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
        digest: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                digest[key] = (
                    value.shape, _stable_hash_bytes(value.tobytes(), 8))
            elif isinstance(value, dict):
                digest[key] = sorted((str(k), str(v))
                                     for k, v in value.items())[:64]
            elif isinstance(value, (str, int, float, bool, type(None))):
                digest[key] = value
        return digest

    def _signature(self, kind: str, action_name: str, payload: Mapping[str, Any]) -> str:
        return _stable_hash_bytes(
            json.dumps({"kind": kind, "action": action_name, "payload": self._payload_digest(
                payload)}, sort_keys=True, default=str).encode(),
            12,
        )

    def _make(self, kind: str, action_name: str, payload: dict[str, Any], source: str = "induced") -> WorldProgram:
        signature = self._signature(kind, action_name, payload)
        existing = self.signature_index.get(signature)
        if existing is not None:
            return self.programs[existing]
        self._counter += 1
        pid = f"P{self._counter:04d}"
        program = WorldProgram(pid, kind, action_name, payload, source=source)
        self.programs[pid] = program
        self.signature_index[signature] = pid
        self.generated_sources[pid] = program.to_source()
        return program

    @staticmethod
    def _anchor(scene: Scene, action: ActionSpec, event: Event) -> Point:
        return FastSpatialActionHash._anchor(scene, action, event)

    def _induce_candidates(self, transition: Transition, before: Scene, after: Scene) -> list[WorldProgram]:
        action = transition.action
        candidates: list[WorldProgram] = []
        candidates.append(
            self._make(
                "exact_replay",
                action.name,
                {
                    "before_key": before.exact_key,
                    "after_grid": after.grid.copy(),
                },
            )
        )

        x, y = self._anchor(before, action, transition.event)
        for size in (3, 5):
            before_patch = FastSpatialActionHash._patch_at(
                before.grid, x, y, size)
            after_patch = FastSpatialActionHash._patch_at(
                after.grid, x, y, size)
            if not np.array_equal(before_patch, after_patch):
                controlled = next(
                    (e for e in before.entities if e.entity_id == before.controlled_entity_id), None)
                candidates.append(
                    self._make(
                        "local_patch_replace",
                        action.name,
                        {
                            "size": size,
                            "before_patch": before_patch,
                            "after_patch": after_patch,
                            "controlled_color": None if controlled is None else controlled.color,
                        },
                    )
                )

        for entity_id, dx, dy in transition.event.entity_moves:
            entity = next(
                (e for e in before.entities if e.entity_id == entity_id), None)
            if entity is not None and (dx or dy):
                candidates.append(
                    self._make(
                        "translation",
                        action.name,
                        {
                            "dx": int(dx),
                            "dy": int(dy),
                            "selector_color": entity.color,
                            "background": before.background,
                        },
                    )
                )
                if entity.entity_id == before.controlled_entity_id:
                    candidates.append(
                        self._make("drag_component", action.name, {"dx": int(dx), "dy": int(dy), "background": before.background})
                    )

        # gravity
        moves = transition.event.entity_moves
        if len(moves) > 0:
            dirs = set((dx, dy) for _, dx, dy in moves if (dx or dy))
            if len(dirs) == 1:
                dx, dy = list(dirs)[0]
                if (dx == 0 and dy != 0) or (dy == 0 and dx != 0):
                    moved_eids = set(eid for eid, mdx, mdy in moves if (mdx or mdy))
                    if len(moved_eids) > 1 or (len(moved_eids) == 1 and list(moved_eids)[0] != before.controlled_entity_id):
                        candidates.append(self._make("gravity", action.name, {"dx": int(np.sign(dx)), "dy": int(np.sign(dy))}))

        # Rotations and Symmetry Flips
        if after.grid.shape == before.grid.shape:
            for k in (1, 2, 3):
                if np.array_equal(np.rot90(before.grid, k=k), after.grid):
                    candidates.append(self._make("rot90", action.name, {"k": k}))
            if np.array_equal(np.fliplr(before.grid), after.grid):
                candidates.append(self._make("flip_h", action.name, {}))
            if np.array_equal(np.flipud(before.grid), after.grid):
                candidates.append(self._make("flip_v", action.name, {}))
            if np.all(before.grid + after.grid == 9):
                candidates.append(self._make("color_invert", action.name, {"max_color": 9}))

        changed = np.where(before.grid != after.grid)
        mapping: dict[int, int] = {}
        mapping_valid = len(changed[0]) > 0
        for yy, xx in zip(*changed):
            src, dst = int(before.grid[yy, xx]), int(after.grid[yy, xx])
            if src in mapping and mapping[src] != dst:
                mapping_valid = False
                break
            mapping[src] = dst
        if mapping_valid and mapping and len(mapping) <= 4:
            candidates.append(self._make(
                "color_map", action.name, {"mapping": mapping}))

        if len(changed[0]) > 0:
            ys, xs = changed
            if mapping_valid:
                # line_connect
                if len(xs) > 1:
                    is_horiz = len(set(ys)) == 1
                    is_vert = len(set(xs)) == 1
                    if is_horiz or is_vert:
                        if is_horiz:
                            if np.max(xs) - np.min(xs) + 1 == len(xs):
                                y = ys[0]
                                min_x, max_x = np.min(xs), np.max(xs)
                                left_bound = min_x == 0 or before.grid[y, min_x - 1] != before.background
                                right_bound = max_x == before.width - 1 or before.grid[y, max_x + 1] != before.background
                                if left_bound and right_bound:
                                    candidates.append(self._make("line_connect", action.name, {"color": int(after.grid[ys[0], xs[0]]), "p1": (int(min_x), int(y)), "p2": (int(max_x), int(y)), "background": before.background}))
                        elif is_vert:
                            if np.max(ys) - np.min(ys) + 1 == len(ys):
                                x = xs[0]
                                min_y, max_y = np.min(ys), np.max(ys)
                                top_bound = min_y == 0 or before.grid[min_y - 1, x] != before.background
                                bottom_bound = max_y == before.height - 1 or before.grid[max_y + 1, x] != before.background
                                if top_bound and bottom_bound:
                                    candidates.append(self._make("line_connect", action.name, {"color": int(after.grid[ys[0], xs[0]]), "p1": (int(x), int(min_y)), "p2": (int(x), int(max_y)), "background": before.background}))

                # copy_pattern
                if len(xs) > 1:
                    min_x, min_y = np.min(xs), np.min(ys)
                    new_shape = set((x - min_x, y - min_y) for x, y in zip(xs, ys))
                    for e in before.entities:
                        if len(e.cells) == len(new_shape):
                            e_min_x = min(c[0] for c in e.cells)
                            e_min_y = min(c[1] for c in e.cells)
                            e_shape = set((c[0] - e_min_x, c[1] - e_min_y) for c in e.cells)
                            if new_shape == e_shape:
                                offsets: list[tuple[int, int]] = []
                                colors: list[int] = []
                                color_match = True
                                for (nx, ny) in sorted(new_shape):
                                    before_color = int(before.grid[e_min_y + ny, e_min_x + nx])
                                    after_color = int(after.grid[min_y + ny, min_x + nx])
                                    if after_color != before_color:
                                        color_match = False
                                        break
                                    offsets.append((int(nx), int(ny)))
                                    colors.append(before_color)
                                if color_match:
                                    candidates.append(self._make("copy_pattern", action.name, {"source_entity_id": e.entity_id, "offsets": tuple(offsets), "colors": tuple(colors)}))
                                    break

                # count_and_fill
                count = len(xs)
                min_x, max_x = np.min(xs), np.max(xs)
                min_y, max_y = np.min(ys), np.max(ys)
                if (max_x - min_x + 1) * (max_y - min_y + 1) == count:
                    offsets = tuple(sorted((int(x - min_x), int(y - min_y)) for x, y in zip(xs, ys)))
                    source_sizes = Counter(len(e.cells) for e in before.entities)
                    if source_sizes.get(count, 0) == 1:
                        candidates.append(self._make("count_and_fill", action.name, {"count": count, "target_color": int(after.grid[ys[0], xs[0]]), "offsets": offsets}))
            
            # conditional_recolor
            if not mapping_valid:
                target_colors = set(after.grid[changed])
                if len(target_colors) == 1:
                    target_c = list(target_colors)[0]
                    source_colors = set(before.grid[changed])
                    if len(source_colors) == 1 and target_c not in source_colors:
                        source_c = int(next(iter(source_colors)))
                        neighbor_counts: Counter[int] = Counter()
                        valid = True
                        for x, y in zip(xs, ys):
                            local_neighbors = {
                                int(before.grid[ny, nx])
                                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                                if 0 <= nx < before.width and 0 <= ny < before.height and int(before.grid[ny, nx]) != source_c
                            }
                            if not local_neighbors:
                                valid = False
                                break
                            for color in local_neighbors:
                                neighbor_counts[color] += 1
                        if valid and neighbor_counts:
                            neighbor_color, support = neighbor_counts.most_common(1)[0]
                            if support >= len(xs):
                                candidates.append(self._make("conditional_recolor", action.name, {"source_color": source_c, "target_color": int(target_c), "neighbor_color": int(neighbor_color)}))
                    
        data = action.data_dict
        if "x" in data and "y" in data and len(changed[0]) > 1:
            cx, cy = int(data["x"]), int(data["y"])
            if 0 <= cx < after.width and 0 <= cy < after.height:
                if after.grid[cy, cx] != before.grid[cy, cx]:
                    target_color = after.grid[cy, cx]
                    if np.all(after.grid[changed] == target_color):
                        changed_set = set(zip(changed[1], changed[0]))
                        if (cx, cy) in changed_set:
                            q = [(cx, cy)]
                            visited = {(cx, cy)}
                            while q:
                                curr_x, curr_y = q.pop(0)
                                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                                    nx, ny = curr_x + dx, curr_y + dy
                                    if (nx, ny) in changed_set and (nx, ny) not in visited:
                                        visited.add((nx, ny))
                                        q.append((nx, ny))
                            if len(visited) == len(changed_set):
                                candidates.append(self._make("flood_fill", action.name, {"x": cx, "y": cy, "target_color": int(target_color)}))

        data = action.data_dict
        if "x" in data and "y" in data and 0 <= data["x"] < before.width and 0 <= data["y"] < before.height:
            click_color = int(before.grid[data["y"], data["x"]])
            component = next((c for c in before.components if (
                data["x"], data["y"]) in c.cells), None)
            if component is not None:
                after_values = [int(after.grid[cy, cx])
                                for cx, cy in component.cells]
                if after_values and all(v == before.background for v in after_values):
                    candidates.append(
                        self._make(
                            "component_delete",
                            action.name,
                            {"clicked_color": click_color,
                                "background": before.background},
                        )
                    )
                elif after_values and len(set(after_values)) == 1 and after_values[0] != click_color:
                    candidates.append(
                        self._make(
                            "component_recolor",
                            action.name,
                            {"clicked_color": click_color,
                                "target_color": after_values[0], "background": before.background},
                        )
                    )

        if transition.event.changed_count > max(8, int(before.grid.size * 0.08)) or before.field_mode:
            rules: dict[bytes, int] = {}
            conflicts: set[bytes] = set()
            padded = np.pad(before.grid, 1, mode="constant",
                            constant_values=-1)
            for yy in range(before.height):
                for xx in range(before.width):
                    patch = np.ascontiguousarray(
                        padded[yy:yy+3, xx:xx+3], dtype=np.int16).tobytes()
                    target = int(after.grid[yy, xx])
                    if patch in rules and rules[patch] != target:
                        conflicts.add(patch)
                    else:
                        rules[patch] = target
            for key in conflicts:
                rules.pop(key, None)
            if len(rules) >= 8:
                candidates.append(self._make(
                    "cellular_rule", action.name, {"rules": rules}))
        return candidates

    def _verify(self, program: WorldProgram, transition: Transition, before_scene: Scene | None = None) -> bool:
        if transition.before_grid is None or transition.after_grid is None:
            return False
        replay_id = _stable_hash_bytes(
            repr(
                (
                    transition.level,
                    transition.step_index,
                    transition.before_exact,
                    transition.action.key,
                    transition.after_exact,
                )
            ).encode(),
            12,
        )
        if replay_id in program.verified_replay_ids:
            return False
        predicted = program.simulate(
            transition.before_grid, transition.action, before_scene)
        if predicted is None or predicted.shape != transition.after_grid.shape:
            return True
        program.verified_replay_ids.add(replay_id)
        program.eligible_replays += 1
        exact = np.array_equal(predicted, transition.after_grid)
        accuracy = float(np.mean(predicted == transition.after_grid))
        program.cell_accuracy_sum += accuracy
        if exact:
            program.exact_matches += 1
            program.support += 1
            program.current_level_support += 1
        elif accuracy >= self.config.program_min_cell_accuracy:
            program.support += 1
            program.current_level_support += 1
        else:
            program.conflicts += 1
            program.current_level_conflicts += 1
        program.evidence.append(transition.step_index)
        if (
            program.current_level_support >= self.config.program_min_support
            and program.current_level_conflicts <= max(1, program.current_level_support // 2)
            and program.support >= self.config.program_min_support
            and program.confidence >= self.config.program_min_confidence
            and program.cell_accuracy >= self.config.program_min_cell_accuracy
        ):
            program.status = "verified"
        elif program.current_level_conflicts > program.current_level_support + 1:
            program.status = "rejected"
        return True

    def _trim(self) -> None:
        if len(self.programs) <= self.config.max_programs:
            return
        ranked = sorted(
            self.programs.values(),
            key=lambda p: (
                p.status == "verified",
                p.kind != "exact_replay",
                p.confidence,
                p.support - p.conflicts,
                max(p.evidence) if p.evidence else -1,
            ),
            reverse=True,
        )[: self.config.max_programs]
        self.programs = {p.program_id: p for p in ranked}
        self.signature_index = {
            self._signature(p.kind, p.action_name, p.payload): p.program_id
            for p in ranked
        }
        self.generated_sources = {
            pid: source for pid, source in self.generated_sources.items() if pid in self.programs
        }

    def record(self, transition: Transition, before: Scene, after: Scene) -> None:
        if not self.config.enable_programs:
            return
        self._prediction_cache.clear()
        self._simulation_cache.clear()
        self.recent_transitions.append(transition)
        candidates = self._induce_candidates(transition, before, after)
        
        verify_budget = self.config.program_verify_limit
        replay_slice = list(self.recent_transitions)[-self.config.program_replay_window:]
        
        # Verify newly induced candidates against recent replay slice
        # Use reversed order to evaluate most recent transitions first
        for program in candidates:
            if verify_budget <= 0:
                break
            for replay in reversed(replay_slice):
                if verify_budget <= 0:
                    break
                if self._verify(program, replay, before if replay is transition else None):
                    verify_budget -= 1
                    
        # Verify existing retained programs against the single new transition
        for program in list(self.programs.values()):
            if verify_budget <= 0:
                break
            if program not in candidates and program.status != "rejected":
                if self._verify(program, transition, before):
                    verify_budget -= 1
                    
        self._trim()

    def predict_grid(
        self,
        grid: np.ndarray,
        action: ActionSpec,
        scene: Scene | None = None,
        allow_provisional: bool = False,
    ) -> ProgramPrediction | None:
        # Only replay-safe programs may drive planning. Exact-state replay is
        # safe after one observation because it applies solely to the identical
        # state; every generalized program must reach ``verified`` status.
        anchor: tuple[Any, ...] = ()
        if scene is not None:
            controlled = next(
                (e for e in scene.entities if e.entity_id == scene.controlled_entity_id), None)
            if controlled is not None:
                anchor = (controlled.color, round(
                    controlled.centroid[0], 2), round(controlled.centroid[1], 2))
        cache_key = (_stable_hash_bytes(np.ascontiguousarray(
            grid, dtype=np.int16).tobytes(), 10), action.key, anchor, allow_provisional)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        options: list[tuple[float, WorldProgram, np.ndarray]] = []
        for program in self.programs.values():
            if program.status == "rejected" or program.action_name != action.name:
                continue
            if program.kind != "exact_replay" and program.status != "verified":
                if not (
                    allow_provisional
                    and self.config.allow_transferred_program_one_step
                    and program.status == "transferred_provisional"
                ):
                    continue
            confidence = 1.0 if program.kind == "exact_replay" else program.confidence
            if program.kind != "exact_replay" and confidence < self.config.program_min_confidence:
                continue
            coordinate_dependent = program.kind in {
                "local_patch_replace", "component_delete", "component_recolor"}
            semantic_action_key: Any = action.key if coordinate_dependent else (
                action.name, ())
            sim_key = (
                program.program_id, cache_key[0], semantic_action_key, anchor if coordinate_dependent else ())
            if sim_key in self._simulation_cache:
                predicted = self._simulation_cache[sim_key]
            else:
                predicted = program.simulate(grid, action, scene)
                if len(self._simulation_cache) > 4096:
                    self._simulation_cache.clear()
                self._simulation_cache[sim_key] = predicted
            if predicted is None or predicted.shape != grid.shape:
                continue
            generality_bonus = 0.08 if program.kind not in {
                "exact_replay", "local_patch_replace"} else 0.0
            options.append((confidence + generality_bonus, program, predicted))
        if not options:
            if len(self._prediction_cache) > 2048:
                self._prediction_cache.clear()
            self._prediction_cache[cache_key] = None
            return None
        options.sort(key=lambda item: item[0], reverse=True)
        _, program, predicted = options[0]
        alternatives = [p for _, _, p in options[1:4]]
        uncertainty = 0.0
        if alternatives:
            disagreement = [1.0 - float(np.mean(predicted == alt))
                            for alt in alternatives]
            uncertainty = float(np.mean(disagreement))
        result = ProgramPrediction(
            grid=np.ascontiguousarray(predicted, dtype=np.int16),
            confidence=1.0 if program.kind == "exact_replay" else program.confidence,
            program_id=program.program_id,
            kind=program.kind,
            expected_effect=f"{program.kind}:changed={int(np.count_nonzero(predicted != grid))}",
            uncertainty=uncertainty,
        )
        if len(self._prediction_cache) > 2048:
            self._prediction_cache.clear()
        self._prediction_cache[cache_key] = result
        return result

    def ingest_model_program(self, spec: Mapping[str, Any], transitions: Sequence[Transition]) -> WorldProgram | None:
        """Accept only a small declarative DSL and activate it only after replay."""
        try:
            kind = str(spec.get("kind", ""))
            action = str(spec.get("action", "")).upper()
            params = dict(spec.get("params", {}))
        except Exception:
            return None
        allowed = {"translation", "color_map", "component_delete", "component_recolor",
                   "line_connect", "drag_component", "gravity", "flood_fill",
                   "copy_pattern", "conditional_recolor", "count_and_fill"}
        if kind not in allowed or not action.startswith("ACTION"):
            return None
        payload: dict[str, Any]
        try:
            if kind == "translation":
                payload = {
                    "dx": int(params["dx"]),
                    "dy": int(params["dy"]),
                    "selector_color": int(params["selector_color"]),
                    "background": int(params.get("background", 0)),
                }
            elif kind == "color_map":
                payload = {"mapping": {int(k): int(v) for k, v in dict(params["mapping"]).items()}}
            elif kind == "component_delete":
                payload = {"background": int(params.get("background", 0)), "clicked_color": int(params.get("clicked_color", -1))}
            elif kind == "component_recolor":
                payload = {"target_color": int(params["target_color"]), "background": int(params.get("background", 0)), "clicked_color": int(params.get("clicked_color", -1))}
            elif kind == "line_connect":
                if "p1" in params and "p2" in params:
                    x1, y1 = params["p1"]
                    x2, y2 = params["p2"]
                    payload = {"color": int(params["color"]), "p1": (int(x1), int(y1)), "p2": (int(x2), int(y2)), "background": int(params.get("background", 0))}
                else:
                    payload = {"color": int(params["color"]), "direction": str(params["direction"]), "background": int(params.get("background", 0))}
            elif kind == "drag_component":
                payload = {"dx": int(params["dx"]), "dy": int(params["dy"]), "background": int(params.get("background", 0))}
            elif kind == "gravity":
                payload = {"dx": int(params["dx"]), "dy": int(params["dy"]), "background": int(params.get("background", 0))}
            elif kind == "flood_fill":
                payload = {"x": int(params["x"]), "y": int(params["y"]), "target_color": int(params["target_color"])}
            elif kind == "copy_pattern":
                payload = {"source_entity_id": str(params.get("source_entity_id", "")), "offsets": tuple((int(x), int(y)) for x, y in params["offsets"]), "colors": tuple(int(v) for v in params["colors"])}
            elif kind == "conditional_recolor":
                payload = {"source_color": int(params["source_color"]), "target_color": int(params["target_color"]), "neighbor_color": int(params["neighbor_color"])}
            elif kind == "count_and_fill":
                payload = {"count": int(params["count"]), "target_color": int(params["target_color"]), "offsets": tuple((int(x), int(y)) for x, y in params["offsets"])}
            else:
                return None
        except Exception:
            return None
        self._prediction_cache.clear()
        self._simulation_cache.clear()
        program = self._make(kind, action, payload, source="local_model")
        for transition in transitions[-64:]:
            self._verify(program, transition)
        return program if program.status == "verified" else None

    def summary(self, limit: int = 8) -> list[dict[str, Any]]:
        ranked = sorted(self.programs.values(), key=lambda p: (
            p.status == "verified", p.confidence, p.support), reverse=True)
        return [
            {
                "id": p.program_id,
                "kind": p.kind,
                "action": p.action_name,
                "confidence": round(p.confidence, 3),
                "replay_accuracy": round(p.replay_accuracy, 3),
                "cell_accuracy": round(p.cell_accuracy, 3),
                "status": p.status,
                "source": p.source,
                "source_code": self.generated_sources.get(p.program_id, "")[:180],
            }
            for p in ranked[:limit]
        ]


class GoalAlignmentVerifier:
    """Level-local goal and safety verification.

    Raw colour-specific invariants are intentionally reset at every level
    boundary. They remain available across death/retry attempts inside the same
    level, where they are useful, but cannot pollute later level semantics.
    """

    def __init__(self, config: Config, goals: GoalHypothesisManager) -> None:
        self.config = config
        self.goals = goals
        self.reset_level()

    def reset_level(self) -> None:
        self.progress_events = 0
        self.progress_preserved_colors: Counter[int] = Counter()
        self.fatal_color_losses: Counter[int] = Counter()
        self.safe_change_ratios: deque[float] = deque(maxlen=24)

    def observe(self, transition: Transition, before: Scene, after: Scene) -> None:
        """Learn level-local invariants from repeated progress/fatal evidence."""
        if not self.config.enable_alignment_gate:
            return
        before_counts = _grid_color_counts(before.grid)
        after_counts = _grid_color_counts(after.grid)
        if transition.event.level_delta > 0 or transition.event.win:
            self.progress_events += 1
            self.safe_change_ratios.append(
                float(np.count_nonzero(before.grid != after.grid)) /
                max(1, before.grid.size)
            )
            for color, count in before_counts.items():
                if color != before.background and after_counts.get(color, 0) == count:
                    self.progress_preserved_colors[color] += 1
        if transition.event.game_over:
            for color, count in before_counts.items():
                if color != before.background and after_counts.get(color, 0) < count:
                    self.fatal_color_losses[color] += 1
        return

    def verify(
        self,
        scene: Scene,
        action: ActionSpec,
        legal_names: set[str],
        prediction: ProgramPrediction | None,
        is_probe: bool,
    ) -> AlignmentDecision:
        active = self.goals.active(
            limit=4) if self.config.enable_goal_inference else []
        goal_ids = tuple(g.goal_id for g in active)
        if action.name not in legal_names:
            return AlignmentDecision(False, -10.0, -1.0, 1.0, goal_ids, ("illegal_action",))
        data = action.data_dict
        if ("x" in data) != ("y" in data):
            return AlignmentDecision(False, -10.0, -1.0, 1.0, goal_ids, ("incomplete_coordinates",))
        if "x" in data and not (0 <= data["x"] < scene.width and 0 <= data["y"] < scene.height):
            return AlignmentDecision(False, -10.0, -1.0, 1.0, goal_ids, ("coordinate_out_of_bounds",))



        if not self.config.enable_alignment_gate:
            return AlignmentDecision(True, 0.0, 0.0, 0.0, goal_ids, ("alignment_disabled",))

        reasons: list[str] = []
        if prediction is None:
            return AlignmentDecision(
                is_probe,
                -0.01 if is_probe else -0.25,
                0.0,
                0.35,
                goal_ids,
                ("unmodelled_probe" if is_probe else "unmodelled_nonprobe",),
            )

        predicted = prediction.grid
        if predicted.shape != scene.grid.shape:
            return AlignmentDecision(False, -5.0, -1.0, 1.0, goal_ids, ("shape_mismatch",))
        changed = int(np.count_nonzero(predicted != scene.grid))
        change_ratio = changed / max(1, scene.grid.size)
        goal_delta = self.goals.prediction_delta(
            scene, predicted) if self.config.enable_goal_inference else 0.0
        risk = 0.0
        score = 1.8 * goal_delta + 0.7 * \
            prediction.confidence - 0.8 * prediction.uncertainty
        if changed == 0:
            risk += 0.35
            score -= 0.9
            reasons.append("predicted_noop")
        if change_ratio > 0.75 and goal_delta <= 0.02:
            risk += 0.8
            score -= 1.5
            reasons.append("unjustified_global_change")

        before_counts = _grid_color_counts(scene.grid)
        after_counts = _grid_color_counts(predicted)

        # Safety Check: Prevent repeating known fatal color losses
        for color, count in before_counts.items():
            if color != scene.background and after_counts.get(color, 0) < count:
                if self.fatal_color_losses.get(color, 0) > 0:
                    return AlignmentDecision(False, -20.0, -1.0, 1.0, goal_ids, (f"fatal_color_loss_c{color}",))
        collect_targets = {
            int(g.params.get("color", -999))
            for g in active
            if g.kind == "collect_color" and g.confidence >= 0.25
        }
        for color, count in before_counts.items():
            if color == scene.background or count > 6:
                continue
            lost = count - after_counts.get(color, 0)
            if lost > 0 and color not in collect_targets:
                risk += min(0.55, 0.12 * lost)
                score -= 0.18 * lost
                reasons.append(f"rare_color_loss:{color}")

        enough_progress = self.progress_events >= self.config.alignment_min_progress_events
        if enough_progress:
            for color, preserved in self.progress_preserved_colors.items():
                if preserved < self.config.alignment_preservation_support:
                    continue
                if preserved / max(1, self.progress_events) < self.config.alignment_preservation_ratio:
                    continue
                if after_counts.get(color, 0) != before_counts.get(color, 0):
                    # Repeated preservation is a cautionary penalty, not by itself
                    # a hard prohibition. Goal evidence can override it.
                    risk += 0.28
                    score -= 0.38
                    reasons.append(
                        f"level_progress_invariant_violation:{color}")

        for color, fatal_support in self.fatal_color_losses.items():
            if fatal_support < self.config.alignment_fatal_min_support:
                continue
            if after_counts.get(color, 0) < before_counts.get(color, 0):
                risk += min(0.72, 0.30 + 0.12 * fatal_support)
                score -= 0.48
                reasons.append(f"repeated_level_fatal_loss:{color}")

        if self.safe_change_ratios and goal_delta <= 0.0:
            observed_max = max(self.safe_change_ratios)
            if change_ratio > max(0.35, observed_max * 1.8):
                risk += 0.35
                score -= 0.45
                reasons.append("outside_observed_progress_envelope")

        controlled = next(
            (e for e in scene.entities if e.entity_id == scene.controlled_entity_id), None)
        if controlled is not None and after_counts.get(controlled.color, 0) == 0:
            risk += 1.0
            score -= 2.0
            reasons.append("controlled_entity_disappears")

        if goal_delta > 0.02:
            reasons.append(f"goal_progress:{goal_delta:.3f}")
        elif goal_delta < -0.03:
            risk += 0.35
            score -= 0.5
            reasons.append(f"goal_regression:{goal_delta:.3f}")
        if prediction.confidence < self.config.program_min_confidence:
            risk += 0.3
            score -= 0.3
            reasons.append("low_model_confidence")

        allowed = score >= self.config.alignment_min_score and risk < 1.15
        if is_probe and prediction.confidence < 0.8:
            allowed = risk < 0.75 and score > -0.45
        return AlignmentDecision(allowed, score, goal_delta, risk, goal_ids, tuple(reasons or ["aligned_neutral"]))


# ---------------------------------------------------------------------------
# Action mapping and bounded model-based path planning
# ---------------------------------------------------------------------------


class ActionDynamics:
    def __init__(self) -> None:
        self.vectors: dict[str, Counter[tuple[int, int]]
                           ] = defaultdict(Counter)
        self.action_stats: dict[str, Counter[str]] = defaultdict(Counter)
        self.effect_signatures: dict[str, Counter[str]] = defaultdict(Counter)

    def reset_all(self) -> None:
        self.vectors.clear()
        self.action_stats.clear()
        self.effect_signatures.clear()

    def record(self, action: ActionSpec, event: Event, scene: Scene | None) -> None:
        stats = self.action_stats[action.name]
        stats["observed"] += 1
        if action.data:
            stats["targeted"] += 1
        else:
            stats["simple"] += 1
        if event.no_op:
            stats["noop"] += 1
        if event.game_over:
            stats["death"] += 1
        if event.level_delta > 0 or event.win:
            stats["progress"] += 1
        if event.topology_change:
            stats["topology"] += 1
        if event.changed_count > 0:
            stats["changed"] += 1
        if event.effect_signature:
            self.effect_signatures[action.name][event.effect_signature] += 1

        controlled_id = None if scene is None else scene.controlled_entity_id
        if not action.data and controlled_id is not None:
            for entity_id, dx, dy in event.entity_moves:
                if entity_id == controlled_id and (dx or dy):
                    ndx = 0 if dx == 0 else (1 if dx > 0 else -1)
                    ndy = 0 if dy == 0 else (1 if dy > 0 else -1)
                    self.vectors[action.name][(ndx, ndy)] += 1
                    stats["movement_like"] += 1
                elif dx or dy:
                    stats["noncontrolled_motion"] += 1

        if scene is not None:
            change_ratio = event.changed_count / max(1, scene.grid.size)
            if change_ratio <= 0.03:
                stats["local"] += 1
            elif change_ratio >= 0.25:
                stats["global"] += 1
            else:
                stats["regional"] += 1
            if event.appeared_colors or event.disappeared_colors:
                stats["recolor_or_spawn"] += 1
            if event.changed_count > 0 and not event.entity_moves:
                stats["field_effect"] += 1

    def action_for_vector(self, vector: tuple[int, int], legal_names: set[str]) -> str | None:
        ranked: list[tuple[int, str]] = []
        for action_name, counts in self.vectors.items():
            if action_name not in legal_names or not counts:
                continue
            best_vector, support = counts.most_common(1)[0]
            if best_vector == vector:
                ranked.append((support, action_name))
        return max(ranked)[1] if ranked else None

    def reliable_vectors(self, legal_names: set[str]) -> dict[tuple[int, int], str]:
        result: dict[tuple[int, int], tuple[int, str]] = {}
        for name, counts in self.vectors.items():
            if name not in legal_names or not counts:
                continue
            vector, support = counts.most_common(1)[0]
            total = sum(counts.values())
            if support >= 1 and support / total >= 0.65:
                if vector not in result or support > result[vector][0]:
                    result[vector] = (support, name)
        return {vector: name for vector, (_, name) in result.items()}

    def summary_for_prompt(self, legal_names: set[str], limit: int = 8) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, stats in self.action_stats.items():
            if name not in legal_names:
                continue
            observed = max(1, stats["observed"])
            sig = self.effect_signatures.get(name, Counter())
            dominant_sig, dominant_support = ("", 0)
            if sig:
                dominant_sig, dominant_support = sig.most_common(1)[0]
            rows.append({
                "action": name,
                "observed": observed,
                "noop_rate": round(stats["noop"] / observed, 3),
                "death_rate": round(stats["death"] / observed, 3),
                "progress_rate": round(stats["progress"] / observed, 3),
                "movement_rate": round(stats["movement_like"] / observed, 3),
                "topology_rate": round(stats["topology"] / observed, 3),
                "targeted_rate": round(stats["targeted"] / observed, 3),
                "scope_hint": (
                    "global" if stats["global"] >= max(stats["local"], stats["regional"]) else
                    "regional" if stats["regional"] >= stats["local"] else
                    "local"
                ),
                "dominant_effect_signature": dominant_sig[:16],
                "dominant_effect_support": dominant_support,
            })
        rows.sort(key=lambda row: (row["observed"], row["progress_rate"], -row["noop_rate"]), reverse=True)
        return rows[:limit]

    def prior_for_action(self, action_name: str, program_kind: str | None = None) -> tuple[float, list[str]]:
        stats = self.action_stats.get(action_name)
        if not stats:
            return 0.0, []
        observed = max(1, stats["observed"])
        reasons: list[str] = []
        prior = 0.0
        noop_rate = stats["noop"] / observed
        death_rate = stats["death"] / observed
        progress_rate = stats["progress"] / observed
        if progress_rate > 0.0:
            prior += 0.35 * progress_rate
            reasons.append(f"progress_rate={progress_rate:.2f}")
        if noop_rate > 0.55:
            prior -= 0.45 * noop_rate
            reasons.append(f"high_noop={noop_rate:.2f}")
        if death_rate > 0.15:
            prior -= 0.65 * death_rate
            reasons.append(f"death_rate={death_rate:.2f}")
        if program_kind:
            kind = str(program_kind)
            if kind in {"translation", "drag_component"} and stats["movement_like"] > 0:
                prior += 0.25
                reasons.append("movement_semantics_match")
            if kind in {"conditional_recolor", "color_map", "component_recolor", "flood_fill"} and stats["recolor_or_spawn"] > 0:
                prior += 0.20
                reasons.append("recolor_semantics_match")
            if kind in {"component_delete", "count_and_fill"} and stats["topology"] > 0:
                prior += 0.18
                reasons.append("topology_semantics_match")
        return prior, reasons


@dataclass(frozen=True, slots=True)
class NavigationTarget:
    score: float = 0.0
    entity_id: str = ""
    cells: tuple[Point, ...] = ()
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class PathPlannerResult:
    action: ActionSpec | None = None
    diagnostics: Any = None


class PathPlannerDiagnostics(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass(frozen=True, slots=True)
class ControlledAssembly:
    centroid: tuple[float, float] = (0.0, 0.0)
    cells: tuple[Point, ...] = ()
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    member_entity_ids: tuple[str, ...] = ()
    representative_entity_id: str | None = None
    confidence: float = 1.0


class ControlInference:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.recent_observations: list[tuple[tuple[tuple[str, int, int], ...], bool]] = []
        self._last_diagnostics: dict[str, Any] = {}

    def record(self, transition: Transition) -> None:
        if transition.event.entity_moves:
            self.recent_observations.append((transition.event.entity_moves, bool(transition.action.data)))
            if len(self.recent_observations) > self.config.control_group_max_gap + 2:
                self.recent_observations.pop(0)

    def controlled_assembly(self, scene: Scene) -> ControlledAssembly | None:
        self._last_diagnostics = {
            "control_group_candidate_count": 0,
            "control_group_joint_moves": 0,
            "control_group_total_moves": 0,
            "control_group_confidence": 0.0,
            "control_group_coherence": 0.0,
            "control_group_membership_stability": 0.0,
            "control_group_spatial_coherence": 0.0,
            "control_group_complex_ratio": 0.0,
            "control_group_rejection_reason": "",
        }
        if not scene.controlled_entity_id:
            return None
        
        controlled_e = next((x for x in scene.entities if x.entity_id == scene.controlled_entity_id), None)
        if not controlled_e:
            return None
            
        co_movers = {scene.controlled_entity_id}
        joint_moves = 0
        total_moves = 0
        coherent_match_sum = 0.0
        complex_moves = 0
        support_sets: list[set[str]] = []
        
        if self.config.enable_control_groups and self.recent_observations:
            for move_set, is_complex in self.recent_observations:
                ctrl_move = next(( (dx, dy) for eid, dx, dy in move_set if eid == scene.controlled_entity_id ), None)
                if ctrl_move is not None:
                    total_moves += 1
                    if is_complex:
                        complex_moves += 1
                    cdx, cdy = ctrl_move
                    matching = {eid for eid, dx, dy in move_set if dx == cdx and dy == cdy}
                    if len(co_movers) == 1:
                        co_movers.update(matching)
                    else:
                        overlap = co_movers.intersection(matching)
                        if len(overlap) / len(co_movers) >= self.config.control_group_membership_overlap:
                            co_movers = overlap
                        else:
                            co_movers.intersection_update(matching)
                    if len(co_movers) > 1:
                        joint_moves += 1
                        support_sets.append(set(matching))
                        coherent_match_sum += len(matching.intersection(co_movers)) / max(1, len(co_movers))
        
        self._last_diagnostics["control_group_candidate_count"] = len(co_movers)
        self._last_diagnostics["control_group_joint_moves"] = joint_moves
        self._last_diagnostics["control_group_total_moves"] = total_moves
        complex_ratio = complex_moves / max(1, total_moves)
        self._last_diagnostics["control_group_complex_ratio"] = complex_ratio
        if total_moves < self.config.control_applicability_min_observations:
            self._last_diagnostics["control_group_rejection_reason"] = "insufficient_observations"
        elif complex_ratio > self.config.control_complex_action_ratio:
            self._last_diagnostics["control_group_rejection_reason"] = "complex_ratio_too_high"
        
        if (
            total_moves >= self.config.control_applicability_min_observations
            and complex_ratio <= self.config.control_complex_action_ratio
            and len(co_movers) >= self.config.control_group_min_members
            and len(co_movers) <= self.config.control_group_max_members
        ):
            confidence = joint_moves / max(1, total_moves)
            coherence = coherent_match_sum / max(1, joint_moves)
            stability = 0.0
            if len(support_sets) >= 2:
                overlaps = []
                for prev, curr in zip(support_sets, support_sets[1:]):
                    overlaps.append(len(prev.intersection(curr)) / max(1, len(prev.union(curr))))
                stability = float(sum(overlaps) / max(1, len(overlaps)))
            elif support_sets:
                stability = 1.0
            self._last_diagnostics["control_group_confidence"] = confidence
            self._last_diagnostics["control_group_coherence"] = coherence
            self._last_diagnostics["control_group_membership_stability"] = stability
            if (
                joint_moves >= self.config.control_group_min_support
                and confidence >= self.config.control_group_min_confidence
                and coherence >= self.config.control_group_min_coherence
                and stability >= self.config.control_group_membership_overlap
            ):
                cells = []
                for eid in co_movers:
                    e = next((x for x in scene.entities if x.entity_id == eid), None)
                    if e:
                        cells.extend(e.cells)
                if cells:
                    cells = list(set(cells))
                    xs = [c[0] for c in cells]
                    ys = [c[1] for c in cells]
                    cx = sum(xs) // len(xs)
                    cy = sum(ys) // len(ys)
                    bbox = (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
                    bbox_area = bbox[2] * bbox[3]
                    spatial_coherence = len(cells) / max(1, bbox_area)
                    self._last_diagnostics["control_group_spatial_coherence"] = spatial_coherence
                    if spatial_coherence >= self.config.control_group_min_spatial_coherence:
                        return ControlledAssembly(
                            centroid=(cx, cy),
                            cells=tuple(cells),
                            bbox=bbox,
                            member_entity_ids=tuple(sorted(list(co_movers))),
                            representative_entity_id=scene.controlled_entity_id,
                            confidence=confidence,
                        )
                    self._last_diagnostics["control_group_rejection_reason"] = "spatial_coherence_too_low"
            else:
                self._last_diagnostics["control_group_rejection_reason"] = "confidence_or_coherence_too_low"
                
        return ControlledAssembly(
            centroid=controlled_e.centroid,
            cells=controlled_e.cells,
            bbox=controlled_e.bbox,
            member_entity_ids=(controlled_e.entity_id,),
            representative_entity_id=controlled_e.entity_id,
            confidence=1.0,
        )

    def diagnostics(self, scene: Scene) -> dict[str, Any]:
        return {
            "controlled_entity_id": scene.controlled_entity_id,
            "entity_confidence": scene.entity_confidence,
            **self._last_diagnostics,
        }


class TerrainPassabilityModel:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.successes: Counter[int] = Counter()
        self.failures: Counter[int] = Counter()

    def passable_colors(self, scene: Scene) -> set[int]:
        colors = {scene.background}
        for color, success_count in self.successes.items():
            failure_count = self.failures[color]
            total = success_count + failure_count * self.config.passability_failure_penalty
            if total > 0:
                confidence = success_count / total
                if success_count >= self.config.passability_min_support and confidence >= self.config.passability_min_confidence:
                    colors.add(color)
        for color, failure_count in self.failures.items():
            success_count = self.successes.get(color, 0)
            total = success_count + failure_count * self.config.passability_failure_penalty
            if total > 0:
                failure_confidence = (failure_count * self.config.passability_failure_penalty) / total
                if failure_count >= self.config.passability_min_support and failure_confidence >= self.config.passability_min_confidence:
                    colors.discard(color)
        return colors

    def passability_mask(self, scene: Scene) -> np.ndarray:
        colors = list(self.passable_colors(scene))
        mask = np.isin(scene.grid, colors)
        return mask

    def record(self, transition: Transition, scene: Scene) -> None:
        action = transition.action
        event = transition.event
        if action.name.startswith("ACTION"):
            data = action.data_dict
            if "x" in data and "y" in data:
                x, y = int(data["x"]), int(data["y"])
                if 0 <= x < scene.width and 0 <= y < scene.height:
                    target_color = int(scene.grid[y, x])
                    if target_color != scene.background:
                        if event.no_op:
                            self.failures[target_color] += 1
                        elif event.changed_count > 0:
                            self.successes[target_color] += 1

    def diagnostics(self, scene: Scene) -> dict[str, Any]:
        return {
            "passable_colors": list(self.passable_colors(scene)),
            "successes": dict(self.successes),
            "failures": dict(self.failures),
        }


class PathPlanner:
    _CARDINAL_VECTORS: frozenset[tuple[int, int]] = frozenset(
        {(1, 0), (-1, 0), (0, 1), (0, -1)})

    def __init__(self, config: Config, dynamics: ActionDynamics, control_inference: ControlInference | None = None, passability: TerrainPassabilityModel | None = None) -> None:
        if control_inference is None:
            control_inference = ControlInference(config)
        if passability is None:
            passability = TerrainPassabilityModel(config)
        self.config = config
        self.dynamics = dynamics
        self.control_inference = control_inference
        self.passability = passability
        self.reset_level()

    def reset_level(self) -> None:
        self.current_step = 0
        self.recent_anchors: deque[Point] = deque(
            maxlen=max(4, self.config.path_cycle_window))
        self.target_cooldown_until: dict[str, int] = {}
        self.active_target_key: str | None = None
        self.last_path_length: int | None = None
        self.no_progress_count = 0
        self.path_cycle_count = 0

    def reset_attempt(self) -> None:
        self.recent_anchors.clear()
        self.active_target_key = None
        self.last_path_length = None
        self.no_progress_count = 0

    def set_step(self, step: int) -> None:
        self.current_step = max(self.current_step, int(step))
        expired = [k for k, until in self.target_cooldown_until.items()
                   if self.current_step >= until]
        for k in expired:
            self.target_cooldown_until.pop(k, None)

    @staticmethod
    def _target_key(entity_id: str, contact_mode: str) -> str:
        return f"{entity_id}:{contact_mode or 'any'}"

    def _target_is_cooled(self, entity_id: str) -> bool:
        prefix = f"{entity_id}:"
        return any(k.startswith(prefix) and self.current_step < until for k, until in self.target_cooldown_until.items())

    def _controlled(self, scene: Scene) -> ControlledAssembly | None:
        return self.control_inference.controlled_assembly(scene)

    @staticmethod
    def _anchor(assembly: ControlledAssembly) -> Point:
        return (int(round(assembly.centroid[0])), int(round(assembly.centroid[1])))

    def _candidate_targets(self, scene: Scene, controlled: ControlledAssembly) -> list[NavigationTarget]:
        targets: list[NavigationTarget] = []
        controlled_ids = set(controlled.member_entity_ids)
        for entity in scene.entities:
            if entity.entity_id in controlled_ids or entity.area <= 0:
                continue
            if self._target_is_cooled(entity.entity_id):
                continue
            centroid = (int(round(entity.centroid[0])), int(
                round(entity.centroid[1])))
            targets.append(NavigationTarget(score=1.0, entity_id=entity.entity_id,
                           cells=entity.cells, bbox=entity.bbox, centroid=centroid))
        return targets

    def _snapshot(
        self,
        scene: Scene,
        legal_names: set[str],
        *,
        attempted: bool,
        status: str,
        controlled: ControlledAssembly | None = None,
        candidate_target_count: int = 0,
        reachable_target_count: int = 0,
        passable_cell_count: int = 0,
        explored_cell_count: int = 0,
        assembly_footprint_cell_count: int = 0,
        assembly_footprint_bbox: tuple[int, int, int, int] | None = None,
        valid_anchor_count: int = 0,
        invalid_anchor_count: int = 0,
        blocked_by_terrain_count: int = 0,
        blocked_by_boundary_count: int = 0,
        target_goal_anchor_count: int = 0,
        target_contact_mode: str = "",
        closest_unreachable_distance: float | None = None,
        selected_target_entity_id: str | None = None,
        path_length: int = 0,
        path_cycle_detected: bool = False,
        path_no_progress_count: int | None = None,
        selected_action: str | None = None,
        selected_vector: tuple[int, int] | None = None,
        selected_target: Point | None = None
    ) -> PathPlannerDiagnostics:
        controlled = controlled or self._controlled(scene)
        vector_actions = self.dynamics.reliable_vectors(legal_names)
        cardinal = {v: a for v, a in vector_actions.items()
                    if v in self._CARDINAL_VECTORS}
        reliable_vectors = tuple(
            sorted((int(v[0]), int(v[1]), str(a)) for v, a in vector_actions.items()))

        control = self.control_inference.diagnostics(scene)
        terrain = self.passability.diagnostics(scene)

        return PathPlannerDiagnostics(
            attempted=attempted,
            status=status,
            field_mode=bool(scene.field_mode),
            entity_confidence=float(scene.entity_confidence),
            entity_confidence_threshold=float(
                self.config.entity_confidence_threshold),
            controlled_entity_id=(
                None if controlled is None else controlled.representative_entity_id),
            controlled_entity_controllability=(
                0.0 if controlled is None else float(controlled.confidence)),
            entity_count=len(scene.entities),
            reliable_vector_count=len(vector_actions),
            cardinal_vector_count=len(cardinal),
            candidate_target_count=int(candidate_target_count),
            reachable_target_count=int(reachable_target_count),
            passable_cell_count=int(passable_cell_count),
            explored_cell_count=int(explored_cell_count),
            reliable_vectors=reliable_vectors,

            planner_applicability=str(control.get(
                "planner_applicability", "undetermined")),
            simple_action_observations=int(
                control.get("simple_action_observations", 0)),
            complex_action_observations=int(
                control.get("complex_action_observations", 0)),
            simple_motion_events=int(control.get("simple_motion_events", 0)),
            complex_motion_events=int(control.get("complex_motion_events", 0)),
            control_candidate_count=int(
                control.get("control_candidate_count", 0)),
            best_control_candidate_id=control.get("best_control_candidate_id"),
            best_control_candidate_signature=control.get(
                "best_control_candidate_signature"),
            best_control_score=float(control.get("best_control_score", 0.0)),
            second_control_score=float(
                control.get("second_control_score", 0.0)),
            control_score_margin=float(
                control.get("control_score_margin", 0.0)),
            control_support=int(control.get("control_support", 0)),
            control_confidence=float(control.get("control_confidence", 0.0)),
            control_mean_persistence=float(
                control.get("control_mean_persistence", 0.0)),
            control_distinct_actions=int(
                control.get("control_distinct_actions", 0)),
            control_mapping_consistency=float(
                control.get("control_mapping_consistency", 0.0)),
            control_group_count=int(control.get("control_group_count", 0)),
            best_control_group_id=control.get("best_control_group_id"),
            best_control_group_score=float(
                control.get("best_control_group_score", 0.0)),
            best_control_group_support=int(
                control.get("best_control_group_support", 0)),
            best_control_group_member_count=int(
                control.get("best_control_group_member_count", 0)),
            best_control_group_members=tuple(
                control.get("best_control_group_members", ())),
            best_control_group_coherence=float(
                control.get("best_control_group_coherence", 0.0)),
            best_control_group_spatial_coherence=float(
                control.get("best_control_group_spatial_coherence", 0.0)),
            best_control_group_membership_stability=float(
                control.get("best_control_group_membership_stability", 0.0)),
            control_promotion_kind=str(
                control.get("control_promotion_kind", "")),
            control_promotion_reason=str(
                control.get("control_promotion_reason", "")),
            control_rejection_reason=str(
                control.get("control_rejection_reason", "")),
            provisional_motion_observations=int(
                control.get("provisional_motion_observations", 0)),
            promoted_control_signature=control.get(
                "promoted_control_signature"),
            promoted_control_entity_id=control.get(
                "promoted_control_entity_id"),
            promoted_control_group_id=control.get("promoted_control_group_id"),
            promoted_control_member_signatures=tuple(
                control.get("promoted_control_member_signatures", ())),
            promoted_control_member_ids=tuple(
                control.get("promoted_control_member_ids", ())),
            controlled_composite_anchor=control.get(
                "controlled_composite_anchor"),
            controlled_composite_bbox=control.get("controlled_composite_bbox"),
            control_reidentifications=int(
                control.get("control_reidentifications", 0)),

            learned_passable_color_count=int(
                terrain.get("learned_passable_color_count", 0)),
            learned_passable_colors=tuple(
                int(v) for v in terrain.get("learned_passable_colors", ())),
            passability_success_observations=int(
                terrain.get("passability_success_observations", 0)),
            passability_failure_observations=int(
                terrain.get("passability_failure_observations", 0)),
            passability_successful_moves=int(
                terrain.get("passability_successful_moves", 0)),
            passability_failed_moves=int(
                terrain.get("passability_failed_moves", 0)),
            passability_boundary_failures=int(
                terrain.get("passability_boundary_failures", 0)),

            assembly_footprint_cell_count=int(assembly_footprint_cell_count),
            assembly_footprint_bbox=assembly_footprint_bbox,
            valid_anchor_count=int(valid_anchor_count),
            invalid_anchor_count=int(invalid_anchor_count),
            blocked_by_terrain_count=int(blocked_by_terrain_count),
            blocked_by_boundary_count=int(blocked_by_boundary_count),
            target_goal_anchor_count=int(target_goal_anchor_count),
            target_contact_mode=target_contact_mode,
            closest_unreachable_distance=(
                None if closest_unreachable_distance is None else float(closest_unreachable_distance)),
            selected_target_entity_id=selected_target_entity_id,
            path_length=int(path_length),
            path_cycle_detected=bool(
                path_cycle_detected or self.path_cycle_count > 0),
            path_no_progress_count=int(
                self.no_progress_count if path_no_progress_count is None else path_no_progress_count),
            cooled_target_count=len(self.target_cooldown_until),
            selected_action=selected_action,
            selected_vector=selected_vector,
            selected_target=selected_target
        )

    def not_consulted(self, scene: Scene, legal_names: set[str], status: str) -> PathPlannerDiagnostics:
        controlled = self._controlled(scene)
        target_count = 0 if controlled is None else len(
            self._candidate_targets(scene, controlled))
        passable_colors = set(self.passability.passable_colors(scene))
        passable_count = int(np.count_nonzero(
            np.isin(scene.grid, tuple(passable_colors))))
        return self._snapshot(
            scene, legal_names, attempted=False, status=status, controlled=controlled,
            candidate_target_count=target_count, passable_cell_count=passable_count,
            assembly_footprint_cell_count=(
                0 if controlled is None else len(controlled.cells)),
            assembly_footprint_bbox=(
                None if controlled is None else controlled.bbox)
        )

    def plan(self, scene: Scene, legal_names: set[str], step: int) -> PathPlannerResult:
        self.set_step(step)
        controlled = self._controlled(scene)
        if not self.config.enable_path_planner or scene.field_mode or controlled is None:
            diag = self.not_consulted(
                scene, legal_names, "path_planner_disabled_or_not_applicable")
            return PathPlannerResult(action=None, diagnostics=diag)

        vector_actions = self.dynamics.reliable_vectors(legal_names)
        cardinal = {v: a for v, a in vector_actions.items()
                    if v in self._CARDINAL_VECTORS}
        if len(cardinal) < 2:
            diag = self.not_consulted(
                scene, legal_names, "insufficient_cardinal_vectors")
            return PathPlannerResult(action=None, diagnostics=diag)

        start = self._anchor(controlled)
        passable_colors = set(self.passability.passable_colors(scene))
        passable = np.isin(scene.grid, tuple(passable_colors))
        h, w = passable.shape
        if not (0 <= start[0] < w and 0 <= start[1] < h):
            diag = self.not_consulted(
                scene, legal_names, "start_out_of_bounds")
            return PathPlannerResult(action=None, diagnostics=diag)

        passable[start[1], start[0]] = True

        targets = self._candidate_targets(scene, controlled)
        for target in targets:
            goals: set[Point] = set()
            tx, ty = target.centroid
            for gx, gy in ((tx + 1, ty), (tx - 1, ty), (tx, ty + 1), (tx, ty - 1), (tx, ty)):
                if 0 <= gx < w and 0 <= gy < h and passable[gy, gx]:
                    goals.add((gx, gy))
            if not goals:
                continue

            queue: deque[Point] = deque([start])
            parent: dict[Point, tuple[Point, tuple[int, int]]] = {}
            seen = {start}
            reached: Point | None = None

            while queue and len(seen) < self.config.path_max_anchor_states:
                point = queue.popleft()
                if point in goals:
                    reached = point
                    break
                for vector in cardinal:
                    nx, ny = point[0] + vector[0], point[1] + vector[1]
                    nxt = (nx, ny)
                    if 0 <= nx < w and 0 <= ny < h and passable[ny, nx] and nxt not in seen:
                        seen.add(nxt)
                        parent[nxt] = (point, vector)
                        queue.append(nxt)

            if reached is None or reached == start:
                continue

            cursor = reached
            first_vector: tuple[int, int] | None = None
            path_len = 0
            while cursor != start:
                prev, vector = parent[cursor]
                first_vector = vector
                cursor = prev
                path_len += 1

            if first_vector is not None:
                action_name = cardinal[first_vector]
                spec = ActionSpec(
                    name=action_name,
                    source="path_planner",
                    predicted_effect=f"move{first_vector}",
                    score=3.0,
                )
                diag = self._snapshot(
                    scene, legal_names, attempted=True, status="path_found",
                    controlled=controlled, candidate_target_count=len(targets),
                    reachable_target_count=1, passable_cell_count=int(np.count_nonzero(passable)),
                    explored_cell_count=len(seen), assembly_footprint_cell_count=len(controlled.cells),
                    assembly_footprint_bbox=controlled.bbox, selected_target_entity_id=target.entity_id,
                    path_length=path_len, selected_action=action_name, selected_vector=first_vector,
                    selected_target=target.centroid
                )
                return PathPlannerResult(action=spec, diagnostics=diag)

        diag = self._snapshot(
            scene, legal_names, attempted=True, status="no_reachable_targets",
            controlled=controlled, candidate_target_count=len(targets), passable_cell_count=int(np.count_nonzero(passable))
        )
        return PathPlannerResult(action=None, diagnostics=diag)

    def next_action(self, scene: Scene, legal_names: set[str], time_budget_sec: float | None = None) -> PathPlannerResult:
        return self.plan(scene, legal_names, self.current_step)

    def observe_execution(self, before: Scene, after: Scene, action: ActionSpec, event: Event, step: int, diagnostics: Mapping[str, Any] | None) -> None:
        self.set_step(step)
        if action.source != "path_planner" or not isinstance(diagnostics, Mapping):
            return

        target_id = str(diagnostics.get("selected_target_entity_id") or "")
        contact_mode = str(diagnostics.get("target_contact_mode") or "")
        if not target_id:
            return
        target_key = self._target_key(target_id, contact_mode)

        before_anchor_raw = diagnostics.get("controlled_composite_anchor")
        before_anchor: Point | None = None
        if isinstance(before_anchor_raw, Sequence) and len(before_anchor_raw) >= 2:
            try:
                before_anchor = (int(round(float(before_anchor_raw[0]))), int(
                    round(float(before_anchor_raw[1]))))
            except (TypeError, ValueError):
                pass

        after_assembly = self._controlled(after)
        after_anchor = None if after_assembly is None else self._anchor(
            after_assembly)
        if after_anchor is not None:
            self.recent_anchors.append(after_anchor)
        elif before_anchor is not None:
            self.recent_anchors.append(before_anchor)

        try:
            path_length = int(diagnostics.get("path_length", 0) or 0)
        except (TypeError, ValueError):
            path_length = 0

        target_raw = diagnostics.get("selected_target")
        target: Point | None = None
        if isinstance(target_raw, Sequence) and len(target_raw) >= 2:
            try:
                target = (int(round(float(target_raw[0]))), int(
                    round(float(target_raw[1]))))
            except (TypeError, ValueError):
                pass

        progress = bool(event.level_delta > 0 or event.win)
        if before_anchor is not None and after_anchor is not None and target is not None:
            before_distance = abs(
                before_anchor[0] - target[0]) + abs(before_anchor[1] - target[1])
            after_distance = abs(
                after_anchor[0] - target[0]) + abs(after_anchor[1] - target[1])
            progress = progress or after_distance < before_distance
        if self.active_target_key == target_key and self.last_path_length is not None and path_length > 0 and path_length < self.last_path_length:
            progress = True

        anchors = list(self.recent_anchors)
        two_cycle = bool(len(anchors) >= 4 and anchors[-4] == anchors[-2]
                         and anchors[-3] == anchors[-1] and anchors[-2] != anchors[-1])
        if two_cycle:
            self.path_cycle_count += 1

        if self.active_target_key != target_key:
            self.active_target_key = target_key
            self.no_progress_count = 0
        self.no_progress_count = 0 if progress else self.no_progress_count + 1
        self.last_path_length = path_length if path_length > 0 else self.last_path_length

        should_cool = bool(event.game_over or two_cycle or self.no_progress_count >= max(
            2, self.config.path_no_progress_limit))
        if should_cool:
            self.target_cooldown_until[target_key] = self.current_step + \
                max(1, self.config.path_target_cooldown_steps)
            self.active_target_key = None
            self.last_path_length = None
            self.no_progress_count = 0
            self.recent_anchors.clear()


# ---------------------------------------------------------------------------
# Optional milestone-gated local text model
# ---------------------------------------------------------------------------


RELATED_MECHANISM_FAMILIES: dict[str, set[str]] = {
    "line_or_beam": {"movement_control", "gravity_or_fall", "line_or_beam", "translation"},
    "movement_control": {"line_or_beam", "drag_or_push", "movement_control", "translation", "gravity_or_fall"},
    "drag_or_push": {"movement_control", "drag_or_push", "translation", "topology_change"},
    "gravity_or_fall": {"movement_control", "line_or_beam", "gravity_or_fall"},
    "targeted_recolor": {"component_delete", "targeted_recolor", "recoloration", "conditional_recolor"},
    "component_delete": {"targeted_recolor", "component_delete", "conditional_recolor"},
    "topology_change": {"drag_or_push", "movement_control", "topology_change"},
}

MECHANISM_TO_PROGRAM_KINDS: dict[str, set[str]] = {
    "movement_control": {"translation", "drag_component", "gravity"},
    "targeted_recolor": {"conditional_recolor", "component_recolor", "color_map"},
    "component_delete": {"component_delete"},
    "drag_or_push": {"drag_component", "translation"},
    "line_or_beam": {"line_connect"},
    "flood_or_fill": {"flood_fill", "count_and_fill"},
    "gravity_or_fall": {"gravity", "drag_component"},
    "copy_or_stamp": {"copy_pattern", "local_patch_replace"},
    "count_or_trigger": {"count_and_fill", "cellular_rule"},
    "topology_switch": {"cellular_rule", "line_connect"},
}

_OLLAMA_LOCK = threading.Lock()


class OptionalLocalReasoner:
    """Lazy, milestone-gated local causal-LM adapter with safe grid tools."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.calls = 0
        self.calls_this_level = 0
        self.last_call_step = -10_000
        self.last_latency_sec = 0.0
        self.total_latency_sec = 0.0
        self.abductions_parsed = 0
        self.abductions_selected = 0
        self.llm_action_progress = 0
        self.llm_program_progress = 0
        self.llm_abduction_prior_shift_progress = 0
        self._failed = False
        self._model = None

        self.model_decisions_used = 0
        self.model_decisions_used_this_level = 0
        self.reasoner_decision_attempts = 0
        self.reasoner_decision_skips_by_reason = {
            "skipped_due_to_low_remaining_time": 0,
            "skipped_due_to_projected_latency": 0,
            "skipped_due_to_budget_exhaustion": 0,
            "skipped_due_to_lock_unavailable": 0,
            "skipped_due_to_model_unavailable": 0,
            "skipped_due_to_cooldown_gate": 0,
            "skipped_due_to_llm_backoff": 0,
        }
        self.reasoner_decision_successes = 0
        self.reasoner_consultations_this_level = 0
        
        self.llm_illegal_action_rejections = 0
        self.llm_repetitive_kind_rejections = 0
        self.llm_empty_abduction_count = 0
        self.llm_parsed_abduction_count = 0
        self.llm_diversity_pruned_count = 0
        self.llm_impossible_hypothesis_rejections = 0
        self.llm_unsupported_family_rejections = 0
        self.llm_family_repeat_rejections = 0
        self.llm_schema_valid_but_unexecutable = 0
        self.llm_filtered_for_action_mismatch = 0
        self.llm_rejected_by_alignment = 0
        self.llm_rejected_by_semantic_gate = 0
        self.llm_rejected_by_dead_signature = 0
        self.llm_rejected_by_probe_budget = 0
        self.llm_rejected_by_coordinate_fatigue = 0
        self.llm_semantic_override_attempts = 0
        self.llm_semantic_override_rejections = 0
        self.recent_abductions: list[tuple[str, str]] = []
        self.recent_mismatch_rejections: deque[str] = deque(maxlen=5)
        self.recent_impossible_rejections: deque[str] = deque(maxlen=5)
        self.recent_family_repeat_rejections: deque[str] = deque(maxlen=5)
        self.recent_schema_rejections: deque[str] = deque(maxlen=5)
        self.failed_verifications: dict[tuple[str, str], int] = {}

    @property
    def model_ready(self) -> bool:
        return (
            self.config.enable_model
            and bool(self.config.model_path)
            and not self._failed
        )

    @property
    def budget_available(self) -> bool:
        return (
            self.model_decisions_used < self.config.model_call_budget
            and self.model_decisions_used_this_level < self.config.model_call_budget_per_level
        )

    @property
    def available(self) -> bool:
        return self.model_ready and self.budget_available

    def reset_level(self) -> None:
        self.calls_this_level = 0
        self.last_call_step = -10_000
        self.abductions_parsed = 0
        self.abductions_selected = 0
        self.llm_action_progress = 0
        self.llm_program_progress = 0
        self.llm_abduction_prior_shift_progress = 0
        self.model_decisions_used_this_level = 0
        self.reasoner_consultations_this_level = 0
        self.failed_verifications.clear()
        self.recent_mismatch_rejections.clear()
        self.recent_impossible_rejections.clear()
        self.recent_family_repeat_rejections.clear()
        self.recent_schema_rejections.clear()
        self.recent_abductions.clear()

    def _load(self) -> bool:
        if self._model is not None:
            return True
        # Ping local Ollama server and auto-detect active model tag
        import urllib.request
        import json
        import time
        for _ in range(5):
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode('utf-8'))
                        models = data.get("models", [])
                        if models:
                            self.detected_tag = models[0].get("name")
                        else:
                            self.detected_tag = os.environ.get("DEWMA_MODEL_TAG", "qwen2.5-coder:7b")
                        self._model = "ollama"
                        return True
            except Exception:
                time.sleep(2)
        # Try in-process llama_cpp loader if model_path file exists
        model_path = self.config.model_path
        if model_path and os.path.exists(model_path):
            try:
                from llama_cpp import Llama
                self._model = Llama(model_path=model_path, n_ctx=2048, verbose=False)
                print(f"[llama_cpp] Successfully loaded in-process LLM from {model_path}", file=sys.stderr)
                return True
            except Exception as exc:
                print(f"[llama_cpp] Failed to load model from {model_path}: {exc}", file=sys.stderr)

        self._failed = True
        return False

    def can_call(self, step: int, milestone: bool, is_stagnant: bool = False) -> bool:
        cooldown = max(2, self.config.model_cooldown_steps // 2) if is_stagnant else self.config.model_cooldown_steps
        return (
            (milestone or is_stagnant)
            and self.available
            and step - self.last_call_step >= cooldown
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        start_idx = text.find("{")
        if start_idx == -1:
            return None
            
        end_idx = text.rfind("}")
        if end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx:end_idx + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
                
        brace_count = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            char = text[i]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[start_idx:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
        return None

    def _is_impossible_hypothesis(self, scene: Scene, spec: dict) -> tuple[bool, str]:
        params = spec.get("params", {})
        if not isinstance(params, dict):
            return False, ""
            
        color_counts = dict(scene.color_counts)
        
        # Color checks (must exist in current grid)
        for key in ["source_color", "neighbor_color", "selector_color"]:
            color = params.get(key)
            if isinstance(color, int) and 0 <= color <= 15:
                if color_counts.get(color, 0) == 0:
                    return True, f"color {color} not found in grid"
                    
        # Background color check (optional, but if it's supposed to exist and it's the only one...)
        # Usually background can be 0 even if it's completely filled, so skip background check.
        
        # Coordinate checks
        x = params.get("x")
        y = params.get("y")
        if isinstance(x, int) and (x < 0 or x >= scene.grid.shape[1]):
            return True, f"x={x} out of bounds"
        if isinstance(y, int) and (y < 0 or y >= scene.grid.shape[0]):
            return True, f"y={y} out of bounds"
            
        return False, ""

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return [OptionalLocalReasoner._json_safe(v) for v in value]
        if isinstance(value, list):
            return [OptionalLocalReasoner._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): OptionalLocalReasoner._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _generate_json(self, prompt: str, step: int) -> dict[str, Any] | None:
        if not self.config.enable_model or self._failed:
            return None

        # Thread-safe non-blocking mutex lock for Ollama concurrency protection.
        # If another game thread is querying Ollama, do not block or crash—
        # non-blocking acquire allows fast fallback to deterministic Candidate Generator!
        acquired = _OLLAMA_LOCK.acquire(blocking=False)
        if not acquired:
            return None

        started = time.monotonic()
        text = ""
        parsed = None
        error = None
        try:
            self.calls += 1
            self.calls_this_level += 1
            self.last_call_step = step
            if self._model == "ollama":
                import urllib.request
                import json
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=json.dumps({
                        "model": getattr(self, "detected_tag", os.environ.get("DEWMA_MODEL_TAG", "qwen2.5-coder:7b")),
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "24h",
                        "options": {
                            "temperature": 0.4,
                            "num_predict": self.config.model_max_new_tokens
                        }
                    }).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                    text = result.get("response", "").strip()
                print(f"[Ollama LLM Call #{self.calls}] Tag: {getattr(self, 'detected_tag', 'unknown')} | Latency: {time.monotonic()-started:.2f}s | Output: {text[:80]}...", file=sys.stderr)
                parsed = self._extract_json(text)
                return parsed
            else:
                # In-process llama_cpp inference
                res = self._model(
                    prompt,
                    max_tokens=self.config.model_max_new_tokens,
                    temperature=0.0,
                    stop=["}\n", "}\r\n"]
                )
                text = res["choices"][0]["text"].strip()
                parsed = self._extract_json(text)
                return parsed
        except Exception as exc:
            error = str(exc)
            print(f"[LLM Reasoner Transient Error] {exc}", file=sys.stderr)
            return None
        finally:
            _OLLAMA_LOCK.release()
            self.last_latency_sec = max(0.0, time.monotonic() - started)
            self.total_latency_sec += self.last_latency_sec
            try:
                os.makedirs(self.config.trace_dir, exist_ok=True)
                if hasattr(self, "get_game_id"):
                    game_id = self.get_game_id()
                else:
                    game_id = getattr(self, "game_id", "unknown")
                with open(os.path.join(self.config.trace_dir, f"llm_forensics_{game_id}.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": time.time(),
                        "step": step,
                        "prompt": prompt,
                        "raw_output": text,
                        "parsed_output": parsed,
                        "error": error
                    }) + "\n")
            except Exception:
                pass

    def propose(
        self,
        scene: Scene,
        legal_names: Sequence[str],
        memory: TraceMemory,
        step: int,
        goals_summary: Sequence[Mapping[str, Any]] = (),
        programs_summary: Sequence[Mapping[str, Any]] = (),
        action_semantics_summary: Sequence[Mapping[str, Any]] = (),
        response_frames: Sequence[np.ndarray] = (),
        feedback: str = "",
        bypass_can_call: bool = False,
    ) -> list[AbductionProposal] | None:
        if not bypass_can_call:
            if not self.can_call(step, milestone=True) or not self._load():
                return None
        else:
            if not self._load():
                return None

        inspector = GridInspector(
            scene.grid, scene.previous_grid, response_frames)
        repl = SafeRepl(inspector)
        recent = [
            {
                "action": t.action.name,
                "data": t.action.data_dict,
                "effect": {
                    "changed": t.event.changed_count,
                    "noop": t.event.no_op,
                    "progress": t.event.level_delta,
                    "death": t.event.game_over,
                    "signature": t.event.effect_signature,
                },
            }
            for t in list(memory.transitions)[-max(1, self.config.llm_recent_transitions_limit):]
        ]
        
        controlled_shift = {"status": "unknown"}
        if scene.controlled_entity_id:
            last_event = None
            if memory.transitions:
                last_event = list(memory.transitions)[-1].event
            
            if last_event and getattr(last_event, "entity_moves", None) is not None:
                found = False
                for eid, dx, dy in last_event.entity_moves:
                    if eid == scene.controlled_entity_id:
                        controlled_shift = {"status": "known", "dx": dx, "dy": dy, "moved": bool(dx or dy)}
                        found = True
                        break
                if not found:
                    if any(e.entity_id == scene.controlled_entity_id for e in scene.entities):
                        controlled_shift = {"status": "known", "dx": 0, "dy": 0, "moved": False}
                    else:
                        controlled_shift = {"status": "disappeared"}
        
        system_context = (
            "You control an unknown ARC-AGI-3 environment. You may inspect the "
            "grid, execute python to analyze the state, or output "
            "an abductive explanation of recent transitions. Emit ONE JSON object only per response.\n"
            "Inspection schema: "
            '{"type":"inspect","expression":"state.connected_components()"}\n'
            "Python schema (for data analysis only): "
            '{"type":"python","code":"for t in recent_transitions: print(t[\'action\'])"}\n'
            "Final schema (infer the mechanism, relational structure, or control logic that caused the observed transition!): "
            '{"type":"abduction","mechanism_family":"movement_control","mechanism_confidence":0.85,"priority_program_families":["translation"],"invariants_to_preserve":["color_0"],"recommended_probe_type":"directional_move","suggested_programs":[{"kind":"conditional_recolor","action":"ACTION1","params":{"source_color":2,"target_color":3,"neighbor_color":4},"why":"..."}, {"action":"...","why":"..."}]}\n'
            "CRITICAL: You MUST include 'mechanism_family', 'priority_program_families', and the nested 'suggested_programs' array to propose a structural hypothesis!\n"
            "Allowed state methods: summary(), shape(), find_color(v), count(v), patch(x,y,r), "
            "diff_last_frame(), changed_cells(), color_histogram(), connected_components(color), delta_summary(), symmetry_hints(). "
            "Prefer stored evidence and reversible progress. Do not request a physical "
            "probe unless it can change the decision. Complex actions require x,y.\n"
            "MANDATORY: Do not propose slight variations of the same hypothesis. You must output at least two distinctly different hypotheses from different puzzle families (e.g. one conditional_recolor and one component_delete) if the cause is ambiguous. Verification is strict, so diversity maximizes our chances of finding the true mechanism.\n"
            "In Python execution, `recent_transitions` (list of dicts) is available in the global namespace. Use print() to output results.\n"
            f"legal_actions={list(legal_names)}\n"
            f"state_summary={inspector.summary()}\n"
            f"mechanism_snapshot={inspector.mechanism_snapshot(recent, controlled_shift)}\n"
            f"entity_confidence={scene.entity_confidence:.3f}, field_mode={scene.field_mode}\n"
            f"recent_transitions={json.dumps(recent, separators=(',', ':'))}\n"
            f"candidate_goals={json.dumps(list(goals_summary)[:self.config.llm_goals_limit], separators=(',', ':'), default=str)}\n"
            f"verified_programs={json.dumps(list(programs_summary)[:self.config.llm_programs_limit], separators=(',', ':'), default=str)}\n"
            f"action_semantics={json.dumps(list(action_semantics_summary)[:self.config.llm_action_semantics_limit], separators=(',', ':'), default=str)}\n"
            "Examples of Supported Program Hypotheses (add inside suggested_programs):\n"
            "- conditional_recolor: {\"kind\":\"conditional_recolor\",\"action\":\"ACTION1\",\"params\":{\"source_color\":2,\"target_color\":3,\"neighbor_color\":4}}\n"
            "- component_delete: {\"kind\":\"component_delete\",\"action\":\"ACTION2\",\"params\":{\"background\":0}}\n"
            "- component_recolor: {\"kind\":\"component_recolor\",\"action\":\"ACTION2\",\"params\":{\"target_color\":5}}\n"
            "- translation: {\"kind\":\"translation\",\"action\":\"ACTION6\",\"params\":{\"dx\":1,\"dy\":0,\"selector_color\":3,\"background\":0}}\n"
            "- drag_component: {\"kind\":\"drag_component\",\"action\":\"ACTION6\",\"params\":{\"dx\":0,\"dy\":1,\"background\":0}}\n"
            "- line_connect: {\"kind\":\"line_connect\",\"action\":\"ACTION3\",\"params\":{\"color\":2,\"p1\":[2,3],\"p2\":[8,3]}}\n"
            "- flood_fill: {\"kind\":\"flood_fill\",\"action\":\"ACTION6\",\"params\":{\"x\":4,\"y\":5,\"target_color\":1}}\n"
            "- gravity: {\"kind\":\"gravity\",\"action\":\"ACTION1\",\"params\":{\"dx\":0,\"dy\":1,\"background\":0}}\n"
            "- copy_pattern: {\"kind\":\"copy_pattern\",\"action\":\"ACTION4\",\"params\":{\"offsets\":[[0,1],[1,0]],\"colors\":[3,4]}}\n"
            "- count_and_fill: {\"kind\":\"count_and_fill\",\"action\":\"ACTION5\",\"params\":{\"offsets\":[[0,0],[0,1]],\"target_color\":4}}\n"
            "Use action_semantics and mechanism_snapshot to infer whether an action behaves like movement, recolor, deletion, topology change, or targeted control.\n"
            "A program is a highly valued hypothesis and will be replay-verified before use.\n"
        )

        history_str = ""
        seen_expressions: set[str] = set()
        max_rounds = 3
        for round_index in range(max_rounds):
            if not self.available:
                return None

            if feedback:
                instruction = f"PREVIOUS ATTEMPT FAILED: {feedback}\nPlease analyze the failure and propose corrected hypotheses.\n"
            elif round_index == 0:
                instruction = "MANDATORY: Since this is your first response for this step, you MUST use the `inspect` or `python` tool to explore the state. Do NOT guess an abduction yet.\n"
            else:
                instruction = "MANDATORY: You have already inspected the state. You MUST now propose an abduction hypothesis using the final abduction schema.\n"

            dyn_hint_parts = []
            if len(self.recent_mismatch_rejections) > 0:
                dyn_hint_parts.append("recent proposals failed legality/action-family matching (e.g. recent recolor hypotheses mismatched movement-like effects or invalid topology)")
            if len(self.recent_impossible_rejections) > 0:
                dyn_hint_parts.append("hypotheses referenced colors/objects not present in the scene")
            if len(self.recent_family_repeat_rejections) > 0:
                dyn_hint_parts.append("repeated same hypothesis pattern without progress (diversify beyond simple recolor!)")
            if len(self.recent_schema_rejections) > 0:
                dyn_hint_parts.append("proposed actions lacked required parameters (e.g. missing coordinates for ACTION6)")
            if self.reasoner_decision_successes > 0 and getattr(memory, "no_op_streak", 0) >= 2:
                dyn_hint_parts.append("recent accepted proposals produced no progress (reevaluate the mechanism!)")
            
            dyn_hint = ""
            if dyn_hint_parts:
                dyn_hint = f"HINT: Recent proposals failed because: {'; '.join(dyn_hint_parts)}. Analyze recent_transitions carefully to fix this.\n"
            prompt = system_context + dyn_hint + instruction + history_str
            obj = self._generate_json(prompt, step)
            if not obj:
                return None

            response_type = str(
                obj.get("type", "action" if "action" in obj else "")).lower()
            if response_type == "python":
                if round_index >= max_rounds - 1:
                    history_str += "python_denied=tool_round_limit_reached\nReturn final abduction JSON now.\n"
                    continue
                code = str(obj.get("code", ""))
                if not code or code in seen_expressions:
                    history_str += "python_error=empty_or_repeated_code\nReturn final abduction JSON now.\n"
                    continue
                seen_expressions.add(code)
                try:
                    result_str = repl.exec_python_script(code, list(memory.transitions))
                    history_str += (
                        f"python_{round_index + 1}_result:\n{result_str}\n"
                    )
                except Exception as exc:
                    history_str += (
                        f"python_{round_index + 1}_error={type(exc).__name__}:"
                        f"{str(exc)[:180]}\nFix your python script or return the final abduction JSON.\n"
                    )
                continue

            if response_type == "inspect":
                if round_index >= self.config.model_tool_rounds:
                    history_str += "inspection_denied=tool_round_limit_reached\nReturn final abduction JSON now.\n"
                    continue
                expression = str(obj.get("expression", ""))[:240].strip()
                if not expression or expression in seen_expressions:
                    history_str += "inspection_error=empty_or_repeated_expression\nReturn final abduction JSON now.\n"
                    continue
                seen_expressions.add(expression)
                try:
                    result = self._json_safe(repl.run(expression))
                    rendered = json.dumps(result, separators=(",", ":"))
                    if len(rendered) > 2400:
                        rendered = rendered[:2400] + "...<truncated>"
                    history_str += (
                        f"inspection_{round_index + 1}_expression={expression!r}\n"
                        f"inspection_{round_index + 1}_result={rendered}\n"
                    )
                except Exception as exc:
                    history_str += (
                        f"inspection_{round_index + 1}_error={type(exc).__name__}:{str(exc)[:180]}\nReturn final abduction JSON now.\n"
                    )
                continue
            if response_type == "abduction":
                puzzle_family = obj.get("mechanism_family") or obj.get("puzzle_family") or "mechanism_inference"
                if isinstance(puzzle_family, str):
                    puzzle_family = puzzle_family.strip()
                
                # 1. Parse mechanism_confidence
                try:
                    conf = float(obj.get("mechanism_confidence", 0.5))
                    conf = max(0.0, min(1.0, conf))
                except (ValueError, TypeError):
                    conf = 0.5
                memory.llm_last_mechanism_confidence = conf

                # 2. Parse recommended_probe_type with 6-step TTL
                rec_probe = str(obj.get("recommended_probe_type", "")).strip().lower()
                if rec_probe:
                    memory.recommended_probe_type = rec_probe
                    memory.recommended_probe_ttl = step + 6

                # 3. Store mechanism-level inferences in memory with confidence-weighting and 12-step TTL
                if isinstance(obj.get("priority_program_families"), list):
                    memory.priority_program_families = [str(x) for x in obj.get("priority_program_families", [])]
                    memory.priority_program_ttl = step + 12
                if isinstance(obj.get("invariants_to_preserve"), list):
                    memory.invariants_to_preserve = [str(x) for x in obj.get("invariants_to_preserve", [])]
                
                mech_fam = str(obj.get("mechanism_family", "")).strip()
                if mech_fam in memory.mechanism_scores:
                    # Bounded confidence-weighted evidence bump (0.05 for 0.1 conf up to 0.40 for 1.0 conf)
                    bump = max(0.05, min(0.40, 0.40 * conf))
                    memory.mechanism_scores[mech_fam] = min(1.0, memory.mechanism_scores[mech_fam] + bump)
                    
                proposals_data = obj.get("suggested_programs")
                if not isinstance(proposals_data, list):
                    proposals_data = []
                
                supported_kinds = {
                    "conditional_recolor", "translation", "component_delete", "component_recolor",
                    "line_connect", "drag_component", "flood_fill", "gravity", "copy_pattern",
                    "count_and_fill", "color_map", "local_patch_replace", "cellular_rule"
                }
                results: list[AbductionProposal] = []
                for prop in proposals_data:
                    if not isinstance(prop, dict):
                        continue
                    
                    kind = prop.get("kind")
                    if kind and kind not in supported_kinds:
                        self.llm_unsupported_family_rejections += 1
                        history_str += f"abduction_error=unsupported_program_kind ({kind})\n"
                        continue
                    
                    is_impossible, reason = self._is_impossible_hypothesis(scene, prop)
                    if is_impossible:
                        self.llm_impossible_hypothesis_rejections += 1
                        self.recent_impossible_rejections.append(reason)
                        history_str += f"abduction_error=impossible_hypothesis_rejected ({reason})\n"
                        continue

                    name = str(prop.get("action", "")).upper()
                    if not name:
                        self.llm_schema_valid_but_unexecutable += 1
                        self.recent_schema_rejections.append("missing_action_name")
                        history_str += "abduction_error=missing_action_name\n"
                        continue
                    data: tuple[tuple[str, int], ...] = ()
                    if prop.get("x") is not None or prop.get("y") is not None:
                        try:
                            x_val, y_val = prop.get("x"), prop.get("y")
                            if x_val is not None and y_val is not None:
                                data = (("x", int(x_val)), ("y", int(y_val)))
                        except (ValueError, TypeError):
                            pass
                    confidence = float(prop.get("confidence", 0.5))
                    action_spec = ActionSpec(
                        name=name,
                        data=data,
                        source="milestone_model_repl",
                        predicted_effect=str(prop.get("why", "model proposal"))[:160],
                        score=2.0 + max(0.0, min(1.0, confidence)),
                        goal_ids=tuple(str(x) for x in prop.get("goal_ids", [])[:4]) if isinstance(prop.get("goal_ids"), list) else (),
                    )
                    program_spec = prop if prop.get("kind") else None
                    results.append(AbductionProposal(action_spec, program_spec, puzzle_family=puzzle_family if isinstance(puzzle_family, str) else None))
                
                if results or puzzle_family:
                    if not results:
                        results.append(AbductionProposal(None, puzzle_family=puzzle_family if isinstance(puzzle_family, str) else None))
                    
                    diverse_results = []
                    seen_pairs = set()
                    for res in results:
                        fam = res.puzzle_family or "none"
                        knd = res.program_spec.get("kind", "none") if res.program_spec else "none"
                        act = res.action.name if res.action else "none"
                        pair = (fam, knd, act)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            diverse_results.append(res)
                        else:
                            self.llm_diversity_pruned_count += 1
                            
                    if not diverse_results and results:
                        diverse_results = [results[0]]
                        
                    return diverse_results
                return None
            return None
        return None

    def propose_relocalization(self, scene: Scene, win_pattern: dict[str, Any], step: int = 0) -> dict[str, Any] | None:
        if not self._load():
            return None
        win_action = win_pattern.get("action", "ACTION6")
        win_coords = win_pattern.get("coords", (scene.width // 2, scene.height // 2))
        win_family = win_pattern.get("family", "targeted_recolor")
        
        prompt = (
            f"You are an ARC-AGI-3 level transfer assistant.\n"
            f"Previous level was SOLVED by action '{win_action}' at target coordinates {win_coords} under mechanism family '{win_family}'.\n"
            f"Current Level grid size: {scene.width}x{scene.height}, components: {len(scene.components)}.\n"
            f"Relocalize where the corresponding target component is located in this new level grid.\n"
            f"Respond ONLY with a single JSON object:\n"
            f'{{"type": "relocalize", "x": <int>, "y": <int>, "confidence": <float 0.0-1.0>, "why": "<reasoning>"}}\n'
        )
        try:
            obj = self._generate_json(prompt, step)
            if not obj:
                return None
            x = int(obj.get("x", -1))
            y = int(obj.get("y", -1))
            conf = float(obj.get("confidence", 0.5))
            if 0 <= x < scene.width and 0 <= y < scene.height:
                return {"x": x, "y": y, "confidence": conf, "why": obj.get("why", "")}
        except Exception:
            pass
        return None

# ---------------------------------------------------------------------------
# Candidate generation, virtual-first arbitration, and metacognition
# ---------------------------------------------------------------------------


def _normalize_legal_actions(frame: FrameData) -> list[GameAction]:
    """Return only actions explicitly advertised by the current frame."""
    raw = getattr(frame, "available_actions", None)
    actions: list[GameAction] = []
    if raw:
        for item in raw:
            if isinstance(item, GameAction):
                actions.append(item)
                continue
            name = getattr(item, "name", None)
            if name is None and isinstance(item, int):
                try:
                    actions.append(GameAction[f"ACTION{item}"])
                    continue
                except Exception:
                    pass
                try:
                    actions.append(GameAction(item))
                    continue
                except Exception:
                    pass
            name = (name or str(item)).split(".")[-1].upper()
            try:
                actions.append(GameAction[name])
            except Exception:
                pass
            if name.isdigit():
                try:
                    actions.append(GameAction[f"ACTION{name}"])
                except Exception:
                    pass
    deduped: list[GameAction] = []
    seen: set[str] = set()
    for action in actions:
        if action.name not in seen:
            seen.add(action.name)
            deduped.append(action)
    return deduped


def _component_signature(scene: Scene, x: int, y: int) -> str:
    component = next((c for c in scene.components if (x, y) in c.cells), None)
    if component is None:
        region = (x // 8, y // 8)
        color = int(
            scene.grid[y, x]) if 0 <= y < scene.height and 0 <= x < scene.width else -1
        return f"raw:c{color}:r{region[0]}_{region[1]}"
    area_bucket = "s" if component.area <= 4 else "m" if component.area <= 16 else "l"
    return f"obj:c{component.color}:a{area_bucket}:sh{component.shape_key}:b{int(component.touches_border)}"


def _action_signature(scene: Scene, spec: ActionSpec) -> str:
    data = spec.data_dict
    if "x" in data and "y" in data:
        return f"{spec.name}:{_component_signature(scene, data['x'], data['y'])}"
    field_bucket = "field" if scene.field_mode else "entity"
    return f"{spec.name}:{field_bucket}:{scene.abstract_key[:8]}"


class CandidateGenerator:
    def __init__(
        self,
        config: Config,
        memory: TraceMemory,
        dead: DeadSignatureMemory,
        spatial_hash: FastSpatialActionHash,
        world_model: CausalWorldModel,
        hypotheses: HypothesisMemory,
        dynamics: ActionDynamics | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.dead = dead
        self.spatial_hash = spatial_hash
        self.world_model = world_model
        self.hypotheses = hypotheses
        self.dynamics = dynamics
        self.coordinate_visits: Counter[tuple[int, int]] = Counter()

    def reset_level(self) -> None:
        self.coordinate_visits.clear()

    def _extract_object_centric_anchors(self, scene: Scene) -> list[tuple[float, Point, str]]:
        anchors: list[tuple[float, Point, str]] = []
        try:
            from scipy.ndimage import label
            non_bg = (scene.grid != scene.background)
            labeled, num_features = label(non_bg)
            for obj_id in range(1, min(num_features + 1, 24)):
                ys, xs = np.where(labeled == obj_id)
                if len(xs) == 0:
                    continue
                cx = int(round(float(np.mean(xs))))
                cy = int(round(float(np.mean(ys))))
                min_x, max_x = int(np.min(xs)), int(np.max(xs))
                min_y, max_y = int(np.min(ys)), int(np.max(ys))
                bx = (min_x + max_x) // 2
                by = (min_y + max_y) // 2
                score_c = 1.75 - 0.15 * self.coordinate_visits[(cx, cy)]
                anchors.append((score_c, (cx, cy), "object_centroid_anchor"))
                if (bx, by) != (cx, cy):
                    score_b = 1.60 - 0.15 * self.coordinate_visits[(bx, by)]
                    anchors.append((score_b, (bx, by), "object_bbox_center"))
        except Exception:
            pass
        return anchors

    def _complex_coordinates(self, scene: Scene) -> list[tuple[float, Point, str]]:
        total = max(1, scene.grid.size)
        color_freq = {color: count / total for color,
                      count in scene.color_counts}
        proposals: list[tuple[float, Point, str]] = []

        # Object-derived candidates are used only when representation confidence is adequate.
        if not scene.field_mode:
            for comp in scene.components:
                x, y = int(round(comp.centroid[0])), int(
                    round(comp.centroid[1]))
                is_border = (y <= 2 or x <= 2 or y >= scene.height - 3 or x >= scene.width - 3)
                if is_border:
                    score = 0.8 - 0.20 * self.coordinate_visits[(x, y)]
                else:
                    rarity = 1.0 - min(1.0, color_freq.get(comp.color, 1.0) * 12.0)
                    small = 1.0 / math.sqrt(max(1, comp.area))
                    score = 2.0 * rarity + 1.1 * small - 0.20 * self.coordinate_visits[(x, y)]
                proposals.append((score, (x, y), "rare_small_component"))
                x0, y0, x1, y1 = comp.bbox
                for point in {(x0, y0), (x1, y0), (x0, y1), (x1, y1)}:
                    proposals.append((score - 0.35, point, "component_corner"))

        # Bounded local continuation search around transferred/regrounded targets
        reground_coords = self.memory.regrounded_winning_coords or self.memory.transferred_winning_coords
        if (self.memory.post_breakthrough_window_active or self.memory.current_level_index > 0) and reground_coords is not None:
            tx, ty = reground_coords
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < scene.width and 0 <= ny < scene.height:
                        dist = max(abs(dx), abs(dy))
                        local_score = 3.0 - 0.45 * dist - 0.15 * self.coordinate_visits[(nx, ny)]
                        proposals.append((local_score, (nx, ny), "continuation_local_neighborhood"))

        # Raw representation is always available and dominates in field mode.
        for x, y in scene.changed_cells[-48:]:
            proposals.append(
                (1.5 - 0.15 * self.coordinate_visits[(x, y)], (x, y), "recent_change"))
            # Fine 3x3 local neighborhood probing around recent visual changes
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < scene.width and 0 <= ny < scene.height:
                        proposals.append(
                            (1.35 - 0.15 * self.coordinate_visits[(nx, ny)], (nx, ny), "recent_change_neighborhood"))

        # Rare colors from the exact grid, independent of object segmentation.
        for color, count in sorted(scene.color_counts, key=lambda item: item[1]):
            if color == scene.background:
                continue
            ys, xs = np.where(scene.grid == color)
            if len(xs) == 0:
                continue
            sample_indices = np.linspace(
                0, len(xs) - 1, num=min(8, len(xs)), dtype=int)
            rarity = 1.0 - min(1.0, count / total * 16.0)
            for idx in sample_indices:
                point = (int(xs[idx]), int(ys[idx]))
                is_border = (point[1] <= 2 or point[0] <= 2 or point[1] >= scene.height - 3 or point[0] >= scene.width - 3)
                base = 1.2 if not is_border else 0.4
                proposals.append(
                    (base + rarity - 0.12 * self.coordinate_visits[point], point, "raw_rare_color"))

        # Component-Delete Coordinate Hardening: Restrict to active non-background connected components with payoff ranking
        is_comp_delete = (self.memory.top_mechanism_family == "component_delete" and self.memory.top_mechanism_confidence >= 0.28 and len(self.memory.transitions) >= 4)
        if is_comp_delete:
            self.memory.component_delete_component_locked = True
            # Sort components: small isolated components first (highest payoff)
            sorted_comps = sorted(scene.components, key=lambda c: c.area)
            for comp in sorted_comps:
                if comp.color != scene.background:
                    cx, cy = int(round(comp.centroid[0])), int(round(comp.centroid[1]))
                    if cx <= 2 or cy <= 2 or cx >= scene.width - 3 or cy >= scene.height - 3:
                        continue
                    size_bonus = 1.0 if comp.area <= 4 else (0.5 if comp.area <= 12 else 0.0)
                    proposals.append((3.2 + size_bonus - 0.15 * self.coordinate_visits[(cx, cy)], (cx, cy), "comp_delete_live_centroid"))
                    x0, y0, x1, y1 = comp.bbox
                    bx, by = (x0 + x1) // 2, (y0 + y1) // 2
                    proposals.append((3.0 + size_bonus - 0.15 * self.coordinate_visits[(bx, by)], (bx, by), "comp_delete_live_bbox_center"))
                    for cell in comp.cells[:6]:
                        if cell[0] > 2 and cell[1] > 2 and cell[0] < scene.width - 3 and cell[1] < scene.height - 3:
                            proposals.append((2.8 + size_bonus - 0.15 * self.coordinate_visits[cell], cell, "comp_delete_live_cell"))
                    self.memory.component_delete_payoff_bias_used = True

        # Structured Line-or-Beam Proposals: Interpolate ray/axis coordinates between matching endpoints and corner closures
        is_line_beam = (self.memory.top_mechanism_family == "line_or_beam" and self.memory.top_mechanism_confidence >= 0.28 and len(self.memory.transitions) >= 4)
        if is_line_beam:
            self.memory.line_beam_structured_candidates_used = True
            for color, count in scene.color_counts:
                if color == scene.background or count < 2 or count > 8:
                    continue
                ys, xs = np.where(scene.grid == color)
                for i in range(len(xs)):
                    p1_x, p1_y = int(xs[i]), int(ys[i])
                    if p1_x <= 2 or p1_y <= 2 or p1_x >= scene.width - 3 or p1_y >= scene.height - 3:
                        continue
                    for j in range(i + 1, len(xs)):
                        p2_x, p2_y = int(xs[j]), int(ys[j])
                        if p2_x <= 2 or p2_y <= 2 or p2_x >= scene.width - 3 or p2_y >= scene.height - 3:
                            continue
                        if p1_x == p2_x:
                            min_y, max_y = min(p1_y, p2_y), max(p1_y, p2_y)
                            for y_mid in range(min_y, max_y + 1):
                                p = (p1_x, y_mid)
                                score = 2.8 - 0.12 * self.coordinate_visits[p]
                                proposals.append((score, p, "line_beam_collinear_y"))
                        elif p1_y == p2_y:
                            min_x, max_x = min(p1_x, p2_x), max(p1_x, p2_x)
                            for x_mid in range(min_x, max_x + 1):
                                p = (x_mid, p1_y)
                                score = 2.8 - 0.12 * self.coordinate_visits[p]
                                proposals.append((score, p, "line_beam_collinear_x"))
                        else:
                            # Orthogonal L-junction corner closure candidates
                            p_c1 = (p1_x, p2_y)
                            p_c2 = (p2_x, p1_y)
                            proposals.append((2.6 - 0.12 * self.coordinate_visits[p_c1], p_c1, "line_beam_corner_closure"))
                            proposals.append((2.6 - 0.12 * self.coordinate_visits[p_c2], p_c2, "line_beam_corner_closure"))
                        mid = ((p1_x + p2_x) // 2, (p1_y + p2_y) // 2)
                        proposals.append((3.0 - 0.12 * self.coordinate_visits[mid], mid, "line_beam_midpoint"))
                        self.memory.line_beam_closure_bias_used = True

        # Multi-Click Cell Cycle Persistence: Allow immediate follow-up clicks if interior cell is in active transition
        if (
            self.memory.current_level_index == 0
            and not self.memory.post_breakthrough_window_active
            and self.memory.last_changed_coord is not None
            and self.memory.coord_cycle_clicks_remaining > 0
        ):
            cx, cy = self.memory.last_changed_coord
            if 2 < cx < scene.width - 2 and 2 < cy < scene.height - 2 and scene.grid[cy, cx] != scene.background:
                proposals.append((2.4 - 0.10 * self.coordinate_visits[(cx, cy)], (cx, cy), "cell_cycle_persistence"))
                self.memory.cell_cycle_persistence_used = True

        # Drag / Push Source and Target Socket Anchors for drag_or_push & movement_control
        is_drag_push = (self.memory.top_mechanism_family in ("drag_or_push", "movement_control") and len(self.memory.transitions) >= 3)
        if is_drag_push and self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
            for comp in scene.components:
                if comp.color != scene.background:
                    cx, cy = int(round(comp.centroid[0])), int(round(comp.centroid[1]))
                    if 0 <= cx < scene.width and 0 <= cy < scene.height:
                        proposals.append((3.2 - 0.12 * self.coordinate_visits[(cx, cy)], (cx, cy), "drag_source_entity"))
                    # Target destination along orthogonal projection
                    for dx, dy in ((0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (-2, 2), (2, -2), (-2, -2)):
                        tx, ty = cx + dx, cy + dy
                        if 0 <= tx < scene.width and 0 <= ty < scene.height and scene.grid[ty, tx] == scene.background:
                            proposals.append((2.7 - 0.10 * self.coordinate_visits[(tx, ty)], (tx, ty), "drag_destination_socket"))

        # Systematic Sequential Component Sweep for targeted_recolor (Interior non-border components)
        if (
            self.memory.current_level_index == 0
            and not self.memory.post_breakthrough_window_active
            and (self.memory.top_mechanism_family == "targeted_recolor" or "targeted_recolor" in self.memory.competing_mechanism_families)
            and scene.components
        ):
            interior_comps = [c for c in scene.components if c.color != scene.background and not c.touches_border and c.centroid[1] > 2.5 and c.centroid[0] > 2.5]
            unswept_comps = [
                c for c in interior_comps
                if (int(round(c.centroid[0])), int(round(c.centroid[1]))) not in self.memory.visited_sweep_component_anchors
            ]
            target_comps = unswept_comps if unswept_comps else interior_comps
            if target_comps:
                sorted_comps = sorted(target_comps, key=lambda c: (c.area, self.coordinate_visits[(int(round(c.centroid[0])), int(round(c.centroid[1])))]))
                active_comp = sorted_comps[0]
                cx, cy = int(round(active_comp.centroid[0])), int(round(active_comp.centroid[1]))
                self.memory.sequential_component_sweep_active = True
                if 0 <= cx < scene.width and 0 <= cy < scene.height:
                    proposals.append((2.5 - 0.10 * self.coordinate_visits[(cx, cy)], (cx, cy), "sequential_sweep_centroid"))
                for cell in active_comp.cells[:3]:
                    proposals.append((2.3 - 0.10 * self.coordinate_visits[cell], cell, "sequential_sweep_cell"))

        # Pair Midpoints for rare matching color pairs (Symmetry / Alignment Anchors)
        for color, count in scene.color_counts:
            if color == scene.background or count != 2:
                continue
            ys, xs = np.where(scene.grid == color)
            if len(xs) == 2:
                mid_x = (int(xs[0]) + int(xs[1])) // 2
                mid_y = (int(ys[0]) + int(ys[1])) // 2
                point = (mid_x, mid_y)
                proposals.append((1.8 - 0.12 * self.coordinate_visits[point], point, "pair_midpoint_anchor"))

        # Color Boundary Entropy Anchors (Transitions between distinct interior colors)
        try:
            grid = scene.grid
            h_boundaries = (grid[:, :-1] != grid[:, 1:]).copy()
            v_boundaries = (grid[:-1, :] != grid[1:, :]).copy()
            # Zero out border UI rows so interior puzzle boundaries are extracted
            h_boundaries[:3, :] = False
            h_boundaries[-3:, :] = False
            h_boundaries[:, :3] = False
            h_boundaries[:, -3:] = False
            v_boundaries[:3, :] = False
            v_boundaries[-3:, :] = False
            v_boundaries[:, :3] = False
            v_boundaries[:, -3:] = False
            hy, hx = np.where(h_boundaries)
            vy, vx = np.where(v_boundaries)
            for x, y in zip(hx[:24], hy[:24]):
                p = (int(x), int(y))
                proposals.append((1.15 - 0.15 * self.coordinate_visits[p], p, "color_boundary_anchor"))
            for x, y in zip(vx[:24], vy[:24]):
                p = (int(x), int(y))
                proposals.append((1.15 - 0.15 * self.coordinate_visits[p], p, "color_boundary_anchor"))
        except Exception:
            pass

        # Object Centroid & Bounding Box Center Anchors
        centroids = self._extract_object_centric_anchors(scene)
        proposals.extend(centroids)



        # Stagnation boost: when in a recent state loop or NO-OP streak, boost unvisited frontier locations
        is_stagnant = self.memory.recent_state_loop(12) or self.memory.no_op_streak >= self.config.no_progress_cooldown_steps
        stagnation_boost = 0.85 if is_stagnant else 0.0

        # Resolution-adaptive unexplored frontier grid ensures coverage for any resolution.
        step_x = max(3, scene.width // 6)
        step_y = max(3, scene.height // 6)
        for y in range(step_y // 2, scene.height, step_y):
            for x in range(step_x // 2, scene.width, step_x):
                visits = self.coordinate_visits[(x, y)]
                frontier_score = 0.55 - 0.18 * visits + (stagnation_boost if visits == 0 else 0.0)
                proposals.append((frontier_score, (x, y), "coarse_frontier"))

        proposals.extend(
            [
                (0.4, (scene.width // 2, scene.height // 2), "center"),
                (0.25, (1, 1), "corner"),
                (0.25, (scene.width - 2, 1), "corner"),
                (0.25, (1, scene.height - 2), "corner"),
                (0.25, (scene.width - 2, scene.height - 2), "corner"),
            ]
        )

        # In Component-Delete mode: strongly filter generic background-only coordinate candidates
        if is_comp_delete:
            live_comp_proposals = [
                (score, point, why) for score, point, why in proposals
                if (0 <= point[0] < scene.width and 0 <= point[1] < scene.height and scene.grid[point[1], point[0]] != scene.background)
            ]
            if live_comp_proposals:
                proposals = live_comp_proposals

        best_by_point: dict[Point, tuple[float, str]] = {}
        for score, point, why in proposals:
            x, y = point
            if not (0 <= x < scene.width and 0 <= y < scene.height):
                continue
            previous = best_by_point.get(point)
            if previous is None or score > previous[0]:
                best_by_point[point] = (score, why)
        ranked = sorted(
            ((score, point, why)
             for point, (score, why) in best_by_point.items()),
            reverse=True,
        )
        return ranked[: self.config.max_complex_candidates]

    def generate(
        self,
        scene: Scene,
        legal_actions: Sequence[GameAction],
        use_hypotheses: bool = True,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        legal_names = {a.name for a in legal_actions}
        for action in legal_actions:
            if action is GameAction.RESET:
                continue
            if action.is_complex():
                # Domain-General ARC-AGI-3 Precondition: ACTION7 (drop / place) requires an active drag in progress or paired execution
                if action.name == "ACTION7" and not self.memory.atomic_drag_drop_paired and len(self.memory.macro_replay_queue) == 0:
                    if self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
                        continue

                for coord_score, (x, y), why in self._complex_coordinates(scene):
                    if (self.memory.mode == "exploitation_mode" or self.memory.post_breakthrough_window_active):
                        self.memory.exploitation_noop_blacklist_checks += 1
                        if (action.name, x, y) in self.memory.exploitation_noop_blacklist or (x, y) in self.memory.exploitation_noop_neighborhood_blacklist:
                            self.memory.exploitation_noop_blacklist_hits += 1
                            continue
                    spec = ActionSpec(
                        name=action.name,
                        data=(("x", x), ("y", y)),
                        source="complex_candidate",
                        score=coord_score,
                    )
                    signature = _action_signature(scene, spec)
                    if self.config.enable_dead_signatures and self.dead.is_dead(signature):
                        continue
                    tried = self.memory.tried_count(scene.exact_key, spec)
                    known, confidence, effect = self.spatial_hash.predict(
                        scene, spec) if self.config.enable_spatial_hash else (False, 0.0, None)
                    if use_hypotheses:
                        hypothesis_known, hypothesis_conf, hypothesis_effect = self.hypotheses.predict(
                            signature)
                        information_value = self.hypotheses.information_value(
                            signature)
                        goal_bonus = self.hypotheses.goal_bonus(signature)
                    else:
                        hypothesis_known, hypothesis_conf, hypothesis_effect = False, 0.0, None
                    is_probe = not known and not hypothesis_known and tried == 0
                    is_noop_hyp = hypothesis_known and hypothesis_effect == "noop"
                    is_noop_hash = known and effect == "noop"
                    noop_history_count = self.dead.noops.get(signature, 0)
                    
                    score = (
                        coord_score
                        + (1.2 * confidence if not is_noop_hash else -2.0 * confidence)
                        + (0.8 * hypothesis_conf if not is_noop_hyp else -1.5 * hypothesis_conf)
                        + 0.55 * information_value
                        + goal_bonus
                        - 0.85 * tried
                        - 0.35 * noop_history_count
                    )
                    rationale = [why, f"hyp_info={information_value:.2f}"]
                    if (known and not is_noop_hash) or (hypothesis_known and not is_noop_hyp):
                        score += 1.0
                        if known:
                            rationale.append(f"spatial_hash:{confidence:.2f}")
                        if hypothesis_known:
                            rationale.append(
                                f"hypothesis:{hypothesis_conf:.2f}")
                        spec = ActionSpec(
                            name=spec.name,
                            data=spec.data,
                            source="spatial_hash" if known else "predictive_hypothesis",
                            predicted_effect=effect or hypothesis_effect,
                            score=score,
                        )
                    candidates.append(
                        Candidate(spec, signature, is_probe, score, rationale))
            else:
                spec = ActionSpec(name=action.name, source="simple_candidate")
                signature = _action_signature(scene, spec)
                if self.config.enable_dead_signatures and self.dead.is_dead(signature):
                    continue
                tried = self.memory.tried_count(scene.exact_key, spec)
                known_next, replay_conf = self.world_model.replay_predict(
                    scene.exact_key, spec)
                known_hash, hash_conf, effect = self.spatial_hash.predict(
                    scene, spec) if self.config.enable_spatial_hash else (False, 0.0, None)
                if use_hypotheses:
                    hypothesis_known, hypothesis_conf, hypothesis_effect = self.hypotheses.predict(
                        signature)
                    information_value = self.hypotheses.information_value(
                        signature)
                    goal_bonus = self.hypotheses.goal_bonus(signature)
                else:
                    hypothesis_known, hypothesis_conf, hypothesis_effect = False, 0.0, None
                    information_value, goal_bonus = 0.0, 0.0
                success = self.memory.action_success_rate(action.name)
                noop = self.memory.action_noop_rate(action.name)
                is_probe = tried == 0 and known_next is None and not known_hash and not hypothesis_known
                is_noop_hyp = hypothesis_known and hypothesis_effect == "noop"
                is_noop_replay = known_next == scene.exact_key
                is_noop_hash = known_hash and effect == "noop"
                score = (
                    1.25 * success
                    - 1.1 * noop
                    - 1.2 * tried
                    + (0.7 * hypothesis_conf if not is_noop_hyp else -1.5 * hypothesis_conf)
                    + 0.45 * information_value
                    + goal_bonus
                )
                rationale = [
                    f"success={success:.2f}",
                    f"noop={noop:.2f}",
                    f"hyp_info={information_value:.2f}",
                ]
                if known_next is not None:
                    if is_noop_replay:
                        score -= 2.5 * replay_conf
                        rationale.append("known_self_loop_noop")
                    else:
                        score += 1.8 * replay_conf
                        rationale.append(f"replay={replay_conf:.2f}")
                if known_hash:
                    if is_noop_hash:
                        score -= 2.0 * hash_conf
                        rationale.append("known_hash_noop")
                    else:
                        score += 1.4 * hash_conf
                        rationale.append(f"spatial_hash={hash_conf:.2f}")
                if hypothesis_known:
                    if not is_noop_hyp:
                        score += 0.9 * hypothesis_conf
                        rationale.append(f"hypothesis={hypothesis_conf:.2f}")
                if tried == 0:
                    score += 0.8
                    rationale.append("untried_here")

                # Collision-Responsive Orthogonal Steering on Level 0
                if self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
                    if self.memory.recommended_orthogonal_turn == "vertical":
                        if action.name in ("ACTION1", "ACTION2"):
                            score += 2.5
                            rationale.append("orthogonal_turn_bonus")
                        elif action.name in ("ACTION3", "ACTION4"):
                            score -= 3.0
                            rationale.append("collision_avoidance_penalty")
                    elif self.memory.recommended_orthogonal_turn == "horizontal":
                        if action.name in ("ACTION3", "ACTION4"):
                            score += 2.5
                            rationale.append("orthogonal_turn_bonus")
                        elif action.name in ("ACTION1", "ACTION2"):
                            score -= 3.0
                            rationale.append("collision_avoidance_penalty")
                if self.dynamics is not None and not scene.field_mode and scene.controlled_entity_id is not None:
                    reliable = self.dynamics.reliable_vectors(legal_names)
                    simple_actions = [a for a in legal_actions if not a.is_complex() and a is not GameAction.RESET]
                    if len(reliable) < min(4, len(simple_actions)):
                        if tried == 0:
                            score += 3.0
                            rationale.append("dynamics_calibration_probe")
                candidates.append(
                    Candidate(
                        ActionSpec(
                            name=action.name,
                            source="replay" if known_next is not None else (
                                "spatial_hash" if known_hash else (
                                    "predictive_hypothesis" if hypothesis_known else "simple_candidate"
                                )
                            ),
                            predicted_effect=effect or known_next or hypothesis_effect,
                            score=score,
                        ),
                        signature,
                        is_probe,
                        score,
                        rationale,
                    )
                )
        if not candidates and legal_actions:
            for action in legal_actions:
                if action is GameAction.RESET:
                    continue
                if action.is_complex():
                    for coord_score, (x, y), why in self._complex_coordinates(scene):
                        spec = ActionSpec(
                            name=action.name,
                            data=(("x", x), ("y", y)),
                            source="complex_fallback",
                            score=coord_score - 10.0,
                        )
                        signature = _action_signature(scene, spec)
                        candidates.append(
                            Candidate(spec, signature, True, coord_score - 10.0, ["dead_signature_fallback"])
                        )
                else:
                    spec = ActionSpec(name=action.name, source="simple_fallback")
                    signature = _action_signature(scene, spec)
                    candidates.append(
                        Candidate(
                            ActionSpec(name=action.name, source="simple_fallback", score=-10.0),
                            signature,
                            True,
                            -10.0,
                            ["dead_signature_fallback"],
                        )
                    )
        return candidates

    def mark_committed(self, spec: ActionSpec) -> None:
        data = spec.data_dict
        if "x" in data and "y" in data:
            self.coordinate_visits[(data["x"], data["y"])] += 1


@dataclass(slots=True)
class CounterfactualPlan:
    first_action: ActionSpec
    remaining: tuple[ActionSpec, ...]
    score: float
    prediction: ProgramPrediction
    alignment: AlignmentDecision



def _evaluate_state_completion_and_terminal(
    grid: np.ndarray,
    prev_grid: np.ndarray,
    scene: Scene,
    top_mechanism_family: str,
    cand_spec: ActionSpec | None = None,
) -> tuple[float, float, bool, bool]:
    """
    Evaluates completion progress, closure, and terminal conditions for a predicted grid.
    Returns: (completion_bonus, terminal_bonus, line_closure_used, delete_payoff_used)
    """
    comp_bonus = 0.0
    term_bonus = 0.0
    closure_used = False
    payoff_used = False

    bg = scene.background
    if top_mechanism_family == "component_delete":
        # Measure non-background mass and live components
        prev_non_bg = np.count_nonzero(prev_grid != bg)
        curr_non_bg = np.count_nonzero(grid != bg)
        if curr_non_bg < prev_non_bg:
            reduction = prev_non_bg - curr_non_bg
            comp_bonus += min(3.0, 1.0 + 0.2 * reduction)
            payoff_used = True
        # Terminal condition: clean or nearly clean board
        if curr_non_bg <= 3 and curr_non_bg < prev_non_bg:
            term_bonus += 4.5
        elif curr_non_bg <= 8:
            term_bonus += 2.5

    elif top_mechanism_family == "line_or_beam":
        # Check alignment / span closure
        if cand_spec and cand_spec.data:
            c_dict = dict(cand_spec.data)
            cx, cy = c_dict.get("x"), c_dict.get("y")
            if cx is not None and cy is not None and 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]:
                # Reward placement along continuous row/col spans
                col_match = np.count_nonzero(grid[:, cx] != bg)
                row_match = np.count_nonzero(grid[cy, :] != bg)
                if col_match >= 3 or row_match >= 3:
                    comp_bonus += 2.2
                    closure_used = True
        # Terminal condition: fully connected beam/path (few remaining disconnected endpoints)
        if np.count_nonzero(grid != prev_grid) > 0 and closure_used:
            term_bonus += 3.0

    elif top_mechanism_family == "targeted_recolor":
        prev_colors = set(np.unique(prev_grid)) - {bg}
        curr_colors = set(np.unique(grid)) - {bg}
        if len(curr_colors) < len(prev_colors):
            comp_bonus += 2.4
        if len(curr_colors) <= 2 and len(curr_colors) < len(prev_colors):
            term_bonus += 3.5

    elif top_mechanism_family in ("movement_control", "drag_or_push"):
        if np.count_nonzero(grid != prev_grid) > 0:
            comp_bonus += 1.5
            # Terminal condition: displacement to boundary or socket
            if cand_spec and cand_spec.data:
                c_dict = dict(cand_spec.data)
                cx, cy = c_dict.get("x"), c_dict.get("y")
                if cx is not None and cy is not None:
                    if cx in (0, grid.shape[1] - 1) or cy in (0, grid.shape[0] - 1):
                        term_bonus += 2.0

    return comp_bonus, term_bonus, closure_used, payoff_used

class CounterfactualPlanner:
    """Bounded beam search through verified programs on previously unseen states."""

    def __init__(
        self,
        config: Config,
        programs: ExecutableProgramLibrary,
        goals: GoalHypothesisManager,
        alignment: GoalAlignmentVerifier,
        memory: Any = None,
    ) -> None:
        self.config = config
        self.programs = programs
        self.goals = goals
        self.alignment = alignment
        self.memory = memory

    def plan(
        self,
        scene: Scene,
        candidates: Sequence[Candidate],
        legal_names: set[str],
    ) -> CounterfactualPlan | None:
        if not self.config.enable_counterfactual_planner:
            return None
        base_actions: list[ActionSpec] = []
        seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            if candidate.spec.key in seen:
                continue
            # Prune fatigued interior coordinates (>= 4 visits on Level 0) from counterfactual planning
            if self.memory and self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active and candidate.spec.data:
                c_dict = dict(candidate.spec.data)
                if "x" in c_dict and "y" in c_dict:
                    # Do not prune UI palette selector row (y <= 2)
                    if c_dict["y"] > 2 and c_dict["x"] > 2:
                        if self.memory.spatial_visits_by_action.get((candidate.spec.name, c_dict["x"], c_dict["y"]), 0) >= 4:
                            continue
            seen.add(candidate.spec.key)
            base_actions.append(candidate.spec)
            if len(base_actions) >= self.config.counterfactual_candidate_limit:
                break
        if not base_actions:
            return None

        # node = (score, grid, first, sequence, confidence_product, last_prediction)
        beam: list[tuple[float, np.ndarray, ActionSpec | None, tuple[ActionSpec, ...], float, ProgramPrediction | None]] = [
            (0.0, scene.grid, None, (), 1.0, None)
        ]
        best: CounterfactualPlan | None = None
        visited = {scene.exact_key}
        max_depth = self.config.counterfactual_depth
        prog_count = len(self.programs.programs) if hasattr(self.programs, "programs") else 0
        if prog_count >= 1:
            max_depth = max(max_depth, 3)
            if prog_count >= 3:
                max_depth = max(max_depth, 4)
        for depth in range(1, max_depth + 1):
            expanded: list[tuple[float, np.ndarray, ActionSpec | None,
                                 tuple[ActionSpec, ...], float, ProgramPrediction | None]] = []
            for parent_score, grid, first, sequence, confidence_product, _ in beam:
                for action in base_actions:
                    prediction = self.programs.predict_grid(
                        grid, action, scene if depth == 1 else None)
                    if prediction is None:
                        continue
                    key = _stable_hash_bytes(prediction.grid.tobytes())
                    if key in visited:
                        continue
                    visited.add(key)
                    first_action = first or ActionSpec(
                        name=action.name,
                        data=action.data,
                        source="counterfactual_program",
                        predicted_effect=prediction.expected_effect,
                        score=action.score,
                        program_id=prediction.program_id,
                        predicted_state_key=key,
                        goal_ids=tuple(
                            g.goal_id for g in self.goals.active(4)),
                    )
                    new_sequence = sequence + (action,)
                    new_confidence = confidence_product * \
                        max(0.05, prediction.confidence)
                    goal_delta = self.goals.prediction_delta(
                        scene, prediction.grid)
                    novelty = 0.08 * depth
                    # Mechanism completion and terminal condition scoring
                    top_fam = getattr(self, "memory", None).top_mechanism_family if hasattr(self, "memory") else ""
                    comp_bonus, term_bonus, closure_used, payoff_used = _evaluate_state_completion_and_terminal(
                        prediction.grid, grid, scene, top_fam, action
                    )
                    step_score = (
                        parent_score
                        + 2.4 * goal_delta
                        + 0.75 * new_confidence
                        - 0.32 * depth
                        - 0.8 * prediction.uncertainty
                        + novelty
                        + comp_bonus
                        + term_bonus
                    )
                    decision = self.alignment.verify(scene, first_action, legal_names, prediction if depth == 1 else self.programs.predict_grid(
                        scene.grid, first_action, scene), False)
                    if depth == 1 and decision.allowed and (prediction.kind != "exact_replay" or goal_delta > 0.05):
                        plan = CounterfactualPlan(
                            first_action, (), step_score + decision.score, prediction, decision)
                        if best is None or plan.score > best.score:
                            best = plan
                    if decision.risk < 1.0 and new_confidence >= 0.20:
                        expanded.append(
                            (step_score, prediction.grid, first_action, new_sequence, new_confidence, prediction))
            expanded.sort(key=lambda n: n[0], reverse=True)
            beam = expanded[: self.config.counterfactual_beam]
            if not beam:
                break
            for node_score, _, first_action, seq, _, prediction in beam:
                if first_action is None or prediction is None:
                    continue
                first_prediction = self.programs.predict_grid(
                    scene.grid, first_action, scene)
                if first_prediction is None:
                    continue
                decision = self.alignment.verify(
                    scene, first_action, legal_names, first_prediction, False)
                if not decision.allowed:
                    continue
                remaining = seq[1: 1 + self.config.plan_queue_limit]
                plan = CounterfactualPlan(first_action, tuple(
                    remaining), node_score + decision.score, first_prediction, decision)
                if best is None or plan.score > best.score:
                    best = plan
        return best if best is not None and best.score > 0.10 else None


class MetacognitiveController:
    def __init__(
        self,
        config: Config,
        memory: TraceMemory,
        dead: DeadSignatureMemory,
        world_model: CausalWorldModel,
        candidate_generator: CandidateGenerator,
        path_planner: PathPlanner,
        reasoner: OptionalLocalReasoner,
        goals: GoalHypothesisManager,
        programs: ExecutableProgramLibrary,
        counterfactual: CounterfactualPlanner,
        alignment: GoalAlignmentVerifier,
        hypotheses: HypothesisMemory,
    ) -> None:
        self.config = config
        self.memory = memory
        self.dead = dead
        self.world_model = world_model
        self.candidate_generator = candidate_generator
        self.path_planner = path_planner
        self.reasoner = reasoner
        self.goals = goals
        self.programs = programs
        self.counterfactual = counterfactual
        self.alignment = alignment
        self.hypotheses = hypotheses
        self.plan_queue: deque[ActionSpec] = deque(
            maxlen=config.plan_queue_limit)
        self.last_model_milestone = ""
        self.last_response_frames: tuple[np.ndarray, ...] = ()
        self.counterfactual_streak = 0
        self.failed_consultations_this_level = 0
        self.consecutive_fallback_steps = 0
        self.fallback_loop_streak = 0
        self.counterfactual_fallback_streak = 0
        self.alignment_fallback_streak = 0
        self.steps_since_progress = 0
        self.progress_events_this_level = 0
        self.progress_density = 0.0
        self.recent_lock_skip_count = 0
        self.reasoner_lock_backoff_steps = 0
        self.reasoner_suppressed = False
        self.reasoner_suppression_reason = ""
        self.stuck_mode_activations = 0
        self.stagnation_override_count_this_level = 0
        self.llm_step_meta: dict[str, Any] = {}
        self.perception = None
        self.post_breakthrough_stuck_mode_uses = 0
        self.post_breakthrough_fallback_uses = 0

    def reset_level(self) -> None:
        self.plan_queue.clear()
        self.last_model_milestone = ""
        self.last_response_frames = ()
        self.counterfactual_streak = 0
        self.failed_consultations_this_level = 0
        self.consecutive_fallback_steps = 0
        self.fallback_loop_streak = 0
        self.counterfactual_fallback_streak = 0
        self.alignment_fallback_streak = 0
        self.steps_since_progress = 0
        self.progress_events_this_level = 0
        self.progress_density = 0.0
        self.recent_lock_skip_count = 0
        self.reasoner_lock_backoff_steps = 0
        self.reasoner_suppressed = False
        self.reasoner_suppression_reason = ""
        self.stuck_mode_activations = 0
        self.stagnation_override_count_this_level = 0

    def _milestone(self, scene: Scene, step: int) -> tuple[bool, str]:
        last = self.memory.last()
        if step <= 1:
            return True, "level_start"
        if last is not None:
            if last.event.game_over:
                return True, "death"
            if last.event.topology_change:
                return True, "topology_change"
            if last.event.subframe_count > 1 and last.event.transient_changed_count > 0:
                return True, "animation_novelty"
            if last.event.changed_count > 0 and len(self.memory.transitions) <= 3:
                return True, "first_significant_transition"
        if self.memory.recent_state_loop(self.config.loop_window):
            return True, "loop"
        if self.memory.no_op_streak >= self.config.no_progress_consecutive_threshold:
            return True, "stagnation"
        if scene.field_mode and scene.entity_confidence < 0.35:
            return True, "representation_conflict"
        return False, ""

    def _predict(
        self,
        scene: Scene,
        spec: ActionSpec,
        profile: RuntimeProfile,
    ) -> ProgramPrediction | None:
        if not profile.use_programs:
            return None
        return self.programs.predict_grid(scene.grid, spec, scene)

    def _reground_winning_target(self, scene: Scene) -> tuple[tuple[int, int] | None, tuple[int, int] | None, float, str]:
        if not self.memory.last_winning_coords or not self.memory.post_breakthrough_window_active:
            return None, None, 0.0, "no_winning_coords_or_inactive_window"
        
        w_x, w_y = self.memory.last_winning_coords
        components = scene.components
        if not components:
            return None, None, 0.0, "no_components_in_scene"

        target_color = self.memory.last_winning_component_color
        target_area = max(1, self.memory.last_winning_component_area)
        target_aspect = self.memory.last_winning_component_aspect
        target_shape = self.memory.last_winning_component_shape_hash

        best_comp = None
        best_score = 0.0
        best_delta = (0, 0)

        for comp in components:
            c_x, c_y = int(round(comp.centroid[0])), int(round(comp.centroid[1]))
            # 1. Color match score
            color_score = 1.0 if (target_color is not None and comp.color == target_color) else (0.4 if target_color is None else 0.0)
            
            # 2. Shape Key exact match
            shape_score = 1.0 if (target_shape and comp.shape_key == target_shape) else 0.0
            
            # 3. Area similarity score
            area_ratio = min(comp.area, target_area) / max(comp.area, target_area)
            
            # 4. Aspect ratio similarity score
            bbox_w = comp.bbox[2] - comp.bbox[0] + 1
            bbox_h = comp.bbox[3] - comp.bbox[1] + 1
            aspect = bbox_w / max(1, bbox_h)
            aspect_sim = 1.0 - min(1.0, abs(aspect - target_aspect) / max(0.2, target_aspect))
            
            # 5. Normalized centroid proximity
            norm_dist = math.hypot((c_x - w_x) / max(1, scene.width), (c_y - w_y) / max(1, scene.height))
            proximity_score = max(0.0, 1.0 - norm_dist)
            
            # Weighted total similarity
            score = 0.35 * color_score + 0.20 * shape_score + 0.20 * area_ratio + 0.15 * aspect_sim + 0.10 * proximity_score
            
            if score > best_score:
                best_score = score
                best_comp = comp
                best_delta = (c_x - w_x, c_y - w_y)

        if best_comp is not None and best_score >= 0.50:
            c_x, c_y = int(round(best_comp.centroid[0])), int(round(best_comp.centroid[1]))
            regrounded = (max(0, min(scene.width - 1, c_x)), max(0, min(scene.height - 1, c_y)))
            return regrounded, best_delta, round(best_score, 2), "component_matched"

        return None, None, round(best_score, 2), "low_component_similarity"

    def _update_post_breakthrough_window(self, step: int = 0) -> None:
        if not self.memory.post_breakthrough_window_active:
            return
        
        # Contradiction check: repeated no-ops
        if self.memory.no_op_streak >= 5:
            self.memory.post_breakthrough_window_active = False
            self.memory.post_breakthrough_aborted_reason = "high_noop_contradiction"
            return
        # Contradiction check: repeated deaths
        if self.memory.recent_death_count(5) >= 2:
            self.memory.post_breakthrough_window_active = False
            self.memory.post_breakthrough_aborted_reason = "death_risk_contradiction"
            return

        # Contradiction check: zero state changes across recent steps
        recent = list(self.memory.transitions)[-6:]
        if len(recent) >= 6 and all(t.event.changed_count == 0 for t in recent):
            self.memory.post_breakthrough_window_active = False
            self.memory.post_breakthrough_aborted_reason = "zero_state_change_contradiction"
            return

        # Decrement window steps
        self.memory.post_breakthrough_window_steps_remaining -= 1
        if self.memory.post_breakthrough_window_steps_remaining <= 0:
            self.memory.post_breakthrough_window_active = False
            self.memory.post_breakthrough_aborted_reason = "window_expired"

    def _update_mechanism_beliefs(self, step: int = 0) -> None:
        if not self.memory.transitions:
            return

        # Stagnation / TTL decay for priority program families
        if (self.memory.priority_program_ttl > 0 and step > self.memory.priority_program_ttl) or self.steps_since_progress >= 12:
            self.memory.priority_program_families.clear()
            self.memory.priority_program_ttl = 0

        if self.memory.recommended_probe_ttl > 0 and step > self.memory.recommended_probe_ttl:
            self.memory.recommended_probe_type = ""
            self.memory.recommended_probe_ttl = 0
        
        # Positive and negative evidence accumulation across recent transitions
        triggered_detectors = set()
        recent = list(self.memory.transitions)[-10:]
        for t in recent:
            moves = getattr(t.event, "entity_moves", ())
            
            # Direct Detector 1: drag_or_push
            if len(moves) >= 1 and t.action.name == "ACTION6":
                self.memory.mechanism_scores["drag_or_push"] = min(1.0, self.memory.mechanism_scores["drag_or_push"] + 0.25)
                self.memory.mechanism_scores["targeted_recolor"] = max(0.02, self.memory.mechanism_scores["targeted_recolor"] - 0.05)
                triggered_detectors.add("drag_or_push")
            elif len(moves) == 1:
                self.memory.mechanism_scores["movement_control"] = min(1.0, self.memory.mechanism_scores["movement_control"] + 0.15)
                triggered_detectors.add("movement_control")
            elif len(moves) > 1:
                # Direct Detector 2: gravity_or_fall
                self.memory.mechanism_scores["gravity_or_fall"] = min(1.0, self.memory.mechanism_scores["gravity_or_fall"] + 0.25)
                triggered_detectors.add("gravity_or_fall")
            else:
                # Negative evidence against movement/gravity when transitions produce 0 movement
                self.memory.mechanism_scores["movement_control"] = max(0.02, self.memory.mechanism_scores["movement_control"] - 0.04)
                self.memory.mechanism_scores["gravity_or_fall"] = max(0.02, self.memory.mechanism_scores["gravity_or_fall"] - 0.04)

            if t.event.topology_change:
                self.memory.mechanism_scores["topology_switch"] = min(1.0, self.memory.mechanism_scores["topology_switch"] + 0.2)
                self.memory.mechanism_scores["line_or_beam"] = min(1.0, self.memory.mechanism_scores["line_or_beam"] + 0.15)
                triggered_detectors.add("topology_switch")
            
            # Direct Detector 3: flood_or_fill
            if t.event.changed_count >= 8 and len(moves) == 0:
                self.memory.mechanism_scores["flood_or_fill"] = min(1.0, self.memory.mechanism_scores["flood_or_fill"] + 0.30)
                triggered_detectors.add("flood_or_fill")
            elif t.event.changed_count > 0 and len(moves) == 0:
                self.memory.mechanism_scores["targeted_recolor"] = min(1.0, self.memory.mechanism_scores["targeted_recolor"] + 0.15)
                triggered_detectors.add("targeted_recolor")
            elif t.event.changed_count == 0:
                # Negative evidence against recolor/flood when no pixels changed
                self.memory.mechanism_scores["targeted_recolor"] = max(0.02, self.memory.mechanism_scores["targeted_recolor"] - 0.03)
                self.memory.mechanism_scores["flood_or_fill"] = max(0.02, self.memory.mechanism_scores["flood_or_fill"] - 0.03)

            # Direct Detector 4: component_delete
            if t.event.disappeared_colors or (t.event.changed_count > 0 and len(t.event.appeared_colors) == 0 and len(moves) == 0):
                self.memory.mechanism_scores["component_delete"] = min(1.0, self.memory.mechanism_scores["component_delete"] + 0.25)
                triggered_detectors.add("component_delete")
            elif t.event.changed_count > 0 and not t.event.disappeared_colors:
                self.memory.mechanism_scores["component_delete"] = max(0.02, self.memory.mechanism_scores["component_delete"] - 0.04)

        self.memory.mechanism_detector_triggered = ",".join(sorted(triggered_detectors))
                
        # Normalize and compute top belief
        total = sum(self.memory.mechanism_scores.values()) or 1.0
        prev_top = self.memory.top_mechanism_family
        sorted_fams = sorted(self.memory.mechanism_scores.items(), key=lambda x: -x[1])
        top_cand_family = sorted_fams[0][0]

        # Mechanism-Family Collapse Breaker on Level 0
        if self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
            if top_cand_family:
                self.memory.family_stagnation_steps[top_cand_family] = self.memory.family_stagnation_steps.get(top_cand_family, 0) + 1
                if (
                    self.memory.family_stagnation_steps[top_cand_family] >= 20
                    and self.memory.level_progress_events == 0
                    and len(sorted_fams) > 1
                ):
                    penalized_family = top_cand_family
                    promoted_family = sorted_fams[1][0]
                    # Temporarily down-rank stalled dominant family and boost runner-up
                    self.memory.mechanism_scores[penalized_family] = max(0.05, self.memory.mechanism_scores.get(penalized_family, 0.5) * 0.30)
                    self.memory.mechanism_scores[promoted_family] = min(1.0, self.memory.mechanism_scores.get(promoted_family, 0.2) + 0.45)
                    self.memory.mechanism_collapse_breaker_used = True
                    self.memory.mechanism_family_penalized = penalized_family
                    self.memory.mechanism_family_promoted = promoted_family
                    self.memory.family_stagnation_steps[penalized_family] = 0
                    # Re-sort after applying breaker
                    total = sum(self.memory.mechanism_scores.values()) or 1.0
                    sorted_fams = sorted(self.memory.mechanism_scores.items(), key=lambda x: -x[1])

        self.memory.top_mechanism_family = sorted_fams[0][0]
        self.memory.top_mechanism_confidence = round(sorted_fams[0][1] / max(1.0, total), 3)
        self.memory.competing_mechanism_families = [f[0] for f in sorted_fams[:3]]
        self.memory.early_classified_mechanism = self.memory.top_mechanism_family
        
        if prev_top != self.memory.top_mechanism_family and len(self.memory.transitions) >= 5:
            self.memory.mechanism_shift_event = True
            
        # Mode management: discovery vs exploitation
        if self.memory.top_mechanism_confidence >= 0.32 and self.steps_since_progress <= 25:
            self.memory.mode = "exploitation_mode"
        else:
            self.memory.mode = "discovery_mode"

    def _check_severe_stagnation(self) -> tuple[bool, str]:
        target_budget = 120
        pressure = min(2.0, self.memory.level_steps / target_budget) if target_budget > 0 else 0.0
        is_early_level = self.memory.current_level_index <= 1
        no_level_progress = self.memory.level_progress_events == 0

        # Early-level breakout acceleration: trigger earlier before budget is burned
        if is_early_level and no_level_progress:
            accelerated_steps_thresh = max(14, int(30 - 10 * pressure))
            if self.steps_since_progress >= accelerated_steps_thresh:
                return True, "early_breakout_accelerated"
            accelerated_noop_thresh = max(6, int(12 - 4 * pressure))
            if self.memory.no_op_streak >= accelerated_noop_thresh:
                return True, "early_noop_accelerated"

        if self.steps_since_progress >= 30:
            return True, "steps_since_progress"
        if self.memory.no_op_streak >= 12:
            return True, "no_op_streak"
        repeated = self.memory.repeated_effect_signature(window=8)
        if repeated is not None and repeated[1] >= 4:
            return True, "repeated_effect_signature"
        if self.counterfactual_fallback_streak >= 20:
            return True, "counterfactual_fallback_streak"
        if self.fallback_loop_streak >= 20:
            return True, "fallback_loop_streak"
        # Progressless churn signal: moving/changing without solving
        if self.steps_since_progress >= 25 and self.memory.no_op_streak == 0 and repeated is not None and repeated[1] >= 3:
            return True, "progressless_churn"
        return False, ""

    def _priority_family_boost(self, kind: str | None) -> float:
        if not kind or not self.memory.priority_program_families:
            return 0.0
        for idx, prio in enumerate(self.memory.priority_program_families):
            if kind == prio or kind in MECHANISM_TO_PROGRAM_KINDS.get(prio, set()):
                return max(0.2, 0.8 - 0.2 * idx)
        return 0.0

    def _probe_matches_recommendation(self, candidate: Candidate, scene: Scene, step: int = 0) -> tuple[bool, float]:
        rec = self.memory.recommended_probe_type
        if not rec or (self.memory.recommended_probe_ttl > 0 and step > self.memory.recommended_probe_ttl):
            return False, 0.0
        name = candidate.spec.name
        data = candidate.spec.data_dict
        x, y = data.get("x"), data.get("y")
        matched = False
        
        if rec == "directional_move":
            if name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7", "ACTION8"):
                matched = True
        elif rec == "targeted_click":
            if name == "ACTION6" and x is not None and y is not None:
                matched = True
        elif rec == "boundary_test":
            if x is not None and y is not None and (x == 0 or x == scene.width - 1 or y == 0 or y == scene.height - 1):
                matched = True
        elif rec == "component_interaction":
            if x is not None and y is not None and (0 <= x < scene.width and 0 <= y < scene.height):
                if scene.grid[y, x] != scene.background:
                    matched = True
        elif rec == "color_trigger":
            if x is not None and y is not None and (0 <= x < scene.width and 0 <= y < scene.height):
                counts = dict(scene.color_counts)
                c = scene.grid[y, x]
                if c != scene.background and counts.get(c, 999) <= 12:
                    matched = True
        elif rec == "topology_test":
            if name in ("ACTION6", "ACTION3", "ACTION5"):
                matched = True
                
        return matched, (0.75 if matched else 0.0)

    def _is_promising_state(self, scene: Scene, step: int, profile: RuntimeProfile) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        
        # Hard exclusions: severe stagnation with no progress, or excessive deaths
        if self.memory.no_op_streak >= 12 and self.progress_events_this_level == 0:
            return False, []
        if self.memory.recent_death_count(10) >= 3:
            return False, []
            
        # Signal 1: First level progress already occurred, active post-breakthrough window, or level > 0 continuation
        if self.progress_events_this_level > 0 or self.memory.level_progress_events > 0:
            reasons.append("level_progress_occurred")
        if self.memory.post_breakthrough_window_active:
            reasons.append(f"post_breakthrough_window_active_{self.memory.transferred_winning_family}")
        if self.memory.current_level_index > 0 and self.memory.regrounding_used:
            reasons.append("level_continuation_regrounded_active")
            
        # Signal 2: High mechanism confidence
        if self.memory.top_mechanism_confidence >= 0.35:
            reasons.append(f"high_mechanism_confidence_{self.memory.top_mechanism_family}")
            
        # Signal 3: Active follow-through or priority program families
        if self.memory.llm_follow_through_window > 0:
            reasons.append("active_follow_through_window")
        if self.memory.priority_program_families:
            reasons.append("priority_program_families_active")
            
        # Signal 4: Recent productive transitions
        recent = list(self.memory.transitions)[-8:]
        productive = [t for t in recent if t.event.changed_count > 0 and not t.event.no_op and not t.event.game_over]
        if len(productive) >= 3:
            reasons.append(f"frequent_productive_transitions_{len(productive)}")
            
        # Signal 5: Active verified program family with positive payoff
        top_fam = self.memory.top_mechanism_family
        if top_fam in self.memory.llm_family_payoff:
            p_data = self.memory.llm_family_payoff[top_fam]
            score = p_data.get("changed", 0) + 3 * p_data.get("progress", 0) - p_data.get("noop", 0)
            if score >= 3:
                reasons.append(f"productive_family_payoff_{top_fam}")
                
        is_promising = len(reasons) >= 1 and profile.use_counterfactual
        return is_promising, reasons

    def _promising_state_deep_search(
        self,
        scene: Scene,
        candidates: Sequence[Candidate],
        legal_actions: Sequence[GameAction],
        legal_names: Sequence[str],
        profile: RuntimeProfile,
        remaining_sec: float | None = None,
        promising_reasons: Sequence[str] = (),
    ) -> tuple[ActionSpec | None, list[ActionSpec], dict[str, Any]]:
        t_start = time.monotonic()
        meta = {
            "promising_state_detected": True,
            "promising_state_reasons": list(promising_reasons),
            "deep_search_used": False,
            "deep_search_depth": 0,
            "deep_search_width": 0,
            "deep_search_nodes_evaluated": 0,
            "deep_search_best_score": -999.0,
            "deep_search_time_ms": 0.0,
            "deep_search_selected_family": "",
            "deep_search_aborted_reason": "",
        }

        # Runtime tier budget assignment
        tier = getattr(profile, "tier", "A9")
        if tier == "A9":
            max_depth, max_width, max_nodes = 8, 8, 45
            time_cap_sec = 1.5
        elif tier == "A8":
            max_depth, max_width, max_nodes = 5, 6, 25
            time_cap_sec = 1.0
        elif tier == "A7":
            max_depth, max_width, max_nodes = 3, 4, 12
            time_cap_sec = 0.5
        else:
            meta["deep_search_aborted_reason"] = "disabled_for_tier"
            return None, [], meta

        if remaining_sec is not None and remaining_sec < 45.0:
            meta["deep_search_aborted_reason"] = "insufficient_wall_time"
            return None, [], meta

        max_allowed_time = min(time_cap_sec, (remaining_sec * 0.05) if remaining_sec else time_cap_sec)
        meta["deep_search_depth"] = max_depth
        meta["deep_search_width"] = max_width

        # Filter and rank initial seed candidates
        scored_seeds: list[tuple[float, Candidate]] = []
        top_fam = self.memory.top_mechanism_family
        prio_fams = self.memory.priority_program_families
        top_kinds = MECHANISM_TO_PROGRAM_KINDS.get(top_fam, set())

        reground_coords = self.memory.regrounded_winning_coords or self.memory.transferred_winning_coords
        for cand in candidates:
            if cand.spec.name not in legal_names or self.dead.is_dead(cand.signature):
                continue
            pred = self._predict(scene, cand.spec, profile)
            decision = self._verify(scene, cand.spec, legal_names, pred, cand.is_probe or (pred is None), profile)
            p_kind = getattr(pred, "kind", "") if pred else ""
            is_mech_aligned = (p_kind and (p_kind in top_kinds or p_kind == top_fam or any(p_kind == pf for pf in prio_fams)))
            
            if not decision.allowed:
                # Relax seed admission on Level 0 when candidate is mechanism-aligned
                if self.memory.current_level_index == 0 and decision.score >= -1.5 and is_mech_aligned:
                    pass
                else:
                    continue
            
            seed_score = cand.score + decision.score
            applied_transfer_bias = False
            if p_kind:
                if p_kind == top_fam or p_kind in top_kinds:
                    seed_score += 1.5
                elif any(p_kind == pf or p_kind in MECHANISM_TO_PROGRAM_KINDS.get(pf, set()) for pf in prio_fams):
                    seed_score += 1.0
            if self.memory.post_breakthrough_window_active or self.memory.current_level_index > 0:
                if p_kind and (p_kind == self.memory.transferred_winning_program_kind or p_kind == self.memory.transferred_winning_family):
                    seed_score += 2.0
                    applied_transfer_bias = True
                if cand.spec.name == self.memory.transferred_winning_action_name:
                    seed_score += 1.5
                    applied_transfer_bias = True
                if reground_coords and cand.spec.data:
                    c_dict = dict(cand.spec.data)
                    if "x" in c_dict and "y" in c_dict:
                        dist = max(abs(c_dict["x"] - reground_coords[0]), abs(c_dict["y"] - reground_coords[1]))
                        if dist <= 2:
                            seed_score += 1.8
                            applied_transfer_bias = True
                        elif dist <= 4:
                            seed_score += 0.9
                            applied_transfer_bias = True
            scored_seeds.append((seed_score, cand, applied_transfer_bias))

        if not scored_seeds:
            meta["deep_search_aborted_reason"] = "no_valid_seed_candidates"
            return None, [], meta

        scored_seeds.sort(key=lambda x: -x[0])
        initial_frontier = scored_seeds[:max_width]

        # Beam structure: (total_score, action_path, current_grid, current_scene, last_pred_kind, applied_transfer_bias)
        beams: list[tuple[float, list[ActionSpec], np.ndarray, Scene, str, bool]] = []
        for s_score, cand, s_bias in initial_frontier:
            pred = self._predict(scene, cand.spec, profile)
            next_grid = pred.grid if (pred and pred.grid is not None) else scene.grid
            p_kind = getattr(pred, "kind", "") if pred else ""
            beams.append((s_score, [cand.spec], next_grid, scene, p_kind, s_bias))

        nodes_evaluated = len(initial_frontier)
        best_path: list[ActionSpec] = []
        best_score = -999.0
        best_family = ""
        best_bias_beneficiary = (initial_frontier[0][2] if initial_frontier else False)

        # Multi-step mental beam search rollouts
        for d in range(2, max_depth + 1):
            if (time.monotonic() - t_start) >= max_allowed_time or nodes_evaluated >= max_nodes:
                break

            next_beams: list[tuple[float, list[ActionSpec], np.ndarray, Scene, str, bool]] = []
            for b_score, path, b_grid, b_scene, b_kind, b_bias in beams:
                sub_scene = self.perception.perceive(b_grid, b_scene.level, path[-1]) if getattr(self, "perception", None) is not None else b_scene
                sub_candidates = self.candidate_generator.generate(sub_scene, legal_actions, use_hypotheses=profile.use_hypotheses)
                
                for s_cand in sub_candidates[:max_width]:
                    nodes_evaluated += 1
                    if (time.monotonic() - t_start) >= max_allowed_time or nodes_evaluated >= max_nodes:
                        break

                    sig = _action_signature(sub_scene, s_cand.spec)
                    if self.dead.is_dead(sig):
                        continue
                    if s_cand.spec.data:
                        c_dict = dict(s_cand.spec.data)
                        if "x" in c_dict and "y" in c_dict:
                            if self.memory.spatial_visits.get((c_dict["x"], c_dict["y"]), 0) >= 3:
                                continue

                    pred = self._predict(sub_scene, s_cand.spec, profile)
                    dec = self._verify(sub_scene, s_cand.spec, legal_names, pred, False, profile)
                    if not dec.allowed:
                        continue

                    # Evaluate multi-step trajectory
                    step_reward = s_cand.score + dec.score
                    p_kind = getattr(pred, "kind", "") if pred else b_kind
                    applied_traj_bias = b_bias
                    
                    if pred is not None:
                        if pred.expected_effect is not None:
                            step_reward += 1.2
                        if p_kind in top_kinds or p_kind == top_fam:
                            step_reward += 1.0
                        if getattr(pred, "predicted_level_delta", 0) > 0:
                            step_reward += 6.0
                        if pred.grid is not None and not np.array_equal(pred.grid, b_grid):
                            step_reward += 1.5
                            # Add completion and terminal condition bonuses to deep search
                            c_bonus, t_bonus, _, _ = _evaluate_state_completion_and_terminal(
                                pred.grid, b_grid, sub_scene, top_fam, s_cand.spec
                            )
                            step_reward += (c_bonus + t_bonus)
                    else:
                        step_reward += 0.5
                    
                    if self.memory.post_breakthrough_window_active:
                        if p_kind and (p_kind == self.memory.transferred_winning_program_kind or p_kind == self.memory.transferred_winning_family):
                            step_reward += 1.0
                            applied_traj_bias = True

                    # Action economy discount (prefer shorter paths to progress)
                    step_reward -= 0.05 * d
                    new_score = b_score + step_reward
                    next_grid = pred.grid if (pred and pred.grid is not None) else b_grid
                    new_path = path + [s_cand.spec]
                    next_beams.append((new_score, new_path, next_grid, sub_scene, p_kind, applied_traj_bias))

                    if new_score > best_score and len(new_path) >= 2:
                        best_score = new_score
                        best_path = new_path
                        best_family = p_kind
                        best_bias_beneficiary = applied_traj_bias

            if not next_beams:
                break
            next_beams.sort(key=lambda x: -x[0])
            beams = next_beams[:max_width]

        t_elapsed_ms = (time.monotonic() - t_start) * 1000.0
        meta.update({
            "deep_search_used": bool(best_path and best_score > 1.0),
            "deep_search_nodes_evaluated": nodes_evaluated,
            "deep_search_best_score": round(best_score, 3) if best_path else -999.0,
            "deep_search_time_ms": round(t_elapsed_ms, 2),
            "deep_search_selected_family": best_family,
            "post_breakthrough_bias_used": bool(best_path and best_score > 1.0 and best_bias_beneficiary and self.memory.post_breakthrough_window_active),
        })

        if best_path and best_score > 1.0:
            if best_bias_beneficiary and self.memory.post_breakthrough_window_active:
                self.memory.post_breakthrough_bias_used = True
            first_action = best_path[0]
            remaining = best_path[1:]
            return first_action, remaining, meta

        meta["deep_search_aborted_reason"] = "no_high_scoring_branch" if best_path else "search_exhausted"
        return None, [], meta

    def _verify(
        self,
        scene: Scene,
        spec: ActionSpec,
        legal_names: set[str],
        prediction: ProgramPrediction | None,
        is_probe: bool,
        profile: RuntimeProfile,
        is_severe_stagnation: bool = False,
        is_llm_proposal: bool = False,
    ) -> AlignmentDecision:
        if spec.name not in legal_names:
            return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("illegal_action",))
        data = spec.data_dict
        if ("x" in data) != ("y" in data):
            return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("incomplete_coordinates",))
        if "x" in data and not (0 <= data["x"] < scene.width and 0 <= data["y"] < scene.height):
            return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("coordinate_out_of_bounds",))
            
        action_stats = self.path_planner.dynamics.action_stats.get(spec.name, Counter())
        observed = action_stats.get("observed", 0)
        
        base_decision = None
        if profile.use_alignment:
            base_decision = self.alignment.verify(scene, spec, legal_names, prediction, is_probe)
            if not base_decision.allowed:
                return base_decision
                
        # Bounded Coordinate Exclusion & Fatigue (exempt palette rows / border selectors)
        if "x" in data and "y" in data:
            is_palette_or_border = (data["y"] <= 2 or data["x"] <= 2 or data["y"] >= scene.height - 3 or data["x"] >= scene.width - 3)
            if not is_palette_or_border:
                exact_visits = self.memory.spatial_visits_by_action.get((spec.name, data["x"], data["y"]), 0)
                if exact_visits >= 3:
                    return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("coordinate_fatigue",))
            
            visits = self.memory.spatial_visits.get((data["x"], data["y"]), 0)
            if visits > 0 and not is_palette_or_border:
                penalty = visits * 2.0
                if base_decision is not None:
                    base_decision = AlignmentDecision(base_decision.allowed, base_decision.score - penalty, base_decision.goal_delta, base_decision.risk, base_decision.goal_ids, base_decision.reasons + ("spatial_penalty",))
                else:
                    base_decision = AlignmentDecision(True, -penalty, 0.0, 1.0, (), ("spatial_penalty",))

        # Hard-gate persistent deterministic no-ops
        if observed >= 10 and not is_llm_proposal:
            noop_rate = action_stats.get("noop", 0) / observed
            if noop_rate >= 0.9 and not is_probe:
                return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("semantic_gating_persistent_noop",))
                
        # Semantic Hard-Gating
        if observed >= 5:
            is_movement_hypothesis = prediction is not None and prediction.kind in ("translation", "gravity", "drag_component")
            is_recolor_hypothesis = prediction is not None and prediction.kind in ("conditional_recolor", "component_recolor")
            
            if is_movement_hypothesis and action_stats.get("movement_like", 0) == 0:
                if is_severe_stagnation and is_llm_proposal:
                    if base_decision is not None:
                        base_decision = AlignmentDecision(base_decision.allowed, base_decision.score - 5.0, base_decision.goal_delta, base_decision.risk, base_decision.goal_ids, base_decision.reasons + ("semantic_override_penalty",))
                    else:
                        base_decision = AlignmentDecision(True, -5.0, 0.0, 1.0, (), ("semantic_override_penalty",))
                elif is_severe_stagnation:
                    if base_decision is not None:
                        base_decision = AlignmentDecision(base_decision.allowed, base_decision.score - 5.0, base_decision.goal_delta, base_decision.risk, base_decision.goal_ids, base_decision.reasons + ("semantic_gating_movement_penalty",))
                    else:
                        base_decision = AlignmentDecision(True, -5.0, 0.0, 1.0, (), ("semantic_gating_movement_penalty",))
                else:
                    return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("semantic_gating_not_movement_capable",))
                
            if is_recolor_hypothesis and action_stats.get("movement_like", 0) > observed * 0.5:
                if is_severe_stagnation and is_llm_proposal:
                    if base_decision is not None:
                        base_decision = AlignmentDecision(base_decision.allowed, base_decision.score - 5.0, base_decision.goal_delta, base_decision.risk, base_decision.goal_ids, base_decision.reasons + ("semantic_override_penalty",))
                    else:
                        base_decision = AlignmentDecision(True, -5.0, 0.0, 1.0, (), ("semantic_override_penalty",))
                else:
                    # Down-rank significantly
                    if base_decision is not None:
                        base_decision = AlignmentDecision(base_decision.allowed, base_decision.score - 5.0, base_decision.goal_delta, base_decision.risk, base_decision.goal_ids, base_decision.reasons + ("semantic_gating_movement_penalty",))
                    else:
                        if is_severe_stagnation:
                            base_decision = AlignmentDecision(True, -5.0, 0.0, 1.0, (), ("semantic_gating_movement_penalty",))
                        else:
                            return AlignmentDecision(False, -5.0, -1.0, 1.0, (), ("semantic_gating_movement_penalty",))
                    
            if action_stats.get("death", 0) >= max(3, observed * 0.4):
                if "x" in data and "y" in data:
                    visits = self.memory.spatial_visits.get((data["x"], data["y"]), 0)
                    if visits >= 2:
                        return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("semantic_gating_high_death_context",))

        if base_decision is not None:
            return base_decision
            
        return AlignmentDecision(True, 0.0, 0.0, 0.0, (), ("tier_alignment_bypass",))

    def _aligned_known_candidate(
        self,
        scene: Scene,
        candidate: Candidate,
        legal_names: set[str],
        profile: RuntimeProfile,
    ) -> tuple[float, AlignmentDecision, ProgramPrediction | None]:
        prediction = self._predict(scene, candidate.spec, profile)
        if prediction is None and not candidate.is_probe:
            decision = AlignmentDecision(
                True, 0.05, 0.0, 0.15,
                tuple(g.goal_id for g in self.goals.active(
                    4)) if profile.use_goals else (),
                ("evidence_backed_without_full_grid_prediction",),
            )
        else:
            decision = self._verify(
                scene, candidate.spec, legal_names, prediction, candidate.is_probe, profile)
        score = candidate.score + decision.score
        if prediction is not None:
            score += 0.8 * prediction.confidence - 0.6 * prediction.uncertainty

        # 1. Post-Breakthrough Priority: Transferred winning actions/regrounded coords outrank finish mode and fallback
        if self.memory.post_breakthrough_window_active or self.memory.current_level_index > 0:
            self.memory.post_breakthrough_priority_preserved = True
            self.memory.post_breakthrough_preempt_block_reason = "transferred_level_priors_active"
            p_kind = getattr(prediction, "kind", "") if prediction else ""
            if p_kind and (p_kind == self.memory.transferred_winning_program_kind or p_kind == self.memory.transferred_winning_family):
                score += 4.5
                self.memory.productive_branch_commitment_used = True
                self.memory.post_breakthrough_local_branch_reused = True
            if candidate.spec.name == self.memory.transferred_winning_action_name:
                score += 3.5
                self.memory.productive_branch_commitment_used = True
                self.memory.post_breakthrough_local_branch_reused = True
            if self.memory.regrounded_winning_coords and candidate.spec.data:
                c_dict = dict(candidate.spec.data)
                if "x" in c_dict and "y" in c_dict:
                    dist = max(abs(c_dict["x"] - self.memory.regrounded_winning_coords[0]), abs(c_dict["y"] - self.memory.regrounded_winning_coords[1]))
                    if dist <= 2:
                        score += 4.0
                        self.memory.productive_branch_commitment_used = True
                        self.memory.post_breakthrough_local_branch_reused = True

        # 2. Productive-Branch Commitment & Structured Persistence Dominance during Level 0 finish mode or active persistence
        elif (
            self.memory.current_level_index == 0
            and not self.memory.post_breakthrough_window_active
            and (self.memory.near_terminal_finish_mode_active or self.memory.structured_branch_persistence_used)
        ):
            target_family = self.memory.near_terminal_finish_family or self.memory.productive_branch_family_hint
            top_kinds = MECHANISM_TO_PROGRAM_KINDS.get(target_family, set())
            p_kind = getattr(prediction, "kind", "") if prediction else ""
            
            # Check bound branch matching (source, anchor neighborhood, action name)
            is_bound_branch_match = False
            bound_action = self.memory.finish_bound_branch_action_name or self.memory.productive_branch_action_name
            bound_anchor = self.memory.finish_bound_branch_anchor or self.memory.productive_branch_anchor
            
            if bound_action and candidate.spec.name == bound_action:
                if bound_anchor and candidate.spec.data:
                    c_dict = dict(candidate.spec.data)
                    if "x" in c_dict and "y" in c_dict:
                        bx, by = bound_anchor
                        if max(abs(c_dict["x"] - bx), abs(c_dict["y"] - by)) <= 2:
                            is_bound_branch_match = True
                elif not candidate.spec.data:
                    is_bound_branch_match = True

            if p_kind and (p_kind == target_family or p_kind in top_kinds):
                score += 3.5
                self.memory.productive_branch_commitment_used = True
                self.memory.finish_branch_continuation_family = target_family
                self.memory.finish_branch_continuation_kept_control = True
                self.memory.structured_branch_persistence_kept_control = True
                if is_bound_branch_match:
                    score += 1.0
                    self.memory.finish_bound_branch_kept_control = True
            elif is_bound_branch_match:
                score += 3.5
                self.memory.productive_branch_commitment_used = True
                self.memory.finish_bound_branch_kept_control = True
                self.memory.finish_branch_continuation_kept_control = True
                self.memory.structured_branch_persistence_kept_control = True
            elif candidate.spec.source in (self.memory.near_terminal_finish_source, self.memory.productive_branch_source):
                score += 3.0
                self.memory.productive_branch_commitment_used = True
                self.memory.finish_branch_continuation_kept_control = True
                self.memory.structured_branch_persistence_kept_control = True

        return score, decision, prediction

    def choose(
        self,
        scene: Scene,
        legal_actions: Sequence[GameAction],
        step: int,
        response_frames: Sequence[np.ndarray] = (),
        runtime_tier: str = "A9",
        remaining_sec: float | None = None,
    ) -> tuple[ActionSpec, bool, dict[str, Any]]:
        profile = _runtime_profile(runtime_tier, self.config)
        legal_names = {action.name for action in legal_actions}
        self.last_response_frames = tuple(response_frames)
        self.reasoner_suppressed = False
        self.reasoner_suppression_reason = ""
        self.memory.post_breakthrough_bias_used = False
        self.memory.post_breakthrough_continuation_bias = 0.0
        self.memory.post_breakthrough_local_search_used = False
        self.memory.counterfactual_streak_renewed = False
        self.memory.mechanism_collapse_breaker_used = False
        self.memory.mechanism_family_penalized = ""
        self.memory.mechanism_family_promoted = ""
        self.memory.component_delete_component_locked = False
        self.memory.line_beam_structured_candidates_used = False
        self.memory.counterfactual_completion_bias = 0.0
        self.memory.terminal_condition_bonus = 0.0
        self.memory.mechanism_completion_bonus = 0.0
        self.memory.productive_search_convergence_pressure = 0.0
        self.memory.line_beam_closure_bias_used = False
        self.memory.component_delete_payoff_bias_used = False
        self.memory.atomic_drag_drop_paired = False
        self.memory.sequential_component_sweep_active = False
        self.memory.orthogonal_collision_steer_used = False
        self.memory.cell_cycle_persistence_used = False
        self.memory.productive_branch_commitment_used = False
        self.memory.near_terminal_finish_gate_family_consensus = False
        self.memory.near_terminal_finish_gate_branch_consensus = False
        self.memory.near_terminal_finish_gate_completion_stability = False
        self.memory.near_terminal_finish_gate_allowed = False
        self.memory.finish_branch_continuation_kept_control = False
        self.memory.finish_branch_continuation_break_reason = ""
        self.memory.finish_mode_preempted_counterfactual = False
        self.memory.finish_mode_preempt_block_reason = "" 
        self.llm_step_meta = {
            "reasoner_consulted_this_step": False,
            "reasoner_parsed_abduction_this_step": False,
            "reasoner_surviving_proposals_this_step": 0,
            "stagnation_override_used": False,
        }
        
        if not profile.use_programs:
            self.plan_queue.clear()

        # 0) Stagnation breakout: issue RESET if trapped in a 16+ NO-OP loop
        if self.memory.no_op_streak >= self.config.no_progress_exhaustion_threshold and "RESET" in legal_names and step + 30 <= self.config.max_actions:
            self.memory.no_op_streak = 0
            self.plan_queue.clear()
            return ActionSpec(name="RESET", source="stagnation_breakout_reset"), False, {"stage": "stagnation_breakout_reset"}

        # 0.5) Fallback loop escape
        # Trace evidence (e.g., cd82, bp35) showed the agent churning in fallback without progress.
        if self.consecutive_fallback_steps >= 50 and legal_actions:
            self.consecutive_fallback_steps = 0
            first = legal_actions[0]
            data_fallback: tuple[tuple[str, int], ...] = ()
            if first.is_complex():
                data_fallback = (("x", scene.width // 2), ("y", scene.height // 2))
            return ActionSpec(name=first.name, data=data_fallback, source="fallback_loop_escape"), True, {"stage": "fallback_loop_escape"}

        # 1) Exact observed progress and replay graph are available in every tier.
        replay = (
            self.memory.find_replay_progress(scene.exact_key)
            or self.world_model.known_action(scene.exact_key)
            or self.world_model.plan_to_progress(scene.exact_key)
        )
        if replay is not None and replay.name in legal_names:
            prediction = self._predict(scene, replay, profile)
            decision = self._verify(
                scene, replay, legal_names, prediction, False, profile)
            if decision.allowed or prediction is None:
                return replay, False, {
                    "stage": "verified_progress_replay_or_graph_plan",
                    "alignment": round(decision.score, 3),
                    "goals": list(decision.goal_ids),
                    "profile": profile.tier,
                }

        # 2) Only A9 can continue a program-verified queue.
        while profile.use_programs and self.plan_queue:
            queued = self.plan_queue.popleft()
            if queued.name not in legal_names:
                continue
            signature = _action_signature(scene, queued)
            if self.dead.is_dead(signature):
                self.plan_queue.clear()
                continue
            data = queued.data_dict
            if "x" in data and "y" in data:
                if self.memory.spatial_visits.get((data["x"], data["y"]), 0) >= 3:
                    self.plan_queue.clear()
                    continue
            prediction = self._predict(scene, queued, profile)
            decision = self._verify(
                scene, queued, legal_names, prediction, prediction is None, profile)
            # Permissive Queue Continuation for branch-consistent actions
            is_branch_queued = (
                self.memory.productive_branch_signature != ""
                and queued.name == self.memory.productive_branch_action_name
                and not self.dead.is_dead(signature)
            )
            if decision.allowed or (is_branch_queued and not any(r in ("illegal_action", "coordinate_out_of_bounds") for r in decision.reasons)):
                pred_grid = prediction.grid if prediction and prediction.grid is not None else None
                spec = ActionSpec(
                    name=queued.name, data=queued.data, source="verified_plan_queue",
                    predicted_effect=prediction.expected_effect if prediction else None, score=queued.score,
                    program_id=prediction.program_id if prediction else None,
                    predicted_state_key=_stable_hash_bytes(pred_grid.tobytes()) if pred_grid is not None else None,
                    goal_ids=decision.goal_ids,
                )
                return spec, False, {"stage": "verified_plan_queue", "alignment": round(decision.score, 3)}
            self.plan_queue.clear()

        # Scored mechanism belief update & mode setting
        self._update_mechanism_beliefs(step=step)
        self._update_post_breakthrough_window(step=step)

        # 0a) Component Re-Grounding on Level Transition
        if self.memory.post_breakthrough_window_active and self.memory.level_steps <= 1 and self.memory.regrounded_winning_coords is None:
            regrounded, r_delta, r_conf, r_reason = self._reground_winning_target(scene)
            self.memory.regrounded_winning_coords = regrounded
            self.memory.regrounding_delta = r_delta
            self.memory.regrounding_confidence = r_conf
            self.memory.regrounding_used = bool(regrounded is not None)
            self.memory.regrounding_failed_reason = r_reason
            if regrounded is not None:
                self.memory.transferred_winning_coords = regrounded

            # 0b) Bounded LLM Relocalization if component re-grounding is ambiguous
            if (regrounded is None or r_conf < 0.60) and profile.use_model and remaining_sec is not None and remaining_sec > 90.0 and not _OLLAMA_LOCK.locked():
                self.memory.levelup_relocalization_llm_attempted = True
                win_pattern = {
                    "action": self.memory.transferred_winning_action_name,
                    "coords": self.memory.last_winning_coords,
                    "family": self.memory.transferred_winning_family,
                }
                reloc = self.reasoner.propose_relocalization(scene, win_pattern)
                if reloc is None:
                    self.memory.regrounding_failed_reason = "llm_relocalization_no_response"
                    if self.memory.post_breakthrough_window_active and regrounded is None:
                        self.memory.post_breakthrough_aborted_reason = "continuation_relocalization_failed"
                elif reloc.get("confidence", 0.0) < 0.50:
                    self.memory.regrounding_failed_reason = "llm_relocalization_low_confidence"
                    self.memory.levelup_relocalization_llm_confidence = float(reloc.get("confidence", 0.0))
                    if self.memory.post_breakthrough_window_active and regrounded is None:
                        self.memory.post_breakthrough_aborted_reason = "continuation_relocalization_failed"
                else:
                    self.memory.regrounded_winning_coords = (reloc["x"], reloc["y"])
                    self.memory.transferred_winning_coords = (reloc["x"], reloc["y"])
                    self.memory.levelup_relocalization_llm_used = True
                    self.memory.levelup_relocalization_llm_confidence = float(reloc["confidence"])
                    self.memory.regrounding_used = True
                    self.memory.regrounding_failed_reason = "llm_relocalization_success" 

        # 2a-0) Instant Reflex Fast-Path (Sub-millisecond execution for forced/unambiguous winning moves)
        active_goals = self.goals.active(limit=2)
        if active_goals and active_goals[0].confidence > 0.85:
            top_goal = active_goals[0]
            for action in legal_actions:
                data_tuple = tuple(sorted((k, int(v)) for k, v in action.data.items())) if getattr(action, "data", None) else ()
                spec_cand = ActionSpec(name=action.name, data=data_tuple)
                pred = self._predict(scene, spec_cand, profile)
                if pred is not None:
                    gate = self._verify(scene, spec_cand, legal_names, pred, False, profile)
                    if gate.allowed and gate.score > 1.2 and not self.dead.is_dead(_action_signature(scene, spec_cand)):
                        spec = ActionSpec(
                            name=spec_cand.name, data=spec_cand.data, source="instant_reflex",
                            predicted_effect=pred.expected_effect, score=spec_cand.score + gate.score + 2.0,
                            program_id=pred.program_id,
                            predicted_state_key=_stable_hash_bytes(pred.grid.tobytes()),
                            goal_ids=gate.goal_ids,
                        )
                        return spec, False, {
                            "stage": "instant_reflex", "final_action_source": "instant_reflex",
                            "instant_reflex_used": True, "alignment": round(gate.score, 3), **self.llm_step_meta
                        }

        # 2b-1) Bounded Goal-Directed Macro Replay for Productive Programs
        while self.memory.macro_replay_queue:
            queued_spec = self.memory.macro_replay_queue.popleft()
            if queued_spec.name not in legal_names:
                self.memory.macro_replay_queue.clear()
                self.memory.macro_replay_aborted_reason = "illegal_action"
                break
            prediction = self._predict(scene, queued_spec, profile)
            decision = self._verify(scene, queued_spec, legal_names, prediction, False, profile)
            signature = _action_signature(scene, queued_spec)
            if not decision.allowed or self.dead.is_dead(signature):
                self.memory.macro_replay_queue.clear()
                self.memory.macro_replay_aborted_reason = "safety_or_dead_signature"
                break
            
            spec = ActionSpec(
                name=queued_spec.name,
                data=queued_spec.data,
                source="macro_replay",
                predicted_effect=queued_spec.predicted_effect if prediction is None else prediction.expected_effect,
                score=queued_spec.score + decision.score,
                program_id=self.memory.macro_replay_program_id or (prediction.program_id if prediction else None),
                predicted_state_key=None if prediction is None else _stable_hash_bytes(prediction.grid.tobytes()),
                goal_ids=decision.goal_ids,
            )
            applied_macro_bias = False
            if self.memory.post_breakthrough_window_active:
                if queued_spec.name == self.memory.transferred_winning_action_name or self.memory.macro_replay_family == self.memory.transferred_winning_family:
                    self.memory.post_breakthrough_bias_used = True
                    applied_macro_bias = True
            return spec, False, {
                "stage": "macro_replay",
                "final_action_source": "macro_replay",
                "macro_replay_active": True,
                "macro_replay_family": self.memory.macro_replay_family,
                "macro_replay_program_id": self.memory.macro_replay_program_id,
                "macro_replay_steps_remaining": len(self.memory.macro_replay_queue),
                "post_breakthrough_bias_used": applied_macro_bias,
                **self.llm_step_meta
            }

        candidates = self.candidate_generator.generate(
            scene, legal_actions, use_hypotheses=profile.use_hypotheses)
        action_semantics_summary = self.path_planner.dynamics.summary_for_prompt(legal_names)

        # 2b-2) Post-Breakout Follow-Through Mode (Exploit LLM insight window)
        if self.memory.llm_follow_through_window > 0:
            ft_family = self.memory.llm_follow_through_family
            ft_coords = self.memory.llm_follow_through_coords
            ft_programs = self.memory.llm_promoted_program_ids
            ft_candidates = []
            for candidate in candidates:
                prediction = self._predict(scene, candidate.spec, profile)
                decision = self._verify(scene, candidate.spec, legal_names, prediction, False, profile)
                if not decision.allowed or self.dead.is_dead(candidate.signature):
                    continue
                
                # Counterfactual Sandbox: mental 1-step validation
                if prediction is not None and prediction.grid is not None:
                    pred_sig = _stable_hash_bytes(prediction.grid.tobytes())
                    if self.dead.is_dead(pred_sig):
                        self.memory.counterfactual_pruned_count += 1
                        continue

                ft_score = candidate.score + decision.score
                
                # Active Family Payoff & Early Mechanism Conditioning Boost
                if prediction is not None:
                    p_kind = getattr(prediction, 'kind', '')
                    if p_kind == self.memory.early_classified_mechanism:
                        ft_score += 0.8
                    if p_kind in self.memory.llm_family_payoff:
                        p_score = self.memory.llm_family_payoff[p_kind].get("changed", 0) + 3 * self.memory.llm_family_payoff[p_kind].get("progress", 0) - self.memory.llm_family_payoff[p_kind].get("noop", 0)
                        ft_score += max(-1.0, min(1.5, 0.1 * p_score))
                if prediction is not None and prediction.kind == ft_family:
                    ft_score += 0.8
                if prediction is not None and prediction.program_id in ft_programs:
                    ft_score += 1.0
                applied_ft_transfer_bias = False
                if prediction is not None:
                    ft_score += self._priority_family_boost(prediction.kind)
                if self.memory.post_breakthrough_window_active:
                    if prediction and (prediction.kind == self.memory.transferred_winning_program_kind or prediction.kind == self.memory.transferred_winning_family):
                        ft_score += 1.2
                        applied_ft_transfer_bias = True
                    if candidate.spec.name == self.memory.transferred_winning_action_name:
                        ft_score += 0.8
                        applied_ft_transfer_bias = True
                if ft_coords and candidate.spec.data:
                    cx, cy = candidate.spec.data_dict.get("x"), candidate.spec.data_dict.get("y")
                    if cx is not None and cy is not None:
                        dist = max(abs(cx - ft_coords[0]), abs(cy - ft_coords[1]))
                        if dist <= 2:
                            ft_score += 0.6
                        elif cx == ft_coords[0] or cy == ft_coords[1]:
                            ft_score += 0.3
                ft_candidates.append((ft_score, candidate, decision, prediction, applied_ft_transfer_bias))
            if ft_candidates:
                ft_candidates.sort(key=lambda row: row[0], reverse=True)
                score, best, decision, prediction, applied_ft_transfer_bias = ft_candidates[0]
                if applied_ft_transfer_bias:
                    self.memory.post_breakthrough_bias_used = True
                    self.memory.post_breakthrough_continuation_bias = 1.2
                    if ft_coords and best.spec.data:
                        cx, cy = best.spec.data_dict.get("x"), best.spec.data_dict.get("y")
                        if cx is not None and cy is not None and max(abs(cx - ft_coords[0]), abs(cy - ft_coords[1])) <= 2:
                            self.memory.post_breakthrough_local_search_used = True
                spec = ActionSpec(
                    name=best.spec.name, data=best.spec.data, source="llm_follow_through",
                    predicted_effect=best.spec.predicted_effect if prediction is None else prediction.expected_effect,
                    score=score,
                    program_id=None if prediction is None else prediction.program_id,
                    goal_ids=decision.goal_ids,
                )
                return spec, False, {
                    "stage": "llm_follow_through",
                    "final_action_source": "llm_follow_through",
                    "follow_through_family": ft_family,
                    "follow_through_window": self.memory.llm_follow_through_window,
                    "score": round(score, 3),
                    "post_breakthrough_bias_used": applied_ft_transfer_bias,
                    **self.llm_step_meta
                }

        # 2c) Evidence-Guided Novelty Explorer
        is_persisting = (self.memory.structured_branch_persistence_used and self.memory.structured_branch_persistence_steps_remaining > 0)
        is_live_branch = (
            self.memory.productive_branch_signature != ""
            and self.memory.productive_branch_streak >= 2
            and (step - self.memory.productive_branch_last_effective_step) <= 3
            and (
                self.memory.counterfactual_completion_bias > 0.0
                or self.memory.productive_search_convergence_pressure >= 1.0
                or (self.memory.productive_branch_anchor is not None and self.memory.productive_branch_anchor[1] > 2 and self.memory.productive_branch_anchor[0] > 2)
            )
        )
        repeated_effect = self.memory.repeated_effect_signature()
        if (
            self.steps_since_progress >= max(18, self.config.no_progress_exhaustion_threshold)
            or self.memory.recent_death_count(10) >= 2
            or (repeated_effect is not None and repeated_effect[1] >= 4)
        ):
            exploratory: list[tuple[float, Candidate, AlignmentDecision]] = []
            repeated_sig = None if repeated_effect is None else repeated_effect[0]
            recent_actions = {t.action.name for t in list(self.memory.transitions)[-6:]}
            for candidate in candidates:
                if candidate.spec.data:
                    c_dict = dict(candidate.spec.data)
                    cx, cy = c_dict.get("x"), c_dict.get("y")
                    if cx is not None and cy is not None:
                        if (self.memory.mode == "exploitation_mode" or self.memory.post_breakthrough_window_active):
                            self.memory.exploitation_noop_blacklist_checks += 1
                        if (candidate.spec.name, cx, cy) in self.memory.exploitation_noop_blacklist:
                            self.memory.exploitation_noop_blacklist_hits += 1
                            continue
                        if (self.memory.mode == "exploitation_mode" or self.memory.post_breakthrough_window_active) and (cx, cy) in self.memory.exploitation_noop_neighborhood_blacklist:
                            self.memory.exploitation_noop_blacklist_hits += 1
                            continue
                prediction = self._predict(scene, candidate.spec, profile)
                decision = self._verify(
                    scene, candidate.spec, legal_names, prediction, True, profile)
                if not decision.allowed:
                    continue
                
                evidence_score = 0.0
                action_stats = self.path_planner.dynamics.action_stats.get(candidate.spec.name, Counter())
                observed = max(1, action_stats.get("observed", 0))
                
                # Base exploration value for less-used actions
                evidence_score += 2.0 / observed
                
                # Reward historical topology/movement behavior
                if action_stats.get("topology", 0) > 0:
                    evidence_score += 1.5
                if action_stats.get("movement_like", 0) > 0:
                    evidence_score += 1.0
                    
                # Heavy penalty for repeating recent patterns
                if candidate.spec.name in recent_actions:
                    evidence_score -= 1.5
                    
                # Penalize exact predicted effect repetition or null effects
                if prediction is not None:
                    if prediction.expected_effect is None:
                        if observed >= 5:
                            evidence_score -= 3.0
                    else:
                        eff_count = self.path_planner.dynamics.effect_signatures[candidate.spec.name].get(prediction.expected_effect, 0)
                        evidence_score -= (eff_count * 0.5)
                        if repeated_sig and prediction.expected_effect == repeated_sig:
                            evidence_score -= 3.0
                else:
                    if observed >= 5:
                        evidence_score -= 3.0
                        
                # Reward predicted structural changes
                if prediction is not None:
                    if prediction.grid.shape != scene.grid.shape:
                        evidence_score += 2.0
                
                # Penalize repeated local patch hash usage
                visits_penalty = 0.0
                if candidate.spec.data:
                    data_dict = dict(candidate.spec.data)
                    x, y = data_dict.get("x"), data_dict.get("y")
                    if x is not None and y is not None:
                        visits_penalty = float(self.memory.spatial_visits.get((x, y), 0))
                
                evidence_score -= 0.5 * visits_penalty

                if self.dead.is_dead(candidate.signature):
                    evidence_score -= 2.0
                    
                exploratory.append((evidence_score + decision.score, candidate, decision))
            if exploratory:
                # Anti-churn guard: during active continuation window or structured persistence, skip stuck-mode
                if self.memory.post_breakthrough_window_active:
                    self.post_breakthrough_stuck_mode_uses += 1
                    if self.post_breakthrough_stuck_mode_uses >= 2:
                        self.memory.post_breakthrough_window_active = False
                        self.memory.post_breakthrough_aborted_reason = "continuation_fallback_forced"
                        self.post_breakthrough_stuck_mode_uses = 0
                is_persisting = (self.memory.structured_branch_persistence_used and self.memory.structured_branch_persistence_steps_remaining > 0)
                is_live_branch = (
                    self.memory.productive_branch_signature != ""
                    and self.memory.productive_branch_streak >= 2
                    and (step - self.memory.productive_branch_last_effective_step) <= 3
                )
                if self.memory.post_breakthrough_window_active:
                    self.memory.productive_branch_preempt_blocked = True
                    self.memory.productive_branch_preempt_block_reason = "post_breakthrough_continuation_active"
                elif is_persisting:
                    self.memory.productive_branch_preempt_blocked = True
                    self.memory.productive_branch_preempt_block_reason = "structured_persistence_active"
                elif is_live_branch:
                    self.memory.productive_branch_preempt_blocked = True
                    self.memory.productive_branch_preempt_block_reason = "live_productive_branch_active"

                # Inline guard: Never allow stuck-mode exploration to steal control when post-breakthrough, structured persistence, or live branch is active
                if not self.memory.post_breakthrough_window_active and not is_persisting and not is_live_branch:
                    exploratory.sort(key=lambda row: row[0], reverse=True)
                    score, best, decision = exploratory[0]
                    self.stuck_mode_activations += 1
                    spec = ActionSpec(
                        name=best.spec.name, data=best.spec.data, source="stuck_mode_exploration",
                        predicted_effect=best.spec.predicted_effect, score=best.spec.score + score,
                        goal_ids=decision.goal_ids,
                    )
                    return spec, True, {"stage": "stuck_mode_exploration", "final_action_source": "stuck_mode_exploration", "score": round(score, 3)}
            
        # 2b) Forced Structural Probe (Boredom Override)
        if self.memory.same_family_streak >= 10 and not self.memory.post_breakthrough_window_active and not is_persisting and not is_live_branch:
            dominant_family = self.memory.transitions[-1].action.name if self.memory.transitions else ""
            probes = []
            for candidate in candidates:
                if candidate.spec.name == dominant_family:
                    continue
                prediction = self._predict(scene, candidate.spec, profile)
                decision = self._verify(scene, candidate.spec, legal_names, prediction, True, profile)
                if not decision.allowed:
                    continue
                rarity = -self.memory.global_action_outcomes[candidate.spec.name].total()
                # Prioritize spatial regions with low visit counts
                spatial_penalty = 0
                if candidate.spec.data:
                    data_dict = dict(candidate.spec.data)
                    x, y = data_dict.get("x"), data_dict.get("y")
                    if x is not None and y is not None:
                        # True spatial rarity: penalize heavily visited locations.
                        visits = self.memory.spatial_visits.get((x, y), 0)
                        if visits == 0:
                            rarity += 5.0  # Massive boost for completely unvisited coordinates
                        else:
                            rarity -= visits
                        if candidate.spec.name == "ACTION6":
                            rarity += 1.0 # Boost ACTION6 if it's not the dominant family
                probes.append((rarity, candidate, decision))
            
            if probes:
                probes.sort(key=lambda row: row[0], reverse=True)
                score, best, decision = probes[0]
                spec = ActionSpec(
                    name=best.spec.name, data=best.spec.data, source="forced_structural_probe",
                    predicted_effect=best.spec.predicted_effect, score=score, goal_ids=decision.goal_ids
                )
                self.memory.same_family_streak = 0
                return spec, True, {"stage": "forced_structural_probe", "final_action_source": "forced_structural_probe", "score": round(score, 3)}

        # 3) A9-only bounded counterfactual planning with anti-hijack grounding.
        is_looping = self.memory.recent_state_loop(self.config.loop_window)
        cf_plan = self.counterfactual.plan(
            scene, candidates, legal_names) if profile.use_counterfactual else None
        if cf_plan is not None:
            if cf_plan.prediction is not None and cf_plan.first_action is not None and (self.counterfactual_streak < 15 or profile.use_programs):
                self.counterfactual_streak += 1
                self.plan_queue.extend(cf_plan.remaining)
                applied_cf_bias = False
                if self.memory.post_breakthrough_window_active:
                    p_kind = getattr(cf_plan.prediction, "kind", "")
                    if p_kind and (p_kind == self.memory.transferred_winning_program_kind or p_kind == self.memory.transferred_winning_family):
                        self.memory.post_breakthrough_bias_used = True
                        applied_cf_bias = True
                    elif cf_plan.first_action.name == self.memory.transferred_winning_action_name:
                        self.memory.post_breakthrough_bias_used = True
                        applied_cf_bias = True
                if applied_cf_bias:
                    self.memory.post_breakthrough_continuation_bias = 2.0
                
                # Productive search convergence pressure on Level 0
                if self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
                    if self.steps_since_progress >= 20 and len(self.memory.transitions) >= 15:
                        self.memory.productive_search_convergence_pressure = min(2.5, (self.steps_since_progress - 15) * 0.1)
                
                # Assign completion and terminal condition telemetry
                if cf_plan.prediction and cf_plan.prediction.grid is not None:
                    c_b, t_b, cl_u, py_u = _evaluate_state_completion_and_terminal(
                        cf_plan.prediction.grid, scene.grid, scene, self.memory.top_mechanism_family, cf_plan.first_action
                    )
                    self.memory.counterfactual_completion_bias = c_b
                    self.memory.terminal_condition_bonus = t_b
                    self.memory.mechanism_completion_bonus = c_b
                    if cl_u:
                        self.memory.line_beam_closure_bias_used = True
                    if py_u:
                        self.memory.component_delete_payoff_bias_used = True

                stage_name = "counterfactual_program_search" if profile.use_programs and not self.dead.is_dead("counterfactual_program_search") and self.counterfactual_streak < 15 else "counterfactual_fallback"
                return cf_plan.first_action, False, {
                    "stage": stage_name,
                    "final_action_source": "counterfactual_program" if "program" in stage_name else "counterfactual_search",
                    "program": cf_plan.prediction.program_id,
                    "program_kind": cf_plan.prediction.kind,
                    "alignment": round(cf_plan.alignment.score, 3),
                    "goal_delta": round(cf_plan.alignment.goal_delta, 3),
                    "goals": list(cf_plan.alignment.goal_ids),
                    "plan_length": 1 + len(cf_plan.remaining),
                    "post_breakthrough_bias_used": applied_cf_bias,
                    "counterfactual_completion_bias": self.memory.counterfactual_completion_bias,
                    "terminal_condition_bonus": self.memory.terminal_condition_bonus,
                }
        else:
            self.counterfactual_streak = 0

        # 4) Deterministic navigation for A7/A8/A9 profiles.
        path_res = self.path_planner.next_action(
            scene, legal_names) if profile.use_path_planner else None
        path_action = path_res.action if hasattr(
            path_res, "action") else path_res
        if path_action is not None:
            signature = _action_signature(scene, path_action)
            prediction = self._predict(scene, path_action, profile)
            decision = self._verify(
                scene, path_action, legal_names, prediction, prediction is None, profile)
            if not self.dead.is_dead(signature) and decision.allowed:
                spec = ActionSpec(
                    name=path_action.name, data=path_action.data, source=path_action.source,
                    predicted_effect=path_action.predicted_effect if prediction is None else prediction.expected_effect,
                    score=path_action.score + decision.score,
                    program_id=None if prediction is None else prediction.program_id,
                    predicted_state_key=None if prediction is None else _stable_hash_bytes(
                        prediction.grid.tobytes()),
                    goal_ids=decision.goal_ids,
                )
                self.counterfactual_streak += 1
                return spec, prediction is None, {"stage": "path_planner", "final_action_source": "path_planner", "alignment": round(decision.score, 3)}
            else:
                self.counterfactual_streak = 0

        # 5) Replay/hash/hypothesis arbitration remains the A5 safety core.
        ranked: list[tuple[float, Candidate,
                           AlignmentDecision, ProgramPrediction | None, bool, float, bool]] = []
        for candidate in candidates:
            score, decision, prediction = self._aligned_known_candidate(
                scene, candidate, legal_names, profile)
            if decision.allowed and not candidate.is_probe:
                applied_bias = False
                cand_continuation_bias = 0.0
                is_local_search = False
                if prediction is not None:
                    score += self._priority_family_boost(prediction.kind)
                    if prediction.kind == self.memory.top_mechanism_family:
                        score += 0.8
                    if self.memory.post_breakthrough_window_active or self.memory.current_level_index > 0:
                        reground_coords = self.memory.regrounded_winning_coords or self.memory.transferred_winning_coords
                        if prediction.kind == self.memory.transferred_winning_program_kind or prediction.kind == self.memory.transferred_winning_family:
                            score += 2.0
                            applied_bias = True
                            cand_continuation_bias += 2.0
                        if candidate.spec.name == self.memory.transferred_winning_action_name:
                            score += 1.5
                            applied_bias = True
                            cand_continuation_bias += 1.5
                        if reground_coords and candidate.spec.data:
                            cd = dict(candidate.spec.data)
                            if "x" in cd and "y" in cd:
                                dist = max(abs(cd["x"] - reground_coords[0]), abs(cd["y"] - reground_coords[1]))
                                if dist <= 2:
                                    score += 1.8
                                    applied_bias = True
                                    cand_continuation_bias += 1.8
                                    is_local_search = True
                ranked.append((score, candidate, decision, prediction, applied_bias, cand_continuation_bias, is_local_search))
        if ranked:
            ranked.sort(key=lambda row: row[0], reverse=True)
            score, best, decision, prediction, applied_bias, best_cont_bias, best_local_used = ranked[0]
            if applied_bias:
                self.memory.post_breakthrough_bias_used = True
                self.memory.post_breakthrough_continuation_bias = best_cont_bias
            if best_local_used:
                self.memory.post_breakthrough_local_search_used = True
            spec = ActionSpec(
                name=best.spec.name, data=best.spec.data, source=best.spec.source,
                predicted_effect=best.spec.predicted_effect if prediction is None else prediction.expected_effect,
                score=score,
                program_id=None if prediction is None else prediction.program_id,
                predicted_state_key=None if prediction is None else _stable_hash_bytes(
                    prediction.grid.tobytes()),
                goal_ids=decision.goal_ids,
            )
            return spec, False, {
                "stage": "mental_replay_hash_goal_gate" if profile.use_alignment else "mental_replay_hash",
                "final_action_source": "verified_program" if spec.program_id else "mental_replay",
                "score": round(score, 3),
                "alignment": round(decision.score, 3),
                "goal_delta": round(decision.goal_delta, 3),
                "rationale": best.rationale[:4],
                "post_breakthrough_bias_used": applied_bias,
            }

        # 5b) Promising-State Bounded Deep Search (Multi-Step Mental Rollout)
        is_promising, promising_reasons = self._is_promising_state(scene, step, profile)
        if is_promising:
            ds_first, ds_remaining, ds_meta = self._promising_state_deep_search(
                scene, candidates, legal_actions, legal_names, profile, remaining_sec=remaining_sec, promising_reasons=promising_reasons
            )
            self.llm_step_meta.update(ds_meta)
            if ds_first is not None:
                self.plan_queue.extend(ds_remaining)
                spec = ActionSpec(
                    name=ds_first.name, data=ds_first.data, source="promising_state_deep_search",
                    predicted_effect=ds_first.predicted_effect, score=ds_meta.get("deep_search_best_score", 0.0),
                    program_id=ds_first.program_id,
                    goal_ids=ds_first.goal_ids,
                )
                return spec, False, {
                    "stage": "promising_state_deep_search",
                    "final_action_source": "promising_state_deep_search",
                    "plan_length": 1 + len(ds_remaining),
                    **self.llm_step_meta
                }
        else:
            self.llm_step_meta.update({
                "promising_state_detected": False,
                "promising_state_reasons": [],
                "deep_search_used": False,
                "deep_search_depth": 0,
                "deep_search_width": 0,
                "deep_search_nodes_evaluated": 0,
                "deep_search_best_score": -999.0,
                "deep_search_time_ms": 0.0,
                "deep_search_selected_family": "",
                "deep_search_aborted_reason": "not_promising",
            })

        # 6) Milestone-gated local model for A7/A8/A9 only with safe stagnation override.
        milestone, milestone_name = self._milestone(scene, step)
        is_stagnant_now = is_looping or self.memory.no_op_streak >= self.config.no_progress_consecutive_threshold
        
        stagnation_override_active = False
        long_stagnation = (self.steps_since_progress >= 40 or self.memory.no_op_streak >= 12)
        if (
            not (milestone or is_stagnant_now)
            and long_stagnation
            and self.stagnation_override_count_this_level < 3
            and (remaining_sec is None or remaining_sec > 120)
            and self.reasoner.budget_available
            and not _OLLAMA_LOCK.locked()
        ):
            is_stagnant_now = True
            stagnation_override_active = True
            self.stagnation_override_count_this_level += 1
            milestone_name = "stagnation_override"
        
        if profile.use_model and (milestone or is_stagnant_now):
            self.reasoner.reasoner_decision_attempts += 1
            skip_reason = None
            
            # Cooldown and milestone checks
            cooldown = max(2, self.config.model_cooldown_steps // 2) if is_stagnant_now else self.config.model_cooldown_steps
            has_milestone = (milestone or is_stagnant_now)
            cooldown_satisfied = (step - self.reasoner.last_call_step >= cooldown)
            
            if not self.reasoner.model_ready:
                skip_reason = "skipped_due_to_model_unavailable"
            elif self.reasoner_lock_backoff_steps > 0:
                self.reasoner_lock_backoff_steps -= 1
                skip_reason = "skipped_due_to_lock_backoff"
                self.reasoner_suppressed = True
                self.reasoner_suppression_reason = "lock_backoff"
            elif self.reasoner.reasoner_consultations_this_level >= 10 or self.failed_consultations_this_level >= 4:
                skip_reason = "skipped_due_to_llm_backoff"
                self.reasoner_suppressed = True
                self.reasoner_suppression_reason = "failed_consultation_backoff" if self.failed_consultations_this_level >= 4 else "budget_backoff"
            elif not self.reasoner.budget_available:
                skip_reason = "skipped_due_to_budget_exhaustion"
                self.reasoner_suppressed = True
                self.reasoner_suppression_reason = "budget_backoff"
            elif self.reasoner.calls_this_level == 0 and not is_stagnant_now:
                # Force stagger for the first call unless severely stagnant
                game_id = getattr(self.reasoner, "game_id", "unknown")
                warmup = self.config.llm_level_warmup_steps + (hash(game_id) % 5)
                if step < warmup:
                    skip_reason = "skipped_due_to_warmup_gate"
                    self.reasoner_suppressed = True
                    self.reasoner_suppression_reason = "warmup_stagger"
            elif not (has_milestone and cooldown_satisfied):
                skip_reason = "skipped_due_to_cooldown_gate"
                self.reasoner_suppressed = True
                self.reasoner_suppression_reason = "no_payoff_backoff"
            elif remaining_sec is not None:
                est_latency = max(10.0, self.reasoner.last_latency_sec)
                projected_time = est_latency * 3.0
                if remaining_sec < self.config.min_time_for_llm_sec:
                    skip_reason = "skipped_due_to_low_remaining_time"
                elif remaining_sec < projected_time:
                    skip_reason = "skipped_due_to_projected_latency"
            
            # Lock availability check
            # Trace evidence showed lock-starvation in bp35. Introduce explicit lock backoff.
            if not skip_reason and _OLLAMA_LOCK.locked():
                self.recent_lock_skip_count += 1
                if self.recent_lock_skip_count >= 3:
                    if self.memory.no_op_streak >= 40 and remaining_sec is not None and remaining_sec > 1800:
                        self.reasoner_lock_backoff_steps = 5
                    else:
                        self.reasoner_lock_backoff_steps = 10
                    skip_reason = "skipped_due_to_lock_backoff"
                    self.reasoner_suppressed = True
                    self.reasoner_suppression_reason = "lock_backoff"
                else:
                    skip_reason = "skipped_due_to_lock_unavailable"
                
            if skip_reason:
                if skip_reason not in self.reasoner.reasoner_decision_skips_by_reason:
                    self.reasoner.reasoner_decision_skips_by_reason[skip_reason] = 0
                self.reasoner.reasoner_decision_skips_by_reason[skip_reason] += 1
            else:
                current_feedback = ""
                # One model consultation (even if it does refinement rounds) counts as one decision-budget unit
                self.reasoner.model_decisions_used += 1
                self.reasoner.model_decisions_used_this_level += 1
                self.reasoner.reasoner_consultations_this_level += 1
                
                self.llm_step_meta["reasoner_consulted_this_step"] = True
                self.llm_step_meta["stagnation_override_used"] = stagnation_override_active
                surviving_proposals_this_step = 0
                for refinement_round in range(3):
                    proposals = self.reasoner.propose(
                        scene, sorted(legal_names), self.memory, step,
                        goals_summary=self.goals.summary() if profile.use_goals else (),
                        programs_summary=self.programs.summary() if profile.use_programs else (),
                        action_semantics_summary=action_semantics_summary,
                        response_frames=response_frames,
                        feedback=current_feedback,
                        bypass_can_call=True,
                    )
                    if proposals is not None:
                        self.llm_step_meta["reasoner_parsed_abduction_this_step"] = True
                        self.reasoner.llm_parsed_abduction_count += 1
                        best_spec = None
                        best_is_probe = False
                        best_meta = None
                        best_score = -999.0
                        
                        failed_summaries = []
                        
                        for proposal in proposals:
                            fam = proposal.puzzle_family or "none"
                            kind = proposal.program_spec.get("kind", "none") if proposal.program_spec else "none"
                            
                            # Check hypothesis retirement
                            model_action = proposal.action
                            if fam != "none" and model_action is not None:
                                pair_key = (fam, model_action.name)
                                if self.reasoner.failed_verifications.get(pair_key, 0) >= 3:
                                    failed_summaries.append(f"Skipping family/action pair {pair_key} due to repeated verification failures in this level.")
                                    continue
                            
                            if fam != "none" or kind != "none":
                                recent = self.reasoner.recent_abductions
                                pair = (fam, kind)
                                if recent.count(pair) >= 3:
                                    self.reasoner.llm_repetitive_kind_rejections += 1
                                    self.reasoner.llm_family_repeat_rejections += 1
                                    self.reasoner.recent_family_repeat_rejections.append(str(pair))
                                    failed_summaries.append(f"Skipping heavily repeated pattern {pair} without progress.")
                                    continue
                                recent.append(pair)
                                if len(recent) > 10:
                                    recent.pop(0)

                            if proposal.puzzle_family is not None:
                                self.goals.adjust_priors(proposal.puzzle_family)
                            if proposal.program_spec is not None and profile.use_programs:
                                self.reasoner.abductions_parsed += 1
                                self.programs.ingest_model_program(
                                    proposal.program_spec, list(self.memory.transitions))
                            
                            model_action = proposal.action
                            if model_action is not None:
                                is_severe_stagnation, stagnation_signal = self._check_severe_stagnation()
                                prior_kind = proposal.program_spec.get("kind") if proposal.program_spec else None
                                semantic_prior, semantic_reasons = self.path_planner.dynamics.prior_for_action(
                                    model_action.name, prior_kind)
                                # Modest translation bonus during stagnation
                                if prior_kind == "translation" and is_severe_stagnation:
                                    semantic_prior += 0.6
                                # Family cooldown penalty if family repeatedly failed
                                if prior_kind and self.memory.llm_family_cooldown.get(prior_kind, 0) >= 2:
                                    semantic_prior -= 0.8
                                if model_action.name not in legal_names:
                                    self.reasoner.llm_illegal_action_rejections += 1
                                    self.reasoner.llm_filtered_for_action_mismatch += 1
                                    self.reasoner.recent_mismatch_rejections.append(model_action.name)
                                    failed_summaries.append(f"Action {model_action.name} is not a valid legal action.")
                                    continue
                                if model_action.name == "ACTION6" and not ("x" in model_action.data_dict and "y" in model_action.data_dict):
                                    self.reasoner.llm_illegal_action_rejections += 1
                                    self.reasoner.llm_schema_valid_but_unexecutable += 1
                                    self.reasoner.recent_schema_rejections.append("missing_coordinates")
                                    failed_summaries.append("ACTION6 requires coordinates but none provided.")
                                    continue
                                signature = _action_signature(scene, model_action)
                                is_probe = self.memory.tried_count(
                                    scene.exact_key, model_action) == 0
                                prediction = self._predict(scene, model_action, profile)
                                is_dead_sig = self.dead.is_dead(signature)
                                
                                self.llm_step_meta["llm_override_eligible"] = is_severe_stagnation
                                self.llm_step_meta["llm_severe_stagnation_signal"] = stagnation_signal
                                
                                if not is_severe_stagnation:
                                    self.llm_step_meta["llm_override_block_reason"] = "not_stagnant"
                                elif self.memory.escape_hatch_used:
                                    self.llm_step_meta["llm_override_block_reason"] = "budget_exhausted"
                                elif is_dead_sig:
                                    self.llm_step_meta["llm_override_block_reason"] = "dead_signature"

                                decision = self._verify(
                                    scene, model_action, legal_names, prediction, is_probe, profile, is_severe_stagnation=is_severe_stagnation, is_llm_proposal=True)
                                
                                is_probe_exhausted = is_probe and self.memory.probes_this_level >= self.config.max_physical_probes_per_level
                                
                                allowed = decision.allowed
                                if "semantic_override_penalty" in decision.reasons:
                                    self.reasoner.llm_semantic_override_attempts += 1
                                    
                                if not allowed and is_severe_stagnation and not self.memory.escape_hatch_used and not is_dead_sig:
                                    non_negotiable = [r for r in decision.reasons if r in ("illegal_action", "incomplete_coordinates", "coordinate_out_of_bounds", "semantic_gating_high_death_context", "coordinate_fatigue")]
                                    if decision.score >= -6.0 and not non_negotiable:
                                        allowed = True
                                        self.memory.escape_hatch_used = True
                                        decision = AlignmentDecision(True, decision.score, decision.goal_delta, decision.risk, decision.goal_ids, decision.reasons + ("escape_hatch_override",))
                                        self.llm_step_meta["llm_override_block_reason"] = ""
                                    else:
                                        if non_negotiable:
                                            self.llm_step_meta["llm_override_block_reason"] = non_negotiable[0]
                                        else:
                                            self.llm_step_meta["llm_override_block_reason"] = "score_too_low"
                                            
                                if allowed and is_severe_stagnation and self.llm_step_meta.get("llm_override_block_reason") not in ("budget_exhausted", "dead_signature"):
                                    if "escape_hatch_override" not in decision.reasons:
                                        self.llm_step_meta["llm_override_block_reason"] = "no_override_needed"
                                
                                if allowed and "semantic_override_penalty" in decision.reasons:
                                    self.llm_step_meta["llm_semantic_override_used"] = True
                                    self.llm_step_meta["llm_breakout_used_this_step"] = True
                                
                                if not allowed or is_dead_sig:
                                    if "semantic_override_penalty" in decision.reasons:
                                        self.reasoner.llm_semantic_override_rejections += 1
                                    if is_dead_sig:
                                        self.reasoner.llm_rejected_by_dead_signature += 1
                                        failed_summaries.append(f"Action {model_action.name} rejected due to dead signature.")
                                    else:
                                        reasons_str = ",".join(decision.reasons)
                                        if any(r.startswith("semantic_gating") for r in decision.reasons):
                                            self.reasoner.llm_rejected_by_semantic_gate += 1
                                        elif any(r.startswith("coordinate_fatigue") for r in decision.reasons):
                                            self.reasoner.llm_rejected_by_coordinate_fatigue += 1
                                        else:
                                            self.reasoner.llm_rejected_by_alignment += 1
                                        failed_summaries.append(f"Action {model_action.name} rejected: {reasons_str} (score {round(decision.score, 2)})")
                                        
                                    if fam != "none":
                                        pair_key = (fam, model_action.name)
                                        self.reasoner.failed_verifications[pair_key] = self.reasoner.failed_verifications.get(pair_key, 0) + 1
                                else:
                                    surviving_proposals_this_step += 1
                                    self.llm_step_meta["reasoner_surviving_proposals_this_step"] = surviving_proposals_this_step
                                    self.last_model_milestone = milestone_name
                                    
                                    if fam != "none":
                                        self.reasoner.failed_verifications.pop((fam, model_action.name), None)
                                        
                                    score = model_action.score + decision.score + semantic_prior + self._priority_family_boost(prior_kind)
                                    if score > best_score:
                                        best_score = score
                                        best_spec = ActionSpec(
                                            name=model_action.name, data=model_action.data, source=model_action.source,
                                            predicted_effect=model_action.predicted_effect if prediction is None else prediction.expected_effect,
                                            score=score,
                                            program_id=None if prediction is None else prediction.program_id,
                                            predicted_state_key=None if prediction is None else _stable_hash_bytes(
                                                prediction.grid.tobytes()),
                                            goal_ids=decision.goal_ids,
                                        )
                                        best_is_probe = is_probe
                                        fam_name = prior_kind or "translation"
                                        self.memory.llm_recent_committed_family = fam_name
                                        best_meta = {
                                            "stage": "milestone_model_goal_verified" if profile.use_alignment else "milestone_model_verified",
                                            "final_action_source": "llm_probe" if is_probe else "llm_program",
                                            "final_action_from_llm": True,
                                            "llm_family": fam_name,
                                            "alignment": round(decision.score, 3),
                                            "semantic_prior": round(semantic_prior, 3),
                                            "semantic_reasons": semantic_reasons[:3],
                                            "milestone": milestone_name,
                                            **self.llm_step_meta
                                        }
                        
                        if best_spec is not None:
                            # Smarter failed consultation reset: reset backoff and lock state on successful model proposal.
                            self.reasoner.reasoner_decision_successes += 1
                            self.failed_consultations_this_level = 0
                            self.recent_lock_skip_count = 0
                            self.reasoner_lock_backoff_steps = 0
                            self.reasoner_suppressed = False
                            return best_spec, best_is_probe, best_meta
                        
                        if failed_summaries:
                            current_feedback = " | ".join(failed_summaries)
                        else:
                            current_feedback = "All proposals rejected due to parsing or validation constraints."
                    else:
                        self.reasoner.llm_empty_abduction_count += 1
                        break
                
                # Smarter failed consultation backoff: only strongly penalize if we actually parsed something and it was fully rejected
                if current_feedback and current_feedback != "All proposals rejected due to parsing or validation constraints.":
                    self.failed_consultations_this_level += 1

        # 7) Physical probes; EIG proxy comes from conditional hypothesis memory.
        probes: list[tuple[float, Candidate, AlignmentDecision]] = []
        if self.memory.probes_this_level < self.config.max_physical_probes_per_level or self.memory.no_op_streak >= self.config.no_progress_cooldown_steps:
            for candidate in candidates:
                if not candidate.is_probe:
                    continue
                prediction = self._predict(scene, candidate.spec, profile)
                decision = self._verify(
                    scene, candidate.spec, legal_names, prediction, True, profile)
                if not decision.allowed:
                    continue
                eig = self.hypotheses.information_value(
                    candidate.signature) if profile.use_hypotheses else 0.0
                probe_matched, probe_boost = self._probe_matches_recommendation(candidate, scene, step=step)
                score = candidate.score + 0.65 * eig + decision.score - 0.45 + probe_boost
                if self.memory.recent_state_loop(self.config.loop_window) or self.memory.no_op_streak >= self.config.no_progress_cooldown_steps:
                    score += 0.65
                applied_probe_transfer_bias = False
                if self.memory.post_breakthrough_window_active:
                    if self.memory.transferred_winning_coords and candidate.spec.data:
                        c_dict = dict(candidate.spec.data)
                        if "x" in c_dict and "y" in c_dict:
                            if abs(c_dict["x"] - self.memory.transferred_winning_coords[0]) + abs(c_dict["y"] - self.memory.transferred_winning_coords[1]) <= 2:
                                score += 1.0
                                applied_probe_transfer_bias = True
                probes.append((score, candidate, decision, probe_matched, applied_probe_transfer_bias))
        if probes:
            probes.sort(key=lambda row: row[0], reverse=True)
            score, best, decision, probe_matched, applied_probe_transfer_bias = probes[0]
            if applied_probe_transfer_bias:
                self.memory.post_breakthrough_bias_used = True
            self.memory.recommended_probe_type = ""
            self.memory.recommended_probe_ttl = 0
            spec = ActionSpec(
                name=best.spec.name, data=best.spec.data,
                source="goal_discriminating_probe" if profile.use_goals else "hypothesis_discriminating_probe",
                predicted_effect=best.spec.predicted_effect, score=score, goal_ids=decision.goal_ids,
            )
            return spec, True, {
                "stage": "admissible_goal_discriminating_probe" if profile.use_goals else "admissible_hypothesis_probe",
                "final_action_source": "goal_discriminating_probe" if profile.use_goals else "hypothesis_discriminating_probe",
                "score": round(score, 3),
                "probe_budget_used": self.memory.probes_this_level,
                "goals": list(decision.goal_ids),
                "rationale": best.rationale[:4],
                "is_discriminating_probe": probe_matched,
                "probe_recommendation_present": bool(self.memory.recommended_probe_type),
                "post_breakthrough_bias_used": applied_probe_transfer_bias,
            }

        # 8) Least harmful advertised action, with no invented actions.
        if candidates:
            if self.memory.post_breakthrough_window_active:
                self.post_breakthrough_fallback_uses += 1
                if self.post_breakthrough_fallback_uses >= 2:
                    self.memory.post_breakthrough_window_active = False
                    self.memory.post_breakthrough_aborted_reason = "continuation_fallback_forced"
                    self.post_breakthrough_fallback_uses = 0

            allowed_safest: list[tuple[float, Candidate]] = []
            disallowed_safest: list[tuple[float, Candidate]] = []
            for candidate in candidates:
                prediction = self._predict(scene, candidate.spec, profile)
                decision = self._verify(
                    scene, candidate.spec, legal_names, prediction, candidate.is_probe, profile)
                entry = (candidate.score + decision.score - decision.risk, candidate)
                if decision.allowed:
                    allowed_safest.append(entry)
                else:
                    disallowed_safest.append(entry)
            
            target_list = allowed_safest if allowed_safest else disallowed_safest
            target_list.sort(key=lambda row: row[0], reverse=True)
            stage_name = "alignment_constrained_fallback" if profile.use_alignment else "deterministic_fallback"
            return target_list[0][1].spec, target_list[0][1].is_probe, {
                "stage": stage_name,
                "final_action_source": "alignment_fallback" if profile.use_alignment else "deterministic_fallback"
            }

        reset = next((a for a in legal_actions if a.name == "RESET"), None)
        if reset is not None:
            return ActionSpec(name="RESET", source="fail_closed_reset"), False, {"stage": "fail_closed_reset"}
        if legal_actions:
            first_action = legal_actions[0]
            data_fallback: tuple[tuple[str, int], ...] = ()
            if first_action.is_complex():
                data_fallback = (("x", scene.width // 2), ("y", scene.height // 2))
            return ActionSpec(name=first_action.name, data=data_fallback, source="absolute_fallback"), False, {"stage": "absolute_legal_fallback"}
        raise RuntimeError(
            "ARC-AGI-3 frame advertised no executable legal action")


# ---------------------------------------------------------------------------
# Main Kaggle agent
# ---------------------------------------------------------------------------


class MyAgent(Agent):
    """DEWMA-ARC v4 agent with level-scoped alignment and global runtime control."""

    MAX_ACTIONS = _env_int("DEWMA_MAX_ACTIONS", 400)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = Config(max_actions=self.MAX_ACTIONS)
        self.dynamics = ActionDynamics()
        self._construct_modules()
        self.pending_scene: Scene | None = None
        self.pending_action: ActionSpec | None = None
        self.pending_was_probe = False
        self.pending_signature = ""
        self.current_level = 0
        self.last_state: GameState | None = None
        self.step_index = 0
        self.runtime = RuntimeBudget(self.config, str(
            getattr(self, "game_id", f"agent-{id(self)}")))
        if hasattr(self, "controller") and hasattr(self.controller, "reasoner"):
            self.controller.reasoner.game_id = self.runtime.game_id
        self.diagnostic_logger = DiagnosticTraceLogger(self.config)
        self.pending_reasoning: dict[str, Any] = {}
        self.pending_decision_latency_sec = 0.0

    def __del__(self) -> None:
        try:
            if hasattr(self, "diagnostic_logger"):
                self.diagnostic_logger.flush(
                    str(getattr(self, "game_id", "unknown")))
        except Exception:
            pass

    def _construct_modules(self) -> None:
        self.perception = PerceptionSystem(self.config)
        self.memory = TraceMemory()
        self.dead = DeadSignatureMemory(self.config.no_op_dead_threshold)
        self.spatial_hash = FastSpatialActionHash(self.config)
        self.world_model = CausalWorldModel()
        self.hypotheses = HypothesisMemory()
        self.goals = GoalHypothesisManager(self.config)
        self.programs = ExecutableProgramLibrary(self.config)
        self.path_planner = PathPlanner(self.config, self.dynamics)
        self.reasoner = OptionalLocalReasoner(self.config)
        self.reasoner.get_game_id = lambda: getattr(self, "game_id", "unknown")
        self.generator = CandidateGenerator(
            self.config,
            self.memory,
            self.dead,
            self.spatial_hash,
            self.world_model,
            self.hypotheses,
            self.dynamics,
        )
        self.alignment = GoalAlignmentVerifier(self.config, self.goals)
        self.counterfactual = CounterfactualPlanner(
            self.config, self.programs, self.goals, self.alignment, self.memory
        )
        self.controller = MetacognitiveController(
            self.config,
            self.memory,
            self.dead,
            self.world_model,
            self.generator,
            self.path_planner,
            self.reasoner,
            self.goals,
            self.programs,
            self.counterfactual,
            self.alignment,
            self.hypotheses,
        )
        self.controller.perception = self.perception

    @property
    def name(self) -> str:
        profile = "model" if self.config.enable_model else "det"
        return f"{super().name}.dewma-{profile}.v4-hardened.{self.MAX_ACTIONS}"

    @staticmethod
    def _state_name(frame: FrameData) -> str:
        return getattr(getattr(frame, "state", None), "name", str(getattr(frame, "state", ""))).upper()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        won = self._state_name(latest_frame) == "WIN"
        budget_stop = self.runtime.in_finalization_reserve()
        if won:
            self.runtime.mark_game_complete()
        if won or budget_stop:
            self.diagnostic_logger.flush(
                str(getattr(self, "game_id", "unknown")))
        return won or budget_stop

    def _full_reset(self) -> None:
        self.dynamics.reset_all()
        self._construct_modules()
        self.pending_scene = None
        self.pending_action = None
        self.pending_was_probe = False
        self.pending_signature = ""
        self.current_level = 0
        self.step_index = 0
        self.pending_reasoning = {}
        self.pending_decision_latency_sec = 0.0

    def _attempt_reset(self, level: int) -> None:
        """Reset transient execution state after death, preserving level learning."""
        self.perception.reset()
        self.memory.reset_attempt()
        self.controller.plan_queue.clear()
        self.controller.last_model_milestone = ""
        self.pending_scene = None
        self.pending_action = None
        self.pending_was_probe = False
        self.pending_signature = ""
        self.pending_reasoning = {}
        self.pending_decision_latency_sec = 0.0
        self.current_level = level

    def _level_reset(self, new_level: int, first_scene: Scene | None = None, preserve_perception: bool = False) -> None:
        if not preserve_perception:
            self.perception.reset()
        self.memory.reset_level()
        self.dead.reset_level()
        self.world_model.reset_level()
        self.hypotheses.reset_level()
        self.programs.reset_level()
        self.goals.reset_level(first_scene)
        self.alignment.reset_level()
        self.generator.reset_level()
        self.controller.reset_level()
        self.reasoner.reset_level()
        self.pending_scene = None
        self.pending_action = None
        self.pending_was_probe = False
        self.pending_signature = ""
        self.pending_reasoning = {}
        self.pending_decision_latency_sec = 0.0
        self.current_level = new_level

    @staticmethod
    def _to_game_action(
        spec: ActionSpec,
        legal_actions: Sequence[GameAction],
        reasoning: Mapping[str, Any],
    ) -> GameAction:
        legal_by_name = {action.name: action for action in legal_actions}
        action = legal_by_name.get(spec.name)
        if action is None:
            reset = legal_by_name.get("RESET")
            if reset is None:
                raise RuntimeError(
                    f"Proposed action {spec.name} is not in available_actions")
            action = reset
        if action.is_complex():
            data = spec.data_dict
            x = int(data.get("x", 16))
            y = int(data.get("y", 16))
            action.set_data({"x": max(0, min(63, x)), "y": max(0, min(63, y))})
        try:
            action.reasoning = dict(reasoning)
        except Exception:
            pass
        return action

    @staticmethod
    def _diagnostic_entity_deltas(event: Event) -> list[dict[str, Any]]:
        return [
            asdict(DiagnosticEntityDelta(entity_id=eid, dx=dx, dy=dy))
            for eid, dx, dy in event.entity_moves
        ]

    def _record_pending_transition(
        self,
        scene: Scene,
        latest_frame: FrameData,
        previous_level: int,
        sequence: FrameSequence,
        animation: AnimationTrace,
    ) -> Event | None:
        if self.pending_scene is None or self.pending_action is None:
            return None
        level_delta = max(
            0, int(getattr(latest_frame, "levels_completed", 0)) - previous_level)
        event = _event_from_scenes(
            self.pending_scene,
            scene,
            self.perception.latest_entity_moves,
            level_delta,
            latest_frame.state,
            animation,
        )
        transition = Transition(
            before_exact=self.pending_scene.exact_key,
            before_abstract=self.pending_scene.abstract_key,
            action=self.pending_action,
            after_exact=scene.exact_key,
            after_abstract=scene.abstract_key,
            event=event,
            level=self.current_level,
            step_index=self.step_index,
            before_grid=self.pending_scene.grid.copy(),
            after_grid=scene.grid.copy(),
            response_frames=tuple(
                frame.copy() for frame in sequence.grids[-self.config.animation_history_limit:]),
        )
        pending_tier = str(self.pending_reasoning.get(
            "runtime_tier", self.runtime.tier()))
        profile = _runtime_profile(pending_tier, self.config)
        # Reset per-step transition telemetry flags at start of transition processing
        self.memory.productive_branch_family_wobble_tolerated = False
        self.memory.productive_branch_family_wobble_reason = ""
        self.memory.productive_window_extended = False
        self.memory.productive_window_extension_reason = ""
        self.memory.productive_branch_collision_recovery_used = False
        self.memory.productive_branch_collision_recovery_variant = ""
        self.memory.productive_branch_collision_recovery_preserved_family = False
        self.memory.productive_branch_preempt_blocked = False
        self.memory.productive_branch_preempt_block_reason = ""
        self.memory.collision_soft_retry_used = False
        self.memory.collision_soft_retry_variant = ""
        self.memory.record(transition, self.pending_was_probe)
        self.dead.record(self.pending_signature, event)

        # 1. Productive Counterfactual Streak Renewal (Properly scoped to counterfactual stages)
        pending_stage = str(self.pending_reasoning.get("stage", ""))
        pending_src = str(self.pending_reasoning.get("final_action_source", ""))
        is_cf_selected = ("counterfactual" in pending_stage or "counterfactual" in pending_src or (self.pending_action and "counterfactual" in getattr(self.pending_action, "source", "")))
        if (
            is_cf_selected
            and event.changed_count > 0
            and not event.no_op
            and not event.game_over
        ):
            self.controller.counterfactual_streak = 0
            self.memory.counterfactual_streak_renewed = True

        # Continuation Failure Accounting
        if self.memory.post_breakthrough_window_active:
            self.memory.post_breakthrough_attempts += 1
            if event.changed_count > 0 and not event.no_op and not event.game_over:
                self.memory.post_breakthrough_effective_attempts += 1
            if event.no_op:
                self.memory.post_breakthrough_noop_attempts += 1
                self.memory.post_breakthrough_failed_attempts += 1
            elif event.game_over:
                self.memory.post_breakthrough_failed_attempts += 1
        
        # 0. Bounded Near-Terminal Finish Mode & Productive Branch Dominance on Level 0
        if self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
            # Update rolling recent history
            if event.changed_count > 0 and not event.no_op:
                c_score = float(self.memory.counterfactual_completion_bias + self.memory.terminal_condition_bonus)
                self.memory.recent_completion_scores.append(c_score)
                if len(self.memory.recent_completion_scores) > 6:
                    self.memory.recent_completion_scores.pop(0)
                self.memory.recent_mechanism_families.append(self.memory.top_mechanism_family)
                if len(self.memory.recent_mechanism_families) > 6:
                    self.memory.recent_mechanism_families.pop(0)
                p_stage_curr = str(self.pending_reasoning.get("stage", ""))
                self.memory.recent_action_sources.append(p_stage_curr)
                if len(self.memory.recent_action_sources) > 6:
                    self.memory.recent_action_sources.pop(0)

            # Check Active Finish Mode Updates / Exits & Soft Retries
            if self.memory.near_terminal_finish_mode_active:
                self.memory.finish_branch_continuation_steps += 1
                if event.no_op:
                    self.memory.finish_mode_noop_streak += 1
                else:
                    self.memory.finish_mode_noop_streak = 0
                    # 7. Extend productive window only when real evidence of non-noop progress continues near anchor
                    if self.memory.near_terminal_finish_steps_remaining <= 2 and self.memory.finish_branch_continuation_steps < 8:
                        self.memory.near_terminal_finish_steps_remaining += 2
                        self.memory.productive_window_extended = True
                        self.memory.productive_window_extension_reason = "continuous_local_non_noop_change"

                if event.level_delta > 0 or self.memory.current_level_index > 0:
                    self.memory.near_terminal_finish_mode_active = False
                    self.memory.near_terminal_finish_exit_reason = "level_cleared"
                    self.memory.finish_branch_continuation_break_reason = "level_cleared"
                elif getattr(event, "death", False) or event.game_over:
                    # 8. Collision-aware local retry before abandonment
                    if self.memory.collision_soft_retry_count < 2:
                        self.memory.collision_soft_retry_count += 1
                        self.memory.collision_soft_retry_used = True
                        self.memory.productive_branch_collision_recovery_used = True
                        self.memory.collision_soft_retry_variant = "orthogonal_turn" if self.memory.recommended_orthogonal_turn else "neighbor_variant"
                        self.memory.productive_branch_collision_recovery_variant = self.memory.collision_soft_retry_variant
                        self.memory.productive_branch_collision_recovery_preserved_family = True
                        self.memory.near_terminal_finish_steps_remaining = 2
                    else:
                        self.memory.near_terminal_finish_mode_active = False
                        self.memory.near_terminal_finish_exit_reason = "fatal_collision"
                        self.memory.finish_branch_continuation_break_reason = "fatal_collision"
                        # Collision-aware finish-family suppression
                        fam = self.memory.near_terminal_finish_family
                        if fam:
                            self.memory.finish_family_collision_counts[fam] = self.memory.finish_family_collision_counts.get(fam, 0) + 1
                            if self.memory.finish_family_collision_counts[fam] >= 2:
                                self.memory.finish_family_collision_suppressed = True
                                self.memory.finish_family_collision_suppressed_family = fam
                                self.memory.finish_family_collision_suppressed_until_step = self.step_index + 30
                elif self.memory.finish_mode_noop_streak >= 2:
                    self.memory.near_terminal_finish_mode_active = False
                    self.memory.near_terminal_finish_exit_reason = "noop_streak_exhaustion"
                    self.memory.finish_branch_continuation_break_reason = "noop_streak_exhaustion"
                elif self.memory.top_mechanism_confidence >= 0.65 and self.memory.top_mechanism_family != self.memory.near_terminal_finish_family:
                    # 6. Promote branch persistence across related family wobble
                    related = RELATED_MECHANISM_FAMILIES.get(self.memory.near_terminal_finish_family, set())
                    if self.memory.top_mechanism_family in related and self.memory.finish_mode_noop_streak == 0:
                        self.memory.productive_branch_family_wobble_tolerated = True
                        self.memory.productive_branch_family_wobble_reason = f"tolerated_wobble_to_{self.memory.top_mechanism_family}"
                    else:
                        self.memory.near_terminal_finish_mode_active = False
                        self.memory.near_terminal_finish_exit_reason = "mechanism_family_shift"
                        self.memory.finish_branch_continuation_break_reason = "mechanism_family_shift"
                else:
                    self.memory.near_terminal_finish_steps_remaining -= 1
                    if self.memory.near_terminal_finish_steps_remaining <= 0:
                        self.memory.near_terminal_finish_mode_active = False
                        self.memory.near_terminal_finish_exit_reason = "finish_window_expired"
                        self.memory.finish_branch_continuation_break_reason = "finish_window_expired"

            # 1. Track Concrete Productive Branch in Memory
            p_stage = str(self.pending_reasoning.get("stage", ""))
            p_action_name = self.pending_action.name if self.pending_action else ""
            p_anchor = None
            if self.pending_action and self.pending_action.data:
                p_dict = dict(self.pending_action.data)
                if "x" in p_dict and "y" in p_dict:
                    p_anchor = (p_dict["x"], p_dict["y"])

            if event.changed_count > 0 and not event.no_op and not event.game_over:
                p_sig = f"{p_stage}:{p_action_name}:{p_anchor}"
                is_same_neighborhood = False
                if self.memory.productive_branch_anchor and p_anchor:
                    bx, by = self.memory.productive_branch_anchor
                    px, py = p_anchor
                    if max(abs(px - bx), abs(py - by)) <= 3:
                        is_same_neighborhood = True
                elif not p_anchor and not self.memory.productive_branch_anchor:
                    is_same_neighborhood = True

                if (self.memory.productive_branch_signature == p_sig) or (is_same_neighborhood and self.memory.productive_branch_action_name == p_action_name):
                    self.memory.productive_branch_streak += 1
                else:
                    self.memory.productive_branch_signature = p_sig
                    self.memory.productive_branch_source = p_stage
                    self.memory.productive_branch_action_name = p_action_name
                    self.memory.productive_branch_anchor = p_anchor
                    self.memory.productive_branch_family_hint = self.memory.top_mechanism_family
                    self.memory.productive_branch_program_kind = str(self.pending_reasoning.get("program_kind", ""))
                    self.memory.productive_branch_streak = 1
                self.memory.productive_branch_last_effective_step = self.step_index
            elif event.no_op or event.game_over:
                if self.memory.no_op_streak >= 2:
                    self.memory.productive_branch_streak = 0

            # 4. Structured Branch Persistence: activate and hold control for 6 steps
            has_completion_signal = (
                self.memory.productive_search_convergence_pressure >= 0.5
                or self.memory.counterfactual_completion_bias > 0.0
                or self.memory.consecutive_completion_signals >= 1
            )
            if (
                (self.memory.productive_branch_streak >= 2 or is_cf_selected)
                and event.changed_count > 0
                and not event.no_op
                and not event.game_over
                and has_completion_signal
            ):
                self.memory.structured_branch_persistence_used = True
                self.memory.structured_branch_persistence_steps += 1
                self.memory.structured_branch_persistence_steps_remaining = max(1, 6 - self.memory.structured_branch_persistence_steps)
                self.memory.structured_branch_persistence_break_reason = ""
            elif self.memory.structured_branch_persistence_used:
                if event.no_op and self.memory.finish_mode_noop_streak >= 2:
                    self.memory.structured_branch_persistence_used = False
                    self.memory.structured_branch_persistence_break_reason = "repeated_noop"
                elif event.game_over:
                    self.memory.structured_branch_persistence_used = False
                    self.memory.structured_branch_persistence_break_reason = "game_over"
                elif self.memory.structured_branch_persistence_steps >= 6:
                    self.memory.structured_branch_persistence_used = False
                    self.memory.structured_branch_persistence_break_reason = "window_expired"

            # Check Finish Mode Entry Criteria with Selective Gating & Concrete Branch Binding
            if not self.memory.near_terminal_finish_mode_active and not event.game_over and not event.no_op:
                p_stage = str(self.pending_reasoning.get("stage", ""))
                is_structured = p_stage in ("counterfactual_program_search", "counterfactual_program", "verified_plan_queue", "macro_replay")
                
                # Check collision suppression
                is_suppressed = (
                    self.memory.finish_family_collision_suppressed
                    and self.memory.top_mechanism_family == self.memory.finish_family_collision_suppressed_family
                    and self.step_index < self.memory.finish_family_collision_suppressed_until_step
                )

                # Gate 1: Family Consensus (repeated productive signals from the same mechanism family)
                fam_count = sum(1 for f in self.memory.recent_mechanism_families[-4:] if f == self.memory.top_mechanism_family)
                gate_family = (fam_count >= 3 and len(self.memory.recent_mechanism_families) >= 3 and self.memory.top_mechanism_confidence >= 0.32)

                # Gate 2: Branch Consensus (consecutive non-noop state changes with non-decreasing completion)
                recent_scores = self.memory.recent_completion_scores[-3:]
                gate_branch = (len(recent_scores) >= 2 and all(recent_scores[i] <= recent_scores[i+1] + 0.1 for i in range(len(recent_scores)-1)) and sum(recent_scores) > 1.5)

                # Gate 3: Completion Stability (both completion bias and terminal bonus high)
                gate_stability = (self.memory.counterfactual_completion_bias >= 2.0 and self.memory.terminal_condition_bonus >= 2.5)

                self.memory.near_terminal_finish_gate_family_consensus = gate_family
                self.memory.near_terminal_finish_gate_branch_consensus = gate_branch
                self.memory.near_terminal_finish_gate_completion_stability = gate_stability

                # Check if counterfactual search is exceptionally healthy and should not be prematurely preempted
                is_cf_healthy = (
                    p_stage == "counterfactual_program_search"
                    and self.controller.counterfactual_streak < 3
                    and self.memory.counterfactual_completion_bias >= 2.0
                    and not gate_stability
                    and not gate_family
                )

                if is_cf_healthy:
                    self.memory.finish_mode_preempted_counterfactual = True
                    self.memory.finish_mode_preempt_block_reason = "healthy_counterfactual_preserved"

                # Tightened entry: require non-noop structured change or anchor reuse or rising pressure
                has_anchor_reuse = bool(self.memory.last_changed_coord is not None)
                has_rising_pressure = bool(self.memory.productive_search_convergence_pressure >= 1.5)
                has_structured_change = bool(event.changed_count > 0 and not event.no_op)
                tightened_entry = (has_anchor_reuse or has_rising_pressure or has_structured_change)

                if is_structured and not is_suppressed and not is_cf_healthy and tightened_entry and (gate_family or gate_branch or gate_stability):
                    self.memory.consecutive_completion_signals += 1
                    if self.memory.consecutive_completion_signals >= 2:
                        self.memory.near_terminal_finish_gate_allowed = True
                        self.memory.near_terminal_finish_mode_active = True
                        self.memory.near_terminal_finish_steps_remaining = 6
                        self.memory.near_terminal_finish_family = self.memory.top_mechanism_family
                        self.memory.near_terminal_finish_source = p_stage
                        
                        # Concrete branch binding
                        self.memory.finish_bound_branch_source = p_stage
                        self.memory.finish_bound_branch_action_name = self.pending_action.name if self.pending_action else ""
                        if self.pending_action and self.pending_action.data:
                            p_dict = dict(self.pending_action.data)
                            if "x" in p_dict and "y" in p_dict:
                                self.memory.finish_bound_branch_anchor = (p_dict["x"], p_dict["y"])
                        else:
                            self.memory.finish_bound_branch_anchor = None

                        self.memory.finish_bound_branch_signature = self.memory.productive_branch_signature
                        self.memory.finish_bound_branch_source = self.memory.productive_branch_source or p_stage
                        self.memory.finish_bound_branch_action_name = self.memory.productive_branch_action_name or p_action_name
                        self.memory.near_terminal_finish_trigger_reason = (
                            "family_consensus" if gate_family else ("branch_consensus" if gate_branch else "completion_stability")
                        )
                        self.memory.near_terminal_finish_exit_reason = ""
                        self.memory.finish_mode_noop_streak = 0
                        self.memory.finish_branch_continuation_steps = 0
                        self.memory.finish_branch_continuation_family = self.memory.top_mechanism_family
                        self.memory.finish_bound_branch_kept_control = True
                        self.memory.productive_branch_commitment_used = True
                else:
                    self.memory.consecutive_completion_signals = max(0, self.memory.consecutive_completion_signals - 1)

        # 1. Collision-Responsive Orthogonal Steering on Level 0
        if self.memory.current_level_index == 0 and not self.memory.post_breakthrough_window_active:
            if getattr(event, "death", False) or event.game_over:
                if self.pending_action and self.pending_action.name in ("ACTION3", "ACTION4"):
                    self.memory.recommended_orthogonal_turn = "vertical"
                    self.memory.orthogonal_collision_steer_used = True
                elif self.pending_action and self.pending_action.name in ("ACTION1", "ACTION2"):
                    self.memory.recommended_orthogonal_turn = "horizontal"
                    self.memory.orthogonal_collision_steer_used = True
            elif event.changed_count > 0 and not event.no_op:
                self.memory.recommended_orthogonal_turn = ""

        # 2. Multi-Click Cell Cycle Persistence tracking
        if event.changed_count > 0 and not event.no_op and not event.game_over and self.pending_action and self.pending_action.data:
            p_dict = dict(self.pending_action.data)
            if "x" in p_dict and "y" in p_dict:
                px, py = p_dict["x"], p_dict["y"]
                if self.memory.last_changed_coord == (px, py):
                    self.memory.coord_cycle_clicks_remaining = max(0, self.memory.coord_cycle_clicks_remaining - 1)
                else:
                    self.memory.last_changed_coord = (px, py)
                    self.memory.coord_cycle_clicks_remaining = 2
        else:
            self.memory.coord_cycle_clicks_remaining = max(0, self.memory.coord_cycle_clicks_remaining - 1)

        # 3. Systematic Sequential Component Sweep click decrement and anti-stall
        if self.pending_action and self.pending_action.data:
            p_dict = dict(self.pending_action.data)
            if "x" in p_dict and "y" in p_dict:
                px, py = p_dict["x"], p_dict["y"]
                if self.memory.spatial_visits.get((px, py), 0) >= 2 or event.no_op:
                    self.memory.visited_sweep_component_anchors.add((px, py))
        if self.memory.active_sweep_clicks_remaining > 0:
            if event.no_op or event.changed_count == 0:
                self.memory.active_sweep_clicks_remaining = 0
                self.memory.active_sweep_component_id = None
            else:
                self.memory.active_sweep_clicks_remaining -= 1
                if self.memory.active_sweep_clicks_remaining <= 0:
                    self.memory.active_sweep_component_id = None

        # 4. Atomic Two-Phase Drag/Drop Pairing (ACTION6 -> ACTION7) in drag_or_push / movement_control
        if (
            self.memory.current_level_index == 0
            and not self.memory.post_breakthrough_window_active
            and self.pending_action
            and self.pending_action.name == "ACTION6"
            and not event.no_op
            and not event.game_over
            and self.memory.top_mechanism_family in ("drag_or_push", "movement_control")
            and len(self.memory.macro_replay_queue) == 0
        ):
            p_dict = dict(self.pending_action.data) if self.pending_action.data else {}
            if "x" in p_dict and "y" in p_dict:
                src_x, src_y = p_dict["x"], p_dict["y"]
                moves = getattr(event, "entity_moves", ())
                if moves:
                    m = moves[0]
                    dx = m[1] if isinstance(m, (tuple, list)) and len(m) >= 3 else getattr(m, "dx", 0)
                    dy = m[2] if isinstance(m, (tuple, list)) and len(m) >= 3 else getattr(m, "dy", 0)
                    if dx != 0 or dy != 0:
                        dst_x, dst_y = src_x + dx, src_y + dy
                        if 0 <= dst_x < scene.width and 0 <= dst_y < scene.height:
                            act7 = ActionSpec(name="ACTION7", data=(("x", dst_x), ("y", dst_y)), source="atomic_drag_drop_pair")
                            self.memory.macro_replay_queue.append(act7)
                            self.memory.atomic_drag_drop_paired = True

        # Zero-tolerance NO-OP coordinate blacklisting in exploitation mode
        if (self.memory.mode == "exploitation_mode" or self.memory.post_breakthrough_window_active) and event.no_op and self.pending_action:
            p_data = self.pending_action.data_dict
            if "x" in p_data and "y" in p_data:
                self.memory.exploitation_noop_blacklist.add((self.pending_action.name, p_data["x"], p_data["y"]))
                self.memory.exploitation_noop_neighborhood_blacklist.add((p_data["x"], p_data["y"]))
        # LLM Family Payoff & Follow-Through Window Update
        if self.pending_reasoning.get("final_action_from_llm"):
            fam = self.pending_reasoning.get("llm_family") or self.memory.llm_recent_committed_family or "unspecified"
            self.memory.llm_family_payoff[fam]["committed"] += 1
            if not event.no_op and not event.game_over:
                self.memory.llm_family_payoff[fam]["changed"] += 1
                self.memory.llm_follow_through_window = 3
                self.memory.llm_follow_through_family = fam
                p_data = self.pending_action.data_dict
                if "x" in p_data and "y" in p_data:
                    self.memory.llm_follow_through_coords = (p_data["x"], p_data["y"])
                self.memory.llm_family_cooldown[fam] = 0
                if profile.use_programs:
                    recent_progs = [p.program_id for p in (self.programs.programs.values() if isinstance(self.programs.programs, dict) else self.programs.programs) if getattr(p, 'kind', '') == fam]
                    self.memory.llm_promoted_program_ids = set(recent_progs[:5])
            else:
                self.memory.llm_family_cooldown[fam] += 1
                self.memory.llm_follow_through_window = 0
            if event.level_delta > 0 or event.win:
                self.memory.llm_family_payoff[fam]["progress"] += 1
        else:
            if self.pending_reasoning.get("final_action_source") == "llm_follow_through":
                self.memory.llm_follow_through_recent_steps = 4
            self.memory.llm_follow_through_window = max(0, self.memory.llm_follow_through_window - 1)
        
        # Check if recent follow-through activity produced progress
        if self.memory.llm_follow_through_recent_steps > 0:
            if event.level_delta > 0 or event.win:
                self.memory.follow_through_led_to_progress = True
            self.memory.llm_follow_through_recent_steps -= 1

        # Record winning pattern at progress time
        if event.level_delta > 0 or event.win:
            p_data = self.pending_action.data_dict if self.pending_action else {}
            coords = (p_data["x"], p_data["y"]) if ("x" in p_data and "y" in p_data) else None
            p_kind = self.pending_reasoning.get("program_kind") or self.pending_reasoning.get("program_family") or ""
            if not p_kind and self.pending_action and self.pending_action.program_id:
                p_obj = self.programs.programs.get(self.pending_action.program_id) if hasattr(self.programs, "programs") and isinstance(self.programs.programs, dict) else None
                if p_obj:
                    p_kind = getattr(p_obj, "kind", "")
            
            top_fam = self.memory.top_mechanism_family
            top_conf = self.memory.top_mechanism_confidence
            action_src = self.pending_reasoning.get("final_action_source", "") or getattr(self.pending_action, "source", "")

            winning_comp_color = None
            winning_comp_area = 0
            winning_comp_aspect = 1.0
            winning_comp_shape_hash = ""
            if coords and self.pending_scene:
                wx, wy = coords
                for c in self.pending_scene.components:
                    x0, y0, x1, y1 = c.bbox
                    if x0 <= wx <= x1 and y0 <= wy <= y1:
                        winning_comp_color = c.color
                        winning_comp_area = c.area
                        bbox_w = x1 - x0 + 1
                        bbox_h = y1 - y0 + 1
                        winning_comp_aspect = bbox_w / max(1, bbox_h)
                        winning_comp_shape_hash = c.shape_key
                        break
                if winning_comp_color is None and 0 <= wx < self.pending_scene.width and 0 <= wy < self.pending_scene.height:
                    winning_comp_color = int(self.pending_scene.grid[wy, wx])

            self.memory.last_winning_action_name = self.pending_action.name if self.pending_action else ""
            self.memory.last_winning_program_kind = p_kind
            self.memory.last_winning_family = top_fam if top_fam else ""
            self.memory.last_winning_coords = coords
            self.memory.last_winning_component_color = winning_comp_color
            self.memory.last_winning_component_area = winning_comp_area
            self.memory.last_winning_component_aspect = winning_comp_aspect
            self.memory.last_winning_component_shape_hash = winning_comp_shape_hash
            self.memory.last_winning_source = action_src
            self.memory.last_winning_mechanism_family = top_fam
            self.memory.last_winning_mechanism_confidence = top_conf
            self.memory.last_winning_step_in_level = self.memory.level_steps

        # Seed bounded macro replay if a directional translation or verified program succeeded
        if self.pending_reasoning.get("final_action_source") in ("milestone_model_verified", "milestone_model_goal_verified", "llm_program", "llm_follow_through", "counterfactual_program"):
            if not event.no_op and not event.game_over:
                fam = self.pending_reasoning.get("llm_family") or self.memory.llm_recent_committed_family or "translation"
                p_dict = dict(self.pending_action.data) if self.pending_action and self.pending_action.data else {}
                is_palette_button = "y" in p_dict and p_dict["y"] <= 2
                if not is_palette_button and fam in ("translation", "component_recolor") and len(self.memory.macro_replay_queue) == 0:
                    for _ in range(2):
                        self.memory.macro_replay_queue.append(self.pending_action)
                    self.memory.macro_replay_family = fam
                    self.memory.macro_replay_program_id = str(self.pending_action.program_id or "")

        # Abort macro replay on unexpected no-op or collision
        if self.pending_reasoning.get("final_action_source") == "macro_replay":
            if event.no_op or event.game_over:
                self.memory.macro_replay_queue.clear()
                self.memory.macro_replay_aborted_reason = "unexpected_noop_or_collision" 
        if profile.learn_hypotheses:
            self.hypotheses.record(self.pending_signature, event)
        self.world_model.record(transition)
        self.spatial_hash.record(
            self.pending_scene, scene, self.pending_action, event)
        self.dynamics.record(self.pending_action, event, self.pending_scene)
        if profile.learn_goals:
            self.goals.update(transition, self.pending_scene, scene)
        if profile.learn_alignment:
            self.alignment.observe(transition, self.pending_scene, scene)
        if profile.learn_programs:
            self.programs.record(transition, self.pending_scene, scene)
        if self.config.enable_learned_passability:
            self.path_planner.passability.record(transition, self.pending_scene)
        self.path_planner.control_inference.record(transition)

        data = self.pending_action.data_dict
        anchor_x = int(data.get("x", self.pending_scene.width // 2))
        anchor_y = int(data.get("y", self.pending_scene.height // 2))
        local_patch_hash = _boundary_aware_patch_hash(
            self.pending_scene.grid, anchor_x, anchor_y, radius=1
        )
        active_goals = tuple(g.goal_id for g in self.goals.active(
            6)) if profile.use_goals else ()
        verified_program_ids = tuple(
            row.get("id", "")
            for row in self.programs.summary(6)
            if profile.use_programs and isinstance(row, dict) and row.get("id")
        )
        representation_mode = "field" if scene.field_mode else (
            "entity" if scene.entity_confidence >= self.config.entity_confidence_threshold else "raw"
        )
        record = DiagnosticTraceRecord(
            timestamp=time.time(),
            game_id=str(getattr(self, "game_id", "unknown")),
            level=self.current_level,
            step=self.step_index,
            before_key=transition.before_exact,
            action={"name": self.pending_action.name,
                    **self.pending_action.data_dict},
            after_key=transition.after_exact,
            decision_stage=str(self.pending_reasoning.get(
                "stage", getattr(self.pending_action, "source", ""))) or "fallback_default",
            expected_effect=self.pending_action.predicted_effect,
            observed_effect=event.effect_signature,
            no_op=event.no_op,
            death=event.game_over,
            progress=bool(event.level_delta > 0 or event.win),
            physical_probe=self.pending_was_probe,
            model_called=bool(self.pending_reasoning.get("model_called", False)),
            local_patch_hash=local_patch_hash,
            active_goal_ids=active_goals,
            program_ids=verified_program_ids,
            representation_mode=representation_mode,
            representation_confidence=scene.entity_confidence,
            entity_deltas=self._diagnostic_entity_deltas(event),
            model_latency_sec=self.reasoner.last_latency_sec if self.pending_reasoning.get("model_called", False) else 0.0,
            deterministic_latency_sec=self.pending_decision_latency_sec,
            remaining_wall_time_sec=self.runtime.remaining_sec(),
            runtime_tier=str(self.pending_reasoning.get(
                "runtime_tier", self.runtime.tier())),
            model_decisions_used=int(self.pending_reasoning.get("model_decisions_used", 0)),
            model_decisions_used_this_level=int(self.pending_reasoning.get("model_decisions_used_this_level", 0)),
            reasoner_decision_attempts=int(self.pending_reasoning.get("reasoner_decision_attempts", 0)),
            reasoner_decision_skips_by_reason=dict(self.pending_reasoning.get("reasoner_decision_skips_by_reason", {})),
            reasoner_decision_successes=int(self.pending_reasoning.get("reasoner_decision_successes", 0)),
            reasoner_consultations_this_level=int(self.pending_reasoning.get("reasoner_consultations_this_level", 0)),
            fallback_loop_streak=self.controller.fallback_loop_streak,
            counterfactual_fallback_streak=self.controller.counterfactual_fallback_streak,
            alignment_fallback_streak=self.controller.alignment_fallback_streak,
            steps_since_progress=self.controller.steps_since_progress,
            progress_events_this_level=self.controller.progress_events_this_level,
            progress_density=self.controller.progress_density,
            recent_lock_skip_count=self.controller.recent_lock_skip_count,
            reasoner_lock_backoff_steps=self.controller.reasoner_lock_backoff_steps,
            reasoner_suppressed=self.controller.reasoner_suppressed,
            reasoner_suppression_reason=self.controller.reasoner_suppression_reason,
            llm_illegal_action_rejections=self.controller.reasoner.llm_illegal_action_rejections,
            llm_repetitive_kind_rejections=self.controller.reasoner.llm_repetitive_kind_rejections,
            llm_empty_abduction_count=self.controller.reasoner.llm_empty_abduction_count,
            llm_parsed_abduction_count=self.controller.reasoner.llm_parsed_abduction_count,
            llm_diversity_pruned_count=self.controller.reasoner.llm_diversity_pruned_count,
            llm_impossible_hypothesis_rejections=self.controller.reasoner.llm_impossible_hypothesis_rejections,
            llm_unsupported_family_rejections=self.controller.reasoner.llm_unsupported_family_rejections,
            llm_family_repeat_rejections=self.controller.reasoner.llm_family_repeat_rejections,
            llm_schema_valid_but_unexecutable=self.controller.reasoner.llm_schema_valid_but_unexecutable,
            llm_filtered_for_action_mismatch=self.controller.reasoner.llm_filtered_for_action_mismatch,
            llm_rejected_by_alignment=self.controller.reasoner.llm_rejected_by_alignment,
            llm_rejected_by_semantic_gate=self.controller.reasoner.llm_rejected_by_semantic_gate,
            llm_rejected_by_dead_signature=self.controller.reasoner.llm_rejected_by_dead_signature,
            llm_rejected_by_probe_budget=self.controller.reasoner.llm_rejected_by_probe_budget,
            llm_rejected_by_coordinate_fatigue=self.controller.reasoner.llm_rejected_by_coordinate_fatigue,
            llm_escape_hatch_used=self.controller.memory.escape_hatch_used,
            llm_semantic_override_attempts=self.controller.reasoner.llm_semantic_override_attempts,
            llm_semantic_override_used=self.controller.llm_step_meta.get("llm_semantic_override_used", False),
            llm_semantic_override_rejections=self.controller.reasoner.llm_semantic_override_rejections,
            llm_breakout_used_this_step=self.controller.llm_step_meta.get("llm_breakout_used_this_step", False),
            llm_override_eligible=self.controller.llm_step_meta.get("llm_override_eligible", False),
            llm_override_block_reason=self.controller.llm_step_meta.get("llm_override_block_reason", ""),
            llm_severe_stagnation_signal=self.controller.llm_step_meta.get("llm_severe_stagnation_signal", ""),
            llm_follow_through_active=self.controller.memory.llm_follow_through_window > 0,
            llm_follow_through_family=self.controller.memory.llm_follow_through_family,
            llm_promoted_programs_count=len(self.controller.memory.llm_promoted_program_ids),
            churn_stagnation_detected=self.controller.llm_step_meta.get("llm_severe_stagnation_signal") == "progressless_churn",
            follow_through_led_to_progress=self.controller.memory.follow_through_led_to_progress,
            early_breakout_acceleration_used="early" in str(self.controller.llm_step_meta.get("llm_severe_stagnation_signal", "")),
            level_budget_pressure_signal=round(min(2.0, self.controller.memory.level_steps / 120.0), 3),
            level_steps_elapsed=self.controller.memory.level_steps,
            level_budget_target=120,
            level_budget_pressure=round(min(2.0, self.controller.memory.level_steps / 120.0), 3),
            level_budget_escalation_used=self.controller.memory.level_steps > 120 and self.controller.memory.level_progress_events == 0,
            macro_replay_active=bool(self.pending_reasoning.get("macro_replay_active", False)),
            macro_replay_steps_remaining=len(self.controller.memory.macro_replay_queue),
            macro_replay_family=self.controller.memory.macro_replay_family,
            macro_replay_program_id=self.controller.memory.macro_replay_program_id,
            macro_replay_aborted_reason=self.controller.memory.macro_replay_aborted_reason,
            level_transition_transfer_used=self.controller.memory.level_transition_transfer_used,
            transferred_mechanism_family=self.controller.memory.transferred_mechanism_family,
            early_classified_mechanism=self.controller.memory.early_classified_mechanism,
            instant_reflex_used=bool(self.pending_reasoning.get("instant_reflex_used", False)),
            counterfactual_pruned_count=self.controller.memory.counterfactual_pruned_count,
            subgoal_distance_reward=float(self.pending_reasoning.get("subgoal_reward", 0.0)),
            negative_hypothesis_eliminations=self.controller.memory.negative_hypothesis_eliminations,
            mode=self.controller.memory.mode,
            top_mechanism_family=self.controller.memory.top_mechanism_family,
            top_mechanism_confidence=self.controller.memory.top_mechanism_confidence,
            competing_mechanism_families=self.controller.memory.competing_mechanism_families,
            mechanism_family_scores=dict(self.controller.memory.mechanism_scores),
            priority_program_families=list(self.controller.memory.priority_program_families),
            invariants_to_preserve=list(self.controller.memory.invariants_to_preserve),
            mechanism_shift_event=self.controller.memory.mechanism_shift_event,
            mechanism_aligned_action=bool(self.pending_reasoning.get("mechanism_aligned", False)),
            mechanism_discriminating_probe_used=bool(self.pending_reasoning.get("is_discriminating_probe", False)),
            promising_state_detected=bool(self.pending_reasoning.get("promising_state_detected", False)),
            promising_state_reasons=list(self.pending_reasoning.get("promising_state_reasons", [])),
            deep_search_used=bool(self.pending_reasoning.get("deep_search_used", False)),
            deep_search_depth=int(self.pending_reasoning.get("deep_search_depth", 0)),
            deep_search_width=int(self.pending_reasoning.get("deep_search_width", 0)),
            deep_search_nodes_evaluated=int(self.pending_reasoning.get("deep_search_nodes_evaluated", 0)),
            deep_search_best_score=float(self.pending_reasoning.get("deep_search_best_score", 0.0)),
            deep_search_time_ms=float(self.pending_reasoning.get("deep_search_time_ms", 0.0)),
            deep_search_selected_family=str(self.pending_reasoning.get("deep_search_selected_family", "")),
            deep_search_aborted_reason=str(self.pending_reasoning.get("deep_search_aborted_reason", "")),
            post_breakthrough_window_active=bool(self.memory.post_breakthrough_window_active),
            post_levelup_exploit_steps_remaining=int(self.memory.post_breakthrough_window_steps_remaining),
            transferred_winning_family=str(self.memory.transferred_winning_family),
            transferred_winning_program_kind=str(self.memory.transferred_winning_program_kind),
            transferred_winning_action_name=str(self.memory.transferred_winning_action_name),
            transferred_winning_coords=self.memory.transferred_winning_coords,
            post_breakthrough_aborted_reason=str(self.memory.post_breakthrough_aborted_reason),
            post_breakthrough_bias_used=bool(self.pending_reasoning.get("post_breakthrough_bias_used", self.memory.post_breakthrough_bias_used)),
            post_breakthrough_attempts=int(self.memory.post_breakthrough_attempts),
            post_breakthrough_effective_attempts=int(self.memory.post_breakthrough_effective_attempts),
            post_breakthrough_noop_attempts=int(self.memory.post_breakthrough_noop_attempts),
            post_breakthrough_failed_attempts=int(self.memory.post_breakthrough_failed_attempts),
            post_breakthrough_continuation_bias=float(self.memory.post_breakthrough_continuation_bias),
            post_breakthrough_local_search_used=bool(self.memory.post_breakthrough_local_search_used),
            post_breakthrough_abort_reason=str(self.memory.post_breakthrough_aborted_reason),
            counterfactual_streak_renewed=bool(self.memory.counterfactual_streak_renewed),
            mechanism_collapse_breaker_used=bool(self.memory.mechanism_collapse_breaker_used),
            mechanism_family_penalized=str(self.memory.mechanism_family_penalized),
            mechanism_family_promoted=str(self.memory.mechanism_family_promoted),
            component_delete_component_locked=bool(self.memory.component_delete_component_locked),
            line_beam_structured_candidates_used=bool(self.memory.line_beam_structured_candidates_used),
            counterfactual_completion_bias=float(self.memory.counterfactual_completion_bias),
            terminal_condition_bonus=float(self.memory.terminal_condition_bonus),
            mechanism_completion_bonus=float(self.memory.mechanism_completion_bonus),
            productive_search_convergence_pressure=float(self.memory.productive_search_convergence_pressure),
            line_beam_closure_bias_used=bool(self.memory.line_beam_closure_bias_used),
            component_delete_payoff_bias_used=bool(self.memory.component_delete_payoff_bias_used),
            atomic_drag_drop_paired=bool(self.memory.atomic_drag_drop_paired),
            sequential_component_sweep_active=bool(self.memory.sequential_component_sweep_active),
            orthogonal_collision_steer_used=bool(self.memory.orthogonal_collision_steer_used),
            cell_cycle_persistence_used=bool(self.memory.cell_cycle_persistence_used),
            near_terminal_finish_mode_active=bool(self.memory.near_terminal_finish_mode_active),
            near_terminal_finish_steps_remaining=int(self.memory.near_terminal_finish_steps_remaining),
            near_terminal_finish_family=str(self.memory.near_terminal_finish_family),
            near_terminal_finish_trigger_reason=str(self.memory.near_terminal_finish_trigger_reason),
            near_terminal_finish_exit_reason=str(self.memory.near_terminal_finish_exit_reason),
            productive_branch_commitment_used=bool(self.memory.productive_branch_commitment_used),
            near_terminal_finish_gate_family_consensus=bool(self.memory.near_terminal_finish_gate_family_consensus),
            near_terminal_finish_gate_branch_consensus=bool(self.memory.near_terminal_finish_gate_branch_consensus),
            near_terminal_finish_gate_completion_stability=bool(self.memory.near_terminal_finish_gate_completion_stability),
            near_terminal_finish_gate_allowed=bool(self.memory.near_terminal_finish_gate_allowed),
            finish_branch_continuation_family=str(self.memory.finish_branch_continuation_family),
            finish_branch_continuation_steps=int(self.memory.finish_branch_continuation_steps),
            finish_branch_continuation_kept_control=bool(self.memory.finish_branch_continuation_kept_control),
            finish_branch_continuation_break_reason=str(self.memory.finish_branch_continuation_break_reason),
            finish_family_collision_suppressed=bool(self.memory.finish_family_collision_suppressed),
            finish_family_collision_suppressed_family=str(self.memory.finish_family_collision_suppressed_family),
            finish_family_collision_suppressed_until_step=int(self.memory.finish_family_collision_suppressed_until_step),
            finish_mode_preempted_counterfactual=bool(self.memory.finish_mode_preempted_counterfactual),
            finish_mode_preempt_block_reason=str(self.memory.finish_mode_preempt_block_reason),
            productive_branch_signature=str(self.memory.productive_branch_signature),
            productive_branch_source=str(self.memory.productive_branch_source),
            productive_branch_action_name=str(self.memory.productive_branch_action_name),
            productive_branch_anchor=self.memory.productive_branch_anchor,
            productive_branch_family_hint=str(self.memory.productive_branch_family_hint),
            productive_branch_program_kind=str(self.memory.productive_branch_program_kind),
            productive_branch_streak=int(self.memory.productive_branch_streak),
            productive_branch_last_effective_step=int(self.memory.productive_branch_last_effective_step),
            productive_branch_family_wobble_tolerated=bool(self.memory.productive_branch_family_wobble_tolerated),
            productive_branch_family_wobble_reason=str(self.memory.productive_branch_family_wobble_reason),
            structured_branch_persistence_used=bool(self.memory.structured_branch_persistence_used),
            structured_branch_persistence_steps=int(self.memory.structured_branch_persistence_steps),
            structured_branch_persistence_steps_remaining=int(self.memory.structured_branch_persistence_steps_remaining),
            structured_branch_persistence_kept_control=bool(self.memory.structured_branch_persistence_kept_control),
            structured_branch_persistence_break_reason=str(self.memory.structured_branch_persistence_break_reason),
            finish_bound_branch_signature=str(self.memory.finish_bound_branch_signature),
            finish_bound_branch_source=str(self.memory.finish_bound_branch_source),
            finish_bound_branch_anchor=self.memory.finish_bound_branch_anchor,
            finish_bound_branch_action_name=str(self.memory.finish_bound_branch_action_name),
            finish_bound_branch_kept_control=bool(self.memory.finish_bound_branch_kept_control),
            post_breakthrough_priority_preserved=bool(self.memory.post_breakthrough_priority_preserved),
            post_breakthrough_preempt_block_reason=str(self.memory.post_breakthrough_preempt_block_reason),
            post_breakthrough_local_branch_reused=bool(self.memory.post_breakthrough_local_branch_reused),
            post_breakthrough_local_branch_break_reason=str(self.memory.post_breakthrough_local_branch_break_reason),
            productive_branch_collision_recovery_used=bool(self.memory.productive_branch_collision_recovery_used),
            productive_branch_collision_recovery_variant=str(self.memory.productive_branch_collision_recovery_variant),
            productive_branch_collision_recovery_preserved_family=bool(self.memory.productive_branch_collision_recovery_preserved_family),
            productive_window_extended=bool(self.memory.productive_window_extended),
            productive_window_extension_reason=str(self.memory.productive_window_extension_reason),
            productive_branch_preempt_blocked=bool(self.memory.productive_branch_preempt_blocked),
            productive_branch_preempt_block_reason=str(self.memory.productive_branch_preempt_block_reason),
            collision_soft_retry_used=bool(self.memory.collision_soft_retry_used),
            collision_soft_retry_variant=str(self.memory.collision_soft_retry_variant),
            regrounded_winning_coords=self.memory.regrounded_winning_coords,
            regrounding_delta=self.memory.regrounding_delta,
            regrounding_confidence=float(self.memory.regrounding_confidence),
            regrounding_used=bool(self.memory.regrounding_used),
            regrounding_failed_reason=str(self.memory.regrounding_failed_reason),
            levelup_relocalization_llm_attempted=bool(self.memory.levelup_relocalization_llm_attempted),
            levelup_relocalization_llm_used=bool(self.memory.levelup_relocalization_llm_used),
            levelup_relocalization_llm_confidence=float(self.memory.levelup_relocalization_llm_confidence),
            exploitation_noop_blacklist_checks=int(self.memory.exploitation_noop_blacklist_checks),
            exploitation_noop_blacklist_hits=int(self.memory.exploitation_noop_blacklist_hits),
            mechanism_detector_triggered=str(self.memory.mechanism_detector_triggered),
            llm_top_productive_family=max(self.controller.memory.llm_family_payoff.items(), key=lambda x: x[1].get("changed", 0))[0] if self.controller.memory.llm_family_payoff else "",
            llm_top_unproductive_family=max(self.controller.memory.llm_family_payoff.items(), key=lambda x: (x[1].get("noop", 0) + x[1].get("death", 0)))[0] if self.controller.memory.llm_family_payoff else "",
            llm_family_payoff_summary={
                fam: {
                    "changed": data.get("changed", 0),
                    "progress": data.get("progress", 0),
                    "noop": data.get("noop", 0),
                    "death": data.get("death", 0),
                    "score": data.get("changed", 0) + 3 * data.get("progress", 0) - data.get("noop", 0) - 2 * data.get("death", 0),
                }
                for fam, data in self.controller.memory.llm_family_payoff.items()
            },
            reasoner_consulted_this_step=bool(self.pending_reasoning.get("reasoner_consulted_this_step", False)),
            reasoner_parsed_abduction_this_step=bool(self.pending_reasoning.get("reasoner_parsed_abduction_this_step", False)),
            reasoner_surviving_proposals_this_step=int(self.pending_reasoning.get("reasoner_surviving_proposals_this_step", 0)),
            final_action_from_llm=bool(self.pending_reasoning.get("final_action_from_llm", False)),
            final_action_source=str(self.pending_reasoning.get("final_action_source", self.pending_action.source if self.pending_action else "")),
            stagnation_override_used=bool(self.pending_reasoning.get("stagnation_override_used", False)),
            competing_hypothesis_count=int(self.pending_reasoning.get("competing_hypothesis_count", 0)),
            stuck_mode_activations=self.controller.stuck_mode_activations,
            action_semantics_summary=dict(self.pending_reasoning.get("action_semantics", {})) if isinstance(self.pending_reasoning.get("action_semantics", {}), dict) else {"rows": self.pending_reasoning.get("action_semantics", [])},
        )
        self.diagnostic_logger.record(record)
        # Pulse reset mechanism_shift_event so it only fires on the single shift step
        self.controller.memory.mechanism_shift_event = False

        # Trace evidence showed lock-starvation in bp35; keeping strict fallback loop metrics
        self.controller.steps_since_progress += 1
        
        if event.level_delta > 0 or event.win:
            self.controller.consecutive_fallback_steps = 0
            self.controller.fallback_loop_streak = 0
            self.controller.counterfactual_fallback_streak = 0
            self.controller.alignment_fallback_streak = 0
            self.controller.steps_since_progress = 0
            self.controller.progress_events_this_level += 1
            self.controller.failed_consultations_this_level = 0
            self.controller.recent_lock_skip_count = 0
            self.controller.reasoner_lock_backoff_steps = 0
            self.controller.reasoner_suppressed = False
            self.controller.reasoner.recent_abductions.clear()
        else:
            stage = str(self.pending_reasoning.get("stage", ""))
            if "fallback" in stage:
                self.controller.consecutive_fallback_steps += 1
                self.controller.fallback_loop_streak += 1
                if "counterfactual" in stage:
                    self.controller.counterfactual_fallback_streak += 1
                if "alignment" in stage:
                    self.controller.alignment_fallback_streak += 1
            else:
                self.controller.consecutive_fallback_steps = 0
                self.controller.fallback_loop_streak = 0
                self.controller.counterfactual_fallback_streak = 0
                self.controller.alignment_fallback_streak = 0
                
        # Progress density is used to track the ratio of productive progress to total steps
        self.controller.progress_density = self.controller.progress_events_this_level / max(1, self.step_index)

        if event.topology_change:
            self.dead.invalidate_on_phase_change()
        if self.pending_action.predicted_state_key and self.pending_action.predicted_state_key != scene.exact_key:
            self.controller.plan_queue.clear()
        if self.pending_action.predicted_effect and event.no_op:
            self.controller.plan_queue.clear()
        return event

    def _mandatory_reset(self, latest_frame: FrameData, stage: str, extra: Mapping[str, Any] | None = None) -> GameAction:
        legal = _normalize_legal_actions(latest_frame)
        reset = next((a for a in legal if a.name == "RESET"), None)
        if reset is None:
            # Initial and terminal states are specified to require RESET. Keep a
            # compatibility fallback for toolkit versions that omit it from the
            # metadata while still refusing to enable arbitrary ACTION1..7.
            reset = GameAction.RESET
        try:
            reset.reasoning = {"stage": stage, **dict(extra or {})}
        except Exception:
            pass
        return reset

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        decision_started = time.monotonic()
        if hasattr(self, "game_id"):
            self.runtime.game_id = self.game_id
            if hasattr(self, "controller") and hasattr(self.controller, "reasoner"):
                self.controller.reasoner.game_id = self.game_id
        
        self.step_index += 1
        if bool(getattr(latest_frame, "full_reset", False)):
            self._full_reset()

        state_name = self._state_name(latest_frame)
        if state_name in {"NOT_PLAYED", "NOT_STARTED"}:
            new_level = int(getattr(latest_frame, "levels_completed", 0))
            self._level_reset(new_level)
            return self._mandatory_reset(latest_frame, "mandatory_reset", {"state": state_name})

        sequence = _extract_frame_sequence(latest_frame)
        observed_level = int(getattr(latest_frame, "levels_completed", 0))
        previous_level = self.current_level
        previous_grid = None if self.pending_scene is None else self.pending_scene.grid
        animation = _analyze_animation(previous_grid, sequence)

        if state_name == "GAME_OVER":
            scene = self.perception.perceive(
                sequence.settled, observed_level, self.pending_action)
            fatal_action = None if self.pending_action is None else self.pending_action.name
            event = self._record_pending_transition(
                scene, latest_frame, self.current_level, sequence, animation)
            self.diagnostic_logger.flush(
                str(getattr(self, "game_id", "unknown")))
            self._attempt_reset(observed_level)
            return self._mandatory_reset(
                latest_frame,
                "mandatory_reset_after_death",
                {"fatal_action": fatal_action, "death_recorded": bool(
                    event and event.game_over)},
            )

        level_changed = observed_level != self.current_level
        scene = self.perception.perceive(
            sequence.settled, observed_level, self.pending_action)
        event = self._record_pending_transition(
            scene, latest_frame, previous_level, sequence, animation)
            
        if event is not None and (event.level_delta > 0 or event.win):
            if self.pending_action is not None:
                if str(self.pending_action.source).startswith("milestone_model"):
                    self.reasoner.llm_action_progress += 1
                elif self.pending_action.program_id is not None:
                    prog = self.programs.programs.get(self.pending_action.program_id)
                    if prog is not None and str(prog.source).startswith("llm_abduction"):
                        self.reasoner.llm_program_progress += 1
                        
            active_goals = [g for g in self.goals.hypotheses.values() if g.status == "active"]
            if active_goals:
                current_top = max(active_goals, key=lambda g: g.confidence)
                if str(current_top.source).startswith("llm_abduction"):
                    self.reasoner.llm_abduction_prior_shift_progress += 1

        if level_changed:
            self._level_reset(observed_level, first_scene=None,
                              preserve_perception=False)
            scene = self.perception.perceive(
                sequence.settled, observed_level, None)
            self.goals.reset_level(scene)
            self.current_level = observed_level
        elif not self.goals.hypotheses:
            self.goals.reset_level(scene)

        legal_actions = _normalize_legal_actions(latest_frame)
        legal_names = {a.name for a in legal_actions}
        action_semantics_summary = self.dynamics.summary_for_prompt(legal_names)
        if not legal_actions:
            # Fail closed: do not invent ACTION1..7. RESET is the only safe
            # compatibility action, and the reason is visible in the trace.
            return self._mandatory_reset(latest_frame, "no_advertised_legal_actions")

        if self.memory.no_op_streak >= self.config.severe_stagnation_steps and "RESET" in legal_names:
            spec = ActionSpec(name="RESET", source="stagnation_recovery")
            self.pending_scene = scene
            self.pending_action = spec
            self.pending_was_probe = False
            self.pending_signature = _action_signature(scene, spec)
            return self._to_game_action(
                spec,
                legal_actions,
                {"stage": "controlled_stagnation_reset",
                    "noop_streak": self.memory.no_op_streak},
            )


        self.memory.level_steps += 1
        runtime_tier = self.runtime.tier()
        spec, was_probe, decision = self.controller.choose(
            scene,
            legal_actions,
            self.step_index,
            response_frames=sequence.grids,
            runtime_tier=runtime_tier,
            remaining_sec=self.runtime.remaining_sec(),
        )
        decision = {**decision, "runtime_tier": runtime_tier}
        signature = _action_signature(scene, spec)
        if self.config.enable_dead_signatures and self.dead.is_dead(signature):
            profile = _runtime_profile(runtime_tier, self.config)
            alternatives = [
                c for c in self.generator.generate(scene, legal_actions, use_hypotheses=profile.use_hypotheses)
                if not self.dead.is_dead(c.signature)
            ]
            aligned_alternatives: list[tuple[float, Candidate]] = []
            for candidate in alternatives:
                prediction = self.controller._predict(
                    scene, candidate.spec, profile)
                gate = self.controller._verify(
                    scene, candidate.spec, legal_names, prediction, candidate.is_probe, profile
                )
                if gate.allowed:
                    aligned_alternatives.append(
                        (candidate.score + gate.score, candidate))
            if aligned_alternatives:
                aligned_alternatives.sort(key=lambda row: row[0], reverse=True)
                best = aligned_alternatives[0][1]
                spec, was_probe = best.spec, best.is_probe
                signature = best.signature
                decision = {"stage": "dead_signature_and_alignment_guard"}

        self.generator.mark_committed(spec)
        self.pending_scene = scene
        self.pending_action = spec
        self.pending_was_probe = was_probe
        self.pending_signature = signature
        self.last_state = latest_frame.state

        # Determine truthful mechanism alignment for chosen action via semantic ontology
        chosen_source = str(decision.get("final_action_source", spec.source))
        
        # Resolve true program family/kind from metadata or program library
        chosen_kind = str(decision.get("llm_family") or decision.get("program_kind") or "")
        if not chosen_kind and spec.program_id:
            programs_lib = getattr(self.controller, "programs", None) or getattr(self, "programs", None)
            if programs_lib is not None and hasattr(programs_lib, "programs"):
                prog = programs_lib.programs.get(spec.program_id)
                if prog is not None:
                    chosen_kind = getattr(prog, "kind", "")

        top_fam = self.controller.memory.top_mechanism_family
        prio_fams = self.controller.memory.priority_program_families
        top_kinds = MECHANISM_TO_PROGRAM_KINDS.get(top_fam, set())
        
        is_mech_aligned = False
        if chosen_kind:
            if chosen_kind == top_fam or chosen_kind in top_kinds:
                is_mech_aligned = True
            elif any(chosen_kind == pf or chosen_kind in MECHANISM_TO_PROGRAM_KINDS.get(pf, set()) for pf in prio_fams):
                is_mech_aligned = True
        elif top_fam == "movement_control" and spec.name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
            is_mech_aligned = True
        elif top_fam == "targeted_recolor" and spec.name == "ACTION6":
            is_mech_aligned = True
        elif chosen_source in ("macro_replay", "llm_follow_through", "instant_reflex"):
            is_mech_aligned = True
            
        decision["mechanism_aligned"] = is_mech_aligned

        reasoning: dict[str, Any] = {
            **self.controller.llm_step_meta,
            **decision,
            "source": spec.source,
            "final_action_source": decision.get("final_action_source", spec.source),
            "final_action_from_llm": decision.get("final_action_from_llm", False),
            "model_called": decision.get("reasoner_consulted_this_step", self.controller.llm_step_meta.get("reasoner_consulted_this_step", False)),
            "llm_rejected_by_alignment": self.reasoner.llm_rejected_by_alignment,
            "llm_rejected_by_semantic_gate": self.reasoner.llm_rejected_by_semantic_gate,
            "llm_rejected_by_dead_signature": self.reasoner.llm_rejected_by_dead_signature,
            "llm_rejected_by_probe_budget": self.reasoner.llm_rejected_by_probe_budget,
            "llm_rejected_by_coordinate_fatigue": self.reasoner.llm_rejected_by_coordinate_fatigue,
            "llm_escape_hatch_used": self.controller.memory.escape_hatch_used,
            "llm_semantic_override_attempts": self.reasoner.llm_semantic_override_attempts,
            "llm_semantic_override_used": self.controller.llm_step_meta.get("llm_semantic_override_used", False),
            "llm_semantic_override_rejections": self.reasoner.llm_semantic_override_rejections,
            "llm_breakout_used_this_step": self.controller.llm_step_meta.get("llm_breakout_used_this_step", False),
            "llm_override_eligible": self.controller.llm_step_meta.get("llm_override_eligible", False),
            "llm_override_block_reason": self.controller.llm_step_meta.get("llm_override_block_reason", ""),
            "llm_severe_stagnation_signal": self.controller.llm_step_meta.get("llm_severe_stagnation_signal", ""),
            "llm_follow_through_active": self.controller.memory.llm_follow_through_window > 0,
            "llm_follow_through_family": self.controller.memory.llm_follow_through_family,
            "llm_promoted_programs_count": len(self.controller.memory.llm_promoted_program_ids),
            "churn_stagnation_detected": self.controller.llm_step_meta.get("llm_severe_stagnation_signal") == "progressless_churn",
            "follow_through_led_to_progress": self.controller.memory.follow_through_led_to_progress,
            "early_breakout_acceleration_used": "early" in str(self.controller.llm_step_meta.get("llm_severe_stagnation_signal", "")),
            "level_budget_pressure": round(min(2.0, self.controller.memory.level_steps / 120.0), 3),
            "level_steps_elapsed": self.controller.memory.level_steps,
            "level_budget_target": 120,
            "level_budget_escalation_used": self.controller.memory.level_steps > 120 and self.controller.memory.level_progress_events == 0,
            "macro_replay_active": bool(decision.get("macro_replay_active", False)),
            "macro_replay_steps_remaining": len(self.controller.memory.macro_replay_queue),
            "macro_replay_family": self.controller.memory.macro_replay_family,
            "macro_replay_program_id": self.controller.memory.macro_replay_program_id,
            "macro_replay_aborted_reason": self.controller.memory.macro_replay_aborted_reason,
            "level_transition_transfer_used": self.controller.memory.level_transition_transfer_used,
            "transferred_mechanism_family": self.controller.memory.transferred_mechanism_family,
            "early_classified_mechanism": self.controller.memory.early_classified_mechanism,
            "instant_reflex_used": bool(decision.get("instant_reflex_used", False)),
            "counterfactual_pruned_count": self.controller.memory.counterfactual_pruned_count,
            "subgoal_distance_reward": float(decision.get("subgoal_reward", 0.0)),
            "negative_hypothesis_eliminations": self.controller.memory.negative_hypothesis_eliminations,
            "mode": self.controller.memory.mode,
            "top_mechanism_family": self.controller.memory.top_mechanism_family,
            "top_mechanism_confidence": self.controller.memory.top_mechanism_confidence,
            "competing_mechanism_families": self.controller.memory.competing_mechanism_families,
            "mechanism_family_scores": dict(self.controller.memory.mechanism_scores),
            "priority_program_families": list(self.controller.memory.priority_program_families),
            "invariants_to_preserve": list(self.controller.memory.invariants_to_preserve),
            "mechanism_shift_event": self.controller.memory.mechanism_shift_event,
            "mechanism_aligned_action": bool(decision.get("mechanism_aligned", False)),
            "mechanism_discriminating_probe_used": bool(decision.get("is_discriminating_probe", False)),
            "llm_top_productive_family": max(self.controller.memory.llm_family_payoff.items(), key=lambda x: x[1].get("changed", 0))[0] if self.controller.memory.llm_family_payoff else "",
            "llm_top_unproductive_family": max(self.controller.memory.llm_family_payoff.items(), key=lambda x: (x[1].get("noop", 0) + x[1].get("death", 0)))[0] if self.controller.memory.llm_family_payoff else "",
            "llm_family_payoff_summary": {
                fam: {
                    "changed": data.get("changed", 0),
                    "progress": data.get("progress", 0),
                    "noop": data.get("noop", 0),
                    "death": data.get("death", 0),
                    "score": data.get("changed", 0) + 3 * data.get("progress", 0) - data.get("noop", 0) - 2 * data.get("death", 0),
                }
                for fam, data in self.controller.memory.llm_family_payoff.items()
            },
            "predicted": spec.predicted_effect,
            "program_id": spec.program_id,
            "predicted_state": spec.predicted_state_key,
            "goal_ids": list(spec.goal_ids),
            "probe": was_probe,
            "level": observed_level,
            "step": self.step_index,
            "entity_confidence": round(scene.entity_confidence, 3),
            "field_mode": scene.field_mode,
            "changed": len(scene.changed_cells),
            "animation_frames": animation.frame_count,
            "animation_transient": animation.transient_changed_count,
            "probe_count": self.memory.probes_this_level,
            "model_calls": self.reasoner.calls,
            "active_goals": self.goals.summary(4) if _runtime_profile(runtime_tier, self.config).use_goals else [],
            "verified_programs": self.programs.summary(4) if _runtime_profile(runtime_tier, self.config).use_programs else [],
            "action_semantics": action_semantics_summary,
            "runtime_tier": runtime_tier,
            "runtime_remaining_sec": self.runtime.remaining_sec(),
            "runtime_projected_remaining_sec": self.runtime.projected_remaining_sec(),
            "runtime_projected_margin_ratio": self.runtime.projected_safety_margin_ratio(),
            "runtime_completed_games": self.runtime.completed_games(),
            "runtime_game_p95_sec": self.runtime.measured_game_p95_sec(),
            "model_calls_this_level": self.reasoner.calls_this_level,
            "model_latency_last_sec": round(self.reasoner.last_latency_sec, 4),
            "same_family_streak": self.memory.same_family_streak,
            "counterfactual_streak": self.controller.counterfactual_streak,
            "transitions_until_stabilized": self.goals.transitions_until_stabilized,
            "top_goal_changes_this_level": self.goals.top_goal_changes_this_level,
            "prior_adjustments_this_level": self.goals.prior_adjustments_this_level,
            "stabilized_before_progress": self.goals.stabilized_before_progress,
            "puzzle_family_switch_counter": self.goals.puzzle_family_switch_counter,
            "churn_score": round((self.goals.top_goal_changes_this_level + self.goals.puzzle_family_switch_counter) / max(1, self.goals.transitions_processed), 3),
            "action_family_entropy": round(self.memory.action_family_entropy(), 3),
            "longest_family_streak": self.memory.longest_family_streak(),
            "llm_abductions_parsed": self.reasoner.abductions_parsed,
            "llm_abductions_selected": self.reasoner.abductions_selected,
            "llm_action_progress": self.reasoner.llm_action_progress,
            "llm_program_progress": self.reasoner.llm_program_progress,
            "llm_abduction_prior_shift_progress": self.reasoner.llm_abduction_prior_shift_progress,
            "model_decisions_used": self.reasoner.model_decisions_used,
            "model_decisions_used_this_level": self.reasoner.model_decisions_used_this_level,
            "reasoner_decision_attempts": self.reasoner.reasoner_decision_attempts,
            "reasoner_decision_skips_by_reason": self.reasoner.reasoner_decision_skips_by_reason,
            "reasoner_decision_successes": self.reasoner.reasoner_decision_successes,
            "reasoner_consultations_this_level": self.reasoner.reasoner_consultations_this_level,
            "llm_impossible_hypothesis_rejections": self.reasoner.llm_impossible_hypothesis_rejections,
            "llm_unsupported_family_rejections": self.reasoner.llm_unsupported_family_rejections,
            "llm_family_repeat_rejections": self.reasoner.llm_family_repeat_rejections,
            "llm_schema_valid_but_unexecutable": self.reasoner.llm_schema_valid_but_unexecutable,
            "llm_filtered_for_action_mismatch": self.reasoner.llm_filtered_for_action_mismatch,
            "competing_hypothesis_count": sum(
                1
                for g in self.goals.active(4)
                if self.goals.active(4)
                and g.confidence >= max(x.confidence for x in self.goals.active(4)) - 0.08
            ) if self.goals.hypotheses else 0,
            "stuck_mode_activations": self.controller.stuck_mode_activations,
        }
        if event is not None:
            reasoning["previous_event"] = {
                "changed": event.changed_count,
                "cumulative_changed": event.cumulative_changed_count,
                "transient": event.transient_changed_count,
                "subframes": event.subframe_count,
                "noop": event.no_op,
                "settled_noop": event.settled_no_op,
                "progress": event.level_delta,
                "death": event.game_over,
                "topology": event.topology_change,
            }
        self.pending_reasoning = dict(reasoning)
        self.pending_decision_latency_sec = max(
            0.0, time.monotonic() - decision_started)
            
        if spec.program_id is not None:
            prog = self.programs.programs.get(spec.program_id)
            if prog is not None and str(prog.source).startswith("llm_abduction"):
                self.reasoner.abductions_selected += 1
                
        return self._to_game_action(spec, legal_actions, reasoning)
