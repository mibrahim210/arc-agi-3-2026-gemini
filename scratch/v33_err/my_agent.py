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
        "DEWMA_MODEL_PATH", "/kaggle/input/gemma4-e4b-gguf/gemma4-e4b.gguf").strip()
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

    enable_a8_virtual_first: bool = _env_bool("DEWMA_A8_VIRTUAL_FIRST", True)
    enable_a8_bootstrap_protection: bool = _env_bool(
        "DEWMA_A8_BOOTSTRAP_PROTECTION", False)
    a8_bootstrap_min_cardinal_vectors: int = _env_int(
        "DEWMA_A8_BOOTSTRAP_MIN_CARDINAL_VECTORS", 2)
    a8_bootstrap_min_distinct_simple_actions: int = _env_int(
        "DEWMA_A8_BOOTSTRAP_MIN_DISTINCT_SIMPLE_ACTIONS", 4)
    a8_max_candidates_per_decision: int = _env_int(
        "DEWMA_A8_MAX_CANDIDATES_PER_DECISION", 16)
    a8_min_support: int = _env_int("DEWMA_A8_MIN_SUPPORT", 2)
    a8_reject_confidence: float = _env_float(
        "DEWMA_A8_REJECT_CONFIDENCE", 0.90)
    a8_prefer_confidence: float = _env_float(
        "DEWMA_A8_PREFER_CONFIDENCE", 0.65)
    a8_min_useful_utility: float = _env_float(
        "DEWMA_A8_MIN_USEFUL_UTILITY", 0.15)
    a8_max_hypotheses_per_candidate: int = _env_int(
        "DEWMA_A8_MAX_HYPOTHESES", 6)
    a8_defer_predicted_noop: bool = _env_bool(
        "DEWMA_A8_DEFER_PREDICTED_NOOP", True)
    a8_defer_predicted_death: bool = _env_bool(
        "DEWMA_A8_DEFER_PREDICTED_DEATH", True)

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
    program_cellular_induction_interval: int = _env_int(
        "DEWMA_PROGRAM_CELLULAR_INTERVAL", 4)

    trace_enabled: bool = _env_bool("DEWMA_TRACE_ENABLED", True)
    trace_to_disk: bool = _env_bool("DEWMA_TRACE_TO_DISK", True)
    trace_dir: str = os.getenv("DEWMA_TRACE_DIR", "./traces")
    trace_max_records: int = _env_int("DEWMA_TRACE_MAX_RECORDS", 2048)
    session_limit_sec: int = _env_int(
        "DEWMA_SESSION_LIMIT_SEC", int(8.0 * 3600))
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
class ReasonerProposal:
    action: ActionSpec | None
    program_spec: Mapping[str, Any] | None = None


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

    def summary(self) -> dict[str, Any]:
        hist = self.color_histogram()
        changed = self.changed_cells()
        return {
            "shape": self.shape(),
            "colors": hist,
            "changed_count": len(changed),
            "changed_bbox": _bbox(changed),
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
        self.probes_this_level = 0
        self.total_mental_eliminations = 0

    def reset_level(self) -> None:
        self.transitions.clear()
        self.visits.clear()
        self.actions_by_state.clear()
        self.progress_actions.clear()
        self.no_op_streak = 0
        self.probes_this_level = 0

    def reset_attempt(self) -> None:
        # Preserve learned transitions, state-action counts, global outcomes, and
        # the level-wide probe budget across death/restart attempts.
        self.no_op_streak = 0

    def record(self, transition: Transition, was_probe: bool) -> None:
        self.transitions.append(transition)
        self.visits[transition.after_exact] += 1
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

    def recent_state_loop(self, window: int) -> bool:
        if len(self.transitions) < max(4, window // 2):
            return False
        recent = [t.after_exact for t in list(self.transitions)[-window:]]
        return len(set(recent)) <= max(2, len(recent) // 3)

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
        self.level_start_grid: np.ndarray | None = None
        self.level_start_components = 0
        self._counter = 0
        self._value_cache: dict[str, float] = {}

    def reset_level(self, scene: Scene | None = None) -> None:
        self.hypotheses.clear()
        self._value_cache.clear()
        self.level_start_grid = None if scene is None else scene.grid.copy()
        self.level_start_components = 0 if scene is None else len(
            scene.components)
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

    def _trim(self) -> None:
        if len(self.hypotheses) <= self.config.max_goal_hypotheses:
            return
        ranked = sorted(
            self.hypotheses.values(),
            key=lambda g: (g.status == "active", g.confidence,
                           g.support - g.contradictions),
            reverse=True,
        )[: self.config.max_goal_hypotheses]
        self.hypotheses = {g.goal_id: g for g in ranked}

    def seed(self, scene: Scene) -> None:
        if self.level_start_grid is None:
            self.level_start_grid = scene.grid.copy()
            self.level_start_components = len(scene.components)
        non_background = [(c, n)
                          for c, n in scene.color_counts if c != scene.background]
        non_background.sort(key=lambda item: item[1])
        for index, (color, count) in enumerate(non_background[:4]):
            self._add(
                "collect_color",
                {"color": color, "baseline_count": count},
                0.20 if count <= 12 else 0.10,
                "rare_quantity_prior",
            )
            self._add(
                "reach_or_touch_color",
                {"color": color},
                0.24 if count <= 9 else 0.12,
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
            x1, y1 = self.payload.get("p1", (0, 0))
            x2, y2 = self.payload.get("p2", (0, 0))
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
            x1, y1 = self.payload.get("p1", (0, 0))
            x2, y2 = self.payload.get("p2", (0, 0))
            dx, dy = x2 - x1, y2 - y1
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

    def _verify(self, program: WorldProgram, transition: Transition, before_scene: Scene | None = None) -> None:
        if transition.before_grid is None or transition.after_grid is None:
            return
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
            return
        predicted = program.simulate(
            transition.before_grid, transition.action, before_scene)
        if predicted is None or predicted.shape != transition.after_grid.shape:
            return
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
        # Verify new and existing applicable programs against recent replay.
        for program in candidates:
            for replay in list(self.recent_transitions)[-48:]:
                self._verify(program, replay,
                             before if replay is transition else None)
        for program in list(self.programs.values()):
            if program not in candidates and program.status != "rejected":
                self._verify(program, transition, before)
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
        allowed = {"translation", "color_map",
                   "component_delete", "component_recolor"}
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
                payload = {"mapping": {int(k): int(
                    v) for k, v in dict(params["mapping"]).items()}}
            elif kind == "component_delete":
                payload = {"background": int(params.get("background", 0)), "clicked_color": int(
                    params.get("clicked_color", -1))}
            else:
                payload = {"target_color": int(params["target_color"]), "background": int(
                    params.get("background", 0)), "clicked_color": int(params.get("clicked_color", -1))}
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

    def reset_all(self) -> None:
        self.vectors.clear()

    def record(self, action: ActionSpec, event: Event, controlled_id: str | None) -> None:
        if action.data or controlled_id is None:
            return
        for entity_id, dx, dy in event.entity_moves:
            if entity_id == controlled_id and (dx or dy):
                # Normalize large sprite shifts to one cardinal/diagonal step.
                ndx = 0 if dx == 0 else (1 if dx > 0 else -1)
                ndy = 0 if dy == 0 else (1 if dy > 0 else -1)
                self.vectors[action.name][(ndx, ndy)] += 1

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
    def controlled_assembly(self, scene: Scene) -> ControlledAssembly | None:
        if scene.controlled_entity_id:
            e = next((x for x in scene.entities if x.entity_id ==
                     scene.controlled_entity_id), None)
            if e:
                return ControlledAssembly(
                    centroid=e.centroid,
                    cells=e.cells,
                    bbox=e.bbox,
                    member_entity_ids=(e.entity_id,),
                    representative_entity_id=e.entity_id,
                    confidence=1.0,
                )
        return None

    def diagnostics(self, scene: Scene) -> dict[str, Any]:
        return {
            "controlled_entity_id": scene.controlled_entity_id,
            "entity_confidence": scene.entity_confidence,
        }


class TerrainPassabilityModel:
    def passable_colors(self, scene: Scene) -> set[int]:
        return {scene.background}

    def passability_mask(self, scene: Scene) -> np.ndarray:
        return scene.grid == scene.background

    def diagnostics(self, scene: Scene) -> dict[str, Any]:
        return {
            "passable_colors": list(self.passable_colors(scene)),
        }


class PathPlanner:
    _CARDINAL_VECTORS: frozenset[tuple[int, int]] = frozenset(
        {(1, 0), (-1, 0), (0, 1), (0, -1)})

    def __init__(self, config: Config, dynamics: ActionDynamics, control_inference: ControlInference | None = None, passability: TerrainPassabilityModel | None = None) -> None:
        if control_inference is None:
            control_inference = ControlInference()
        if passability is None:
            passability = TerrainPassabilityModel()
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
        self._failed = False
        self._model = None

    @property
    def available(self) -> bool:
        return (
            self.config.enable_model
            and bool(self.config.model_path)
            and not self._failed
            and self.calls < self.config.model_call_budget
            and self.calls_this_level < self.config.model_call_budget_per_level
        )

    def reset_level(self) -> None:
        self.calls_this_level = 0
        self.last_call_step = -10_000

    def _load(self) -> bool:
        if self._model is not None:
            return True
        # Ping local Ollama server and auto-detect active model tag
        import urllib.request
        import json
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
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
            pass

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
        for candidate in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.S):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

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
        if not self.available:
            return None

        # Thread-safe non-blocking mutex lock for Ollama concurrency protection.
        # If another game thread is querying Ollama, do not block or crashâ€”
        # non-blocking acquire allows fast fallback to deterministic Candidate Generator!
        acquired = _OLLAMA_LOCK.acquire(blocking=False)
        if not acquired:
            return None

        started = time.monotonic()
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
                            "temperature": 0.0,
                            "num_predict": self.config.model_max_new_tokens
                        }
                    }).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                    text = result.get("response", "").strip()
                print(f"[Ollama LLM Call #{self.calls}] Tag: {getattr(self, 'detected_tag', 'unknown')} | Latency: {time.monotonic()-started:.2f}s | Output: {text[:80]}...", file=sys.stderr)
                return self._extract_json(text)
            else:
                # In-process llama_cpp inference
                res = self._model(
                    prompt,
                    max_tokens=self.config.model_max_new_tokens,
                    temperature=0.0,
                    stop=["}\n", "}\r\n"]
                )
                text = res["choices"][0]["text"].strip()
                return self._extract_json(text)
        except Exception as exc:
            print(f"[LLM Reasoner Transient Error] {exc}", file=sys.stderr)
            return None
        finally:
            _OLLAMA_LOCK.release()
            self.last_latency_sec = max(0.0, time.monotonic() - started)
            self.total_latency_sec += self.last_latency_sec

    def propose(
        self,
        scene: Scene,
        legal_names: Sequence[str],
        memory: TraceMemory,
        step: int,
        goals_summary: Sequence[Mapping[str, Any]] = (),
        programs_summary: Sequence[Mapping[str, Any]] = (),
        response_frames: Sequence[np.ndarray] = (),
    ) -> ReasonerProposal | None:
        if not self.can_call(step, milestone=True) or not self._load():
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
            for t in list(memory.transitions)[-8:]
        ]
        prompt = (
            "You control an unknown ARC-AGI-3 environment. You may inspect the "
            "current integer grid or execute Python scripts without spending an action, then return "
            "one legal action. Emit ONE JSON object only per response.\n"
            "Inspection schema: "
            '{"type":"inspect","expression":"state.find_color(3)"}\n'
            "Python execution schema: "
            '{"type":"python","code":"for t in recent_transitions: print(t[\'action\'])"}\n'
            "Final schema: "
            '{"type":"action","action":"ACTION1","x":null,"y":null,'
            '"confidence":0.0,"why":"..."}\n'
            "Allowed state methods: summary(), shape(), find_color(value), count(value), "
            "patch(x,y,radius), diff_last_frame(), changed_cells(), color_histogram(). "
            "Prefer stored evidence and reversible progress. Do not request a physical "
            "probe unless it can change the decision. Complex actions require x,y.\n"
            "In Python execution, `recent_transitions` (list of dicts) is available in the global namespace. Use print() to output results.\n"
            f"legal_actions={list(legal_names)}\n"
            f"state_summary={inspector.summary()}\n"
            f"entity_confidence={scene.entity_confidence:.3f}, field_mode={scene.field_mode}\n"
            f"recent_transitions={json.dumps(recent, separators=(',', ':'))}\n"
            f"candidate_goals={json.dumps(list(goals_summary), separators=(',', ':'), default=str)}\n"
            f"verified_programs={json.dumps(list(programs_summary), separators=(',', ':'), default=str)}\n"
            "Optional program schema: \"program\":{\"kind\":\"translation|color_map|component_delete|component_recolor\",\"action\":\"ACTION1\",\"params\":{...}}. "
            "A program is only a hypothesis and will be replay-verified before use.\n"
        )

        seen_expressions: set[str] = set()
        max_rounds = 8
        for round_index in range(max_rounds):
            if not self.available:
                return None
            obj = self._generate_json(prompt, step)
            if not obj:
                return None

            response_type = str(
                obj.get("type", "action" if "action" in obj else "")).lower()
            if response_type == "python":
                if round_index >= max_rounds - 1:
                    prompt += "python_denied=tool_round_limit_reached\nReturn final action JSON now.\n"
                    continue
                code = str(obj.get("code", ""))
                if not code or code in seen_expressions:
                    prompt += "python_error=empty_or_repeated_code\nReturn final action JSON now.\n"
                    continue
                seen_expressions.add(code)
                try:
                    result_str = repl.exec_python_script(code, list(memory.transitions))
                    prompt += (
                        f"python_{round_index + 1}_result:\n{result_str}\n"
                        "Return another inspection/python or the final action JSON.\n"
                    )
                except Exception as exc:
                    prompt += (
                        f"python_{round_index + 1}_error={type(exc).__name__}:"
                        f"{str(exc)[:180]}\nFix your python script or return the final action JSON.\n"
                    )
                continue

            if response_type == "inspect":
                if round_index >= self.config.model_tool_rounds:
                    prompt += "inspection_denied=tool_round_limit_reached\nReturn final action JSON now.\n"
                    continue
                expression = str(obj.get("expression", ""))[:240].strip()
                if not expression or expression in seen_expressions:
                    prompt += "inspection_error=empty_or_repeated_expression\nReturn final action JSON now.\n"
                    continue
                seen_expressions.add(expression)
                try:
                    result = self._json_safe(repl.run(expression))
                    rendered = json.dumps(result, separators=(",", ":"))
                    if len(rendered) > 2400:
                        rendered = rendered[:2400] + "...<truncated>"
                    prompt += (
                        f"inspection_{round_index + 1}_expression={expression!r}\n"
                        f"inspection_{round_index + 1}_result={rendered}\n"
                        "Return another inspection or the final action JSON.\n"
                    )
                except Exception as exc:
                    prompt += (
                        f"inspection_{round_index + 1}_error={type(exc).__name__}:"
                        f"{str(exc)[:180]}\nReturn final action JSON now.\n"
                    )
                continue

            name = str(obj.get("action", "")).upper()
            if name not in legal_names:
                return None
            data: tuple[tuple[str, int], ...] = ()
            if obj.get("x") is not None or obj.get("y") is not None:
                if obj.get("x") is None or obj.get("y") is None:
                    return None
                x, y = int(obj["x"]), int(obj["y"])
                if not (0 <= x < scene.width and 0 <= y < scene.height):
                    return None
                data = (("x", x), ("y", y))
            confidence = float(obj.get("confidence", 0.5))
            action_spec = ActionSpec(
                name=name,
                data=data,
                source="milestone_model_repl",
                predicted_effect=str(obj.get("why", "model proposal"))[:160],
                score=2.0 + max(0.0, min(1.0, confidence)),
                goal_ids=tuple(str(x) for x in obj.get("goal_ids", [])[
                               :4]) if isinstance(obj.get("goal_ids"), list) else (),
            )
            program_spec = obj.get("program") if isinstance(
                obj.get("program"), dict) else None
            return ReasonerProposal(action_spec, program_spec)
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
                rarity = 1.0 - min(1.0, color_freq.get(comp.color, 1.0) * 12.0)
                small = 1.0 / math.sqrt(max(1, comp.area))
                score = 2.0 * rarity + 1.1 * small + \
                    (0.2 if comp.touches_border else 0.0)
                score -= 0.20 * self.coordinate_visits[(x, y)]
                proposals.append((score, (x, y), "rare_small_component"))
                x0, y0, x1, y1 = comp.bbox
                for point in {(x0, y0), (x1, y0), (x0, y1), (x1, y1)}:
                    proposals.append((score - 0.35, point, "component_corner"))

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
                proposals.append(
                    (1.2 + rarity - 0.12 * self.coordinate_visits[point], point, "raw_rare_color"))

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

        # Color Boundary Entropy Anchors (Transitions between distinct colors)
        try:
            grid = scene.grid
            h_boundaries = (grid[:, :-1] != grid[:, 1:])
            v_boundaries = (grid[:-1, :] != grid[1:, :])
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

        # 2-Point Pair Macro Generator (Source Object -> Target Slot / Midpoint)
        try:
            if centroids:
                # Top source anchor (first component/object centroid)
                s_score, (sx, sy), s_why = centroids[0]
                # High-confidence destination targets (grid center, opposite corner, pair midpoint)
                dests = [
                    (scene.width // 2, scene.height // 2),
                    (max(0, scene.width - sx - 1), max(0, scene.height - sy - 1)),
                ]
                for dx, dy in dests:
                    if (dx, dy) != (sx, sy):
                        macro_score = 2.40 - 0.15 * self.coordinate_visits[(sx, sy)]
                        proposals.append((macro_score, (sx, sy), f"macro_pair_src:dest=({dx},{dy})"))
        except Exception:
            pass


        # Stagnation boost: when in a recent state loop or NO-OP streak, boost unvisited frontier locations
        is_stagnant = self.memory.recent_state_loop(12) or self.memory.no_op_streak >= 5
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
                for coord_score, (x, y), why in self._complex_coordinates(scene):
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


class CounterfactualPlanner:
    """Bounded beam search through verified programs on previously unseen states."""

    def __init__(
        self,
        config: Config,
        programs: ExecutableProgramLibrary,
        goals: GoalHypothesisManager,
        alignment: GoalAlignmentVerifier,
    ) -> None:
        self.config = config
        self.programs = programs
        self.goals = goals
        self.alignment = alignment

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
                    step_score = (
                        parent_score
                        + 2.4 * goal_delta
                        + 0.75 * new_confidence
                        - 0.32 * depth
                        - 0.8 * prediction.uncertainty
                        + novelty
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

    def reset_level(self) -> None:
        self.plan_queue.clear()
        self.last_model_milestone = ""
        self.last_response_frames = ()
        self.counterfactual_streak = 0

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
        if self.memory.no_op_streak >= 3:
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

    def _verify(
        self,
        scene: Scene,
        spec: ActionSpec,
        legal_names: set[str],
        prediction: ProgramPrediction | None,
        is_probe: bool,
        profile: RuntimeProfile,
    ) -> AlignmentDecision:
        if profile.use_alignment:
            return self.alignment.verify(scene, spec, legal_names, prediction, is_probe)
        if spec.name not in legal_names:
            return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("illegal_action",))
        data = spec.data_dict
        if ("x" in data) != ("y" in data):
            return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("incomplete_coordinates",))
        if "x" in data and not (0 <= data["x"] < scene.width and 0 <= data["y"] < scene.height):
            return AlignmentDecision(False, -10.0, -1.0, 1.0, (), ("coordinate_out_of_bounds",))
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
        return score, decision, prediction

    def choose(
        self,
        scene: Scene,
        legal_actions: Sequence[GameAction],
        step: int,
        response_frames: Sequence[np.ndarray] = (),
        runtime_tier: str = "A9",
    ) -> tuple[ActionSpec, bool, dict[str, Any]]:
        profile = _runtime_profile(runtime_tier, self.config)
        legal_names = {action.name for action in legal_actions}
        self.last_response_frames = tuple(response_frames)
        if not profile.use_programs:
            self.plan_queue.clear()

        # 0) Stagnation breakout: issue RESET if trapped in a 16+ NO-OP loop
        if self.memory.no_op_streak >= 16 and "RESET" in legal_names and step + 30 <= self.config.max_actions:
            self.memory.no_op_streak = 0
            self.plan_queue.clear()
            return ActionSpec(name="RESET", source="stagnation_breakout_reset"), False, {"stage": "stagnation_breakout_reset"}

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
            prediction = self._predict(scene, queued, profile)
            decision = self._verify(
                scene, queued, legal_names, prediction, False, profile)
            if decision.allowed and prediction is not None:
                spec = ActionSpec(
                    name=queued.name, data=queued.data, source="verified_plan_queue",
                    predicted_effect=prediction.expected_effect, score=queued.score,
                    program_id=prediction.program_id,
                    predicted_state_key=_stable_hash_bytes(
                        prediction.grid.tobytes()),
                    goal_ids=decision.goal_ids,
                )
                return spec, False, {"stage": "verified_plan_queue", "alignment": round(decision.score, 3)}
            self.plan_queue.clear()

        candidates = self.candidate_generator.generate(
            scene, legal_actions, use_hypotheses=profile.use_hypotheses)

        # 3) A9-only bounded counterfactual planning with anti-hijack grounding.
        is_looping = self.memory.recent_state_loop(self.config.loop_window)
        cf_plan = self.counterfactual.plan(
            scene, candidates, legal_names) if profile.use_counterfactual else None
        if cf_plan is not None:
            if self.counterfactual_streak >= 3 and (self.memory.no_op_streak > 0 or is_looping):
                cf_plan = None
                self.counterfactual_streak = 0
            else:
                self.counterfactual_streak += 1
                self.plan_queue.extend(cf_plan.remaining)
                return cf_plan.first_action, False, {
                    "stage": "counterfactual_program_search",
                    "program": cf_plan.prediction.program_id,
                    "program_kind": cf_plan.prediction.kind,
                    "alignment": round(cf_plan.alignment.score, 3),
                    "goal_delta": round(cf_plan.alignment.goal_delta, 3),
                    "goals": list(cf_plan.alignment.goal_ids),
                    "plan_length": 1 + len(cf_plan.remaining),
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
                return spec, prediction is None, {"stage": "path_planner", "alignment": round(decision.score, 3)}

        # 5) Replay/hash/hypothesis arbitration remains the A5 safety core.
        ranked: list[tuple[float, Candidate,
                           AlignmentDecision, ProgramPrediction | None]] = []
        for candidate in candidates:
            score, decision, prediction = self._aligned_known_candidate(
                scene, candidate, legal_names, profile)
            if decision.allowed and not candidate.is_probe:
                ranked.append((score, candidate, decision, prediction))
        if ranked:
            ranked.sort(key=lambda row: row[0], reverse=True)
            score, best, decision, prediction = ranked[0]
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
                "score": round(score, 3),
                "alignment": round(decision.score, 3),
                "goal_delta": round(decision.goal_delta, 3),
                "rationale": best.rationale[:4],
            }

        # 6) Milestone-gated local model for A7/A8/A9 only.
        milestone, milestone_name = self._milestone(scene, step)
        is_stagnant_now = is_looping or self.memory.no_op_streak >= 3
        if profile.use_model and self.reasoner.can_call(step, milestone, is_stagnant=is_stagnant_now):
            proposal = self.reasoner.propose(
                scene, sorted(legal_names), self.memory, step,
                goals_summary=self.goals.summary() if profile.use_goals else (),
                programs_summary=self.programs.summary() if profile.use_programs else (),
                response_frames=response_frames,
            )
            if proposal is not None:
                if proposal.program_spec is not None and profile.use_programs:
                    self.programs.ingest_model_program(
                        proposal.program_spec, list(self.memory.transitions))
                model_action = proposal.action
                if model_action is not None:
                    signature = _action_signature(scene, model_action)
                    is_probe = self.memory.tried_count(
                        scene.exact_key, model_action) == 0
                    prediction = self._predict(scene, model_action, profile)
                    decision = self._verify(
                        scene, model_action, legal_names, prediction, is_probe, profile)
                    if (
                        not self.dead.is_dead(signature)
                        and decision.allowed
                        and (not is_probe or self.memory.probes_this_level < self.config.max_physical_probes_per_level)
                    ):
                        self.last_model_milestone = milestone_name
                        spec = ActionSpec(
                            name=model_action.name, data=model_action.data, source=model_action.source,
                            predicted_effect=model_action.predicted_effect if prediction is None else prediction.expected_effect,
                            score=model_action.score + decision.score,
                            program_id=None if prediction is None else prediction.program_id,
                            predicted_state_key=None if prediction is None else _stable_hash_bytes(
                                prediction.grid.tobytes()),
                            goal_ids=decision.goal_ids,
                        )
                        return spec, is_probe, {
                            "stage": "milestone_model_goal_verified" if profile.use_alignment else "milestone_model_verified",
                            "milestone": milestone_name,
                            "alignment": round(decision.score, 3),
                        }

        # 7) Physical probes; EIG proxy comes from conditional hypothesis memory.
        probes: list[tuple[float, Candidate, AlignmentDecision]] = []
        if self.memory.probes_this_level < self.config.max_physical_probes_per_level or self.memory.no_op_streak >= 8:
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
                score = candidate.score + 0.65 * eig + decision.score - 0.45
                if self.memory.recent_state_loop(self.config.loop_window) or self.memory.no_op_streak >= 8:
                    score += 0.65
                probes.append((score, candidate, decision))
        if probes:
            probes.sort(key=lambda row: row[0], reverse=True)
            score, best, decision = probes[0]
            spec = ActionSpec(
                name=best.spec.name, data=best.spec.data,
                source="goal_discriminating_probe" if profile.use_goals else "hypothesis_discriminating_probe",
                predicted_effect=best.spec.predicted_effect, score=score, goal_ids=decision.goal_ids,
            )
            return spec, True, {
                "stage": "admissible_goal_discriminating_probe" if profile.use_goals else "admissible_hypothesis_probe",
                "score": round(score, 3),
                "probe_budget_used": self.memory.probes_this_level,
                "goals": list(decision.goal_ids),
                "rationale": best.rationale[:4],
            }

        # 8) Least harmful advertised action, with no invented actions.
        if candidates:
            safest: list[tuple[float, Candidate]] = []
            for candidate in candidates:
                prediction = self._predict(scene, candidate.spec, profile)
                decision = self._verify(
                    scene, candidate.spec, legal_names, prediction, candidate.is_probe, profile)
                safest.append(
                    (candidate.score + decision.score - decision.risk, candidate))
            safest.sort(key=lambda row: row[0], reverse=True)
            return safest[0][1].spec, safest[0][1].is_probe, {
                "stage": "alignment_constrained_fallback" if profile.use_alignment else "deterministic_fallback"
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

    MAX_ACTIONS = _env_int("DEWMA_MAX_ACTIONS", 600)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = Config(max_actions=self.MAX_ACTIONS)
        self.macro_queue: deque[ActionSpec] = deque()
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
        self.diagnostic_logger = DiagnosticTraceLogger(self.config)
        self.pending_reasoning: dict[str, Any] = {}
        self.pending_decision_latency_sec = 0.0

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
            self.config, self.programs, self.goals, self.alignment
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
        self.memory.record(transition, self.pending_was_probe)
        self.dead.record(self.pending_signature, event)
        if profile.learn_hypotheses:
            self.hypotheses.record(self.pending_signature, event)
        self.world_model.record(transition)
        self.spatial_hash.record(
            self.pending_scene, scene, self.pending_action, event)
        self.dynamics.record(self.pending_action, event,
                             self.pending_scene.controlled_entity_id)
        if profile.learn_goals:
            self.goals.update(transition, self.pending_scene, scene)
        if profile.learn_alignment:
            self.alignment.observe(transition, self.pending_scene, scene)
        if profile.learn_programs:
            self.programs.record(transition, self.pending_scene, scene)

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
                "stage", self.pending_action.source)),
            expected_effect=self.pending_action.predicted_effect,
            observed_effect=event.effect_signature,
            no_op=event.no_op,
            death=event.game_over,
            progress=bool(event.level_delta > 0 or event.win),
            physical_probe=self.pending_was_probe,
            model_called=bool(self.pending_reasoning.get(
                "stage", "").startswith("milestone_model")),
            local_patch_hash=local_patch_hash,
            active_goal_ids=active_goals,
            program_ids=verified_program_ids,
            representation_mode=representation_mode,
            representation_confidence=scene.entity_confidence,
            entity_deltas=self._diagnostic_entity_deltas(event),
            model_latency_sec=self.reasoner.last_latency_sec if self.pending_reasoning.get(
                "stage", "").startswith("milestone_model") else 0.0,
            deterministic_latency_sec=self.pending_decision_latency_sec,
            remaining_wall_time_sec=self.runtime.remaining_sec(),
            runtime_tier=str(self.pending_reasoning.get(
                "runtime_tier", self.runtime.tier())),
        )
        self.diagnostic_logger.record(record)

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

        # Process queued 2-step macro action if present
        if self.macro_queue:
            next_spec = self.macro_queue.popleft()
            return self._to_game_action(
                next_spec,
                legal_actions,
                {"stage": "macro_step_2_execution", "queue_len": len(self.macro_queue)},
            )

        runtime_tier = self.runtime.tier()
        spec, was_probe, decision = self.controller.choose(
            scene,
            legal_actions,
            self.step_index,
            response_frames=sequence.grids,
            runtime_tier=runtime_tier,
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

        reasoning: dict[str, Any] = {
            **decision,
            "source": spec.source,
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
            "runtime_tier": runtime_tier,
            "runtime_remaining_sec": self.runtime.remaining_sec(),
            "runtime_projected_remaining_sec": self.runtime.projected_remaining_sec(),
            "runtime_projected_margin_ratio": self.runtime.projected_safety_margin_ratio(),
            "runtime_completed_games": self.runtime.completed_games(),
            "runtime_game_p95_sec": self.runtime.measured_game_p95_sec(),
            "model_calls_this_level": self.reasoner.calls_this_level,
            "model_latency_last_sec": round(self.reasoner.last_latency_sec, 4),
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
        return self._to_game_action(spec, legal_actions, reasoning)
