"""Run `agent/my_agent.py` locally against a real ARC-AGI-3 game.

This is the fast inner-loop: no Docker, no Kaggle round-trip. Uses the
`arc-agi` PyPI package to host the game engine and the ARC-AGI-3-Agents
framework's `Agent.main()` loop to drive it — exactly what the Kaggle
gateway does, just in-process.

Usage:
    .venv/bin/python scripts/play_local.py --game ls20 --max-steps 200
    .venv/bin/python scripts/play_local.py --list
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.exists():
    raise SystemExit(f"Framework not found at {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode


def load_my_agent_class():
    """Import MyAgent from agent/my_agent.py via importlib."""
    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise SystemExit("agent/my_agent.py must define a class named `MyAgent`")
    return module.MyAgent


def play_one_game(MyAgentCls, arc, game_id: str, render_mode: str | None):
    env = arc.make(game_id, render_mode=render_mode)
    if env is None:
        return (game_id, "ENV_CREATE_FAILED", 0, 0)

    agent = MyAgentCls(
        card_id="local-dev",
        game_id=game_id,
        agent_name=f"MyAgent.local.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["local-dev"],
    )
    agent.main()

    final = agent.frames[-1]
    return (
        game_id,
        final.state,
        final.levels_completed,
        agent.action_counter,
    )


def write_run_summary(trace_dir: str, per_game, score_val) -> None:
    path = Path(trace_dir) / "summary.md"
    progressed = [gid for gid, _state, levels, _actions in per_game if levels > 0]
    lines = [
        "# Local Run Summary",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Aggregate scorecard score: {score_val}",
        f"- Games run: {len(per_game)}",
        f"- Games with progress: {len(progressed)}",
        f"- Progressed games: {', '.join(progressed) if progressed else 'None'}",
        "",
        "## Per-game Results",
        "",
        "| Game | Levels | Actions | State |",
        "| --- | ---: | ---: | --- |",
    ]
    for gid, state, levels, actions in sorted(per_game, key=lambda row: row[0]):
        lines.append(f"| {gid} | {levels} | {actions} | {state} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None,
                   help="Game id to play. If omitted, plays ALL available games "
                        "(mirrors what Kaggle does in competition rerun). "
                        "Comma-separated list also accepted, e.g. ls20,vc33.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Per-game cap on actions (overrides MyAgent.MAX_ACTIONS).")
    p.add_argument("--list", action="store_true",
                   help="List available games and exit.")
    p.add_argument("--render", default=None, choices=[None, "terminal"],
                   help="Optional terminal rendering each step.")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of games to run concurrently. Use >1 to better "
                        "approximate Kaggle swarm behavior.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Preserve every benchmark run by default. Individual trace shards and
    # llm_forensics logs will be written under a timestamped archive folder
    # unless the caller explicitly overrides DEWMA_TRACE_DIR.
    trace_dir = os.environ.get("DEWMA_TRACE_DIR", "").strip()
    if not trace_dir:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        trace_root = ROOT / "traces_archive" / run_stamp
        trace_root.mkdir(parents=True, exist_ok=True)
        os.environ["DEWMA_TRACE_DIR"] = str(trace_root)
        trace_dir = str(trace_root)
    print(f"Trace output directory: {trace_dir}\n")

    # NORMAL = local execution; game source is downloaded on first call into
    # ./environment_files/ and cached for subsequent runs.
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()

    if args.list:
        print(f"{len(all_envs)} environments:")
        for e in all_envs:
            print(f"  {e.game_id}: {getattr(e, 'title', '?')}")
        return

    # Resolve which games to play. `arc.make()` accepts the short game id
    # (e.g. "ls20") even though the EnvironmentInfo.game_id includes the
    # version suffix ("ls20-9607627b"), so we normalize to short ids.
    if args.game:
        wanted = {g.strip().split("-")[0] for g in args.game.split(",")}
        game_ids = [e.game_id.split("-")[0] for e in all_envs
                    if e.game_id.split("-")[0] in wanted]
        missing = wanted - set(game_ids)
        if missing:
            raise SystemExit(f"Unknown game id(s): {sorted(missing)}. Run --list.")
    else:
        game_ids = [e.game_id.split("-")[0] for e in all_envs]
        print(f"No --game specified; playing all {len(game_ids)} games "
              f"(this is what Kaggle does in competition rerun).\n")

    MyAgentCls = load_my_agent_class()
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = args.max_steps

    per_game = []
    if args.parallel <= 1:
        for i, game_id in enumerate(game_ids, 1):
            print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
            result = play_one_game(MyAgentCls, arc, game_id, args.render)
            per_game.append(result)
            _, state, levels, actions = result
            print(f"  -> state={state}, levels_completed={levels}, actions={actions}")
    else:
        print(f"Running {len(game_ids)} games with parallelism={args.parallel}\n")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_index = {
                executor.submit(play_one_game, MyAgentCls, arc, game_id, args.render): (i, game_id)
                for i, game_id in enumerate(game_ids, 1)
            }
            for future in as_completed(future_to_index):
                i, game_id = future_to_index[future]
                print(f"=== [{i}/{len(game_ids)}] {game_id} ===")
                result = future.result()
                per_game.append(result)
                _, state, levels, actions = result
                print(f"  -> state={state}, levels_completed={levels}, actions={actions}")

    sc = arc.get_scorecard()
    print("\n========= SUMMARY =========")
    for gid, state, levels, actions in per_game:
        print(f"  {gid:8} levels={levels:3}  actions={actions:5}  state={state}")
    score_val = sc.score if hasattr(sc, "score") else sc
    print(f"\nAggregate scorecard score: {score_val}")
    write_run_summary(trace_dir, per_game, score_val)
    print(f"Run summary written to: {Path(trace_dir) / 'summary.md'}")


if __name__ == "__main__":
    main()
