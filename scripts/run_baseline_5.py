"""Run the 5-Game Calibration Baseline Check (`tn36`, `vc33`, `lp85`, `r11l`, `lf52`).

Usage:
    python scripts/run_baseline_5.py
    python scripts/run_baseline_5.py --max-steps 400 --parallel 5
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_5_GAMES = ["tn36", "vc33", "lp85", "r11l", "lf52"]


def main():
    parser = argparse.ArgumentParser(description="Run 5-game calibration baseline verification.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum action steps per game (default: 400)")
    parser.add_argument("--parallel", type=int, default=5, help="Number of parallel games (default: 5)")
    parser.add_argument("--render", action="store_true", help="Render gameplay window")
    args, unknown = parser.parse_known_args()

    games_arg = ",".join(BASELINE_5_GAMES)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "play_local.py"),
        "--game",
        games_arg,
        "--max-steps",
        str(args.max_steps),
        "--parallel",
        str(args.parallel),
    ]
    if args.render:
        cmd.append("--render")
    if unknown:
        cmd.extend(unknown)

    print(f"=== Running 5-Game Calibration Baseline Check ===")
    print(f"Games ({len(BASELINE_5_GAMES)}): {games_arg}")
    print(f"Max Steps: {args.max_steps} | Parallel: {args.parallel}\n")

    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
