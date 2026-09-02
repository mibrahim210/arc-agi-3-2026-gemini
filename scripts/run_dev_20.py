"""Run the 20-Game Development & Regression Suite.

Includes the 5 solved calibration baselines (`tn36`, `vc33`, `lp85`, `r11l`, `lf52`)
plus the 15 active tuning games.

Usage:
    python scripts/run_dev_20.py
    python scripts/run_dev_20.py --max-steps 400 --parallel 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_20_GAMES = [
    "tn36", "vc33", "lp85", "r11l", "lf52",
    "ar25", "bp35", "cd82", "cn04", "ft09",
    "g50t", "ka59", "m0r0", "re86", "s5i5",
    "sc25", "sk48", "tr87", "tu93", "wa30"
]


def main():
    parser = argparse.ArgumentParser(description="Run 20-game development and regression suite.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum action steps per game (default: 400)")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel games (default: 4)")
    parser.add_argument("--render", action="store_true", help="Render gameplay window")
    args, unknown = parser.parse_known_args()

    games_arg = ",".join(DEV_20_GAMES)
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

    print(f"=== Running 20-Game Development & Regression Suite ===")
    print(f"Games ({len(DEV_20_GAMES)}): {games_arg}")
    print(f"Max Steps: {args.max_steps} | Parallel: {args.parallel}\n")

    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())
