from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.custom_env import ACTIONS, GovMateEnv  # noqa: E402
from training.common import make_raw_env  # noqa: E402


def best_run(algo: str | None) -> tuple[str, str]:
    """Return (algo, run_name) of the best finished sweep run."""
    rows = []
    for d in sorted((ROOT / "logs").glob("*_s[0-9][0-9]")):
        s = d / "summary.json"
        if not s.exists():
            continue
        import json
        res = json.loads(s.read_text())["final_eval"]["standard"]
        rows.append((res["mean_return"], res["success_rate"], d.name.split("_")[0], d.name))
    if algo:
        rows = [r for r in rows if r[2] == algo]
    if not rows:
        sys.exit("No trained runs found in logs/. Train first: uv run python training/sweep.py")
    best = max(rows)
    return best[2], best[3]


def load_selector(run: str):
    algo = run.split("_")[0]
    if algo == "dqn":
        from training.dqn_training import load_actor, make_selector
        return make_selector(load_actor(ROOT / "models" / "dqn" / f"{run}_final.pt"))
    from training.pg_training import load_selector as load_pg
    return load_pg(ROOT / "models" / "pg" / f"{run}_final.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], help="best run of this algorithm")
    ap.add_argument("--run", help="a specific run name, e.g. ppo_s04")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=3, help="GUI speed (steps per second)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scenario", choices=["standard", "hard"], default="standard")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--random", action="store_true", help="random actions instead of a trained agent")
    a = ap.parse_args()

    if a.random:
        rng = np.random.default_rng(a.seed)
        select, label = (lambda obs: int(rng.integers(len(ACTIONS)))), "RANDOM baseline"
    else:
        run = a.run or best_run(a.algo)[1]
        select, label = load_selector(run), f"{run.split('_')[0].upper()} agent ({run})"

    env: GovMateEnv = make_raw_env(a.scenario)
    if not a.no_gui:
        env.render_mode = "human"
        env.metadata["render_fps"] = a.fps
    print(f"\n=== GovMate | {label} | scenario: {a.scenario} | {a.episodes} episodes ===")

    wins, returns = 0, []
    for ep in range(a.episodes):
        obs, info = env.reset(seed=a.seed + ep)
        print(f"\n--- Episode {ep + 1}: citizen goal = {info['goal']} | digital access = "
              f"{info['digital_access']} | documents held = "
              f"{[d for d, v in info['docs_true'].items() if v]}")
        done, total = False, 0.0
        while not done:
            action = select(obs)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
            print(f"  step {env.steps:>2} | {ACTIONS[action]:<30} | {info['outcome']:<8} | "
                  f"reward {r:+6.2f} | days {info['days']:>4.0f} | fees {info['fees_rwf']:>8,.0f} RWF | "
                  f"patience {max(info['patience'], 0):4.1f} | steps done "
                  f"{len(info['completed_steps'])}/{len(env.goals[env.goal])}")
        wins += info["goal_complete"]
        returns.append(total)
        print(f"  => {info['end_reason'].upper()} | return {total:+.2f} | {env.steps} steps | "
              f"{info['days']:.0f} days | {info['fees_rwf']:,.0f} RWF")
        if not a.no_gui:
            for _ in range(a.fps * 2):         # hold the final frame ~2 s
                env.render()
    print(f"\n=== Summary: goal completed in {wins}/{a.episodes} episodes | "
          f"mean return {np.mean(returns):+.2f} ===")
    env.close()


if __name__ == "__main__":
    main()
