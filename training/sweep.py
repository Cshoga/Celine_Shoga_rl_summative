"""Hyperparameter sweeps: 10 configurations per algorithm, run in parallel.

    uv run python training/sweep.py                       # all 4 algorithms, all cores
    uv run python training/sweep.py --algo ppo --workers 4
    uv run python training/sweep.py --frames 20000        # quick, shorter test sweep

Run 00 of each algorithm is the default configuration; runs 01-09 change one
(or two) hyperparameters at a time, so each effect can be read off directly.
Finished runs (logs/<name>/summary.json exists) are skipped, so an interrupted
sweep can simply be restarted. Results are collected into logs/sweep_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

SWEEPS: dict[str, list[dict]] = {
    # value-based: learning rate, discount, target-network update rate, replay size,
    # exploration schedule, double-DQN on/off, replay ratio (updates per batch)
    "dqn": [
        {},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.95},
        {"tau": 0.001},
        {"tau": 0.05},
        {"buffer_size": 5_000},
        {"eps_decay_frames": 20_000},
        {"double_dqn": False},
        {"updates_per_batch": 10},
    ],
    # REINFORCE: learning rate, discount, entropy bonus, return normalisation, batch size
    "reinforce": [
        {},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.95},
        {"entropy_coeff": 0.0},
        {"entropy_coeff": 0.05},
        {"normalize_returns": False},
        {"frames_per_batch": 500},
        {"frames_per_batch": 5_000},
        {"hidden": (64, 64)},
    ],
    # A2C: learning rate, discount, GAE lambda (advantage), entropy, critic weight, rollout length
    "a2c": [
        {},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.95},
        {"gae_lambda": 0.8},
        {"gae_lambda": 1.0},
        {"entropy_coeff": 0.0},
        {"entropy_coeff": 0.05},
        {"frames_per_batch": 2_000},
        {"critic_coeff": 1.0},
    ],
    # PPO: learning rate, clipping range, epochs, entropy, discount, rollout length
    "ppo": [
        {},
        {"lr": 1e-3},
        {"lr": 1e-4},
        {"clip_epsilon": 0.1},
        {"clip_epsilon": 0.3},
        {"ppo_epochs": 10},
        {"entropy_coeff": 0.0},
        {"entropy_coeff": 0.05},
        {"gamma": 0.95},
        {"frames_per_batch": 2_000},
    ],
}


def run_one(job: tuple[str, int, dict, int, int]) -> tuple[str, bool, str]:
    """Train one configuration (executed in a worker process)."""
    algo, idx, overrides, frames, seed = job
    name = f"{algo}_s{idx:02d}"
    try:
        import torch
        torch.set_num_threads(1)           # one thread per worker: no CPU oversubscription
        if algo == "dqn":
            from training.dqn_training import DQNConfig, train
            cfg = DQNConfig()
        else:
            from training.pg_training import default_config, train
            cfg = default_config(algo)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        cfg.run_name, cfg.seed = name, seed
        if frames:
            cfg.total_frames = frames
            if algo == "dqn":
                cfg.eval_every = max(frames // 10, 1000)
                cfg.eps_decay_frames = min(cfg.eps_decay_frames, frames // 2)
            else:
                cfg.eval_every = max(frames // 10, 1000)
        train(cfg)
        return name, True, ""
    except Exception as exc:  # keep the sweep going if one run fails
        import traceback
        return name, False, traceback.format_exc()


def collect_results(algos: list[str]) -> Path:
    """Gather summary.json of every run into one CSV (used for the report tables)."""
    rows = []
    for algo in algos:
        for idx, overrides in enumerate(SWEEPS[algo]):
            name = f"{algo}_s{idx:02d}"
            path = ROOT / "logs" / name / "summary.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            std, hard = s["final_eval"]["standard"], s["final_eval"]["hard"]
            frames_to_80 = ""
            metrics = ROOT / "logs" / name / "metrics.jsonl"
            if metrics.exists():
                for line in metrics.read_text().splitlines():
                    r = json.loads(line)
                    if r.get("eval_success", 0) >= 0.8:
                        frames_to_80 = r["frames"]
                        break
            rows.append({
                "algo": algo, "run": name,
                "changed": json.dumps(overrides, default=list) if overrides else "default",
                "std_return": round(std["mean_return"], 2),
                "std_success": round(std["success_rate"], 3),
                "std_days": round(std["mean_days"], 1),
                "std_fees_rwf": round(std["mean_fees_rwf"]),
                "hard_return": round(hard["mean_return"], 2),
                "hard_success": round(hard["success_rate"], 3),
                "frames_to_80pct_success": frames_to_80,
                "train_time_s": s["train_time_s"],
            })
    out = ROOT / "logs" / "sweep_results.csv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="all", choices=["all", *SWEEPS])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--frames", type=int, default=0, help="override total_frames (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N configs per algo (quick tests)")
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    a = ap.parse_args()

    algos = list(SWEEPS) if a.algo == "all" else [a.algo]
    jobs = []
    for algo in algos:
        for idx, ov in enumerate(SWEEPS[algo]):
            if a.limit and idx >= a.limit:
                break
            done = (ROOT / "logs" / f"{algo}_s{idx:02d}" / "summary.json").exists()
            if done and not a.force:
                continue
            jobs.append((algo, idx, ov, a.frames, a.seed))
    print(f"{len(jobs)} runs to do with {a.workers} workers "
          f"({sum(len(SWEEPS[x]) for x in algos) - len(jobs)} already finished)", flush=True)

    t0, failed = time.time(), []
    if jobs:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(run_one, j) for j in jobs]
            for n, fut in enumerate(as_completed(futures), 1):
                name, ok, err = fut.result()
                print(f"--- finished {n}/{len(jobs)}: {name} {'OK' if ok else 'FAILED'} "
                      f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
                if not ok:
                    failed.append(name)
                    print(err, flush=True)
    out = collect_results(algos)
    print(f"results table: {out}")
    if failed:
        print("FAILED runs (re-run the command to retry):", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
