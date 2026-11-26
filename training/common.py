"""Helpers shared by the DQN and policy-gradient training scripts.

Everything that must be identical across algorithms for a fair comparison lives
here: the environment factory, evaluation protocol (fixed held-out seeds),
scenarios for generalisation tests, logging, and config overrides.
"""
from __future__ import annotations

import ast
import json
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.custom_env import GovMateEnv  # noqa: E402

EVAL_SEED0 = 100_000   # evaluation episodes use seeds never used for training

SCENARIOS = {
    "standard": {},
    "hard": {"digital_access_prob": 0.2, "doc_scale": 0.5, "patience": (6, 9)},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_raw_env(scenario: str = "standard") -> GovMateEnv:
    env = GovMateEnv()
    s, c = SCENARIOS[scenario], env.citizen_cfg
    if "digital_access_prob" in s:
        c["digital_access_prob"] = s["digital_access_prob"]
    if "doc_scale" in s:
        for k in c["doc_hold_probs"]:
            c["doc_hold_probs"][k] *= s["doc_scale"]
    if "patience" in s:
        c["patience"] = {"low": s["patience"][0], "high": s["patience"][1]}
    return env


def make_torchrl_env(seed: int | None = None):
    """GovMate wrapped for TorchRL, with episode-reward and step counters."""
    from torchrl.envs import Compose, GymWrapper, RewardSum, StepCounter, TransformedEnv

    env = TransformedEnv(GymWrapper(GovMateEnv(), device="cpu"),
                         Compose(RewardSum(), StepCounter()))
    if seed is not None:
        env.set_seed(seed)
    return env


@torch.no_grad()
def evaluate(select_action, episodes: int = 100, scenario: str = "standard",
             seed0: int = EVAL_SEED0) -> dict:
    """Run `select_action(obs)->int` on fixed held-out seeds and summarise."""
    env = make_raw_env(scenario)
    rets, ok, days, fees, steps, ends = [], [], [], [], [], {}
    per_goal: dict[str, list] = {g: [] for g in env.goal_names}
    for ep in range(episodes):
        obs, info = env.reset(seed=seed0 + ep)
        goal, done, total = env.goal, False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(select_action(obs))
            total += r
            done = term or trunc
        rets.append(total)
        ok.append(float(info["goal_complete"]))
        days.append(info["days"])
        fees.append(info["fees_rwf"])
        steps.append(env.steps)
        per_goal[goal].append(float(info["goal_complete"]))
        ends[info["end_reason"]] = ends.get(info["end_reason"], 0) + 1
    return {
        "scenario": scenario,
        "episodes": episodes,
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "success_rate": float(np.mean(ok)),
        "mean_days": float(np.mean(days)),
        "mean_fees_rwf": float(np.mean(fees)),
        "mean_steps": float(np.mean(steps)),
        "per_goal_success": {g: (float(np.mean(v)) if v else None) for g, v in per_goal.items()},
        "end_reasons": ends,
    }


class RunLogger:
    """Appends one JSON object per line to logs/<run_name>/metrics.jsonl."""

    def __init__(self, run_dir: Path, cfg) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self.f = open(self.dir / "metrics.jsonl", "w", encoding="utf-8")
        self.t0 = time.time()

    def log(self, **kw) -> None:
        kw["wall_time"] = round(time.time() - self.t0, 2)
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()

    def summary(self, data: dict) -> None:
        (self.dir / "summary.json").write_text(json.dumps(data, indent=2))

    def close(self) -> None:
        self.f.close()


def _cast(text: str, current):
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes")
    if isinstance(current, (list, tuple)):
        return tuple(ast.literal_eval(text))
    if isinstance(current, int):
        return int(float(text))
    if isinstance(current, float):
        return float(text)
    return text


def apply_overrides(cfg, pairs: list[str] | None):
    """Apply 'key=value' overrides to a dataclass config (used by the CLI/sweeps)."""
    names = {f.name for f in fields(cfg)}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if key not in names:
            raise KeyError(f"unknown hyperparameter '{key}'. valid: {sorted(names)}")
        setattr(cfg, key, _cast(value, getattr(cfg, key)))
    return cfg
