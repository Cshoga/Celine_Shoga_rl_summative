"""Sanity tests + baselines for the environment.

Run:  uv run pytest -q           (tests)
      uv run python tests/test_env.py   (prints random vs scripted baselines)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymnasium.utils.env_checker import check_env  # noqa: E402

from environment.custom_env import A, ACTIONS, KIND_ACTION, GovMateEnv  # noqa: E402


def scripted_policy(env: GovMateEnv, state: dict) -> int:
    
    if not env.goal_known:
        return A["clarify_goal"]
    if not env.digital_known:
        return A["ask_eligibility"]
    if not env.digital and not env.alt_channel:
        return A["recommend_alternative_channel"]
    if not state.get("asked"):
        state["asked"] = True
        return A["ask_documents"]
    nxt = env.next_step()
    missing = [d for d in nxt["needs_docs"] if env.docs_known[d] == 0.0
               and not env.docs_true[d]]
    if missing:
        if any(d in env.gatherable for d in missing):
            return A["advise_gather_document"]
        return A["escalate_to_human"]
    return A[KIND_ACTION[nxt["kind"]]]


def run(policy, episodes=200, seed=0):
    env = GovMateEnv()
    returns, done_flags, days, fees, reasons = [], [], [], [], {}
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        state, total, terminated, truncated = {}, 0.0, False, False
        while not (terminated or truncated):
            a = policy(env, state)
            obs, r, terminated, truncated, info = env.step(a)
            total += r
        returns.append(total)
        done_flags.append(info["goal_complete"])
        days.append(info["days"])
        fees.append(info["fees_rwf"])
        reasons[info["end_reason"]] = reasons.get(info["end_reason"], 0) + 1
    return dict(mean_return=float(np.mean(returns)), success_rate=float(np.mean(done_flags)),
                mean_days=float(np.mean(days)), mean_fees=float(np.mean(fees)), end=reasons)


def test_gym_api():
    check_env(GovMateEnv(), skip_render_check=True)


def test_obs_shape_and_range():
    env = GovMateEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,) and obs.min() >= 0.0 and obs.max() <= 1.0


def test_reproducible():
    e1, e2 = GovMateEnv(), GovMateEnv()
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    assert np.array_equal(o1, o2)
    for a in [A["clarify_goal"], A["ask_documents"], A["wait"]]:
        r1 = e1.step(a)
        r2 = e2.step(a)
        assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1]


def test_every_goal_is_completable():
    env = GovMateEnv()
    for goal in env.goal_names:
        ok = False
        for seed in range(20):
            env.reset(seed=seed, options={"goal": goal})
            state, term, trunc = {}, False, False
            while not (term or trunc):
                _, _, term, trunc, info = env.step(scripted_policy(env, state))
            ok = ok or info["goal_complete"]
        assert ok, f"scripted policy never completed {goal}"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("random   :", run(lambda e, s: int(rng.integers(len(ACTIONS)))))
    print("scripted :", run(scripted_policy))
