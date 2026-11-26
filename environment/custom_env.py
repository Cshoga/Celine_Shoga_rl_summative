"""GovMate custom Gymnasium environment.

A simulated citizen arrives with a hidden goal (one of five government
journeys). The agent guides them through a multi-agency dependency graph.
Documents, digital access and patience are partially observable, steps can
fail stochastically (exams, reviews, appointments), and failures can send the
citizen back to an earlier step. See environment/services.yaml for all values.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

ACTIONS = [
    "ask_documents",
    "ask_eligibility",
    "explain_next_step",
    "guide_submit_application",
    "guide_payment",
    "book_appointment",
    "check_status",
    "advise_gather_document",
    "advise_in_person_visit",
    "recommend_alternative_channel",
    "clarify_goal",
    "escalate_to_human",
    "wait",
    "close_session",
]
A = {name: i for i, name in enumerate(ACTIONS)}

# step kind -> the action that performs it
KIND_ACTION = {
    "submit": "guide_submit_application",
    "pay": "guide_payment",
    "book": "book_appointment",
    "visit": "advise_in_person_visit",
    "status": "check_status",
    "lookup": "explain_next_step",
    "gather": "advise_gather_document",
}
KINDS = list(KIND_ACTION)                      
ACTION_KIND = {v: k for k, v in KIND_ACTION.items()}

SUCCESS, REJECTED, DELAYED, NA = 0, 1, 2, 3    # last-outcome codes
OUTCOME_NAMES = ["success", "rejected", "delayed", "n/a"]

DEFAULT_CONFIG = Path(__file__).with_name("services.yaml")


class GovMateEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG,
                 render_mode: str | None = None):
        super().__init__()
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.meta = self.cfg["meta"]
        self.rw = self.cfg["rewards"]
        self.citizen_cfg = self.cfg["citizen"]
        self.docs = list(self.cfg["documents"])
        self.gatherable = self.cfg["gatherable"]
        self.goal_names = list(self.cfg["goals"])
        self.goals = {g: v["steps"] for g, v in self.cfg["goals"].items()}
        self.step_by_id = {s["id"]: s for steps in self.goals.values() for s in steps}

        self.render_mode = render_mode
        self.renderer = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.obs_dim = (len(self.goal_names) + 1) + len(self.docs) + 1 \
            + (len(KINDS) + 1) + 4 + 3 + 1 + 1
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self._reset_state_vars()

    # ------------------------------------------------------------------ state
    def _reset_state_vars(self):
        self.goal = self.goal_names[0]
        self.completed: set[str] = set()
        self.docs_true = {d: False for d in self.docs}
        self.docs_known = {d: 0.5 for d in self.docs}
        self.days = 0.0
        self.fees = 0.0
        self.patience = 10.0
        self.patience_max = 14.0
        self.digital = True
        self.digital_known = False
        self.alt_channel = False
        self.goal_known = False
        self.escalations = 0
        self.steps = 0
        self.last_outcome = NA
        self.last_action = None
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.end_reason = None
        self.action_log: list[tuple] = []

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state_vars()
        c = self.citizen_cfg
        probs = np.array([c["goal_probs"][g] for g in self.goal_names], dtype=float)
        forced = (options or {}).get("goal")
        if forced:
            self.goal = forced
        else:
            self.goal = self.np_random.choice(self.goal_names, p=probs / probs.sum())
            self.goal = str(self.goal)
        for d in self.docs:
            self.docs_true[d] = bool(self.np_random.random() < c["doc_hold_probs"][d])
        lo, hi = c["patience"]["low"], c["patience"]["high"]
        self.patience_max = float(hi)
        self.patience = float(self.np_random.integers(lo, hi + 1))
        self.digital = bool(self.np_random.random() < c["digital_access_prob"])
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    # ----------------------------------------------------------- helpers
    def next_step(self):
        for s in self.goals[self.goal]:
            if s["id"] not in self.completed:
                return s
        return None

    def _remaining_docs(self):
        needed = set()
        for s in self.goals[self.goal]:
            if s["id"] not in self.completed:
                needed.update(s["needs_docs"])
        return needed

    def _revert_to(self, step_id: str):
        ids = [s["id"] for s in self.goals[self.goal]]
        for sid in ids[ids.index(step_id):]:
            self.completed.discard(sid)

    def _sample_days(self, lohi):
        return float(self.np_random.integers(lohi[0], lohi[1] + 1))

    def _run_step(self, step, force=False):
        """Try to execute a service step. Returns (outcome, reward)."""
        missing = [d for d in step["needs_docs"] if not self.docs_true[d]]
        if missing and not force:
            for d in missing:                       # failing reveals the gap
                self.docs_known[d] = 0.0
            self.days += 1
            self.patience -= 1.0
            return REJECTED, self.rw["invalid_action"]
        if (not step["in_person"]) and not self.digital and not self.alt_channel \
                and not force:
            self.days += 2
            self.patience -= 1.0
            return DELAYED, -0.5
        self.fees += step["fee_rwf"]
        self.days += self._sample_days(step["days"])
        if force or self.np_random.random() < step["success_prob"]:
            self.completed.add(step["id"])
            for d in step["needs_docs"]:
                self.docs_true[d] = True
                self.docs_known[d] = 1.0
            for d in step["produces_docs"]:
                self.docs_true[d] = True
                self.docs_known[d] = 1.0
            self.patience = min(self.patience_max, self.patience + 0.5)
            return SUCCESS, self.rw["prerequisite_done"]
        self.patience -= 0.5
        if step["on_fail"].startswith("goto:"):
            self._revert_to(step["on_fail"].split(":", 1)[1])
        return REJECTED, 0.0

    def _penalty(self, kind: str, patience_cost: float):
        self.patience -= patience_cost
        return NA, self.rw[kind]

    # -------------------------------------------------------------- step
    def step(self, action: int):
        name = ACTIONS[int(action)]
        self.steps += 1
        self.last_action = name
        days0, fees0 = self.days, self.fees
        reward = self.rw["step_cost"]
        outcome = NA
        terminated = truncated = False
        end_reason = None
        nxt = self.next_step()

        if name == "clarify_goal":
            if self.goal_known:
                outcome, r = self._penalty("redundant_action", 0.5)
            elif self.np_random.random() < 0.85:
                self.goal_known = True
                self.patience -= 0.5
                outcome, r = SUCCESS, 0.0
            else:
                self.patience -= 0.5
                outcome, r = DELAYED, 0.0       # citizen was vague
            reward += r

        elif name == "ask_documents":
            needed = self._remaining_docs() if self.goal_known else set()
            unknown = [d for d in needed if self.docs_known[d] == 0.5]
            if not unknown:
                outcome, r = self._penalty("redundant_action", 0.5)
            else:
                for d in unknown:
                    self.docs_known[d] = 1.0 if self.docs_true[d] else 0.0
                self.patience -= 0.5
                outcome, r = SUCCESS, 0.0
            reward += r

        elif name == "ask_eligibility":
            if self.digital_known:
                outcome, r = self._penalty("redundant_action", 0.5)
            else:
                self.digital_known = True
                self.patience -= 0.5
                outcome, r = SUCCESS, 0.0
            reward += r

        elif name == "recommend_alternative_channel":
            if self.alt_channel or (self.digital_known and self.digital):
                outcome, r = self._penalty("redundant_action", 0.5)
            else:
                self.alt_channel = True
                self.fees += 500
                self.days += 1
                outcome, r = SUCCESS, 0.0
            reward += r

        elif name == "wait":
            self.days += 1
            outcome = NA

        elif name == "close_session":
            terminated, end_reason = True, "closed_early"
            reward += self.rw["abandon"]       # closing unresolved = failure

        elif name == "escalate_to_human":
            if not nxt or self.escalations >= 2:
                outcome, r = self._penalty("invalid_action", 1.0)
            else:
                self.escalations += 1
                self.goal_known = True
                self.days += 3
                outcome, r = self._run_step(nxt, force=True)
                reward += r - 1.0               # human handover costs the agent
            if outcome == NA:
                reward += r

        elif name == "advise_gather_document":
            if not self.goal_known or not nxt:
                outcome, r = self._penalty("invalid_action", 1.0)
                reward += r
            else:
                known_missing = [d for d in nxt["needs_docs"]
                                 if self.docs_known[d] == 0.0 and not self.docs_true[d]
                                 and d in self.gatherable]
                if known_missing:
                    d = known_missing[0]
                    g = self.gatherable[d]
                    self.fees += g["fee_rwf"]
                    self.days += self._sample_days(g["days"])
                    if self.np_random.random() < g["success_prob"]:
                        self.docs_true[d] = True
                        self.docs_known[d] = 1.0
                        outcome, reward = SUCCESS, reward + self.rw["prerequisite_done"] * 0.5
                    else:
                        self.patience -= 0.5
                        outcome = REJECTED
                elif nxt["kind"] == "gather":
                    outcome, r = self._run_step(nxt)
                    reward += r
                else:
                    outcome, r = self._penalty("invalid_action", 1.0)
                    reward += r

        else:  
            if not self.goal_known or not nxt:
                outcome, r = self._penalty("invalid_action", 1.0)
                reward += r
            elif ACTION_KIND.get(name) == nxt["kind"]:
                outcome, r = self._run_step(nxt)
                reward += r
            elif name == "explain_next_step":
                outcome, r = self._penalty("redundant_action", 0.25)
                reward += r
            else:
                outcome, r = self._penalty("invalid_action", 1.0)   # wrong step type
                reward += r

        self.last_outcome = outcome
        reward += self.rw["per_day_cost"] * (self.days - days0)
        reward += self.rw["per_1000_rwf_cost"] * (self.fees - fees0) / 1000.0

        if not terminated:
            if self.next_step() is None:
                terminated, end_reason = True, "goal_complete"
                reward += self.rw["goal_complete"]
            elif self.patience <= 0:
                terminated, end_reason = True, "abandoned"
                reward += self.rw["abandon"]
            elif self.steps >= self.meta["max_steps"] or self.days >= self.meta["max_days"]:
                truncated, end_reason = True, "timeout"
                
        self.last_reward = float(reward)
        self.episode_return += float(reward)
        self.end_reason = end_reason
        self.action_log.append((self.steps, name, OUTCOME_NAMES[outcome], float(reward)))
        self.action_log = self.action_log[-8:]
        
        info = self._info()
        info["end_reason"] = end_reason
        info["outcome"] = OUTCOME_NAMES[outcome]
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------ observations
    def _obs(self) -> np.ndarray:
        gi = len(self.goal_names)
        goal = np.zeros(gi + 1, np.float32)
        goal[self.goal_names.index(self.goal) if self.goal_known else gi] = 1.0
        docs = np.array([self.docs_known[d] for d in self.docs], np.float32)
        steps = self.goals[self.goal]
        progress = np.array([len(self.completed) / len(steps)], np.float32)
        kind = np.zeros(len(KINDS) + 1, np.float32)
        nxt = self.next_step()
        kind[KINDS.index(nxt["kind"]) if (self.goal_known and nxt) else -1] = 1.0
        outcome = np.zeros(4, np.float32)
        outcome[self.last_outcome] = 1.0
        scalars = np.array([
            min(self.days / self.meta["max_days"], 1.0),
            min(self.fees / self.meta["max_fees_rwf"], 1.0),
            float(np.clip(self.patience / self.patience_max, 0.0, 1.0)),
        ], np.float32)
        digital = np.array([(1.0 if self.digital else 0.0) if self.digital_known else 0.5],
                           np.float32)
        remaining = np.array([1.0 - self.steps / self.meta["max_steps"]], np.float32)
        remaining = np.clip(remaining, 0.0, 1.0)
        # progress 
        if not self.goal_known:
            progress[:] = 0.0
        return np.concatenate([goal, docs, progress, kind, outcome, scalars,
                               digital, remaining]).astype(np.float32)

    def _info(self) -> dict:
        return {
            "goal": self.goal,
            "completed_steps": sorted(self.completed),
            "days": self.days,
            "fees_rwf": self.fees,
            "patience": self.patience,
            "digital_access": self.digital,
            "docs_true": dict(self.docs_true),
            "goal_complete": self.next_step() is None,
            "last_action": self.last_action,
        }

    # ------------------------------------------------------------ render
    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            try:
                from environment.rendering import GovMateRenderer
            except ImportError as exc:  
                raise NotImplementedError(
                    "environment/rendering.py (GovMateRenderer) is not implemented yet"
                ) from exc
            self.renderer = GovMateRenderer(self, self.render_mode)
        return self.renderer.draw()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
