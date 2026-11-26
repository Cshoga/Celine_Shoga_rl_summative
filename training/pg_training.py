"""Policy-gradient training for GovMate with TorchRL: REINFORCE, A2C and PPO.

Run one experiment (overrides are key=value pairs):
    uv run python training/pg_training.py --algo ppo --run_name ppo_test --set lr=0.0003 clip_epsilon=0.2

Components (all from TorchRL): GymWrapper env, ProbabilisticActor (categorical
policy), ValueOperator (critic), SyncDataCollector, GAE, A2CLoss, ClipPPOLoss.
REINFORCE is implemented directly from reward-to-go returns (no critic).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
try:                                   # TorchRL >= 0.15 renamed SyncDataCollector -> Collector
    from torchrl.collectors import Collector as SyncDataCollector
except ImportError:                    # older TorchRL
    from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.common import (ROOT, RunLogger, apply_overrides, evaluate,  # noqa: E402
                             make_torchrl_env, set_seed)

ALGOS = ("reinforce", "a2c", "ppo")


@dataclass
class PGConfig:
    algo: str = "ppo"
    run_name: str = "pg_default"
    seed: int = 0
    # budget
    total_frames: int = 100_000
    frames_per_batch: int = 1_000        # on-policy rollout length per update
    # network / optimisation
    hidden: tuple = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    entropy_coeff: float = 0.01          # exploration bonus (policy entropy)
    # REINFORCE only
    normalize_returns: bool = True
    # A2C / PPO (advantage estimation + critic)
    gae_lambda: float = 0.95
    critic_coeff: float = 0.5
    # PPO only
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 250
    # evaluation
    eval_every: int = 10_000
    eval_episodes: int = 50
    final_eval_episodes: int = 200


def default_config(algo: str) -> PGConfig:
    """Sensible per-algorithm starting points (the sweeps vary these)."""
    cfg = PGConfig(algo=algo, run_name=f"{algo}_default")
    if algo == "reinforce":
        cfg.lr, cfg.frames_per_batch = 1e-3, 2_000
    elif algo == "a2c":
        cfg.lr, cfg.frames_per_batch = 7e-4, 500
    elif algo == "ppo":
        cfg.lr, cfg.frames_per_batch = 3e-4, 1_000
    return cfg


def build_networks(obs_dim: int, n_actions: int, hidden, action_spec, with_critic: bool):
    logit_net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=list(hidden),
                    activation_class=nn.Tanh)
    policy_module = TensorDictModule(logit_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(policy_module, spec=action_spec, in_keys=["logits"],
                               distribution_class=OneHotCategorical, return_log_prob=True)
    critic = None
    if with_critic:
        value_net = MLP(in_features=obs_dim, out_features=1, num_cells=list(hidden),
                        activation_class=nn.Tanh)
        critic = ValueOperator(value_net, in_keys=["observation"])
    return logit_net, actor, critic


def make_selector(logit_net, greedy: bool = True):
    """Action from raw gymnasium observations (greedy = argmax of the policy logits)."""
    def select(obs) -> int:
        with torch.no_grad():
            logits = logit_net(torch.as_tensor(obs, dtype=torch.float32))
            if greedy:
                return int(logits.argmax())
            return int(torch.distributions.Categorical(logits=logits).sample())
    return select


def load_selector(path: str | Path, greedy: bool = True):
    """Rebuild a trained policy from a checkpoint (used by play/main)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    env = make_torchrl_env()
    logit_net, _, _ = build_networks(ckpt["obs_dim"], ckpt["n_actions"], ckpt["cfg"]["hidden"],
                                     env.action_spec, with_critic=False)
    logit_net.load_state_dict(ckpt["logit_state_dict"])
    logit_net.eval()
    return make_selector(logit_net, greedy)


def train(cfg: PGConfig) -> dict:
    assert cfg.algo in ALGOS, f"algo must be one of {ALGOS}"
    set_seed(cfg.seed)
    env = make_torchrl_env(cfg.seed)
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = int(env.action_spec.space.n)

    use_critic = cfg.algo in ("a2c", "ppo")
    logit_net, actor, critic = build_networks(obs_dim, n_actions, cfg.hidden,
                                              env.action_spec, use_critic)
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg.frames_per_batch,
                                  total_frames=cfg.total_frames, device="cpu")

    params = list(actor.parameters()) + (list(critic.parameters()) if use_critic else [])
    optim = torch.optim.Adam(params, lr=cfg.lr)

    advantage_module = loss_module = None
    if use_critic:
        advantage_module = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic,
                               average_gae=True)
    if cfg.algo == "a2c":
        loss_module = A2CLoss(actor, critic, entropy_coeff=cfg.entropy_coeff,
                              critic_coeff=cfg.critic_coeff, loss_critic_type="smooth_l1")
    elif cfg.algo == "ppo":
        loss_module = ClipPPOLoss(actor, critic, clip_epsilon=cfg.clip_epsilon,
                                  entropy_coeff=cfg.entropy_coeff, critic_coeff=cfg.critic_coeff,
                                  loss_critic_type="smooth_l1", normalize_advantage=True)

    run_dir = ROOT / "logs" / cfg.run_name
    model_dir = ROOT / "models" / "pg"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, cfg)

    def checkpoint(tag: str) -> Path:
        path = model_dir / f"{cfg.run_name}_{tag}.pt"
        torch.save({"logit_state_dict": logit_net.state_dict(), "cfg": asdict(cfg),
                    "obs_dim": obs_dim, "n_actions": n_actions, "algo": cfg.algo}, path)
        return path

    def step_optimizer(loss: torch.Tensor) -> None:
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optim.step()

    frames, next_eval, best_eval = 0, cfg.eval_every, -float("inf")
    start = time.time()
    for data in collector:
        frames += data.numel()
        done = data["next", "done"].squeeze(-1)
        ep_returns = data["next", "episode_reward"].squeeze(-1)[done]
        ep_lengths = data["next", "step_count"].squeeze(-1)[done]
        record = {"frames": frames, "fps": round(frames / (time.time() - start), 1)}
        stats: dict[str, list] = {}

        def add(**kw):
            for k, v in kw.items():
                stats.setdefault(k, []).append(float(v))

        if cfg.algo == "reinforce":
            # Monte-Carlo policy gradient with reward-to-go returns (no critic).
            rewards = data["next", "reward"]
            returns = reward2go(rewards, data["next", "done"], cfg.gamma).detach()
            if cfg.normalize_returns:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            dist = actor.get_dist(data)
            logp = dist.log_prob(data["action"])
            entropy = dist.entropy().mean()
            loss = -(logp * returns.squeeze(-1)).mean() - cfg.entropy_coeff * entropy
            step_optimizer(loss)
            add(policy_loss=loss.item(), entropy=entropy.item())

        elif cfg.algo == "a2c":
            with torch.no_grad():
                advantage_module(data)
            out = loss_module(data)
            loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
            step_optimizer(loss)
            add(policy_loss=out["loss_objective"].item(), critic_loss=out["loss_critic"].item(),
                entropy=out["entropy"].item())

        else:  # ppo
            with torch.no_grad():
                advantage_module(data)
            flat = data.reshape(-1)
            n = flat.numel()
            for _ in range(cfg.ppo_epochs):
                perm = torch.randperm(n)
                for i in range(0, n, cfg.minibatch_size):
                    mb = flat[perm[i:i + cfg.minibatch_size]]
                    out = loss_module(mb)
                    loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                    step_optimizer(loss)
                    add(policy_loss=out["loss_objective"].item(),
                        critic_loss=out["loss_critic"].item(), entropy=out["entropy"].item(),
                        clip_fraction=out["clip_fraction"].item())

        for k, v in stats.items():
            record[k] = sum(v) / len(v)
        if len(ep_returns):
            record.update(train_return=float(ep_returns.mean()),
                          train_ep_len=float(ep_lengths.float().mean()),
                          train_episodes=int(len(ep_returns)))

        if frames >= next_eval:
            logit_net.eval()
            ev = evaluate(make_selector(logit_net), cfg.eval_episodes)
            logit_net.train()
            record.update(eval_return=ev["mean_return"], eval_success=ev["success_rate"])
            if ev["mean_return"] > best_eval:
                best_eval = ev["mean_return"]
                checkpoint("best")
            next_eval += cfg.eval_every
            print(f"[{cfg.run_name}] frames={frames:>7} eval_return={ev['mean_return']:7.2f} "
                  f"success={ev['success_rate']:.2f} entropy={record.get('entropy', float('nan')):.3f} "
                  f"fps={record['fps']}", flush=True)
        logger.log(**record)

    collector.shutdown()
    logit_net.eval()
    final = {s: evaluate(make_selector(logit_net), cfg.final_eval_episodes, s)
             for s in ("standard", "hard")}
    checkpoint("final")
    if best_eval == -float("inf"):
        checkpoint("best")
    summary = {"config": asdict(cfg), "train_time_s": round(time.time() - start, 1),
               "best_periodic_eval_return": best_eval, "final_eval": final}
    logger.summary(summary)
    logger.close()
    print(f"[{cfg.run_name}] done in {summary['train_time_s']}s | "
          f"standard return {final['standard']['mean_return']:.2f} "
          f"success {final['standard']['success_rate']:.2f} | "
          f"hard return {final['hard']['mean_return']:.2f} "
          f"success {final['hard']['success_rate']:.2f}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = apply_overrides(default_config(a.algo), a.set)
    if a.run_name:
        cfg.run_name = a.run_name
    train(cfg)
