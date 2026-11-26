"""DQN (value-based) training for GovMate using TorchRL."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.common import (ROOT, RunLogger, apply_overrides, evaluate,  # noqa: E402
                             make_torchrl_env, set_seed)


@dataclass
class DQNConfig:
    run_name: str = "dqn_default"
    seed: int = 0
    # budget
    total_frames: int = 100_000          
    frames_per_batch: int = 100          
    updates_per_batch: int = 50          
    init_random_frames: int = 2_000     
    # network / optimisation
    hidden: tuple = (128, 128)
    lr: float = 1e-3
    batch_size: int = 128
    gamma: float = 0.99
    max_grad_norm: float = 10.0
    # replay + target network
    buffer_size: int = 50_000
    tau: float = 0.01                    # soft target update rate
    double_dqn: bool = True
    # exploration (epsilon-greedy, linear decay)
    eps_init: float = 1.0
    eps_end: float = 0.05
    eps_decay_frames: int = 50_000
    # evaluation
    eval_every: int = 10_000
    eval_episodes: int = 50
    final_eval_episodes: int = 200


def build_q_actor(obs_dim: int, n_actions: int, hidden, action_spec):
    net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=list(hidden),
              activation_class=nn.ReLU)
    actor = QValueActor(module=net, spec=action_spec, in_keys=["observation"])
    return net, actor


def make_selector(actor):
    """Greedy action from raw gymnasium observations (used for evaluation / play)."""
    def select(obs) -> int:
        td = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[])
        with torch.no_grad():
            return int(actor(td)["action"])
    return select


def load_actor(path: str | Path):
    """Rebuild a trained DQN actor from a checkpoint (used by play/main)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    env = make_torchrl_env()
    _, actor = build_q_actor(ckpt["obs_dim"], ckpt["n_actions"], ckpt["cfg"]["hidden"],
                             env.action_spec)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor


def train(cfg: DQNConfig) -> dict:
    set_seed(cfg.seed)
    env = make_torchrl_env(cfg.seed)
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = int(env.action_spec.space.n)

    net, actor = build_q_actor(obs_dim, n_actions, cfg.hidden, env.action_spec)
    exploration = EGreedyModule(spec=env.action_spec, eps_init=cfg.eps_init,
                                eps_end=cfg.eps_end, annealing_num_steps=cfg.eps_decay_frames)
    policy = TensorDictSequential(actor, exploration)

    collector = SyncDataCollector(env, policy, frames_per_batch=cfg.frames_per_batch,
                                  total_frames=cfg.total_frames,
                                  init_random_frames=cfg.init_random_frames, device="cpu")
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)

    loss_module = DQNLoss(actor, action_space=env.action_spec, delay_value=True,
                          double_dqn=cfg.double_dqn)
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_updater = SoftUpdate(loss_module, tau=cfg.tau)
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    run_dir = ROOT / "logs" / cfg.run_name
    model_dir = ROOT / "models" / "dqn"
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir, cfg)

    def checkpoint(tag: str) -> Path:
        path = model_dir / f"{cfg.run_name}_{tag}.pt"
        torch.save({"state_dict": actor.state_dict(), "cfg": asdict(cfg),
                    "obs_dim": obs_dim, "n_actions": n_actions}, path)
        return path

    frames, next_eval, best_eval = 0, cfg.eval_every, -float("inf")
    start = time.time()
    for data in collector:
        n = data.numel()
        frames += n
        rb.extend(data.reshape(-1))

        # ---- episode statistics from the collected batch
        done = data["next", "done"].squeeze(-1)
        ep_returns = data["next", "episode_reward"].squeeze(-1)[done]
        ep_lengths = data["next", "step_count"].squeeze(-1)[done]

        # ---- gradient updates
        losses, qmeans = [], []
        if frames >= cfg.init_random_frames and len(rb) >= cfg.batch_size:
            for _ in range(cfg.updates_per_batch):
                batch = rb.sample()
                loss = loss_module(batch)["loss"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                optim.step()
                target_updater.step()
                losses.append(loss.item())
            with torch.no_grad():
                qmeans.append(net(batch["observation"]).max(-1).values.mean().item())
            collector.update_policy_weights_()
        exploration.step(n)

        record = {"frames": frames, "epsilon": float(exploration.eps),
                  "fps": round(frames / (time.time() - start), 1)}
        if len(ep_returns):
            record.update(train_return=float(ep_returns.mean()),
                          train_ep_len=float(ep_lengths.float().mean()),
                          train_episodes=int(len(ep_returns)))
        if losses:
            record.update(dqn_loss=sum(losses) / len(losses), mean_max_q=qmeans[-1])

        # ---- periodic held-out evaluation (greedy policy)
        if frames >= next_eval:
            actor.eval()
            ev = evaluate(make_selector(actor), cfg.eval_episodes)
            actor.train()
            record.update(eval_return=ev["mean_return"], eval_success=ev["success_rate"])
            if ev["mean_return"] > best_eval:
                best_eval = ev["mean_return"]
                checkpoint("best")
            next_eval += cfg.eval_every
            print(f"[{cfg.run_name}] frames={frames:>7} eps={record['epsilon']:.2f} "
                  f"eval_return={ev['mean_return']:7.2f} success={ev['success_rate']:.2f} "
                  f"loss={record.get('dqn_loss', float('nan')):.4f} fps={record['fps']}", flush=True)
        logger.log(**record)

    collector.shutdown()
    actor.eval()
    final = {s: evaluate(make_selector(actor), cfg.final_eval_episodes, s)
             for s in ("standard", "hard")}
    checkpoint("final")
    if best_eval == -float("inf"):            # never reached an eval point
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
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = apply_overrides(DQNConfig(), a.set)
    if a.run_name:
        cfg.run_name = a.run_name
    train(cfg)
