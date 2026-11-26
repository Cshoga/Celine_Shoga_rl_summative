# Celine_Shoga_rl_summative

# GovMate: RL agent guiding citizens through government services

A custom Gymnasium environment in which an agent guides a simulated citizen through
a multi-agency journey (business registration, driving licence, land tax, passport,
building permit). Four RL algorithms (DQN, REINFORCE, A2C, PPO) are trained with
TorchRL on the same environment and compared.

## Quick start

```bash
uv sync
uv run main.py                      # best trained agent, GUI + verbose terminal log
uv run main.py --algo ppo           # best PPO run
uv run main.py --scenario hard      # harder, unseen citizens
uv run main.py --random             # random-action baseline
uv run main.py --no-gui             # terminal only
```

## Reproduce the experiments

```bash
uv run python tests/test_env.py                 # random vs scripted reference baselines
uv run python training/sweep.py                 # 10 runs x 4 algorithms (parallel, resumable)
uv run python training/make_figures.py          # figures/ + logs/best_runs.csv
uv run python -m pytest -q                      # environment sanity tests
```

Single runs: `uv run python training/dqn_training.py --run_name x --set lr=0.0005 gamma=0.95`
and `uv run python training/pg_training.py --algo ppo --run_name y --set clip_epsilon=0.1`.

## Layout

| Path | Purpose |
|---|---|
| `environment/custom_env.py` | Gymnasium environment (actions, observations, rewards, terminal conditions) |
| `environment/services.yaml` | Service graph: steps, fees, days, success probabilities, sources |
| `environment/rendering.py` | pygame visualisation |
| `training/dqn_training.py`, `training/pg_training.py` | TorchRL training scripts |
| `training/sweep.py` | hyperparameter sweeps (10 configs per algorithm) |
| `training/make_figures.py` | report figures |
| `models/dqn/`, `models/pg/` | saved checkpoints (`*_final.pt`, `*_best.pt`) |
| `logs/` | per-run metrics (`metrics.jsonl`), configs, summaries, `sweep_results.csv` |
| `tests/` | environment tests and baselines |

## Data provenance

Service requirements, fees and processing times come from the official portals and are
recorded with a `source` and `verified` flag per step in `services.yaml`. Citizen behaviour
(documents held, digital access, patience, exam outcomes, review rejections) is simulated:
no real personal data is used.
