"""Watch the scripted expert play GovMate in the pygame window.

Temporary demo until main.py / play.py exist:
    uv run python demo_render.py            # random citizens, slow
    uv run python demo_render.py --episodes 3 --fps 8
"""
import argparse

from environment.custom_env import GovMateEnv
from tests.test_env import scripted_policy

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--fps", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = GovMateEnv(render_mode="human")
env.metadata["render_fps"] = args.fps
for ep in range(args.episodes):
    env.reset(seed=args.seed + ep)
    state, done = {}, False
    while not done:
        _, _, term, trunc, info = env.step(scripted_policy(env, state))
        done = term or trunc
    for _ in range(args.fps * 2):          # hold the final frame for ~2 seconds
        env.render()
env.close()
