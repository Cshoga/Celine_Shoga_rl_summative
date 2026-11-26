"""Create every report figure from logs/ (run after the sweeps finish).

    uv run python training/make_figures.py

Outputs PNGs to figures/ and a best-run table to logs/best_runs.csv.
Best run per algorithm = highest mean return on the standard scenario
(ties broken by success rate) among its sweep runs.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLOR = {"dqn": "#257de0", "reinforce": "#f47230", "a2c": "#00b894", "ppo": "#6a4ce2"}
STYLE = {"dqn": "-", "reinforce": "--", "a2c": "-.", "ppo": ":"}
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def load_runs():
    runs = {a: [] for a in ALGOS}
    for d in sorted((ROOT / "logs").glob("*_s[0-9][0-9]")):
        algo = d.name.split("_")[0]
        if algo not in runs or not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        m = [json.loads(x) for x in (d / "metrics.jsonl").read_text().splitlines() if x.strip()]
        std = s["final_eval"]["standard"]
        runs[algo].append({"name": d.name, "summary": s, "metrics": m, "std": std,
                           "hard": s["final_eval"]["hard"], "cfg": s["config"]})
    return runs


def best_of(rs):
    return max(rs, key=lambda r: (r["std"]["mean_return"], r["std"]["success_rate"]))


def series(run, key):
    xs = [r["frames"] for r in run["metrics"] if key in r and r[key] == r[key]]
    ys = [r[key] for r in run["metrics"] if key in r and r[key] == r[key]]
    return np.array(xs), np.array(ys, dtype=float)


def smooth(y, k=7):
    if len(y) < k:
        return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


def fig_reward_curves(runs, best):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axes[0], "train_return", "Training episode return (smoothed)"),
                           (axes[1], "eval_return", "Held-out evaluation return (greedy policy)")]:
        for a in ALGOS:
            for r in runs[a]:
                x, y = series(r, key)
                if len(x):
                    ax.plot(x, smooth(y) if key == "train_return" else y, color=COLOR[a],
                            alpha=0.13, lw=0.8)
            if a in best:
                x, y = series(best[a], key)
                ax.plot(x, smooth(y) if key == "train_return" else y, color=COLOR[a],
                        ls=STYLE[a], lw=2.2, label=f"{LABEL[a]} (best: {best[a]['name']})")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode return")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cumulative reward: best run per algorithm (thick) over all 10 sweep runs (faint)",
                 x=0.01, ha="left", fontsize=11, y=1.02)
    save(fig, "fig1_reward_curves.png")


def fig_dqn_loss(runs, best):
    if "dqn" not in best:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for r in runs["dqn"]:
        x, y = series(r, "dqn_loss")
        ax.plot(x, smooth(y), color=COLOR["dqn"], alpha=0.15, lw=0.8)
    x, y = series(best["dqn"], "dqn_loss")
    ax.plot(x, smooth(y), color=COLOR["dqn"], lw=2.2, label=f"best run ({best['dqn']['name']})")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("DQN loss (Huber TD error, log scale)")
    ax.set_title("DQN objective during training", loc="left", fontsize=11)
    ax.legend()
    save(fig, "fig2_dqn_loss.png")


def fig_entropy(runs, best):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for a in ("reinforce", "a2c", "ppo"):
        for r in runs[a]:
            x, y = series(r, "entropy")
            ax.plot(x, smooth(y), color=COLOR[a], alpha=0.13, lw=0.8)
        if a in best:
            x, y = series(best[a], "entropy")
            ax.plot(x, smooth(y), color=COLOR[a], ls=STYLE[a], lw=2.2, label=LABEL[a] + " (best run)")
    ax.axhline(np.log(14), color=MUTED, lw=0.8)
    ax.text(0.99, np.log(14), " uniform random policy (ln 14)", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), color=MUTED, fontsize=8)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Policy entropy: exploration to exploitation", loc="left", fontsize=11)
    ax.legend(loc="center right", fontsize=8.5)
    save(fig, "fig3_policy_entropy.png")


def fig_convergence(runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    top = None
    for i, a in enumerate(ALGOS):
        vals = []
        for r in runs[a]:
            hit = next((m["frames"] for m in r["metrics"] if m.get("eval_success", 0) >= 0.8), None)
            vals.append(hit)
        budget = max((r["cfg"]["total_frames"] for r in runs[a]), default=0)
        top = max(top or 0, budget)
        rng = np.random.default_rng(i)
        for v in vals:
            j = i + rng.uniform(-0.12, 0.12)
            if v is None:
                ax.scatter(j, budget * 1.04, facecolor="white", edgecolor=COLOR[a], s=46, zorder=3)
            else:
                ax.scatter(j, v, color=COLOR[a], s=46, zorder=3, edgecolor="white", linewidth=1.2)
        reached = [v for v in vals if v is not None]
        if reached:
            ax.hlines(np.median(reached), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=4)
    ax.set_xticks(range(len(ALGOS)), [LABEL[a] for a in ALGOS])
    ax.set_ylabel("environment steps until eval success >= 80%")
    ax.set_title("Convergence speed across the 10 sweep runs", loc="left", fontsize=11)
    if top:
        ax.set_ylim(0, top * 1.12)
    ax.text(0.01, 0.97, "filled dot = reached 80%; hollow dot (top) = never reached;\nbar = median of runs that reached it",
            transform=ax.transAxes, va="top", fontsize=8, color=MUTED)
    save(fig, "fig4_convergence.png")


def fig_generalization(best):
    algos = [a for a in ALGOS if a in best]
    if not algos:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x, w = np.arange(len(algos)), 0.38
    for ax, key, ylabel in [(axes[0], "mean_return", "mean episode return"),
                            (axes[1], "success_rate", "goal completion rate")]:
        std = [best[a]["std"][key] for a in algos]
        hard = [best[a]["hard"][key] for a in algos]
        ax.bar(x - w / 2 - 0.01, std, w, color=[COLOR[a] for a in algos], label="standard citizens")
        ax.bar(x + w / 2 + 0.01, hard, w, color=[COLOR[a] for a in algos], alpha=0.45,
               hatch="//", edgecolor="white", label="hard citizens (unseen)")
        for xi, (s, h) in enumerate(zip(std, hard)):
            ax.text(xi - w / 2, s, f"{s:.2f}" if key == "success_rate" else f"{s:.1f}", ha="center",
                    va="bottom", fontsize=8)
            ax.text(xi + w / 2, h, f"{h:.2f}" if key == "success_rate" else f"{h:.1f}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_xticks(x, [LABEL[a] for a in algos])
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Generalisation: trained on standard citizens, tested on harder unseen ones "
                 "(low digital access, fewer documents, less patience)", x=0.01, ha="left",
                 fontsize=10.5, y=1.02)
    save(fig, "fig5_generalization.png")


def fig_sweep_bars(runs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, a in zip(axes.ravel(), ALGOS):
        rs = sorted(runs[a], key=lambda r: r["name"])
        vals = [r["std"]["mean_return"] for r in rs]
        names = []
        for i, r in enumerate(rs):
            base = json.loads(json.dumps(runs[a][0]["cfg"]))
            diff = {k: v for k, v in r["cfg"].items()
                    if k not in ("run_name", "seed") and base.get(k) != v}
            names.append("default" if i == 0 else ", ".join(f"{k}={v}" for k, v in diff.items()))
        ax.barh(range(len(rs)), vals, color=COLOR[a], height=0.62)
        ax.set_yticks(range(len(rs)), [f"{n[:34]}" for n in names], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"{LABEL[a]}: return per configuration", loc="left", fontsize=10)
        ax.set_xlabel("mean return (standard scenario)")
        ax.grid(axis="y", visible=False)
    save(fig, "fig6_sweep_by_config.png")


def main():
    runs = load_runs()
    counts = {a: len(v) for a, v in runs.items()}
    print("runs found:", counts)
    best = {a: best_of(v) for a, v in runs.items() if v}
    if not best:
        sys.exit("no finished runs in logs/ - run training/sweep.py first")
    with open(ROOT / "logs" / "best_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "run", "std_return", "std_success", "hard_return", "hard_success"])
        for a, r in best.items():
            w.writerow([a, r["name"], round(r["std"]["mean_return"], 2),
                        round(r["std"]["success_rate"], 3), round(r["hard"]["mean_return"], 2),
                        round(r["hard"]["success_rate"], 3)])
    fig_reward_curves(runs, best)
    fig_dqn_loss(runs, best)
    fig_entropy(runs, best)
    fig_convergence(runs)
    fig_generalization(best)
    fig_sweep_bars(runs)
    print("best runs:", {a: r["name"] for a, r in best.items()})


if __name__ == "__main__":
    main()
