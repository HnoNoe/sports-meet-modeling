# -*- coding: utf-8 -*-
"""受控数值实验：特长生优势 δ 的敏感性扫描。

实验设计：
  实验臂：现状模式（后续加入 Q1 分组、Q2 分级、Q3 特长生方案）；
  实验条件：δ ∈ [0.5, 2.5]，步长 0.25；
  对照方式：所有臂共用同一组 (seed, 届次) 派生种子（公共随机数）；
  观测指标：各单位平均得分、夺冠概率、平均得分的综合公平指数 F。

结论只允许以"在 δ 全区间内均成立"的形式给出，不依赖单点参数。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-14 22 时生成，已经人工核对；实验设计由人工制定。
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring_engine import MeetConfig, simulate_meet, summarize, DATA, RESULTS
from fairness_index import composite

DELTAS = np.round(np.arange(0.5, 2.51, 0.25), 2)
N_REPEATS = 500
W_EQUAL = np.full(4, 0.25)


def load_arms() -> dict:
    """实验臂注册处：新模式在这里加入。"""
    college = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    teams = college.set_index("学院")[["男生人数", "女生人数", "特长生数"]]
    return {"现状": teams}


def run_experiment(deltas=DELTAS, n_repeats=N_REPEATS) -> pd.DataFrame:
    arms = load_arms()
    rows = []
    for delta in deltas:
        for arm_name, teams in arms.items():
            cfg = replace(MeetConfig(), talent_delta=float(delta))
            scores = simulate_meet(teams, cfg, n_repeats=n_repeats)
            mean_scores = scores.mean().to_numpy()
            rows.append({
                "δ": delta,
                "模式": arm_name,
                "得分公平指数F": round(composite(mean_scores, W_EQUAL), 4),
                "冠军队": scores.mean().idxmax(),
                "冠军队平均得分": round(float(mean_scores.max()), 1),
                "末位队": scores.mean().idxmin(),
                "末位队平均得分": round(float(mean_scores.min()), 1),
            })
    return pd.DataFrame(rows)


# >>> AI-GENERATED BLOCK
def plot_sweep(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm, sub in df.groupby("模式"):
        axes[0].plot(sub["δ"], sub["得分公平指数F"], "o-", label=arm)
    axes[0].set_xlabel("特长生能力优势 δ（σ 倍数）")
    axes[0].set_ylabel("平均得分的公平指数 F（越小越公平）")
    axes[0].set_title("δ 扫描：得分不均衡度随 δ 的变化")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for arm, sub in df.groupby("模式"):
        axes[1].plot(sub["δ"], sub["冠军队平均得分"], "o-", label="冠军队")
        axes[1].plot(sub["δ"], sub["末位队平均得分"], "s--", label="末位队")
    axes[1].set_xlabel("特长生能力优势 δ（σ 倍数）")
    axes[1].set_ylabel("平均得分")
    axes[1].set_title("δ 扫描：首尾差距")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    print(f"受控实验：δ ∈ [{DELTAS[0]}, {DELTAS[-1]}]，每档 {N_REPEATS} 届，公共随机数对照")
    df = run_experiment()
    print("\n=== δ 敏感性扫描（现状模式）===")
    print(df.to_string(index=False))

    df.to_csv(RESULTS / "exp_delta_sweep.csv", index=False, encoding="utf-8-sig")
    plot_sweep(df, RESULTS / "fig_exp_delta_sweep.png")
    print(f"\n已保存: {RESULTS / 'exp_delta_sweep.csv'}")
    print(f"已保存: {RESULTS / 'fig_exp_delta_sweep.png'}")


if __name__ == "__main__":
    sys.exit(main())
