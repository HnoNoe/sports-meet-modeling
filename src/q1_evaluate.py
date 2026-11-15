# -*- coding: utf-8 -*-
"""Q1 方法横向评估：四种求解器 × K=12..20 全扫描，统一三维裁决。

裁决维度：
  人口公平 F_pop   规模+性别的真实公平指数（精修后）
  得分公平 F_score 仿真引擎下各组平均得分的公平指数
  改动成本         合并次数 = 20 - K

产出：方法对比总表、方案明细表、F-K 权衡曲线图。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 09 时生成，已经人工核对；评估维度与实验流程由人工设计。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import composite
from scoring_engine import MeetConfig, simulate_meet, RESULTS
from q1_solvers import SOLVERS, load_colleges, refine, scheme_F, solve_matching

K_RANGE = range(12, 21)
N_REPEATS = 200
W4 = np.full(4, 0.25)


def groups_to_teams(groups: list, df: pd.DataFrame) -> pd.DataFrame:
    """分组方案 -> 引擎输入表（组名、男、女、特长生汇总）。"""
    rows = {}
    for k, g in enumerate(groups, 1):
        sub = df[df["学院"].isin(g)]
        rows[f"组{k:02d}"] = {
            "男生人数": int(sub["男生人数"].sum()),
            "女生人数": int(sub["女生人数"].sum()),
            "特长生数": int(sub["特长生数"].sum()),
        }
    return pd.DataFrame(rows).T


def evaluate_scheme(groups: list, df: pd.DataFrame,
                    n_repeats: int = N_REPEATS) -> dict:
    teams = groups_to_teams(groups, df)
    scores = simulate_meet(teams, MeetConfig(), n_repeats=n_repeats)
    mean_scores = scores.mean().to_numpy()
    ranks = scores.rank(ascending=False, axis=1, method="min")
    return {
        "F_pop": round(scheme_F(groups, df), 4),
        "F_score": round(composite(mean_scores, W4), 4),
        "冠军集中度": round(float((ranks == 1).mean().max()), 3),
    }


def scheme_records(name: str, K: int, groups: list, df: pd.DataFrame) -> list:
    recs = []
    for gi, g in enumerate(groups, 1):
        sub = df[df["学院"].isin(g)]
        total = int(sub["总人数"].sum())
        female = int(sub["女生人数"].sum())
        recs.append({"方法": name, "K": K, "组号": gi, "学院": "+".join(g),
                     "总人数": total, "女生人数": female,
                     "女生占比": round(female / total, 4),
                     "特长生数": int(sub["特长生数"].sum())})
    return recs


def main():
    df = load_colleges()
    rows, details = [], []

    def run_one(name, K, groups):
        t0 = time.time()
        g = refine(groups, df)
        ev = evaluate_scheme(g, df)
        rows.append({"方法": name, "K": K, "改动成本": 20 - K,
                     **ev, "用时s": round(time.time() - t0, 2)})
        details.extend(scheme_records(name, K, g, df))
        print(f"{name} K={K} 完成: F_pop={ev['F_pop']}, "
              f"F_score={ev['F_score']}, 冠军集中度={ev['冠军集中度']}")

    # 匹配法：结构决定 K=12，只跑一次
    run_one("匹配法", 12, solve_matching(df))

    # 其余三法 × K 全扫描
    for K in K_RANGE:
        for name, solver in SOLVERS.items():
            run_one(name, K, solver(df, K))

    out = pd.DataFrame(rows).sort_values(["K", "F_pop"])
    out.to_csv(RESULTS / "q1_methods_compare.csv", index=False,
               encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(RESULTS / "q1_schemes_detail.csv",
                                 index=False, encoding="utf-8-sig")

    print("\n=== 各 K 下最优方法（按 F_pop）===")
    best = out.loc[out.groupby("K")["F_pop"].idxmin()]
    print(best[["K", "方法", "F_pop", "F_score", "冠军集中度"]].to_string(index=False))

    print("\n=== 方法汇总 ===")
    print(out.groupby("方法")[["F_pop", "F_score"]].agg(["mean", "min"])
          .round(4).to_string())

    # >>> AI-GENERATED BLOCK: 对比图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, sub in out.groupby("方法"):
        sub = sub.sort_values("K")
        axes[0].plot(sub["K"], sub["F_pop"], "o-", ms=4, label=name)
        axes[1].plot(sub["K"], sub["F_score"], "s--", ms=4, label=name)
    axes[0].set_xlabel("组数 K"); axes[0].set_ylabel("F_pop（人口公平）")
    axes[0].set_title("人口公平 vs K"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("组数 K"); axes[1].set_ylabel("F_score（得分公平）")
    axes[1].set_title("得分公平 vs K（仿真 200 届）")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_q1_methods.png", dpi=150)
    plt.close(fig)
    # <<< AI-GENERATED BLOCK END

    print(f"\n已保存: {RESULTS / 'q1_methods_compare.csv'}")
    print(f"已保存: {RESULTS / 'q1_schemes_detail.csv'}")
    print(f"已保存: {RESULTS / 'fig_q1_methods.png'}")


if __name__ == "__main__":
    sys.exit(main())
