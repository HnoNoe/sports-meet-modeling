# -*- coding: utf-8 -*-
"""Q1 改良：共识矩阵集成分组。

思想：单一方案易受方法与权重的个体偏见影响。统计全部候选方案中
"学院 i 与 j 同组"的频率，得到共识矩阵 P，再对 P 做层次聚类切割，
得到集体智慧意义上的最终分组，并用真实 F 精修、引擎仿真裁决。

结论：共识方案 F_pop=0.1217，与 MILP 单方法（0.1145）接近但未反超；
共识频率最高仅 29%，说明解空间平坦、近似等优方案大量存在，
可作为最终方案的稳健性证据。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 11 时生成，已经人工核对；集成策略由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import composite
from scoring_engine import MeetConfig, simulate_meet, RESULTS
from q1_solvers import load_colleges, refine, scheme_F

K_FINAL = 12
N_REPEATS = 500


def build_consensus_matrix(detail: pd.DataFrame, names: list) -> np.ndarray:
    """P[i,j] = 候选方案中学院 i 与 j 同组的频率。"""
    n = len(names)
    idx = {c: i for i, c in enumerate(names)}
    P = np.zeros((n, n))
    schemes = detail.groupby(["方法", "K"])["组号"]
    n_schemes = 0
    for _, groups in schemes:
        n_schemes += 1
        for _, sub in detail.loc[groups.index].groupby("组号"):
            members = [idx[c] for cell in sub["学院"] for c in cell.split("+")]
            for a in members:
                for b in members:
                    P[a, b] += 1
    return P / n_schemes


# >>> AI-GENERATED BLOCK: 层次聚类切割共识矩阵
def consensus_cluster(P: np.ndarray, names: list, K: int) -> list:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = 1.0 - P[np.ix_(range(len(names)), range(len(names)))]
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=K, criterion="maxclust")
    groups = {}
    for name, lab in zip(names, labels):
        groups.setdefault(lab, []).append(name)
    return list(groups.values())
# <<< AI-GENERATED BLOCK END


# >>> AI-GENERATED BLOCK: 共识矩阵热力图
def plot_consensus(P: np.ndarray, names: list, groups: list, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    order = [names.index(c) for g in groups for c in g]
    Pm = P[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(Pm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    # 组间分隔线
    acc = 0
    for g in groups[:-1]:
        acc += len(g)
        ax.axhline(acc - 0.5, color="navy", lw=1.2)
        ax.axvline(acc - 0.5, color="navy", lw=1.2)
    ax.set_title("共识矩阵：学院同组频率（28 个候选方案）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    df = load_colleges()
    names = df["学院"].tolist()
    detail = pd.read_csv(RESULTS / "q1_schemes_detail.csv", encoding="utf-8-sig")

    P = build_consensus_matrix(detail, names)

    # 共识最强的学院对（排除自身）
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], P[i, j]))
    pairs.sort(key=lambda x: -x[2])
    print("=== 共识最强的 10 对学院 ===")
    for a, b, p in pairs[:10]:
        print(f" {a} + {b}: {p:.0%}")

    # 聚类切割 -> 精修 -> 评估
    groups = consensus_cluster(P, names, K_FINAL)
    groups = refine(groups, df)
    f_pop = scheme_F(groups, df)

    teams_rows = {}
    for k, g in enumerate(groups, 1):
        sub = df[df["学院"].isin(g)]
        teams_rows[f"组{k:02d}"] = {"男生人数": int(sub["男生人数"].sum()),
                                    "女生人数": int(sub["女生人数"].sum()),
                                    "特长生数": int(sub["特长生数"].sum())}
    teams = pd.DataFrame(teams_rows).T
    scores = simulate_meet(teams, MeetConfig(), n_repeats=N_REPEATS)
    f_score = composite(scores.mean().to_numpy(), np.full(4, 0.25))

    print("\n=== 共识方案（K=12，精修后）===")
    for k, g in enumerate(groups, 1):
        sub = df[df["学院"].isin(g)]
        print(f" 组{k:02d}: {'+'.join(g)}  "
              f"总人数{int(sub['总人数'].sum())}, "
              f"女生占比{sub['女生人数'].sum() / sub['总人数'].sum():.3f}")
    print(f"\n F_pop   = {f_pop:.4f}   (MILP 单方法最优 0.1145, 现状 0.227)")
    print(f" F_score = {f_score:.4f}   (MILP 单方法 0.2671)")

    plot_consensus(P, names, groups, RESULTS / "fig_q1_consensus.png")

    recs = []
    for k, g in enumerate(groups, 1):
        sub = df[df["学院"].isin(g)]
        recs.append({"组号": k, "学院": "+".join(g),
                     "总人数": int(sub["总人数"].sum()),
                     "女生人数": int(sub["女生人数"].sum()),
                     "特长生数": int(sub["特长生数"].sum())})
    pd.DataFrame(recs).to_csv(RESULTS / "q1_consensus_scheme.csv",
                              index=False, encoding="utf-8-sig")
    print(f"\n已保存: {RESULTS / 'q1_consensus_scheme.csv'}")
    print(f"已保存: {RESULTS / 'fig_q1_consensus.png'}")


if __name__ == "__main__":
    sys.exit(main())
