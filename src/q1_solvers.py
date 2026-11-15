# -*- coding: utf-8 -*-
"""Q1 求解器库：四种互补方法 + 统一局部精修。

四个生成器的偏科方向各不相同，互为补充：
  MILP    全局最优，但只能用线性代理目标
  LPT     快速、有多机调度 4/3 近似比理论保证，但短视
  蛇形选秀 结构透明可解释，但只考虑排序不考虑组合
  匹配法  两两互补结构最优（blossom 算法，多项式时间），但只允许结对

所有求解器统一接口 solve(df, K) -> list[list[str]]（组 -> 学院名单），
产出后统一经过 refine() 用真实公平指数 F 做局部搜索精修。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 09 时生成，已经人工核对；模型构造与求解器选型由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import composite

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

W_EQUAL = np.full(4, 0.25)


def load_colleges() -> pd.DataFrame:
    return pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")


def scheme_F(groups: list, df: pd.DataFrame) -> float:
    """方案的真实公平指数：规模与性别两属性等权，四度量等权。"""
    sizes, females = [], []
    for g in groups:
        sub = df[df["学院"].isin(g)]
        sizes.append(sub["总人数"].sum())
        females.append(sub["女生人数"].sum())
    return 0.5 * composite(np.array(sizes), W_EQUAL) \
         + 0.5 * composite(np.array(females), W_EQUAL)


def solve_milp(df: pd.DataFrame, K: int) -> list:
    """整数规划：最小化各组总人数、女生人数与均值的相对 L1 偏差。"""
    from scipy.optimize import Bounds, LinearConstraint, milp

    s = df["总人数"].to_numpy(float)
    f = df["女生人数"].to_numpy(float)
    n = len(df)
    S_bar, F_bar = s.sum() / K, f.sum() / K

    n_x = n * K                      # x[i,j]: 学院 i 是否进组 j
    n_var = n_x + 2 * K              # u_j, v_j: 偏差辅助变量

    c = np.zeros(n_var)
    c[n_x:] = 1.0                    # 目标：min Σu_j + Σv_j

    rows, lbs, ubs = [], [], []

    # >>> AI-GENERATED BLOCK: 约束矩阵组装
    A = np.zeros((2 * n + 4 * K + K, n_var))
    r = 0
    for i in range(n):                          # 每学院恰进一组
        A[r, i * K:(i + 1) * K] = 1; rows.append(A[r].copy()); lbs.append(1); ubs.append(1); r += 1
    for j in range(K):                          # 每组非空
        A[r, [i * K + j for i in range(n)]] = 1; rows.append(A[r].copy()); lbs.append(1); ubs.append(np.inf); r += 1
    for j in range(K):                          # u_j ≥ ±(S_j/S̄ − 1)
        idx = [i * K + j for i in range(n)]
        A[r, idx] = s / S_bar; A[r, n_x + j] = -1; rows.append(A[r].copy()); lbs.append(-np.inf); ubs.append(1); r += 1
        A[r, idx] = -s / S_bar; A[r, n_x + j] = -1; rows.append(A[r].copy()); lbs.append(-np.inf); ubs.append(-1); r += 1
    for j in range(K):                          # v_j ≥ ±(F_j/F̄ − 1)
        idx = [i * K + j for i in range(n)]
        A[r, idx] = f / F_bar; A[r, n_x + K + j] = -1; rows.append(A[r].copy()); lbs.append(-np.inf); ubs.append(1); r += 1
        A[r, idx] = -f / F_bar; A[r, n_x + K + j] = -1; rows.append(A[r].copy()); lbs.append(-np.inf); ubs.append(-1); r += 1
    # <<< AI-GENERATED BLOCK END

    A = np.array(rows[:r])
    res = milp(c=c, constraints=LinearConstraint(A, lbs[:r], ubs[:r]),
               bounds=Bounds(0, 1), integrality=np.r_[np.ones(n_x), np.zeros(2 * K)])
    if not res.success:
        raise RuntimeError(f"MILP 求解失败(K={K}): {res.message}")

    x = np.round(res.x[:n_x].reshape(n, K)).astype(int)
    groups = [[] for _ in range(K)]
    for i in range(n):
        groups[int(np.argmax(x[i]))].append(df["学院"].iloc[i])
    return groups


def solve_lpt(df: pd.DataFrame, K: int) -> list:
    """最长处理时间优先：大学院依次放入当前总人数最少的组。"""
    groups = [[] for _ in range(K)]
    totals = np.zeros(K)
    ordered = df.sort_values("总人数", ascending=False)
    for _, row in ordered.iterrows():
        j = int(np.argmin(totals))
        groups[j].append(row["学院"])
        totals[j] += row["总人数"]
    return groups


def solve_snake(df: pd.DataFrame, K: int) -> list:
    """蛇形选秀：按总人数降序，正反向轮流入组。"""
    groups = [[] for _ in range(K)]
    ordered = df.sort_values("总人数", ascending=False)["学院"].tolist()
    order = list(range(K)) + list(range(K - 1, -1, -1))
    for t, name in enumerate(ordered):
        groups[order[t % (2 * K)]].append(name)
    return groups


def solve_matching(df: pd.DataFrame) -> list:
    """两两匹配法：20 个学院选 8 对合并（K=12），blossom 算法求最优匹配。

    加 4 个虚拟节点吸收单飞的学院，完美匹配强制恰好 8 对结对。
    边权 = 结对质量：两院合并后的总人数、女生数越接近双院均值越大。
    """
    import networkx as nx

    s = df["总人数"].to_numpy(float)
    f = df["女生人数"].to_numpy(float)
    names = df["学院"].tolist()
    n = len(df)
    S2, F2 = 2 * s.sum() / 12, 2 * f.sum() / 12

    G = nx.Graph()
    # 4 个虚拟节点：4 条实-虚边占位，其余 16 个实点两两结对成 8 条边
    dummies = [f"_dummy{d}" for d in range(4)]
    G.add_nodes_from(names + dummies)
    for i in range(n):
        for j in range(i + 1, n):
            dev = abs(s[i] + s[j] - S2) / S2 + abs(f[i] + f[j] - F2) / F2
            G.add_edge(names[i], names[j], weight=1.0 - dev)
        for d in dummies:
            G.add_edge(names[i], d, weight=0.0)

    mate = nx.max_weight_matching(G, maxcardinality=True, weight="weight")
    paired, singles = [], []
    for a, b in mate:
        if a.startswith("_dummy"):
            singles.append(b)
        elif b.startswith("_dummy"):
            singles.append(a)
        else:
            paired.append([a, b])
    return paired + [[x] for x in singles]


def refine(groups: list, df: pd.DataFrame, max_iter: int = 300) -> list:
    """局部搜索精修：单学院调组，以真实 F 为目标，直到无改进。"""
    groups = [list(g) for g in groups]
    best = scheme_F(groups, df)
    for _ in range(max_iter):
        improved = False
        for gi in range(len(groups)):
            for college in list(groups[gi]):
                if len(groups[gi]) == 1:
                    continue
                for gj in range(len(groups)):
                    if gi == gj:
                        continue
                    trial = [list(g) for g in groups]
                    trial[gi].remove(college)
                    trial[gj].append(college)
                    f_trial = scheme_F(trial, df)
                    if f_trial < best - 1e-9:
                        groups, best = trial, f_trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return groups


SOLVERS = {"MILP": solve_milp, "LPT": solve_lpt, "蛇形选秀": solve_snake}
