# -*- coding: utf-8 -*-
"""公平指数：基尼、泰尔、标准差、极差率四度量 × 规模、性别、特长生三属性，
两层加权合成综合指数 F（[0,1]，越小越公平），并做全权重敏感性分析。

注：均值固定时变异系数与标准差等价，故度量层不含变异系数；
男生人数可由总人数与女生人数推出，属性层不重复纳入。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-14 22 时生成，已经人工核对；指标体系与加权结构由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

METRIC_NAMES = ["基尼系数", "泰尔指数", "标准差", "极差率"]
ATTR_NAMES = ["规模", "性别", "特长生"]
ATTR_COLS = ["总人数", "女生人数", "特长生数"]


# >>> AI-GENERATED BLOCK
def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.sum() == 0:
        return 0.0
    xs = np.sort(x)
    n = len(xs)
    cum = np.arange(1, n + 1)
    return float((2 * np.sum(cum * xs)) / (n * xs.sum()) - (n + 1) / n)


def theil(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    if mu == 0 or len(x) < 2:
        return 0.0
    r = x / mu
    r = r[r > 0]
    return float(np.mean(r * np.log(r))) / np.log(len(x))


def std_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    if mu == 0 or len(x) < 2:
        return 0.0
    return float(x.std() / (mu * np.sqrt(len(x) - 1)))


def range_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.max() + x.min() == 0:
        return 0.0
    return float((x.max() - x.min()) / (x.max() + x.min()))


METRICS = [gini, theil, std_norm, range_norm]
# <<< AI-GENERATED BLOCK END


def composite(x: np.ndarray, w: np.ndarray) -> float:
    vals = np.array([f(x) for f in METRICS])
    return float(np.dot(np.asarray(w, dtype=float), vals))


def fairness_score(vectors: dict, w: np.ndarray, lam: np.ndarray) -> dict:
    """vectors 为 {属性名: 各组数值向量}；w 与 lam 分别为度量、属性权重。"""
    per_attr = {a: composite(vectors[a], w) for a in ATTR_NAMES}
    f_total = float(np.dot(np.asarray(lam, dtype=float),
                           [per_attr[a] for a in ATTR_NAMES]))
    return {"per_attr": per_attr, "F_综合": f_total}


def load_baseline():
    """现状基线：20 个学院各自成组。"""
    college = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    return {a: college[c].to_numpy() for a, c in zip(ATTR_NAMES, ATTR_COLS)}


def sensitivity_analysis(vectors: dict):
    """7 个权重逐一扰动（OAT）+ Dirichlet 联合采样 5000 次。"""
    grid = np.linspace(0.05, 0.95, 19)
    oat = []
    for i, mname in enumerate(METRIC_NAMES):
        for wi in grid:
            w = np.full(4, (1 - wi) / 3); w[i] = wi
            lam = np.full(3, 1 / 3)
            oat.append((f"度量·{mname}", wi,
                        fairness_score(vectors, w, lam)["F_综合"]))
    for j, aname in enumerate(ATTR_NAMES):
        for lj in grid:
            lam = np.full(3, (1 - lj) / 2); lam[j] = lj
            w = np.full(4, 0.25)
            oat.append((f"属性·{aname}", lj,
                        fairness_score(vectors, w, lam)["F_综合"]))
    df_oat = pd.DataFrame(oat, columns=["权重", "取值", "F_综合"])

    # >>> AI-GENERATED BLOCK
    rng = np.random.default_rng(42)
    n = 5000
    W = rng.dirichlet(np.ones(4), size=n)
    L = rng.dirichlet(np.ones(3), size=n)
    scores = np.array([fairness_score(vectors, W[k], L[k])["F_综合"]
                       for k in range(n)])
    df_glob = pd.DataFrame(W, columns=[f"w_{m}" for m in METRIC_NAMES])
    for j, a in enumerate(ATTR_NAMES):
        df_glob[f"λ_{a}"] = L[:, j]
    df_glob["F_综合"] = scores
    # <<< AI-GENERATED BLOCK END

    return df_oat, df_glob


# >>> AI-GENERATED BLOCK
def plot_sensitivity(df_oat: pd.DataFrame, df_glob: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10.colors
    for k, (name, sub) in enumerate(df_oat.groupby("权重")):
        ls = "-" if name.startswith("度量") else "--"
        axes[0].plot(sub["取值"], sub["F_综合"], ls, marker="o", ms=3,
                     color=colors[k % 10], label=name)
    axes[0].set_xlabel("权重取值（其余权重等比分配剩余份额）")
    axes[0].set_ylabel("综合公平指数 F（越小越公平）")
    axes[0].set_title("全维度单维扰动敏感性（OAT，7 个权重）")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    s = df_glob["F_综合"]
    axes[1].hist(s, bins=50, color="#4C72B0", edgecolor="white")
    axes[1].axvline(s.mean(), color="red", ls="--", label=f"均值 {s.mean():.4f}")
    axes[1].axvline(s.min(), color="green", ls=":", label=f"最小 {s.min():.4f}")
    axes[1].axvline(s.max(), color="orange", ls=":", label=f"最大 {s.max():.4f}")
    axes[1].set_xlabel("综合公平指数 F")
    axes[1].set_ylabel("频次")
    axes[1].set_title(f"全局权重联合采样（Dirichlet, n=5000）\n"
                      f"均值 {s.mean():.4f}，标准差 {s.std():.4f}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    vectors = load_baseline()

    w0, lam0 = np.full(4, 0.25), np.full(3, 1 / 3)
    base = fairness_score(vectors, w0, lam0)

    print("=== 基线场景：现状（20 学院各自成组）===")
    print(f"{'度量':<8}" + "".join(f"{a:>10}" for a in ATTR_NAMES))
    for mname, f in zip(METRIC_NAMES, METRICS):
        print(f"{mname:<8}" + "".join(f"{f(vectors[a]):>10.4f}" for a in ATTR_NAMES))
    print("\n等权基准（w 各 1/4，λ 各 1/3）：")
    for a in ATTR_NAMES:
        print(f" F_{a} = {base['per_attr'][a]:.4f}")
    print(f" F_综合 = {base['F_综合']:.4f}")

    df_oat, df_glob = sensitivity_analysis(vectors)

    print("\n=== 单维扰动影响幅度（降序）===")
    rows = []
    for name, sub in df_oat.groupby("权重"):
        s = sub["F_综合"]
        rows.append((name, s.min(), s.max(), s.max() - s.min()))
    for name, lo, hi, rng_ in sorted(rows, key=lambda r: -r[3]):
        print(f" {name:<12} F ∈ [{lo:.4f}, {hi:.4f}]  波动 {rng_:.4f}")

    s = df_glob["F_综合"]
    print(f"\n全局采样（n=5000）: 均值 {s.mean():.4f}, 标准差 {s.std():.4f}, "
          f"范围 [{s.min():.4f}, {s.max():.4f}]")

    fig_path = RESULTS / "fig_fairness_sensitivity.png"
    plot_sensitivity(df_oat, df_glob, fig_path)
    df_oat.to_csv(RESULTS / "sensitivity_oat.csv", index=False, encoding="utf-8-sig")
    df_glob.to_csv(RESULTS / "sensitivity_global.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: {fig_path}")


if __name__ == "__main__":
    sys.exit(main())
