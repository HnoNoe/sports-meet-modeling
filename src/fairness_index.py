# -*- coding: utf-8 -*-
"""
===============================================================================
公平指数模块 v2：大加权综合指数 + 全维度敏感性分析
赛题：运动会优化比赛模式探索（问题 1 配套评估工具）

------------------------------ AI 使用声明 ------------------------------
【人设计】以下内容由参赛队员设计决定，AI 未参与决策：
  1. 指标体系："互不冲突的全部纳入"原则
     - 度量维度（4 个）：基尼系数、泰尔指数、标准差、极差率
       * 变异系数与标准差在均值固定时完全等价（CV=σ/μ），剔除，避免信息重复
     - 属性维度（3 个）：规模（各组总人数）、性别（各组女生人数）、
       特长生（各组特长生人数）
       * 男生人数 = 总人数 - 女生人数，与已有维度线性相关，不重复纳入
  2. 综合指数两层结构（大加权）：
       F = Σ_a λ_a · F_a,   F_a = Σ_m w_m · f_m(x_a)
       a ∈ {规模, 性别, 特长生}, m ∈ {基尼, 泰尔, 标准差, 极差率}
       λ 和为 1，w 和为 1，共 7 个权重；F ∈ [0,1]，0 = 完全均衡
  3. 各度量归一化方式（人设计）：
     - 基尼系数：天然有界 [0,1]
     - 泰尔指数：T / ln(K)
     - 标准差：  σ / (μ·√(K-1))（相对理论上界）
     - 极差率：  (max-min) / (max+min)
  4. 敏感性分析方案（全维度）：
     - 单维扰动（OAT）：7 个权重逐一从 0 扰动到 1，其余等比分配
     - 全局采样：λ ~ Dirichlet(1,1,1)、w ~ Dirichlet(1,1,1,1) 联合采样 5000 次

【AI 生成】以下代码块为模板化实现，由 AI 按上述设计生成并经队员核对：
  - 四个不平等度量的数值计算函数
  - Dirichlet 采样与统计汇总代码
  - matplotlib 绘图代码

标注约定：
  # >>> AI-GENERATED BLOCK: <说明>
  # <<< AI-GENERATED BLOCK END
===============================================================================
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


# >>> AI-GENERATED BLOCK: 四个不平等度量函数（标准公式实现）
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


# --- 人设计：两层大加权综合指数 ---
def composite(x: np.ndarray, w: np.ndarray) -> float:
    """单一属性向量的四维加权指数。"""
    vals = np.array([f(x) for f in METRICS])
    return float(np.dot(np.asarray(w, dtype=float), vals))


def fairness_score(vectors: dict, w: np.ndarray, lam: np.ndarray) -> dict:
    """
    vectors: {属性名: 各组数值向量}
    w:   4 维度量权重（和为 1）
    lam: 3 维属性权重（和为 1）
    返回各属性指数与综合指数（越小越公平）。
    """
    per_attr = {a: composite(vectors[a], w) for a in ATTR_NAMES}
    f_total = float(np.dot(np.asarray(lam, dtype=float),
                           [per_attr[a] for a in ATTR_NAMES]))
    return {"per_attr": per_attr, "F_综合": f_total}


def load_baseline():
    """基线场景：现状 = 20 个学院各自成组。"""
    college = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    return {a: college[c].to_numpy() for a, c in zip(ATTR_NAMES, ATTR_COLS)}


def sensitivity_analysis(vectors: dict):
    """全维度敏感性分析（7 个权重全部扰动）。"""
    # --- 单维扰动（OAT）：度量权重 4 个 + 属性权重 3 个 ---
    grid = np.linspace(0.05, 0.95, 19)
    oat = []
    for i, mname in enumerate(METRIC_NAMES):          # 扰动度量权重 w_i
        for wi in grid:
            w = np.full(4, (1 - wi) / 3); w[i] = wi
            lam = np.full(3, 1 / 3)
            oat.append((f"度量·{mname}", wi,
                        fairness_score(vectors, w, lam)["F_综合"]))
    for j, aname in enumerate(ATTR_NAMES):            # 扰动属性权重 λ_j
        for lj in grid:
            lam = np.full(3, (1 - lj) / 2); lam[j] = lj
            w = np.full(4, 0.25)
            oat.append((f"属性·{aname}", lj,
                        fairness_score(vectors, w, lam)["F_综合"]))
    df_oat = pd.DataFrame(oat, columns=["权重", "取值", "F_综合"])

    # >>> AI-GENERATED BLOCK: 全局联合采样（λ 与 w 各自 Dirichlet）
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


# >>> AI-GENERATED BLOCK: 敏感性分析绘图（matplotlib 模板代码）
def plot_sensitivity(df_oat: pd.DataFrame, df_glob: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：7 个权重的单维扰动曲线
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

    # 右：全局采样分布
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

    # --- 等权基准 ---
    w0, lam0 = np.full(4, 0.25), np.full(3, 1 / 3)
    base = fairness_score(vectors, w0, lam0)

    print("=== 基线场景：现状（20 学院各自成组）===")
    print("各属性四维明细（行=度量，列=属性）：")
    print(f"{'度量':<8}" + "".join(f"{a:>10}" for a in ATTR_NAMES))
    for mname, f in zip(METRIC_NAMES, METRICS):
        print(f"{mname:<8}" + "".join(f"{f(vectors[a]):>10.4f}" for a in ATTR_NAMES))
    print("\n等权基准（w 各 1/4，λ 各 1/3）：")
    for a in ATTR_NAMES:
        print(f" F_{a} = {base['per_attr'][a]:.4f}")
    print(f" F_综合 = {base['F_综合']:.4f}")

    # --- 全维度敏感性 ---
    df_oat, df_glob = sensitivity_analysis(vectors)

    print("\n=== 单维扰动：各权重对 F_综合 的影响幅度（降序）===")
    rows = []
    for name, sub in df_oat.groupby("权重"):
        s = sub["F_综合"]
        rows.append((name, s.min(), s.max(), s.max() - s.min()))
    for name, lo, hi, rng_ in sorted(rows, key=lambda r: -r[3]):
        print(f" {name:<12} F ∈ [{lo:.4f}, {hi:.4f}]  波动 {rng_:.4f}")

    print("\n=== 全局联合采样（Dirichlet, n=5000）===")
    s = df_glob["F_综合"]
    print(f" F_综合: 均值 {s.mean():.4f}, 标准差 {s.std():.4f}, "
          f"范围 [{s.min():.4f}, {s.max():.4f}]")

    fig_path = RESULTS / "fig_fairness_sensitivity.png"
    plot_sensitivity(df_oat, df_glob, fig_path)
    df_oat.to_csv(RESULTS / "sensitivity_oat.csv", index=False, encoding="utf-8-sig")
    df_glob.to_csv(RESULTS / "sensitivity_global.csv", index=False, encoding="utf-8-sig")

    print(f"\n已保存: {fig_path}")
    print(f"已保存: {RESULTS / 'sensitivity_oat.csv'}")
    print(f"已保存: {RESULTS / 'sensitivity_global.csv'}")


if __name__ == "__main__":
    sys.exit(main())
