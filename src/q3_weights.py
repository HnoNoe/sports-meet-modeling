# -*- coding: utf-8 -*-
"""Q3b：特长生不参赛时的加权积分模式。

特长生全部不参赛时期望得分近似与人数成正比，大院垄断。
为抵消规模优势，对学院 i 的原始得分乘权重 w_i = (T/b_i)^α，
其中 b_i 为该院（无特长生）期望得分、T 为全院均值；
权重由期望得分推导而非人为指定。α=0 退化为不加权，α=1 时期望
加权得分完全拉平。α 过大会使小院权重虚高、单场爆冷决定冠军
（矫枉过正），故扫描 α ∈ [0,1]，以 F 与夺冠概率分布共同选定 α*。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 12 时生成，已经人工核对；加权形式与选参规则由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import composite
from scoring_engine import MeetConfig, simulate_meet, RESULTS, DATA

W4 = np.full(4, 0.25)
N_REP = 500
F_TARGET = 0.05      # 公平性门槛
P_WIN_CAP = 0.30     # 单一学院夺冠概率上限（防矫枉过正）


def weighted_metrics(raw: pd.DataFrame, w: np.ndarray) -> dict:
    s = raw * w
    win_p = (s.rank(ascending=False, axis=1, method="min") == 1).mean()
    return {"F": composite(s.mean().to_numpy(), W4),
            "有奔头数": int((win_p >= 0.05).sum()),
            "最大夺冠概率": float(win_p.max())}


# >>> AI-GENERATED BLOCK: 结果图
def plot_q3b(sweep: pd.DataFrame, w_star: pd.Series, alpha_star: float,
             out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax2 = axes[0].twinx()
    axes[0].plot(sweep["alpha"], sweep["F"], "o-", color="#C44E52",
                 label="F（左轴）")
    ax2.plot(sweep["alpha"], sweep["最大夺冠概率"], "s--", color="#4C72B0",
             label="最大夺冠概率（右轴）")
    axes[0].axvline(alpha_star, color="black", ls=":", lw=1)
    axes[0].text(alpha_star + 0.02, sweep["F"].max() * 0.9,
                 f"α* = {alpha_star}", fontsize=10)
    axes[0].set_xlabel("加权强度 α")
    axes[0].set_ylabel("综合公平指数 F")
    ax2.set_ylabel("最大夺冠概率")
    axes[0].set_title("加权强度扫描：公平性与爆冷风险的权衡")
    lines = axes[0].get_lines() + ax2.get_lines()
    axes[0].legend(lines, [l.get_label() for l in lines], loc="center right")
    axes[0].grid(alpha=0.3)

    ws = w_star.sort_values(ascending=False)
    axes[1].bar(range(len(ws)), ws.values, color="#4C72B0")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set_xticks(range(len(ws)))
    axes[1].set_xticklabels(
        [n.replace("学院", "") for n in ws.index], rotation=45, ha="right",
        fontsize=8)
    axes[1].set_ylabel("积分权重 w_i")
    axes[1].set_title(f"各学院积分权重（α* = {alpha_star}，均值归一）")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    df = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    teams = df.set_index("学院")[["男生人数", "女生人数"]].copy()
    teams["特长生数"] = 0  # 特长生不参赛

    raw = simulate_meet(teams, MeetConfig(), n_repeats=N_REP)
    b = raw.mean().to_numpy()
    T = b.mean()
    print(f"无特长生期望得分：最强 {b.max():.1f}，最弱 {b.min():.1f}，"
          f"比值 {b.max() / b.min():.2f}")

    rows = []
    for alpha in np.arange(0, 1.01, 0.1):
        w = (T / b) ** alpha
        w *= len(w) / w.sum()          # 均值归一
        r = weighted_metrics(raw, w)
        rows.append({"alpha": round(float(alpha), 1), "F": r["F"],
                     "有奔头数": r["有奔头数"],
                     "最大夺冠概率": r["最大夺冠概率"],
                     "权重比(max/min)": float(w.max() / w.min())})
    sweep = pd.DataFrame(rows)
    print("\n=== α 扫描 ===")
    print(sweep.to_string(index=False))

    # 选参：满足公平门槛与爆冷约束的最小 α；无则取 F 最小者
    ok = sweep[(sweep["F"] <= F_TARGET) & (sweep["最大夺冠概率"] <= P_WIN_CAP)]
    alpha_star = float(ok["alpha"].min()) if len(ok) else \
        float(sweep.loc[sweep["F"].idxmin(), "alpha"])
    print(f"\n选定 α* = {alpha_star}")

    w_star = pd.Series((T / b) ** alpha_star, index=teams.index)
    w_star *= len(w_star) / w_star.sum()
    r0 = weighted_metrics(raw, np.ones(len(b)))
    rs = weighted_metrics(raw, w_star.to_numpy())
    print(f"不加权: F={r0['F']:.4f}, 有奔头数={r0['有奔头数']}")
    print(f"加权后: F={rs['F']:.4f}, 有奔头数={rs['有奔头数']}, "
          f"最大夺冠概率={rs['最大夺冠概率']:.3f}")

    sweep.to_csv(RESULTS / "q3b_alpha_sweep.csv", index=False,
                 encoding="utf-8-sig")
    pd.DataFrame({"学院": w_star.index,
                  "权重": w_star.round(4)}).to_csv(
        RESULTS / "q3b_weights.csv", index=False, encoding="utf-8-sig")
    plot_q3b(sweep, w_star, alpha_star, RESULTS / "fig_q3b_weights.png")
    print(f"\n已保存: {RESULTS / 'q3b_alpha_sweep.csv'}")
    print(f"已保存: {RESULTS / 'q3b_weights.csv'}")
    print(f"已保存: {RESULTS / 'fig_q3b_weights.png'}")


if __name__ == "__main__":
    sys.exit(main())
