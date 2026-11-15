# -*- coding: utf-8 -*-
"""Q2：甲乙组分级比赛模式。

思路：以仿真引擎的期望得分作为各学院实力代理，按实力排序后枚举连续切分点，
最小化层内实力方差（层内竞争有悬念）；并用全枚举（2^19）验证连续切分的
最优性。分层后各层单独比赛（层内分配名额、层内单独排名），用引擎评估
每个学院的夺冠概率分布，与现状对比"有奔头的学院数量"。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 11 时生成，已经人工核对；分层目标与评估口径由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import composite
from scoring_engine import MeetConfig, simulate_meet, RESULTS, DATA

N_STRENGTH = 1000   # 估计实力用的仿真届数
N_EVAL = 500        # 评估分层方案用的仿真届数
W4 = np.full(4, 0.25)


def college_strength(df: pd.DataFrame) -> pd.Series:
    """各学院实力 = 现状模式下引擎期望得分。"""
    teams = df.set_index("学院")[["男生人数", "女生人数", "特长生数"]]
    scores = simulate_meet(teams, MeetConfig(), n_repeats=N_STRENGTH)
    return scores.mean()


def best_contiguous_cut(strength: pd.Series) -> dict:
    """排序后枚举 19 个连续切分点，最小化加权层内方差。"""
    s = strength.sort_values(ascending=False)
    v = s.to_numpy()
    best = None
    for cut in range(1, len(v)):
        va, vb = v[:cut], v[cut:]
        # 按层内人数加权：大院多的层方差权重更大
        obj = len(va) * va.var() + len(vb) * vb.var()
        if best is None or obj < best["obj"]:
            best = {"cut": cut, "obj": obj,
                    "甲组": list(s.index[:cut]), "乙组": list(s.index[cut:])}
    return best


def verify_contiguous_optimal(strength: pd.Series) -> tuple:
    """全枚举 2^19 种分层，验证最优分层是否为按实力的连续切分。"""
    v = strength.to_numpy()
    n = len(v)
    best_obj, best_mask = np.inf, None
    for bits in range(1, 1 << (n - 1)):  # 对称去重：固定学院 0 在甲组
        mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool)
        va, vb = v[mask], v[~mask]
        obj = mask.sum() * va.var() + (~mask).sum() * vb.var()
        if obj < best_obj - 1e-12:
            best_obj, best_mask = obj, mask
    # 最优甲组是否恰为实力前 |mask| 名
    top_set = set(np.argsort(-v)[:best_mask.sum()])
    contig = set(np.where(best_mask)[0]) == top_set
    return best_mask, contig


def simulate_tiers(tier_a: list, tier_b: list, df: pd.DataFrame,
                   n_repeats: int = N_EVAL) -> dict:
    """分层仿真：每层单独一场运动会，层内分配名额。"""
    out = {}
    for name, members in (("甲组", tier_a), ("乙组", tier_b)):
        teams = df[df["学院"].isin(members)].set_index("学院")[
            ["男生人数", "女生人数", "特长生数"]]
        scores = simulate_meet(teams, MeetConfig(), n_repeats=n_repeats)
        ranks = scores.rank(ascending=False, axis=1, method="min")
        win_p = (ranks == 1).mean()
        out[name] = {
            "平均得分": scores.mean(),
            "夺冠概率": win_p,
            "F_score": composite(scores.mean().to_numpy(), W4),
            "有奔头学院数": int((win_p >= 0.05).sum()),  # 夺冠概率 ≥5%
        }
    return out


# >>> AI-GENERATED BLOCK: 对比图
def plot_q2(strength: pd.Series, cut: int, res_tier: dict, res_now: dict,
            out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    s = strength.sort_values(ascending=False)
    colors = ["#C44E52"] * cut + ["#4C72B0"] * (len(s) - cut)
    axes[0].bar(range(len(s)), s.values, color=colors)
    axes[0].axvline(cut - 0.5, color="black", ls="--", lw=1.5)
    axes[0].set_xticks(range(len(s)))
    axes[0].set_xticklabels(s.index, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("期望得分（引擎仿真）")
    axes[0].set_title(f"甲乙分层：红=甲组（前 {cut} 名），蓝=乙组")

    cats = ["现状", "甲组", "乙组"]
    vals = [res_now, res_tier["甲组"]["有奔头学院数"],
            res_tier["乙组"]["有奔头学院数"]]
    bars = axes[1].bar(cats, vals, color=["#999999", "#C44E52", "#4C72B0"])
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.1, str(v),
                     ha="center")
    axes[1].set_ylabel("夺冠概率 ≥5% 的学院数")
    axes[1].set_title("竞争性对比：分层让更多学院有奔头")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    df = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")

    print("估计各学院实力（现状模式仿真）……")
    strength = college_strength(df)

    best = best_contiguous_cut(strength)
    print(f"\n最优连续切分：甲组 {best['cut']} 个学院，加权层内方差 {best['obj']:.1f}")

    print("全枚举验证（2^19 种分层）……")
    mask, contig = verify_contiguous_optimal(strength)
    print(f"全枚举最优为连续切分: {contig}")

    print("\n=== 分层方案 ===")
    print("甲组:", "、".join(best["甲组"]))
    print("乙组:", "、".join(best["乙组"]))

    # 现状的竞争性基线
    teams = df.set_index("学院")[["男生人数", "女生人数", "特长生数"]]
    scores_now = simulate_meet(teams, MeetConfig(), n_repeats=N_EVAL)
    win_now = (scores_now.rank(ascending=False, axis=1, method="min") == 1).mean()
    n_now = int((win_now >= 0.05).sum())

    res_tier = simulate_tiers(best["甲组"], best["乙组"], df)

    print(f"\n=== 竞争性对比 ===")
    print(f"现状: 有奔头（夺冠概率≥5%）学院数 = {n_now}")
    for t, r in res_tier.items():
        print(f"{t}: F_score={r['F_score']:.4f}, "
              f"有奔头学院数 = {r['有奔头学院数']}/{len(r['平均得分'])}")
    total = sum(r["有奔头学院数"] for r in res_tier.values())
    print(f"分层后合计: {total} 个学院有奔头（现状 {n_now} 个）")

    recs = []
    for t, members in (("甲组", best["甲组"]), ("乙组", best["乙组"])):
        for m in members:
            recs.append({"层级": t, "学院": m,
                         "期望得分": round(float(strength[m]), 1)})
    pd.DataFrame(recs).to_csv(RESULTS / "q2_tiers.csv",
                              index=False, encoding="utf-8-sig")
    plot_q2(strength, best["cut"], res_tier, n_now,
            RESULTS / "fig_q2_tiers.png")
    print(f"\n已保存: {RESULTS / 'q2_tiers.csv'}")
    print(f"已保存: {RESULTS / 'fig_q2_tiers.png'}")


if __name__ == "__main__":
    sys.exit(main())
