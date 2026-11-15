# -*- coding: utf-8 -*-
"""Q3a：特长生跨学院流动模式。

总人数与性别结构不变，重新分配 582 名特长生的归属学院，
使各学院期望团体总分均衡。方法为注水原理（water-filling）：
t_i = (T - b_i)/m，其中基底 b_i 与边际得分 m 由引擎校准——
设一个按全校性别比构成的幻影单位吸收全部 582 名特长生，
幻影人均得分即 m（总池固定时密度不变，m 与归属无关），
此时各学院得分即基底 b_i。分配结果送回引擎迭代修正。
主方案允许自由流动，另设"每院流出不超过 50%"的有界方案作对照。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 12 时生成，已经人工核对；注水框架与校准设计由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import composite
from scoring_engine import MeetConfig, simulate_meet, RESULTS, DATA

W4 = np.full(4, 0.25)
N_REP = 500       # 迭代内仿真届数
N_FINAL = 1000    # 最终评估届数


def run(df: pd.DataFrame, talents: np.ndarray, n_repeats: int = N_REP,
        cfg: MeetConfig = None) -> pd.DataFrame:
    """以指定特长生归属仿真，返回 (n_repeats × 20) 得分表。"""
    teams = df.set_index("学院")[["男生人数", "女生人数"]].copy()
    teams["特长生数"] = np.asarray(talents, dtype=int)
    return simulate_meet(teams, cfg or MeetConfig(), n_repeats=n_repeats)


def calibrate(df: pd.DataFrame, total: int) -> tuple:
    """幻影单位校准：返回 (基底得分 b, 边际得分 m)。"""
    phantom = df.set_index("学院")[["男生人数", "女生人数", "特长生数"]].copy()
    phantom["特长生数"] = 0
    # 幻影按全校性别比取 100 人（名额占比可忽略，仅为性别拆分服务）
    rm = df["男生人数"].sum() / df["总人数"].sum()
    phantom.loc["幻影单位"] = [round(100 * rm), round(100 * (1 - rm)), total]
    s = simulate_meet(phantom, MeetConfig(), n_repeats=N_REP).mean()
    b = s.drop("幻影单位").to_numpy()
    m = float(s["幻影单位"]) / total
    if m <= 0:
        raise ValueError(f"校准得到非正边际得分 m={m}")
    return b, m


def metrics(scores: pd.DataFrame) -> dict:
    win_p = (scores.rank(ascending=False, axis=1, method="min") == 1).mean()
    return {"F": composite(scores.mean().to_numpy(), W4),
            "有奔头数": int((win_p >= 0.05).sum()),
            "最大夺冠概率": float(win_p.max())}


# >>> AI-GENERATED BLOCK: 整数化与注水分配
def integerize(raw: np.ndarray, total: int, lo: np.ndarray) -> np.ndarray:
    """最大余数法取整，满足下界且总和严格等于 total。"""
    raw = np.maximum(np.asarray(raw, float), lo)
    t = np.maximum(np.floor(raw).astype(int), lo)
    rem = int(total - t.sum())
    if abs(rem) > 3 * len(t):
        raise ValueError("取整偏差异常，请检查输入量级")
    if rem > 0:  # 按小数部分从大到小补
        order = np.argsort(-(raw - np.floor(raw)))
        t[order[:rem]] += 1
    elif rem < 0:  # 从超出下界最多者扣
        order = np.argsort(-(t - lo))
        for k in order[:-rem]:
            t[k] -= 1
    return t


def waterfill(b: np.ndarray, m: float, total: int,
              lo: np.ndarray) -> np.ndarray:
    """t_i = max((T - b_i)/m, lo_i)，二分 T 使总和为 total。"""
    lo_T, hi_T = b.min(), b.max() + m * total
    for _ in range(60):
        T = 0.5 * (lo_T + hi_T)
        if np.maximum((T - b) / m, lo).sum() > total:
            hi_T = T
        else:
            lo_T = T
    return integerize((lo_T - b) / m, total, lo)
# <<< AI-GENERATED BLOCK END


def optimize(df: pd.DataFrame, lo_ratio: float = 0.0, rounds: int = 3) -> dict:
    """注水分配 + 引擎迭代修正。lo_ratio 为每院特长生保留比例下限。"""
    a = df["特长生数"].to_numpy()
    total = int(a.sum())
    lo = np.ceil(lo_ratio * a).astype(int)
    b, m = calibrate(df, total)
    s_now = run(df, a).mean().to_numpy()                    # 现状

    t = waterfill(b, m, total, lo)
    best = None
    for k in range(rounds):
        s = run(df, t).mean().to_numpy()
        f = composite(s, W4)
        if best is None or f < best["F"]:
            best = {"t": t.copy(), "s": s.copy(), "F": f, "轮次": k}
        t = integerize(t + (s.mean() - s) / m, total, lo)   # 按实测偏差修正
    best["b"], best["s_now"], best["m"] = b, s_now, m
    return best


# >>> AI-GENERATED BLOCK: 结果图
def plot_q3a(df: pd.DataFrame, free: dict, bounded: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    names = df["学院"].str.replace("学院", "", regex=False)
    order = np.argsort(-df["特长生数"].to_numpy())
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    w = 0.28
    axes[0].bar(x - w, df["特长生数"].to_numpy()[order], w,
                label="现状", color="#999999")
    axes[0].bar(x, free["t"][order], w, label="自由流动", color="#C44E52")
    axes[0].bar(x + w, bounded["t"][order], w, label="有界流动(保留≥50%)",
                color="#4C72B0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names[order], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("特长生数")
    axes[0].set_title("特长生归属：现状 vs 两种再分配方案")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x - w / 2, free["s_now"][order], w, label="现状",
                color="#999999")
    axes[1].bar(x + w / 2, free["s"][order], w, label="自由流动后",
                color="#C44E52")
    axes[1].axhline(free["s"].mean(), color="black", ls="--", lw=1,
                    label=f"均值 {free['s'].mean():.0f}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names[order], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("期望团体总分")
    axes[1].set_title("期望得分：自由流动后显著拉平")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    df = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    a = df["特长生数"].to_numpy()

    print("=== 自由流动方案 ===")
    free = optimize(df, lo_ratio=0.0)
    print(f"边际得分 m = {free['m']:.3f} 分/人，最优轮次 {free['轮次']}，"
          f"F = {free['F']:.4f}")

    print("\n=== 有界流动方案（每院保留 ≥50%）===")
    bounded = optimize(df, lo_ratio=0.5)
    print(f"最优轮次 {bounded['轮次']}，F = {bounded['F']:.4f}")

    print("\n=== 最终评估（1000 届）===")
    for name, t in (("现状", a), ("自由流动", free["t"]),
                    ("有界流动", bounded["t"])):
        r = metrics(run(df, t, N_FINAL))
        print(f"{name:<8} F={r['F']:.4f}  有奔头数={r['有奔头数']}  "
              f"最大夺冠概率={r['最大夺冠概率']:.3f}")

    print("\n=== δ 敏感性（自由流动 vs 现状 F）===")
    for d in (0.5, 1.0, 1.5, 2.0, 2.5):
        cfg = MeetConfig(talent_delta=d)
        f_now = composite(run(df, a, 300, cfg).mean().to_numpy(), W4)
        f_new = composite(run(df, free["t"], 300, cfg).mean().to_numpy(), W4)
        print(f" δ={d:<3}  现状 {f_now:.4f} -> 流动后 {f_new:.4f}")

    out = pd.DataFrame({"学院": df["学院"], "现状特长生": a,
                        "自由流动": free["t"], "有界流动": bounded["t"],
                        "期望得分_现状": free["s_now"].round(1),
                        "期望得分_自由流动": free["s"].round(1)})
    out.to_csv(RESULTS / "q3a_flow.csv", index=False, encoding="utf-8-sig")
    plot_q3a(df, free, bounded, RESULTS / "fig_q3a_flow.png")
    print(f"\n已保存: {RESULTS / 'q3a_flow.csv'}")
    print(f"已保存: {RESULTS / 'fig_q3a_flow.png'}")


if __name__ == "__main__":
    sys.exit(main())
