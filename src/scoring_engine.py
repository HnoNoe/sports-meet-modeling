# -*- coding: utf-8 -*-
"""
===============================================================================
赛制计分仿真引擎
赛题：运动会优化比赛模式探索（问题 2/3/4 的共用基础设施）

功能：给定若干参赛单位（学院或合并后的组），按赛制规则模拟一届运动会，
      输出各单位的团体总分；重复 N 次即得得分分布与排名概率。

赛制规则（人设计，与题面一致）：
  - 单项前八名得分：第 1 名 9 分，第 2~8 名依次 7,6,5,4,3,2,1 分
  - 团体总分 = 男子项目得分 + 女子项目得分
  - 每项目参赛名额固定，按各单位该性别人数比例分配（最大余数法取整）
  - 普通学生发挥 ~ N(0,1)；特长生发挥 ~ N(δ,1)，δ 为能力优势
  - 每名特长生最多参加 talent_events 个项目，项目随机选择
  - 特长生男女归属按本队性别比例拆分（原始数据未给特长生性别，此为假设）

AI 使用声明：本文件中以 >>> AI-GENERATED BLOCK 标注的代码块为重复性模板
代码（最大余数法取整、随机数生成、绘图等），由 Kimi AI（Moonshot AI 模型）
于 2026 年 11 月 14 日 22 时生成，经人工核对；赛制规则与参数取值由人工设计。
===============================================================================
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# 题面积分规则：第 1 名 9 分，第 2~8 名 7~1 分
POINTS = np.array([9, 7, 6, 5, 4, 3, 2, 1], dtype=float)


@dataclass
class MeetConfig:
    """赛制参数（人设计，默认值可在仿真中做敏感性）。"""
    n_events_male: int = 10        # 男子项目数
    n_events_female: int = 10      # 女子项目数
    slots_per_event: int = 40      # 每项目总参赛名额
    talent_events: int = 3         # 每名特长生最多参加项目数
    talent_delta: float = 1.5      # 特长生能力优势（σ 倍数）
    seed: int = 42


# >>> AI-GENERATED BLOCK: 最大余数法名额分配（模板代码）
def allocate_slots(pop: np.ndarray, slots: int) -> np.ndarray:
    """按人口比例把 slots 个名额分给各单位（最大余数法，总数严格守恒）。"""
    pop = np.asarray(pop, dtype=float)
    if pop.sum() == 0:
        return np.zeros(len(pop), dtype=int)
    raw = slots * pop / pop.sum()
    base = np.floor(raw).astype(int)
    remain = slots - base.sum()
    if remain > 0:
        order = np.argsort(-(raw - base))
        base[order[:remain]] += 1
    return base
# <<< AI-GENERATED BLOCK END


# --- 人设计：单项目模拟——发挥=能力均值偏移+单位正态噪声，前八名按 POINTS 计分 ---
def simulate_event(entries: np.ndarray, talent_flags: np.ndarray,
                   talent_delta: float, rng: np.random.Generator) -> np.ndarray:
    """
    entries:      各单位在本项目的普通生参赛人数
    talent_flags: 各单位在本项目出场的特长生人数
    返回各单位在本项目的得分。
    """
    n_teams = len(entries)
    scores = np.zeros(n_teams)
    team_idx, deltas = [], []
    for k in range(n_teams):
        n_all = entries[k] + talent_flags[k]
        if n_all == 0:
            continue
        team_idx += [k] * n_all
        deltas += [0.0] * entries[k] + [talent_delta] * talent_flags[k]
    if not team_idx:
        return scores
    team_idx = np.array(team_idx)
    perf = np.array(deltas) + rng.standard_normal(len(team_idx))
    top = np.argsort(-perf)[:len(POINTS)]          # 前八名
    np.add.at(scores, team_idx[top], POINTS[:len(top)])
    return scores


def simulate_meet(teams: pd.DataFrame, cfg: MeetConfig,
                  n_repeats: int = 1000) -> pd.DataFrame:
    """
    模拟 n_repeats 届运动会。
    teams: DataFrame，须含列 [男生人数, 女生人数, 特长生数]，索引为单位名。
    返回 (n_repeats × n_teams) 的得分 DataFrame。
    """
    rng = np.random.default_rng(cfg.seed)
    male = teams["男生人数"].to_numpy()
    female = teams["女生人数"].to_numpy()
    talents = teams["特长生数"].to_numpy()

    # --- 人设计：特长生按队内性别比例拆分到男/女子项目 ---
    ratio_m = np.divide(male, male + female,
                        out=np.zeros_like(male, dtype=float),
                        where=(male + female) > 0)
    tal_m = np.round(talents * ratio_m).astype(int)
    tal_f = talents - tal_m

    # --- 人设计：名额分配只与性别人口挂钩（规模优势的来源）---
    slots_m = allocate_slots(male, cfg.slots_per_event)
    slots_f = allocate_slots(female, cfg.slots_per_event)

    n_teams = len(teams)
    out = np.zeros((n_repeats, n_teams))
    n_ev = {"m": cfg.n_events_male, "f": cfg.n_events_female}

    for r in range(n_repeats):
        total = np.zeros(n_teams)
        # >>> AI-GENERATED BLOCK: 特长生参赛项目的随机排程
        for g, n_events, slots, tal in (
                ("m", n_ev["m"], slots_m, tal_m),
                ("f", n_ev["f"], slots_f, tal_f)):
            # tal_event[k, e] = 队 k 在项目 e 出场的特长人数
            tal_event = np.zeros((n_teams, n_events), dtype=int)
            for k in range(n_teams):
                for _ in range(tal[k]):
                    evs = rng.choice(n_events,
                                     size=min(cfg.talent_events, n_events),
                                     replace=False)
                    tal_event[k, evs] += 1
            for e in range(n_events):
                total += simulate_event(slots, tal_event[:, e],
                                        cfg.talent_delta, rng)
        # <<< AI-GENERATED BLOCK END
        out[r] = total

    return pd.DataFrame(out, columns=teams.index)


# >>> AI-GENERATED BLOCK: 结果汇总与绘图（模板代码）
def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """得分分布汇总：均值、标准差、夺冠概率、前三概率。"""
    ranks = scores.rank(ascending=False, axis=1, method="min")
    s = pd.DataFrame({
        "平均得分": scores.mean().round(1),
        "得分标准差": scores.std().round(1),
        "夺冠概率": (ranks == 1).mean().round(3),
        "前三概率": (ranks <= 3).mean().round(3),
    }).sort_values("平均得分", ascending=False)
    return s


def plot_scores(summary: pd.DataFrame, out: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(summary)), summary["平均得分"],
           yerr=summary["得分标准差"], capsize=2,
           color=["#C44E52" if i < 3 else "#4C72B0" for i in range(len(summary))])
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("平均团体总分（误差棒=标准差）")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
# <<< AI-GENERATED BLOCK END


def main():
    """演示：现状模式（20 学院各自参赛）下模拟 1000 届运动会。"""
    college = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    teams = college.set_index("学院")[["男生人数", "女生人数", "特长生数"]]

    cfg = MeetConfig()
    print(f"赛制: 男子 {cfg.n_events_male} 项 + 女子 {cfg.n_events_female} 项, "
          f"每项 {cfg.slots_per_event} 个名额, 特长生优势 δ={cfg.talent_delta}σ")
    print("模拟 1000 届运动会中……")

    scores = simulate_meet(teams, cfg, n_repeats=1000)
    s = summarize(scores)

    print("\n=== 现状模式：各学院得分汇总（按平均得分降序）===")
    print(s.to_string())

    scores.to_csv(RESULTS / "sim_status_quo_scores.csv",
                  index=False, encoding="utf-8-sig")
    s.to_csv(RESULTS / "sim_status_quo_summary.csv", encoding="utf-8-sig")
    plot_scores(s, RESULTS / "fig_sim_status_quo.png",
                "现状模式：1000 届运动会各学院平均团体总分")
    print(f"\n已保存: {RESULTS / 'sim_status_quo_scores.csv'}")
    print(f"已保存: {RESULTS / 'sim_status_quo_summary.csv'}")
    print(f"已保存: {RESULTS / 'fig_sim_status_quo.png'}")


if __name__ == "__main__":
    sys.exit(main())
