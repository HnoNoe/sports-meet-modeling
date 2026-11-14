# -*- coding: utf-8 -*-
"""赛制计分仿真引擎：给定参赛单位（学院或联队），模拟整届运动会并输出团体总分。

规则：男子、女子项目分开比；单项前八名得 9,7,…,1 分；团体总分为男女得分之和。
每项目名额按各单位该性别人数比例分配（最大余数法）。普通生发挥 ~ N(0,1)，
特长生发挥 ~ N(δ,1)，每人最多报 talent_events 项；特长生性别按队内性别比拆分。

为支持受控实验，每一届模拟使用由 (cfg.seed, 届次) 派生的独立种子，
保证不同参赛方案在相同随机环境下对照（公共随机数）。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-14 22 时生成，已经人工核对；赛制规则与参数取值由人工设定。
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

POINTS = np.array([9, 7, 6, 5, 4, 3, 2, 1], dtype=float)  # 题面积分规则


@dataclass
class MeetConfig:
    n_events_male: int = 10     # 男子项目数
    n_events_female: int = 10   # 女子项目数
    slots_per_event: int = 40   # 每项目总名额
    talent_events: int = 3      # 每名特长生最多报名项目数
    talent_delta: float = 1.5   # 特长生能力优势（σ 倍数）
    seed: int = 42


# >>> AI-GENERATED BLOCK
def allocate_slots(pop: np.ndarray, slots: int) -> np.ndarray:
    """最大余数法：名额总数严格守恒。"""
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


def simulate_event(entries: np.ndarray, talents_in: np.ndarray,
                   delta: float, rng: np.random.Generator) -> np.ndarray:
    """单个项目：所有参赛者发挥排序，前八名按 POINTS 计分。"""
    n_teams = len(entries)
    team_idx, deltas = [], []
    for k in range(n_teams):
        n_all = entries[k] + talents_in[k]
        if n_all == 0:
            continue
        team_idx += [k] * n_all
        deltas += [0.0] * entries[k] + [delta] * talents_in[k]
    scores = np.zeros(n_teams)
    if not team_idx:
        return scores
    team_idx = np.array(team_idx)
    perf = np.array(deltas) + rng.standard_normal(len(team_idx))
    top = np.argsort(-perf)[:len(POINTS)]
    np.add.at(scores, team_idx[top], POINTS[:len(top)])
    return scores


def simulate_meet(teams: pd.DataFrame, cfg: MeetConfig,
                  n_repeats: int = 1000) -> pd.DataFrame:
    """模拟 n_repeats 届运动会，返回 (n_repeats × n_teams) 得分表。

    teams 索引为单位名，须含 [男生人数, 女生人数, 特长生数] 三列。
    """
    male = teams["男生人数"].to_numpy()
    female = teams["女生人数"].to_numpy()
    talents = teams["特长生数"].to_numpy()

    ratio_m = np.divide(male, male + female,
                        out=np.zeros_like(male, dtype=float),
                        where=(male + female) > 0)
    tal = {"m": np.round(talents * ratio_m).astype(int)}
    tal["f"] = talents - tal["m"]

    slots = {"m": allocate_slots(male, cfg.slots_per_event),
             "f": allocate_slots(female, cfg.slots_per_event)}
    n_events = {"m": cfg.n_events_male, "f": cfg.n_events_female}

    n_teams = len(teams)
    out = np.zeros((n_repeats, n_teams))
    for r in range(n_repeats):
        rng = np.random.default_rng((cfg.seed, r))  # 公共随机数：跨方案同环境
        total = np.zeros(n_teams)
        for g in ("m", "f"):
            n_ev = n_events[g]
            tal_event = np.zeros((n_teams, n_ev), dtype=int)
            for k in range(n_teams):
                for _ in range(tal[g][k]):
                    evs = rng.choice(n_ev, size=min(cfg.talent_events, n_ev),
                                     replace=False)
                    tal_event[k, evs] += 1
            for e in range(n_ev):
                total += simulate_event(slots[g], tal_event[:, e],
                                        cfg.talent_delta, rng)
        out[r] = total

    return pd.DataFrame(out, columns=teams.index)


# >>> AI-GENERATED BLOCK
def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    ranks = scores.rank(ascending=False, axis=1, method="min")
    return pd.DataFrame({
        "平均得分": scores.mean().round(1),
        "得分标准差": scores.std().round(1),
        "夺冠概率": (ranks == 1).mean().round(3),
        "前三概率": (ranks <= 3).mean().round(3),
    }).sort_values("平均得分", ascending=False)
# <<< AI-GENERATED BLOCK END


if __name__ == "__main__":
    college = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    teams = college.set_index("学院")[["男生人数", "女生人数", "特长生数"]]
    scores = simulate_meet(teams, MeetConfig(), n_repeats=1000)
    print(summarize(scores).to_string())
    sys.exit(0)
