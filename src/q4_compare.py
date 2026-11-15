# -*- coding: utf-8 -*-
"""Q4：四种比赛模式的统一仿真比较与理论对账。

模式：现状、Q1 学院合并（K=12 MILP 方案）、Q2 甲乙分层、Q3 特长生流动
与加权积分。统一用引擎仿真 1000 届，横向比较综合公平指数、有奔头学院数、
冠军集中度（HHI 与最大夺冠概率）。公平度量权重除等权基准外，
用熵权法从各模式指标矩阵客观赋权作稳健性对照（回填公平指数权重来源）。
理论对账：无特长生时单项目各学院期望得分有解析式 36·s_i/40
（前八名为 40 名参赛者的均匀随机子集），与引擎单项目仿真比对。

AI 声明：标注 >>> AI-GENERATED BLOCK 的代码块由 Kimi AI（Moonshot AI 模型）
于 2026-11-15 12 时生成，已经人工核对；比较口径与理论推导由人工设计。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_index import METRICS, METRIC_NAMES, composite
from scoring_engine import (MeetConfig, simulate_meet, simulate_event,
                            allocate_slots, POINTS, RESULTS, DATA)

W4 = np.full(4, 0.25)
N_REP = 1000


def load_modes(df: pd.DataFrame) -> dict:
    """构造各模式的参赛单位表；分层模式特殊处理（两场独立比赛）。"""
    base = df.set_index("学院")[["男生人数", "女生人数", "特长生数"]]
    modes = {"现状": base}

    det = pd.read_csv(RESULTS / "q1_schemes_detail.csv", encoding="utf-8-sig")
    milp = det[(det["方法"] == "MILP") & (det["K"] == 12)]
    merged = {}
    for _, r in milp.iterrows():
        members = r["学院"].split("+")
        merged[r["学院"]] = base.loc[members].sum()
    modes["Q1 合并"] = pd.DataFrame(merged).T.astype(int)

    flow = pd.read_csv(RESULTS / "q3a_flow.csv", encoding="utf-8-sig")
    t3 = base.copy()
    t3["特长生数"] = flow["自由流动"].to_numpy()
    modes["Q3 流动"] = t3

    w3 = pd.read_csv(RESULTS / "q3b_weights.csv", encoding="utf-8-sig")
    t4 = base.copy()
    t4["特长生数"] = 0
    modes["Q3 加权"] = (t4, w3.set_index("学院")["权重"])
    return modes


def tier_mode(df: pd.DataFrame) -> dict:
    """Q2 分层：甲、乙两组各自一场，指标按组内计算后合并。"""
    tiers = pd.read_csv(RESULTS / "q2_tiers.csv", encoding="utf-8-sig")
    base = df.set_index("学院")[["男生人数", "女生人数", "特长生数"]]
    win_p, f_list, n_list = {}, [], []
    for name, grp in tiers.groupby("层级"):
        teams = base.loc[grp["学院"]]
        s = simulate_meet(teams, MeetConfig(), n_repeats=N_REP)
        win_p.update(((s.rank(ascending=False, axis=1, method="min") == 1)
                      .mean()).to_dict())
        f_list.append([f(s.mean().to_numpy()) for f in METRICS])
        n_list.append(len(teams))
    win_p = pd.Series(win_p)
    fvals = np.average(f_list, axis=0, weights=n_list)  # 按学院数加权
    return {"metric_vals": fvals, "win_p": win_p}


# >>> AI-GENERATED BLOCK: 熵权法
def entropy_weights(M: np.ndarray) -> np.ndarray:
    """由 (模式 × 指标) 矩阵客观赋权：差异度越大的指标权重越大。"""
    X = np.asarray(M, float)
    X = X - X.min(axis=0) + 1e-12          # 平移保证非负
    P = X / X.sum(axis=0)
    k = 1.0 / np.log(len(X))
    e = -k * (P * np.log(P)).sum(axis=0)
    d = 1 - e
    return d / d.sum()
# <<< AI-GENERATED BLOCK END


def theory_check(df: pd.DataFrame) -> pd.DataFrame:
    """理论对账：无特长生单项目，期望得分 = 36·s_i/40（解析）vs 引擎。"""
    male = df["男生人数"].to_numpy()
    slots = allocate_slots(male, 40)
    rng = np.random.default_rng(0)
    n = 20000
    acc = np.zeros((n, len(df)))
    zero_talent = np.zeros(len(df), dtype=int)
    for r in range(n):
        acc[r] = simulate_event(slots, zero_talent, 0.0, rng)
    sim = acc.mean(axis=0)
    theo = POINTS.sum() * slots / slots.sum()
    out = pd.DataFrame({"学院": df["学院"], "名额": slots,
                        "理论期望": theo.round(3),
                        "仿真期望": sim.round(3)})
    out["相对偏差%"] = np.divide(sim - theo, theo,
                                 out=np.full_like(theo, np.nan),
                                 where=theo > 0) * 100
    out["相对偏差%"] = out["相对偏差%"].round(2)
    return out


def main():
    df = pd.read_csv(DATA / "college_clean.csv", encoding="utf-8-sig")
    modes = load_modes(df)

    rows, metric_mat = {}, {}
    for name, spec in modes.items():
        if isinstance(spec, tuple):  # Q3 加权：原始得分乘权重
            teams, w = spec
            s = simulate_meet(teams, MeetConfig(), n_repeats=N_REP) \
                * w.to_numpy()
        else:
            s = simulate_meet(spec, MeetConfig(), n_repeats=N_REP)
        win_p = (s.rank(ascending=False, axis=1, method="min") == 1).mean()
        fvals = np.array([f(s.mean().to_numpy()) for f in METRICS])
        metric_mat[name] = fvals
        rows[name] = {"win_p": win_p, "metric_vals": fvals}

    rows["Q2 分层"] = tier_mode(df)
    metric_mat["Q2 分层"] = rows["Q2 分层"]["metric_vals"]

    order = ["现状", "Q1 合并", "Q2 分层", "Q3 流动", "Q3 加权"]
    w_ent = entropy_weights(np.vstack([metric_mat[m] for m in order]))
    print("熵权法度量权重:",
          dict(zip(METRIC_NAMES, np.round(w_ent, 3))))

    table = []
    for name in order:
        r = rows[name]
        p = r["win_p"]
        table.append({
            "模式": name,
            "F(等权)": round(float(np.dot(W4, r["metric_vals"])), 4),
            "F(熵权)": round(float(np.dot(w_ent, r["metric_vals"])), 4),
            "有奔头学院数": int((p >= 0.05).sum()),
            "最大夺冠概率": round(float(p.max()), 3),
            "冠军集中度HHI": round(float((p ** 2).sum()), 3),
        })
    tab = pd.DataFrame(table).set_index("模式")
    print("\n=== 四种（五列）模式横向比较 ===")
    print(tab.to_string())

    chk = theory_check(df)
    err = chk["相对偏差%"].abs().max()
    n_zero = int((chk["名额"] == 0).sum())
    print(f"\n理论对账：单项目期望得分解析式 vs 引擎，"
          f"最大相对偏差 {err:.2f}%（20000 次抽样；"
          f"{n_zero} 个学院男子名额为 0，不纳入偏差统计）")

    tab.to_csv(RESULTS / "q4_compare.csv", encoding="utf-8-sig")
    chk.to_csv(RESULTS / "q4_theory_check.csv", index=False,
               encoding="utf-8-sig")
    pd.DataFrame({"度量": METRIC_NAMES, "熵权": w_ent.round(4)}).to_csv(
        RESULTS / "q4_entropy_weights.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存: {RESULTS / 'q4_compare.csv'}")
    print(f"已保存: {RESULTS / 'q4_theory_check.csv'}")
    print(f"已保存: {RESULTS / 'q4_entropy_weights.csv'}")


if __name__ == "__main__":
    sys.exit(main())
