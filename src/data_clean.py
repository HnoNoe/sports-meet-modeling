# -*- coding: utf-8 -*-
"""
===============================================================================
第一步：数据清洗
赛题：运动会优化比赛模式探索
数据：学院级（20 学院）+ 专业级（104 专业），含体育特长生数

AI 使用声明：本文件中以 >>> AI-GENERATED BLOCK 标注的代码块为重复性模板
代码（文件读取、类型转换、打印输出等），由 Kimi AI（Moonshot AI 模型）于
2026 年 11 月 14 日 22 时生成，经人工核对；其余逻辑由人工设计。
===============================================================================
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NUM_COLS = ["男生人数", "女生人数", "特长生数", "总人数"]


# >>> AI-GENERATED BLOCK: 多编码自动识别的 CSV 读取（模板代码）
def load_raw(path: Path) -> pd.DataFrame:
    """读取原始 CSV，自动尝试常见中文编码。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法识别文件编码: {path}")
# <<< AI-GENERATED BLOCK END


def clean_table(df: pd.DataFrame, name_cols: list, label: str):
    """
    通用清洗流程（规则由人设计，实现由 AI 辅助）：
    去空白 -> 转数值 -> 缺失删行 -> 非整数取整 -> 负数/重复报错
    -> 行内一致性校验 -> 派生列
    """
    report = []

    # >>> AI-GENERATED BLOCK: 列名/文本去空白 + 数值列批量转换
    df.columns = [str(c).strip() for c in df.columns]
    for c in name_cols:
        df[c] = df[c].astype(str).str.strip()
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # <<< AI-GENERATED BLOCK END

    # --- 人设计：缺失值处理策略——整行删除并记录，不做填充 ---
    n_na = int(df.isna().sum().sum())
    report.append(f"[{label}] 缺失值: {n_na}")
    if n_na:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        report.append(f"[{label}] 删除含缺失值行: {before - len(df)}")

    # --- 人设计：人数必须为整数且非负；负数和重复记录视为数据错误，直接报错 ---
    for col in NUM_COLS:
        if (df[col] % 1 != 0).any():
            report.append(f"[{label}] 警告: {col} 含非整数，已四舍五入")
            df[col] = df[col].round()
        neg = int((df[col] < 0).sum())
        if neg:
            raise ValueError(f"[{label}] {col} 存在 {neg} 个负数，数据异常")
        df[col] = df[col].astype(int)

    dup = df.duplicated(subset=name_cols, keep=False)
    if dup.any():
        keys = df.loc[dup, name_cols].drop_duplicates().values.tolist()
        raise ValueError(f"[{label}] 存在重复记录 {keys}，请人工核查")
    report.append(f"[{label}] 重复记录: 无")

    # --- 人设计：行内校验规则 总人数 = 男 + 女；特长生数为子集 ---
    bad_total = df[df["总人数"] != df["男生人数"] + df["女生人数"]]
    if len(bad_total):
        raise ValueError(f"[{label}] {len(bad_total)} 行 总人数 ≠ 男+女:\n{bad_total}")
    report.append(f"[{label}] 行内校验: 总人数 = 男 + 女 ✓")

    bad_talent = df[df["特长生数"] > df["总人数"]]
    if len(bad_talent):
        raise ValueError(f"[{label}] {len(bad_talent)} 行 特长生数 > 总人数:\n{bad_talent}")
    report.append(f"[{label}] 行内校验: 特长生数 ≤ 总人数 ✓")

    # --- 人设计：派生列定义 ---
    df["男生比例"] = (df["男生人数"] / df["总人数"]).round(4)
    df["特长生比例"] = (df["特长生数"] / df["总人数"]).round(4)

    return df, report


def main():
    college = load_raw(DATA / "college_raw.csv")
    major = load_raw(DATA / "major_raw.csv")

    college, rep_c = clean_table(college, ["学院"], "学院级")
    major, rep_m = clean_table(major, ["学院", "专业"], "专业级")

    # --- 人设计：跨表一致性校验（专业汇总必须等于学院数值，逐项核对）---
    cross = []
    agg = major.groupby("学院")[NUM_COLS].sum()
    for _, row in college.iterrows():
        c = row["学院"]
        if c not in agg.index:
            raise ValueError(f"学院 {c} 在专业表中不存在")
        for col in NUM_COLS:
            if int(agg.loc[c, col]) != int(row[col]):
                raise ValueError(
                    f"学院 {c} 的 {col} 不一致: 学院表={row[col]}, 专业汇总={agg.loc[c, col]}")
    cross.append("跨表校验: 专业级汇总 = 学院级数值 ✓ (全部 20 个学院)")

    # --- 人设计：专业数校验 ---
    major_cnt = major.groupby("学院").size()
    for _, row in college.iterrows():
        if int(major_cnt[row["学院"]]) != int(row["专业数"]):
            raise ValueError(
                f"学院 {row['学院']} 专业数不一致: 学院表={row['专业数']}, 实际={major_cnt[row['学院']]}")
    cross.append(f"跨表校验: 各学院专业数 ✓ (合计 {len(major)} 个专业)")

    # --- 人设计：输出排序（学院按总人数降序，专业表按学院分组、组内按总人数降序）---
    college = college.sort_values("总人数", ascending=False).reset_index(drop=True)
    major = major.sort_values(["学院", "总人数"], ascending=[True, False]).reset_index(drop=True)

    college.to_csv(DATA / "college_clean.csv", index=False, encoding="utf-8-sig")
    major.to_csv(DATA / "major_clean.csv", index=False, encoding="utf-8-sig")

    # >>> AI-GENERATED BLOCK: 清洗报告与描述统计的控制台格式化输出
    print("=== 清洗报告 ===")
    for line in rep_c + rep_m + cross:
        print(" -", line)

    total = college[NUM_COLS].sum()
    print("\n=== 全校汇总 ===")
    print(f" 学院 {len(college)} 个, 专业 {len(major)} 个")
    print(f" 男生 {total['男生人数']}, 女生 {total['女生人数']}, "
          f"特长生 {total['特长生数']}, 总人数 {total['总人数']}")

    print("\n=== 学院级数据（按总人数降序）===")
    print(college.to_string(index=False))

    print("\n=== 描述统计（学院级）===")
    print(college[NUM_COLS].describe().round(2).to_string())

    print("\n已保存: data/college_clean.csv, data/major_clean.csv")
    # <<< AI-GENERATED BLOCK END


if __name__ == "__main__":
    sys.exit(main())
