# 运动会优化比赛模式探索

在积分规则（9,7,…,1）不变的前提下，通过学院合并、甲乙分级、特长生流动
或加权积分改革赛制，实现保障公平、兼顾效率的目标。

## 目录

```
data/      数据：college/major 两级，raw 为原始留档，clean 为建模输入
src/       代码：data_clean 清洗，fairness_index 公平指数，
           scoring_engine 赛制仿真，experiment 受控数值实验
docs/      设计文档：审题与 Q1 分析
results/   运行产出（不入库）
```

## 运行

```bash
python src/data_clean.py       # 数据清洗与校验
python src/fairness_index.py   # 公平指数基线与权重敏感性
python src/experiment.py       # 受控实验：δ 敏感性扫描
```

## AI 声明

部分重复性模板代码（文件读写、标准公式、绘图）由 Kimi AI（Moonshot AI 模型）
于 2026-11-14 生成，源文件中以 `>>> AI-GENERATED BLOCK` 标注，均经人工核对。
