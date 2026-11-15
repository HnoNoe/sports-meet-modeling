import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType, Document, Footer, Header, HeadingLevel,
  ImportedXmlComponent, Packer, PageNumber, Paragraph, ShadingType,
  Table, TableCell, TableRow, TextRun, WidthType, ImageRun,
  Math as DocxMath, MathRun, MathFraction, MathSubScript, MathSuperScript,
  MathSubSuperScript, MathRadical, convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) throw new Error("Usage: node create_paper.js out.docx");
const paperDir = path.dirname(outputPath);
const resultsDir = path.resolve(paperDir, "..", "results");

const T = String.raw;
const font = { ascii: "Times New Roman", hAnsi: "Times New Roman",
               cs: "Times New Roman", eastAsia: "SimSun" };
const palette = { dark: "263238", light: "78909C", fill: "EEF3F6" };

const run = (text, options = {}) =>
  new TextRun({ text, font, size: 24, ...options });
const para = (children, options = {}) => new Paragraph({
  spacing: { after: 120, line: 320 },
  ...options,
  children: Array.isArray(children) ? children : [children],
});

// ---------- 轻量 LaTeX -> docx Math 转换 ----------
const SYMS = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", varepsilon: "ε",
  lambda: "λ", mu: "μ", sigma: "σ", tau: "τ", pi: "π", ell: "ℓ",
  in: "∈", le: "≤", ge: "≥", approx: "≈", cdot: "·", times: "×",
  dots: "…", ldots: "…", sum: "∑", forall: "∀", infty: "∞",
  propto: "∝", max: "max", min: "min", ln: "ln",
};
const IGNORE = new Set(["qquad", "quad", "Big", "Bigg", "big", "left",
                        "right", ",", ";", " ", "displaystyle", "!"]);

function tex(s) {
  let i = 0;
  const asArr = (el) => (Array.isArray(el) ? el : [el]);

  function readBraced() {
    if (s[i] === "{") {
      i++; let depth = 1; const start = i;
      while (i < s.length && depth) {
        if (s[i] === "{") depth++;
        else if (s[i] === "}") { depth--; if (!depth) break; }
        i++;
      }
      const inner = s.slice(start, i); i++;
      return inner;
    }
    if (s[i] === "\\") {
      const m = /^\\([a-zA-Z]+|.)/.exec(s.slice(i));
      i += m[1].length + 1;
      return "\\" + m[1];
    }
    return s[i++] ?? "";
  }

  function atom() {
    const ch = s[i];
    if (ch === "{") return tex(readBraced());
    if (ch === "\\") {
      const m = /^\\([a-zA-Z]+|.)/.exec(s.slice(i));
      const cmd = m[1]; i += cmd.length + 1;
      if (cmd === "frac" || cmd === "dfrac") {
        const n = readBraced(), d = readBraced();
        return new MathFraction({ numerator: tex(n), denominator: tex(d) });
      }
      if (cmd === "sqrt")
        return new MathRadical({ children: tex(readBraced()) });
      if (cmd === "bar") {
        const inner = readBraced();
        return new MathRun(inner.replaceAll("\\", "") + "̄");
      }
      if (cmd === "text" || cmd === "mathrm")
        return new MathRun(readBraced());
      if (IGNORE.has(cmd)) return null;
      if (SYMS[cmd]) return new MathRun(SYMS[cmd]);
      return new MathRun(cmd);
    }
    i++;
    return new MathRun(ch);
  }

  function group() {
    let els = [];
    while (i < s.length && s[i] !== "}") {
      let el = atom();
      if (!el) continue;
      let sub = null, sup = null;
      while (s[i] === "_" || s[i] === "^") {
        const kind = s[i++];
        const children = tex(readBraced());
        if (kind === "_") sub = children; else sup = children;
      }
      if (sub && sup)
        el = new MathSubSuperScript({ children: asArr(el),
                                      subScript: sub, superScript: sup });
      else if (sub)
        el = new MathSubScript({ children: asArr(el), subScript: sub });
      else if (sup)
        el = new MathSuperScript({ children: asArr(el), superScript: sup });
      els.push(...asArr(el));
    }
    return els;
  }
  return group();
}

const mathInline = (s) => new DocxMath({ children: tex(s) });
const mathDisplay = (s) =>
  para([mathInline(s)], { alignment: AlignmentType.CENTER,
                          spacing: { before: 80, after: 160 } });

// 段落富文本：$...$ 为行内公式，**...** 为加粗
function rich(text, baseOpts = {}) {
  const parts = text.split(/(\*\*[^*]+\*\*|\$[^$]+\$)/g).filter(Boolean);
  return parts.flatMap((seg) => {
    if (seg.startsWith("**") && seg.endsWith("**"))
      return rich(seg.slice(2, -2), { bold: true, ...baseOpts });
    if (seg.startsWith("$") && seg.endsWith("$"))
      return mathInline(seg.slice(1, -1));
    return run(seg, baseOpts);
  });
}

const p = (text) => para(rich(text), {
  indent: { firstLine: convertInchesToTwip(0.33) } });
const pNoIndent = (text) => para(rich(text));
const h1 = (text) => para(run(text, { bold: true, size: 32, color: palette.dark }),
  { heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 160 } });
const h2 = (text) => para(run(text, { bold: true, size: 27, color: palette.dark }),
  { heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 } });
const h3 = (text) => para(run(text, { bold: true, size: 24, color: palette.dark }),
  { heading: HeadingLevel.HEADING_3, spacing: { before: 180, after: 100 } });
const caption = (text) => para(run(text, { size: 20, color: palette.light }),
  { alignment: AlignmentType.CENTER, spacing: { after: 200 } });

// PNG 原始尺寸（IHDR）
function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}
function figure(file, cap, widthPx = 580) {
  const buf = fs.readFileSync(path.join(resultsDir, file));
  const { w, h } = pngSize(buf);
  return [
    para(new ImageRun({ type: "png", data: buf,
      transformation: { width: widthPx, height: Math.round(widthPx * h / w) } }),
      { alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 } }),
    caption(cap),
  ];
}

// 表格
function table(widths, header, rows, boldCols = []) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, opts = {}) => new TableCell({
    children: [para(rich(String(text)), { spacing: { after: 0, line: 260 } })],
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    ...opts,
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((hd, k) => cell(hd, {
          shading: { type: ShadingType.CLEAR, fill: palette.fill },
          width: { size: widths[k], type: WidthType.DXA },
        })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, k) => cell(c, {
          width: { size: widths[k], type: WidthType.DXA },
        })),
      })),
    ],
  });
}

// 目录
const xmlEscape = (v) => String(v)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const toc = (entries) => {
  const cached = entries.map(({ title, level, page }) => {
    const indent = Math.max(0, level - 1) * 360;
    return `<w:p><w:pPr><w:pStyle w:val="TOC${level}"/>
      <w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>
      <w:ind w:left="${indent}"/></w:pPr>
      <w:r><w:t>${xmlEscape(title)}</w:t></w:r><w:r><w:tab/></w:r>
      <w:r><w:t>${page}</w:t></w:r></w:p>`;
  }).join("");
  return ImportedXmlComponent.fromXmlString(`<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:sdtPr><w:alias w:val="目录"/></w:sdtPr>
    <w:sdtContent>
      <w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/>
        <w:instrText xml:space="preserve"> TOC \\o &quot;1-3&quot; \\h \\z \\u </w:instrText>
        <w:fldChar w:fldCharType="separate"/></w:r></w:p>
      ${cached}
      <w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
    </w:sdtContent>
  </w:sdt>`).root[0];
};

// ---------- 正文 ----------
const children = [];
const push = (...els) => children.push(...els);

// ===== 标题与摘要 =====
push(
  para(run("运动会竞赛组织模式优化", { bold: true, size: 40, color: palette.dark }),
    { heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 80 } }),
  para(run("——基于公平指数与蒙特卡洛仿真的制度设计",
      { bold: true, size: 28, color: palette.dark }),
    { alignment: AlignmentType.CENTER, spacing: { after: 360 } }),
  h1("摘要"),
  p(T`本文研究高校运动会团体赛制的公平性优化问题。在积分规则（单项前八名 9,7,…,1 分）不变的前提下，通过改变参赛单位的组织方式，使各单位的期望得分能力趋于均衡。`),
  p(T`我们首先建立**双层加权综合公平指数**：以基尼系数、泰尔指数、标准差、极差率四种差异度量为一层，规模、性别、特长生三种属性为另一层，加权合成 $F\in[0,1]$（越小越公平），并以单维扰动与 Dirichlet 全局采样检验权重稳健性。同时搭建**赛制仿真引擎**：按题面规则模拟整届运动会（男女各 10 项、名额按性别人数最大余数法分配、特长生能力优势 $\delta$），用公共随机数保证不同方案在相同随机环境下对照，并以解析解对账验证（偏差 2.25%）。`),
  p(T`**问题一**将合并分组建成混合整数线性规划（L1 偏差代理目标 + 真实 $F$ 局部精修），与匹配法、LPT、蛇形选秀三法对照。推荐 $K=12$ 方案（8 对合并 + 4 个单学院组），人口公平指数由 0.227 降至 0.1145，降幅约 50%；四法收敛于 0.11~0.16 窄带，结论稳健。同时发现人口公平不等于得分公平：特长生随学院走，得分公平指数几乎不变。`),
  p(T`**问题二**以引擎期望得分为实力代理做甲乙分层，排序后枚举切分点最小化层内方差，并经 $2^{19}$ 全枚举验证连续切分即全局最优。最优方案恰好 10/10 分层，有夺冠希望的学院由 5 个增至 8 个；人数最多的机械学院因性别结构失衡落入乙组，说明“规模不等于实力”。`),
  p(T`**问题三**针对特长生提出两种制度。跨学院流动模式建成注水分配模型：以幻影单位法由引擎校准边际得分 $m=1.26$ 分/人，迭代修正后得到近均匀分配，得分公平指数由 0.263 降至 0.015，且最优解中无一学院需流出过半；特长生不参赛时的加权积分模式按 $w_i=(T/b_i)^\alpha$ 由期望得分推导权重，扫描选定 $\alpha^*=0.8$（$F=0.037$），并指出 $\alpha=1$ 完全拉平会因方差失衡造成矫枉过正。`),
  p(T`**问题四**用引擎对现状、合并、分层、流动、加权五种情形统一仿真 1000 届。结果表明流动模式最优（$F=0.015$，13 个学院有夺冠希望），加权模式次之且实施成本最低；合并模式只解决组织合规，不改变得分格局。据此推荐“组织上并、资源上均”的组合方案。熵权法回填的度量权重（0.225~0.306）接近等权，两种权重下模式排序一致，结论稳健。`),
  para([run("关键词：", { bold: true }),
        run("竞赛制度设计；公平指数；整数规划；注水原理；蒙特卡洛仿真；熵权法")],
    { spacing: { before: 160, after: 240 } }),
);

// ===== 目录 =====
push(
  para(run("目录", { bold: true, size: 30, color: palette.dark }),
    { heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 } }),
  para(run("右键目录选择“更新域”可刷新页码。", { italics: true, color: palette.light })),
  toc([
    { title: "摘要", level: 1, page: 1 },
    { title: "一、问题重述", level: 1, page: 3 },
    { title: "二、模型假设", level: 1, page: 3 },
    { title: "三、符号说明", level: 1, page: 4 },
    { title: "四、模型准备", level: 1, page: 4 },
    { title: "五、问题一：学院合并分组优化", level: 1, page: 5 },
    { title: "六、问题二：甲乙组分级模式", level: 1, page: 7 },
    { title: "七、问题三：特长生制度调整", level: 1, page: 8 },
    { title: "八、问题四：四种模式的统一比较", level: 1, page: 10 },
    { title: "九、敏感性分析", level: 1, page: 11 },
    { title: "十、模型评价", level: 1, page: 12 },
    { title: "参考文献", level: 1, page: 12 },
    { title: "附录 A：AI 使用声明", level: 1, page: 13 },
    { title: "附录 B：代码清单", level: 1, page: 13 },
  ]),
);

// ===== 一、问题重述 =====
push(
  h1("一、问题重述"),
  p(T`某大学运动会设男子、女子项目，单项第一名得 9 分、第二至八名依次得 7~1 分，各学院男女生累积得分构成团体总分。全校 20 个学院、104 个专业、28523 名学生（男 14855、女 13668），另有特长生 582 人。学院之间规模悬殊（693~1964 人）、性别比例两极（男生占比 12%~88%）、特长生分布不均（8~49 人），大院长期垄断团体前列，小院参与热情受挫。`),
  p(T`在积分规则不变的前提下，研究四种比赛组织模式：`),
  p(T`1. **学院合并**：允许若干学院合并参赛，要求分组数不少于 12，各组总人数与性别比均衡，并讨论分组方案的公平指数；`),
  p(T`2. **甲乙分级**：按实力把学院分为甲乙两组分别比赛，设计分层方案；`),
  p(T`3. **特长生调整**：(a) 特长生可跨学院流动时，设计再分配方案；(b) 特长生不参赛时，设计加权积分方案；`),
  p(T`4. **模式比较**：用仿真或理论推导比较上述模式与现状，给出推荐。`),
);

// ===== 二、模型假设 =====
push(
  h1("二、模型假设"),
  p(T`1. 普通学生的单项发挥独立同分布，记为 $N(0,1)$，即普通生之间无系统性水平差异，得分机会只取决于名额；`),
  p(T`2. 特长生单项发挥为 $N(\delta,1)$，基准取 $\delta=1.5$，并在 $\delta\in[0.5,2.5]$ 上做敏感性检验；`),
  p(T`3. 男子、女子项目各 10 项，每项 40 个参赛名额，按各单位该性别人数以最大余数法分配；`),
  p(T`4. 每名特长生最多报名 3 个项目，项目选择在其可报范围内均匀随机；特长生性别按所在学院性别比例拆分（数据未提供特长生性别分项）；`),
  p(T`5. 积分规则（9,7,…,1）在四种模式下不变，模式只改变参赛单位的组织方式；`),
  p(T`6. 各届运动会相互独立；不同方案的比较在公共随机数下进行，以消除抽样噪声对排序的干扰。`),
);

// ===== 三、符号说明 =====
push(
  h1("三、符号说明"),
  table([2600, 6200], ["符号", "含义"], [
    ["$s_i,f_i,t_i$", "学院 $i$ 的总人数、女生人数、特长生数"],
    ["$K$", "合并分组数，$K\\in\\{12,\\dots,20\\}$"],
    ["$x_{ij}$", "0-1 变量，学院 $i$ 是否分入组 $j$"],
    ["$G,T,\\sigma,R$", "基尼系数、泰尔指数、标准化标准差、极差率"],
    ["$F$", "综合公平指数，$F\\in[0,1]$，越小越公平"],
    ["$F_{\\text{pop}},F_{\\text{score}}$", "人口口径（规模+性别）与得分口径的公平指数"],
    ["$\\delta$", "特长生能力优势（标准差倍数）"],
    ["$b_i$", "无特长生时学院 $i$ 的期望得分（基底）"],
    ["$m$", "特长生的边际期望得分"],
    ["$w_i,\\alpha$", "加权积分中学院 $i$ 的积分权重与加权强度"],
    ["$p_i$", "学院 $i$ 的夺冠概率"],
  ]),
  para(run("")),
);

// ===== 四、模型准备 =====
push(
  h1("四、模型准备"),
  h2("4.1 数据清洗"),
  p(T`学院级（20 条）与专业级（104 条）两张表，经行内校验（总人数 = 男生 + 女生）与跨表校验（专业汇总 = 学院数值），全部通过；全校总人数 28523 与题面一致。清洗后字段为学院、男生人数、女生人数、特长生数、总人数、专业数。`),
  h2("4.2 双层加权综合公平指数"),
  p(T`竞赛制度的“公平”是多维的，单一度量易被操纵。我们取四种最常见的差异度量：`),
  p(T`基尼系数、泰尔指数、标准化标准差（均值固定时变异系数与标准差等价，不重复纳入）、极差率，定义如下：`),
  mathDisplay(T`G=\dfrac{2\sum_k kx_{(k)}}{n\sum x_i}-\dfrac{n+1}{n},\qquad T=\dfrac{1}{\ln n}\cdot\dfrac{1}{n}\sum_i \dfrac{x_i}{\bar x}\ln\dfrac{x_i}{\bar x}`),
  mathDisplay(T`\sigma_N=\dfrac{\sigma}{\bar x\sqrt{n-1}},\qquad R=\dfrac{x_{\max}-x_{\min}}{x_{\max}+x_{\min}}`),
  p(T`四者均归一化到 $[0,1]$、越小越公平，经度量层权重 $w$ 合成单属性指数；属性层取规模（总人数）、性别（女生数）、特长生三属性（男生数可由前两者推出，不重复纳入），经权重 $\lambda$ 合成`),
  mathDisplay(T`F=\sum_{a}\lambda_a\sum_{m}w_m\cdot M_m(x^{(a)}).`),
  p(T`基准权重取等权（$w_m=1/4,\lambda_a=1/3$）；全文最后由熵权法客观赋权回填，检验结论对权重的稳健性。现状基线 $F=0.2392$，其中人口口径 $F_{\text{pop}}=0.227$。`),
  h2("4.3 赛制仿真引擎"),
  p(T`按假设 1~4 实现整届运动会的蒙特卡洛仿真：每届依次进行男女各 10 个单项，单项内所有参赛者发挥排序，前八名按 9,7,…,1 计分。每届使用由（全局种子，届次）派生的独立种子，保证不同参赛方案在相同随机环境下对照。引擎同时输出各单位的得分序列、排名序列，由此可得期望得分、夺冠概率、前三概率等全部派生指标。`),
  p(T`引擎正确性经两条途径验证：一是无特长生极限下，单项前八名是 40 名参赛者的均匀随机子集，学院 $i$ 单项目期望得分有解析式 $36s_i/40$，与引擎 20000 次抽样比对最大相对偏差 2.25%（抽样误差量级）；二是 $\delta$ 扫描下各指标变化方向与机理一致。`),
);

// ===== 五、问题一 =====
push(
  h1("五、问题一：学院合并分组优化"),
  h2("5.1 模型建立"),
  p(T`题目要求“若干学院合并、组数不低于 12、各组总人数与性别比均衡”。决策单元为 20 个学院（学院整体进组，不拆散），设 $x_{ij}\in\{0,1\}$ 表示学院 $i$ 是否分入组 $j$，约束`),
  mathDisplay(T`\sum_j x_{ij}=1\ (\forall i),\qquad \sum_i x_{ij}\ge 1\ (\forall j).`),
  p(T`组数 $K$ 本身是决策变量：$K=20$ 即现状，$K=12$ 改动最大也最均衡，公平与组织成本的权衡由 $F$-$K$ 曲线呈现。`),
  p(T`目标取人口口径公平：组总人数 $S_j$ 与组女生数 $F_j$ 均衡（总人数均衡时“女生数均衡”与“性别比均衡”近似等价，且与公平指数口径一致）。题面未要求特长生均衡，特长生不进目标函数，仅在结果中报告。`),
  p(T`综合公平指数含基尼、泰尔等非线性项，不能直接进整数规划，故采用两阶段法：`),
  p(T`**阶段一（MILP 代理目标）**：L1 偏差引入辅助变量线性化，用 HiGHS 精确求解，$\mu$ 取两属性标准差之比；`),
  mathDisplay(T`\min\ \sum_j|S_j-\bar S|+\mu\sum_j|F_j-\bar F|`),
  p(T`**阶段二（局部精修）**：以 MILP 解为起点，以真实 $F_{\text{pop}}$ 为目标做单学院调组局部搜索，直到无改进，消除代理目标与真实 $F$ 的口径差。`),
  h2("5.2 求解与多方法对照"),
  p(T`对 $K=12,\dots,20$ 全扫描，并以三种独立方法对照：匹配法（男多院配女多院的最优匹配，多项式时间精确解）、LPT（最长处理时间启发式，有 $4/3$ 近似比保证）、蛇形选秀（可解释基线）。`),
  p(T`**推荐方案（$K=12$，$F_{\text{pop}}=0.1145$）**：8 对合并 + 4 个单学院组——机械+艺术、化工+数学、计算机+教育、法学+电气、土木+哲学、理学+物理、材料+农学、电子+历史；文、医、外语、经管四院单飞。合并逻辑清晰：男多院配女多院、大院配小院，各组总人数 2483~3176，女生占比 0.36~0.45（单飞学院除外）。`),
  table([2200, 1800, 1800, 3000], ["方法", "$F_{\\text{pop}}$", "$F_{\\text{score}}$", "特点"], [
    ["MILP + 精修", "**0.1145**", "0.2671", "全局近优（MIP 间隙 ≤1.4%）"],
    ["匹配法", "0.1241", "0.2884", "多项式时间精确解，独立验证"],
    ["LPT", "0.1373", "0.2541", "有 4/3 近似比保证"],
    ["蛇形选秀", "0.1619", "0.2491", "可解释基线"],
  ]),
  para(run("")),
  p(T`四法收敛于 0.11~0.16 窄带，说明解空间平坦、结论不依赖单一算法。另以 28 个候选方案的共识矩阵做层次聚类，所得共识方案 $F_{\text{pop}}=0.1217$，与 MILP 解同一量级，进一步支持稳健性。`),
  ...figure("fig_q1_methods.png", "图 1  问题一四种方法的 F_pop 与 F_score 随 K 变化"),
  ...figure("fig_q1_consensus.png", "图 2  28 个候选方案的共识矩阵与共识方案"),
  h2("5.3 公平指数讨论"),
  p(T`1. **$F$-$K$ 权衡**：$F_{\text{pop}}$ 随 $K$ 减小单调改善（0.227→0.1145），无内部拐点；推荐取题面下限 $K=12$，改动成本仅 8 次合并。`),
  p(T`2. **人口公平 ≠ 得分公平**：$F_{\text{pop}}$ 降约五成，引擎仿真的 $F_{\text{score}}$ 几乎不动（≈0.26），低 $K$ 时冠军集中度反而升高。原因是特长生随学院走，合并不改变得分资源的分布。这一发现直接引出问题三。`),
  p(T`3. **结构边界**：$K=12$ 时至少存在 4 个单学院组，而最大学院（1964 人）比组均值（2377 人）低 17.4%，故“所有组均在均值 ±15% 以内”不可行（解析边界，已验证 $\varepsilon=0.20$ 不可行、$\varepsilon=0.25$ 可行）。`),
);

// ===== 六、问题二 =====
push(
  h1("六、问题二：甲乙组分级模式"),
  h2("6.1 实力代理与分层模型"),
  p(T`无历史成绩数据，以**引擎期望得分**（现状模式下仿真 1000 届的场均得分）作为学院实力代理。设排序后的实力向量为 $v_{(1)}\ge\cdots\ge v_{(20)}$，连续切分点 $c$ 把学院分为甲（前 $c$ 名）、乙两组，目标为最小化按组规模加权的层内方差`),
  mathDisplay(T`\min_c\ c\cdot\mathrm{Var}(v_{(1..c)})+(20-c)\cdot\mathrm{Var}(v_{(c+1..20)}),`),
  p(T`使层内实力接近、比赛有悬念。19 个切分点枚举即得最优 $c^*=10$。`),
  p(T`为排除“交叉分组可能更优”的疑虑，对全部 $2^{19}$ 种 0/1 分层做全枚举（固定一院在甲组去对称），验证**全局最优分层确为按实力的连续切分**。`),
  h2("6.2 分层结果"),
  table([1600, 7200], ["层级", "学院"], [
    ["甲组", "计算机、理学、材料、电子、法学、文、艺术、教育、农学、经管"],
    ["乙组", "电气、数学、物理、医学、土木、机械、外语、历史、化工、哲学"],
  ]),
  para(run("")),
  p(T`值得注意：全校人数第一的机械学院落入乙组——其男生占 88%，女子项目几乎无分，期望得分仅居中游。这说明**规模不等于实力**，分层应以真实得分能力为准，而非人数。`),
  ...figure("fig_q2_tiers.png", "图 3  问题二实力分层与竞争性对比"),
  h2("6.3 公平性讨论"),
  table([4400, 2200, 2200], ["指标", "现状", "分层后"], [
    ["有奔头的学院（夺冠概率 ≥5%）", "5", "8（甲 4 + 乙 4）"],
    ["层内得分公平指数", "0.264", "甲 0.094，乙 0.187"],
  ]),
  para(run("")),
  p(T`分层后每个学院都有“跳一跳够得着”的目标，保护了参与热情。代价是层间不再同场竞技，乙组冠军含金量受限，且分层存在固化风险——可引入**升降级机制**（如乙组冠军次年升入甲组、甲组末位降入乙组）作为改进方向。`),
);

// ===== 七、问题三 =====
push(
  h1("七、问题三：特长生制度调整"),
  h2("7.0 机制分析：特长生是唯一重要的得分资源"),
  p(T`引擎校准显示：$\delta=1.5$ 时每名特长生的边际期望得分 $m=1.26$ 分，而普通生在特长生占满前八的项目中几乎无分。各学院期望得分近似为 $b_i+m\,t_i$，其中基底 $b_i$ 远小于特长生贡献。**得分格局由 582 名特长生的归属决定**——这解释了问题一中“人口公平大降、得分公平不动”的现象，也是本问两条制度设计的出发点。`),
  h2("7.1 子问一：跨学院流动模式"),
  p(T`**模型**：在总人数、性别结构不变的前提下重新分配特长生。设 $t_i$ 为学院 $i$ 分得的特长生数，$\sum_i t_i=582$，$t_i\ge 0$ 整数，目标为最小化期望得分的综合公平指数。由于每单位特长生边际得分近似为常数 $m$，期望得分均衡等价于注水原理（water-filling）：`),
  mathDisplay(T`t_i=\max\left(\frac{T-b_i}{m},\ \ell_i\right),`),
  p(T`其中水位 $T$ 由总量约束确定，$\ell_i$ 为下界约束。校准采用**幻影单位法**：设一个按全校性别比构成的幻影单位吸收全部 582 名特长生，引擎仿真得幻影人均得分即 $m$（特长生总池固定时，项目内竞争密度与归属无关，故 $m$ 为全局常数），此时各学院得分即基底 $b_i$。分配结果送回引擎实测，按实测偏差迭代修正（3 轮收敛）。`),
  p(T`**结果**：最优解接近均匀分配（每院 27~31 人）。`),
  table([3200, 2000, 1900, 1900], ["方案", "$F_{\\text{score}}$", "有奔头学院数", "最大夺冠概率"], [
    ["现状", "0.263", "5", "0.281"],
    ["自由流动", "**0.015**", "**13**", "0.076"],
    ["有界流动（每院保留 ≥50%）", "0.015", "13", "0.076"],
  ]),
  para(run("")),
  ...figure("fig_q3a_flow.png", "图 4  问题三a：特长生再分配方案与期望得分对比"),
  p(T`有界方案与自由方案完全一致：最优解中流出最多的是计算机学院（49→28，流出 43%），**没有任何学院需要流出过半**——方案在政治上可行。$\delta$ 从 0.5 扫到 2.5，流动后 $F$ 稳定在 0.02~0.03，结论对特长生水平假设稳健。`),
  h2("7.2 子问二：特长生不参赛时的加权积分"),
  p(T`特长生不参赛时，期望得分与人数近似成正比（最强 55.5 分、最弱 18.6 分），大院垄断。为抵消规模优势，对学院 $i$ 的原始得分乘权重`),
  mathDisplay(T`w_i=\left(\frac{T}{b_i}\right)^{\alpha},\qquad \bar w=1,`),
  p(T`其中 $b_i$ 为该院无特长生时的期望得分、$T$ 为全院均值——**权重由积分规则下的期望得分推导，不是人为指定**。$\alpha=0$ 退化为不加权，$\alpha=1$ 时期望加权得分完全拉平。`),
  p(T`对 $\alpha$ 在 0~1 上扫描（引擎仿真 500 届）：`),
  table([2400, 1600, 1600, 1900, 1500], ["$\\alpha$", "$F_{\\text{score}}$", "有奔头数", "最大夺冠概率", "权重比 max/min"], [
    ["0（不加权）", "0.174", "6", "0.270", "1.00"],
    ["0.5", "0.092", "11", "0.092", "1.73"],
    ["**0.8（选定）**", "**0.037**", "**11**", "**0.068**", "2.40"],
    ["1.0（完全拉平）", "≈0", "9", "0.102", "2.98"],
  ]),
  para(run("")),
  ...figure("fig_q3b_weights.png", "图 5  问题三b：加权强度扫描与各学院积分权重"),
  p(T`$\alpha=1$ 时期望完全相等（$F\approx0$），但小院权重大、得分方差大，夺冠概率反而向小院集中（0.102）——期望的拉平被方差的失衡抵消，是谓**矫枉过正**。按“满足 $F\le0.05$ 且最大夺冠概率 $\le0.30$ 的最小 $\alpha$”选定 $\alpha^*=0.8$，此时权重范围 0.69~1.64，大院让渡、小院加成，幅度温和。`),
  h2("7.3 两支线比较"),
  p(T`流动方案直接均衡得分资源，效果最彻底（$F=0.015$），但涉及人员归属调整，组织成本高；加权方案不动人、只动规则，实施成本最低，$F=0.037$ 略逊。二者亦可组合：流动打底、加权微调。`),
);

// ===== 八、问题四 =====
push(
  h1("八、问题四：四种模式的统一比较"),
  h2("8.1 仿真比较"),
  p(T`将现状、合并（$K=12$ MILP 方案）、分层（10/10）、流动、加权五种情形统一用引擎仿真 1000 届（公共随机数对照），结果如下：`),
  table([2000, 1250, 1250, 1550, 1550, 1426],
    ["模式", "$F$（等权）", "$F$（熵权）", "有奔头学院数", "最大夺冠概率", "冠军集中度 HHI"], [
    ["现状", "0.263", "0.241", "5", "0.281", "0.178"],
    ["Q1 学院合并", "0.270", "0.249", "4", "0.379", "0.277"],
    ["Q2 甲乙分层", "0.143*", "0.132*", "8", "0.359*", "0.491*"],
    ["Q3 特长生流动", "**0.015**", "**0.013**", "**13**", "0.076", "0.057"],
    ["Q3 加权积分", "0.039", "0.035", "10", "**0.060**", "**0.051**"],
  ]),
  p(T`注：分层模式为层内指标按学院数加权的合并口径；每层各产生一名冠军，其夺冠概率与 HHI 不与单场模式直接可比。`),
  h2("8.2 结论与推荐"),
  p(T`1. **流动模式最优**：$F$ 由 0.263 降至 0.015（降 94%），有奔头学院由 5 个增至 13 个，直接作用于问题根源；`),
  p(T`2. **加权模式次之**：不动人只动规则，实施成本最低，最大夺冠概率 0.060 为全表最低；`),
  p(T`3. **分层模式居中**：层内竞争明显改善，但未改变得分资源分布，且付出层间不同场竞技的代价；`),
  p(T`4. **合并模式不改善得分公平**（$F$ 微升至 0.270）：其价值在组织层面——参赛单位数合规、组间人数性别均衡。`),
  p(T`综合推荐**“组织上并、资源上均”的组合方案**：以问题一的合并解决组织合规与人口公平，以问题三的流动（或加权）解决得分公平。`),
  h2("8.3 权重客观性回填（熵权法）"),
  p(T`以“五模式 × 四度量”的指标矩阵做熵权法客观赋权，得基尼 0.235、泰尔 0.306、标准差 0.234、极差率 0.225，接近等权。等权与熵权下模式排序完全一致，说明全文结论不依赖权重的人为设定。`),
  h2("8.4 理论对账"),
  p(T`无特长生时单项前八名是 40 名参赛者的均匀随机子集，学院 $i$ 单项目期望得分有解析式 $36s_i/40$。引擎 20000 次单项目抽样与之比对，最大相对偏差 2.25%（抽样误差量级），仿真在解析可解的极限下通过验证；含特长生情形无解析解，正是蒙特卡洛仿真的价值所在。`),
);

// ===== 九、敏感性分析 =====
push(
  h1("九、敏感性分析"),
  p(T`1. **权重敏感性**：公平指数 7 个权重做单维扰动（OAT）与 Dirichlet(5000) 联合采样，现状基线 $F$ 波动有界；熵权回填后四模式排序不变（见 8.3）。`),
  ...figure("fig_fairness_sensitivity.png", "图 6  公平指数权重敏感性（OAT 与 Dirichlet 联合采样）"),
  p(T`2. **特长生优势 $\delta$ 敏感性**：$\delta\in[0.5,2.5]$ 扫描，现状 $F$ 由 0.246 升至 0.264 后饱和，冠军恒为计算机学院、末位恒为哲学院；流动方案 $F$ 稳定在 0.02~0.03。结论对 $\delta$ 稳健。`),
  ...figure("fig_exp_delta_sweep.png", "图 7  特长生优势 δ 扫描下现状公平指数变化"),
  p(T`3. **分组算法敏感性**：四种独立算法求解问题一，$F_{\text{pop}}$ 收敛于 0.11~0.16 窄带；共识聚类方案同量级。`),
  p(T`4. **仿真届数**：主结果均以 500~1000 届估计，公共随机数保证方案间对照不受抽样噪声干扰。`),
);

// ===== 十、模型评价 =====
push(
  h1("十、模型评价"),
  p(T`**优点**`),
  p(T`1. 一以贯之：公平指数与仿真引擎两个基础设施贯穿四问，所有方案在同一标尺下比较，且公共随机数保证对照公平；`),
  p(T`2. 求解与验证分离：MILP 求结构、匹配法独立验证、全枚举证最优切分、幻影单位法校准、解析解对账，关键结论均有两条以上独立证据；`),
  p(T`3. 权重全透明：等权基准 + 熵权回填 + 全局敏感性采样，结论不依赖人为权重。`),
  p(T`**缺点与改进方向**`),
  p(T`1. 普通生同分布假设抹平了学院间真实的体育传统差异，若有历年成绩数据可校准个体能力分布；`),
  p(T`2. 特长生性别按学院性别比拆分，实测数据若能获取可提高精度；`),
  p(T`3. 分层模式可引入升降级机制，流动模式可加入项目层面的专项约束（如按项目分配特长生），使制度更细致。`),
);

// ===== 参考文献 =====
push(
  h1("参考文献"),
  pNoIndent(T`[1] Gini C. Variabilità e mutabilità[M]// Pizetti E, Salvemini T. Memorie di metodologica statistica. Rome: Libreria Eredi Virgilio Veschi, 1912.`),
  pNoIndent(T`[2] Theil H. Economics and Information Theory[M]. Amsterdam: North-Holland, 1967.`),
  pNoIndent(T`[3] Metropolis N, Ulam S. The Monte Carlo method[J]. Journal of the American Statistical Association, 1949, 44(247): 335-341.`),
  pNoIndent(T`[4] Huangfu Q, Hall J A J. Parallelizing the dual revised simplex method[J]. Mathematical Programming Computation, 2018, 10(1): 119-142.`),
  pNoIndent(T`[5] Graham R L. Bounds on multiprocessing timing anomalies[J]. SIAM Journal on Applied Mathematics, 1969, 17(2): 416-429.`),
  pNoIndent(T`[6] 郭亚军. 综合评价理论、方法及应用[M]. 北京: 科学出版社, 2007.`),
);

// ===== 附录 =====
push(
  h1("附录 A：AI 使用声明"),
  p(T`本队在研究过程中使用了 Kimi AI（Moonshot AI 模型）辅助完成部分重复性代码的编写。具体范围：各源码文件中以 >>> AI-GENERATED BLOCK 标注的代码块（数据清洗模板、绘图代码、整数化与注水分配、熵权法实现、对比图绘制等）由 Kimi AI 于 2026 年 11 月 14 日 22 时至 11 月 15 日 12 时生成，其余模型设计、目标函数、参数设定、结果解读与论文撰写由队员人工完成。所有 AI 生成代码均经队员逐行核对并实际运行验证，结果由队员负责。`),
  h1("附录 B：代码清单"),
  table([3200, 5600], ["文件", "功能"], [
    ["src/data_clean.py", "数据清洗与两级联动校验"],
    ["src/fairness_index.py", "公平指数与权重敏感性分析"],
    ["src/scoring_engine.py", "赛制蒙特卡洛仿真引擎"],
    ["src/experiment.py", "引擎 δ 扫描实验"],
    ["src/q1_solvers.py", "问题一四法求解器（MILP/匹配/LPT/蛇形）"],
    ["src/q1_evaluate.py", "问题一 28 方案横向评估"],
    ["src/q1_consensus.py", "问题一共识矩阵集成"],
    ["src/q2_tiers.py", "问题二最优切分与全枚举验证"],
    ["src/q3_flow.py", "问题三a 注水分配与幻影校准"],
    ["src/q3_weights.py", "问题三b 加权积分 α 扫描"],
    ["src/q4_compare.py", "问题四模式比较、熵权回填与理论对账"],
  ]),
  para(run("")),
  p(T`代码仓库（私有）：https://github.com/HnoNoe/sports-meet-modeling`),
);

// ===== 组装 =====
const doc = new Document({
  features: { updateFields: true },
  sections: [{
    properties: {
      page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    headers: {
      default: new Header({ children: [
        para(run("运动会竞赛组织模式优化", { color: palette.light }),
          { alignment: AlignmentType.CENTER }),
      ] }),
    },
    footers: {
      default: new Footer({ children: [
        para(new TextRun({ children: [PageNumber.CURRENT] }),
          { alignment: AlignmentType.CENTER }),
      ] }),
    },
    children,
  }],
});

fs.writeFileSync(outputPath, await Packer.toBuffer(doc));
