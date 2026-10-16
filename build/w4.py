"""Week 4: causal questions, audits, reproducibility, wrap-up and career (days 22 to 28)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framework import (CODE, COLS, FIG, FORMULA, LEAD, NOTE, OL, P, QA, TABLE, UL, WARN, Lesson, Paper)

from course import causal
from course.plotting import INK, OCHRE, ROSE, SAND, TEAL


# ============================================================== figures
def fig_causal():
    import matplotlib.pyplot as plt
    df = causal.simulate_chatbot_use(6000, true_effect=-2.0, seed=1)
    est = {"naive difference": causal.naive_difference(df),
           "adjust for age, literacy": causal.regression_adjusted(df, ["age", "literacy"]),
           "adjust incl. baseline loneliness": causal.regression_adjusted(df, ["age", "literacy", "lonely0"]),
           "IPW (all covariates)": causal.ipw_ate(df, ["age", "literacy", "lonely0"])}
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    y = np.arange(len(est))
    ax.barh(y, list(est.values()), color=[SAND, SAND, TEAL, TEAL])
    ax.axvline(-2.0, color=OCHRE, ls="--", label="true effect = -2")
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(list(est.keys()), fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("estimated effect of chatbot use on loneliness"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_items():
    import matplotlib.pyplot as plt
    from course.audit import item_analysis
    from course.data import simulate_vignette_matrix
    u, *_ = simulate_vignette_matrix(80, 40, seed=23)
    u[:, 3] = 1; u[:, 7] = (np.random.default_rng(0).random(80) < 0.5)
    tab = item_analysis(u)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    flagged = (tab["flags"] != "").values
    pr, rr = tab.pass_rate.values, tab.item_rest_r.fillna(0).values
    ax.scatter(pr[~flagged], rr[~flagged], color=TEAL, s=18, label="ok")
    ax.scatter(pr[flagged], rr[flagged], color=OCHRE, s=30, marker="x", label="flagged")
    ax.axhline(0.1, color=SAND, lw=1)
    ax.set_xlabel("pass rate across models"); ax.set_ylabel("item-rest correlation"); ax.legend(fontsize=8)
    ax.set_title("Item audit of a mental-health vignette set (simulated)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_drift():
    import matplotlib.pyplot as plt
    import pandas as pd
    from course.repro import anchor_drift
    rng = np.random.default_rng(24)
    rows = []
    for run in range(12):
        shift = 0.35 if run >= 7 else 0.0
        for i in range(60):
            rows.append({"run": run, "item": i, "score": 3.2 + shift + rng.normal(0, 0.7)})
    out = anchor_drift(pd.DataFrame(rows), baseline_runs=4)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x, m, fl = out.run.values + 1, out["mean"].values, out.flag.values
    ax.plot(x, m, "o-", color=INK)
    ax.fill_between(x, out.lower.values, out.upper.values, color=SAND, alpha=.5, label="control limits from months 1-4")
    ax.scatter(x[fl], m[fl], color=OCHRE, s=60, zorder=3, label="flagged")
    ax.set_xlabel("month"); ax.set_ylabel("mean judge score on anchor set"); ax.legend(fontsize=8)
    ax.set_title("Hosted judge changed silently in month 8 (simulated)", fontsize=10)
    fig.tight_layout()
    return fig


# ============================================================== day 22
D22 = Lesson(
    day=22, slug="causal_inference", title="因果推断:聊天机器人使用能减轻老年人的孤独吗", track="eval",
    goals=["用潜在结果框架说明为什么观察数据的比较会有偏差", "在模拟数据里用回归调整和逆概率加权恢复真实效应",
           "说清这些方法的前提:无未测混杂、重叠性", "用双重差分分析服务层面的上线效果,并判断它能否支持因果结论"],
    slides=[
        ("问题", LEAD("观察数据显示:用了聊天机器人的老年人,8 周后孤独感得分反而更高。聊天机器人让人更孤独吗?")
         + P("不一定。更孤独的人更可能去用它。这正是你在 UTAUT2 研究里关心的采纳问题:谁会使用,本身就不是随机的。")),
        ("潜在结果框架", FORMULA("个体效应 = Yᵢ(1) − Yᵢ(0)", "同一个人用与不用时的结局,只能观察到其中一个")
         + UL(["**随机化**让两组在所有特征上可比,组间差就是平均效应。你的 RCT 就是这样回答因果问题的。",
               "观察数据中,用与不用的人在基线孤独感、数字素养、年龄上都不同,这些是**混杂因素**。"])),
        ("模拟:自我选择造成的偏差", FIG(fig_causal, "模拟:真实效应是降低 2 分。基线孤独感和数字素养同时影响使用和结局。")
         + P("只调整年龄和数字素养不够,遗漏了基线孤独感,估计仍然有偏。")),
        ("三种调整方法", TABLE(["方法", "思路", "关键前提"], [["回归调整", "在模型里控制混杂变量", "模型形式正确"], ["倾向得分加权(IPW)", "按使用概率的倒数加权,构造可比的伪总体", "倾向得分模型正确;重叠性"],
              ["匹配", "为每个使用者找特征相近的未使用者", "可比的对照存在"]])
         + WARN("三种方法都依赖“无未测混杂”。如果孤独的真实原因(例如丧偶)没有被测量,任何调整都救不了。")),
        ("双重差分", FORMULA("DiD = (Y_后 − Y_前)_上线服务 − (Y_后 − Y_前)_未上线服务")
         + P("Habicht 等人(2024)比较了上线与未上线聊天机器人的 NHS 服务在上线前后的转介变化,思路与双重差分接近。前提是**平行趋势**:如果没有上线,两类服务的变化趋势相同。")),
        ("在评测工作中的用处", UL(["“加了某批训练数据后分数上升”:是数据的作用,还是同时改了别的设置?", "线上 A/B 测试是最直接的因果评估。",
               "离线评测分数和线上用户结局之间的关系,本身就是一个预测效度问题。"])),
        ("今日论文", P("OpenAI 与 MIT Media Lab(Phang 等人, 2025, arXiv:2504.03888):一部分是对 300 多万条对话的平台分析,另一部分是近 1000 人、28 天的随机对照试验。")
         + UL(["两部分都发现,极高的使用量与自报的依赖相关。", "阅读重点:平台分析的相关结果,能否说明使用导致依赖?随机化部分回答了什么、没回答什么?"])),
        ("今日作业", OL(["完成 notebook 练习 22.1 到 22.4。", "精读 Phang 等人 2025,回答学习单问题。",
                     "写一段话:用你的 UTAUT2 采纳研究数据,能回答哪些因果问题?不能回答哪些?"])),
    ],
    cells=[
        ("md", """## 1. 观察数据:6000 位老年人"""),
        ("code", """from course.causal import simulate_chatbot_use, naive_difference, regression_adjusted, ipw_ate, did_estimate
obs = simulate_chatbot_use(6000, true_effect=-2.0, seed=1)
obs.groupby("use")[["age", "literacy", "lonely0", "lonely1"]].mean().round(2)"""),
        ("md", """**练习 22.1**:使用者和未使用者的基线孤独感(`lonely0`)差多少?(使用者减未使用者)这说明了什么?"""),
        ("code", """_g = obs.groupby("use").lonely0.mean()
ans_22_1 = None
check("练习 22.1", ans_22_1, float(_g[1] - _g[0]), tol=1e-6)"""),
        ("md", """## 2. 各种估计"""),
        ("code", """est = pd.Series({
    "naive": naive_difference(obs),
    "adjust age + literacy": regression_adjusted(obs, ["age", "literacy"]),
    "adjust + baseline loneliness": regression_adjusted(obs, ["age", "literacy", "lonely0"]),
    "IPW": ipw_ate(obs, ["age", "literacy", "lonely0"]),
}); est.round(3)"""),
        ("md", """**练习 22.2**:哪个估计离真实效应(−2)最远?填它在上表中的名字。"""),
        ("code", """ans_22_2 = None
check("练习 22.2", ans_22_2, (est + 2).abs().idxmax())"""),
        ("md", """## 3. 随机化的版本"""),
        ("code", """rct = simulate_chatbot_use(6000, true_effect=-2.0, randomised=True, seed=2)
print("RCT 组间差:", round(naive_difference(rct), 3))"""),
        ("md", """**练习 22.3**:在 RCT 数据上,把基线孤独感作为协变量调整(`regression_adjusted(rct, ["lonely0"])`)。估计值仍应接近 −2,但标准误更小。用 bootstrap(200 次)比较未调整与调整后估计的标准差,填入**调整后**的标准差。"""),
        ("code", """rng = np.random.default_rng(0)
b_raw, b_adj = [], []
for _ in range(200):
    s = rct.sample(len(rct), replace=True, random_state=int(rng.integers(1e9)))
    b_raw.append(naive_difference(s)); b_adj.append(regression_adjusted(s, ["lonely0"]))
print("未调整 SD:", round(np.std(b_raw), 3), " 调整后 SD:", round(np.std(b_adj), 3))
ans_22_3 = None
check("练习 22.3", ans_22_3, float(np.std(b_adj)), tol=1e-6)"""),
        ("md", """## 4. 双重差分:社区中心上线聊天机器人

10 家社区中心上线了心理支持聊天机器人,10 家没有。结局是每月主动求助的人数(每家平均)。"""),
        ("code", """pre_t, post_t = 42.0, 55.0     # 上线中心:上线前、后
pre_c, post_c = 40.0, 46.0     # 对照中心"""),
        ("md", """**练习 22.4**:计算双重差分估计。然后写一句话:要相信这个数字,需要什么前提?"""),
        ("code", """ans_22_4 = None
check("练习 22.4", ans_22_4, did_estimate(pre_t, post_t, pre_c, post_c), tol=1e-9)"""),
        ("md", """*前提:*"""),
    ],
    paper=Paper("Phang J. et al. (2025). Investigating Affective Use and Emotional Well-being on ChatGPT. arXiv:2504.03888(OpenAI 与 MIT Media Lab 合作)",
                "这篇工作同时有大规模观察分析和随机对照试验,是练习区分相关与因果的好材料。",
                ["平台分析部分发现了什么?它能支持“使用导致依赖”吗?", "RCT 部分的处理条件是什么?结果为什么说是“复杂的”?", "如果研究对象换成香港老年人,你会怎样设计?"],
                "https://arxiv.org/abs/2504.03888"),
    research="你的 UTAUT2 研究关心谁会采纳心理健康应用。把它和效果研究结合起来,可以回答一个重要问题:观察数据中的“使用者效果”有多少来自自我选择?这是数字心理健康研究中常见但常被忽视的偏差来源。",
    interview=[("如何判断评测分数提升是因为模型变好,而不是评测集被污染?", "对照实验:同时测原题和改写题、旧题和训练截止后的新题;如果提升只出现在原题或旧题上,怀疑污染。还要检查同时改动的其他设置,例如裁判版本、提示词。"),
               ("线上 A/B 测试之外,怎么评估聊天机器人对用户的影响?", "准实验方法:双重差分、断点回归、倾向得分方法,并说明各自的前提;观察数据结论要谨慎。")],
    homework=["notebook 练习 22.1 到 22.4", "一段话:UTAUT2 数据能回答和不能回答的因果问题", "学习单论文问题"],
)
D22.solutions = {"ans_22_1": "float(_g[1] - _g[0])", "ans_22_2": "(est + 2).abs().idxmax()", "ans_22_3": "float(np.std(b_adj))",
                 "ans_22_4": "did_estimate(pre_t, post_t, pre_c, post_c)"}

# ============================================================== day 23
D23 = Lesson(
    day=23, slug="validity_audit", title="评测集审计:入库检查、饱和与退役", track="eval",
    goals=["对一个心理健康评测集做题目分析,标出过易、过难和区分度低的题", "检验 rubric 维度之间的冗余",
           "写出可操作的饱和与退役判据", "说明为什么审计不能由评测集的制作方自己完成"],
    slides=[
        ("入库前的审计清单", TABLE(["检查", "问题", "方法"], [["信度", "评分稳定吗", "重测、评分者间一致性"], ["区分度", "能把强弱模型分开吗", "题总相关、IRT 区分度"],
              ["难度分布", "题目都太容易或太难吗", "通过率分布"], ["污染", "题目是否被见过", "第 13 天的方法"], ["覆盖", "是否覆盖要测的范围", "专家评审,对照使用场景"],
              ["独立性", "谁来审计", "不能由制作方自己把关"]])),
        ("题目分析", FIG(fig_items, "模拟:80 个模型 × 40 道情景题。")
         + UL(["**通过率接近 1**:所有模型都能做对,不提供排序信息(除非作为安全底线题)。", "**题总相关低或为负**:可能题目有歧义,或者评分标准有误。"])),
        ("维度冗余", P("如果 rubric 的两个维度相关 0.9,它们可能在测同一件事;合并可以减少评分负担,也让结论更清楚。")
         + NOTE("用相关矩阵或探索性因子分析检查结构效度,这是你在量表开发中做过的事。", "迁移")),
        ("饱和与退役", UL(["**饱和**:领先模型都接近满分,且区间重叠;或多数题被所有模型做对。", "**退役条件**:饱和、污染证据充分、或效度不再成立(例如使用场景变了)。",
               "退役要记录原因和日期,保留旧版本以便比较。"])),
        ("测量的视角", P("Wallach 等人(ICML 2025 立场论文)认为,评测生成式 AI 本质上是社会科学的测量问题:先定义要测的构念,再考察测量工具的信度和效度。")
         + NOTE("这和第 1 天讲的内容一致。心理学背景在这里不是加分项,而是方法本身。", "要点")),
        ("今日作业", OL(["完成 notebook 练习 23.1 到 23.4。", "精读 Wallach 等人 2025,回答学习单问题。",
                     "为你的项目写审计报告:哪些题需要修改或删除?理由是什么?"])),
    ],
    cells=[
        ("md", """## 1. 题目分析"""),
        ("code", """from course.audit import item_analysis, cronbach_alpha, saturation_check
from course.data import simulate_vignette_matrix
u, theta, a, b, items = simulate_vignette_matrix(80, 40, seed=23)
u[:, 3] = 1                                                     # 一道所有模型都能做对的题
u[:, 7] = (np.random.default_rng(0).random(80) < 0.5)           # 一道与能力无关的题(例如评分标准有误)
tab = item_analysis(u, items.item.values)
tab[tab["flags"] != ""]"""),
        ("md", """**练习 23.1**:有几道题被标记?"""),
        ("code", """ans_23_1 = None
check("练习 23.1", ans_23_1, int((tab["flags"] != "").sum()))"""),
        ("md", """## 2. 维度冗余:五个 rubric 维度的相关"""),
        ("code", """from course.data import simulate_counselor_study, DIMENSIONS
st = simulate_counselor_study(seed=23)
wide = st["human"][st["human"].dim.isin(DIMENSIONS)].groupby(["item", "system", "dim"]).score.mean().unstack("dim")
corr = wide.corr(); corr.round(2)"""),
        ("md", """**练习 23.2**:除对角线外,相关最高的一对维度是哪两个?填一个排好序的列表,例如 `["D1", "D3"]`。"""),
        ("code", """_c = corr.where(~np.eye(5, dtype=bool)).stack()
ans_23_2 = None
check("练习 23.2", sorted(ans_23_2) if ans_23_2 else ans_23_2, sorted(_c.idxmax()))"""),
        ("md", """**练习 23.3**:把五个维度当作五道题,计算 Cronbach's α。"""),
        ("code", """ans_23_3 = None
check("练习 23.3", ans_23_3, cronbach_alpha(wide.values), tol=1e-6)"""),
        ("md", """## 3. 饱和判断"""),
        ("code", """scores = {"model_1": (0.97, 0.94, 0.99), "model_2": (0.96, 0.93, 0.98), "model_3": (0.95, 0.91, 0.98), "model_4": (0.81, 0.75, 0.86)}
saturation_check(scores)"""),
        ("md", """**练习 23.4**:如果 model_3 的结果是 (0.90, 0.86, 0.94),这个评测集还算饱和吗?填 `True` 或 `False`。"""),
        ("code", """ans_23_4 = None
check("练习 23.4", ans_23_4, saturation_check({**scores, "model_3": (0.90, 0.86, 0.94)})["saturated"])"""),
    ],
    paper=Paper("Wallach H. et al. (2025). Position: Evaluating Generative AI Systems Is a Social Science Measurement Challenge. ICML 2025.",
                "这篇立场论文把你熟悉的测量理论带进 AI 评测,是你在面试中解释“心理学背景为什么有用”的好依据。",
                ["作者提出的测量框架分为哪几个层次?", "他们如何区分“构念”和“测量工具”?", "用这个框架分析 MentalHealthBench:它要测的构念是什么?测量工具是否足以支持结论?"],
                "https://arxiv.org/abs/2502.00561"),
    research="对已发表的心理健康 LLM 基准做一次系统的“测量审计”(构念定义、信度证据、效度证据、污染风险),可以写成一篇方法学评论。这和你的元分析经验很契合。",
    interview=[("评测集的饱和和退役,你用什么判据?", "领先模型接近上限且区间重叠;多数题被所有模型做对;有污染证据;或者使用场景变化导致效度不再成立。退役要记录原因,保留版本,用于纵向比较。"),
               ("为什么评测集入库的审计不能由制作方自己完成?", "制作方有动机让评测集通过,也容易对自己的设计盲点视而不见。独立审计类似同行评审和独立数据监查委员会。")],
    homework=["notebook 练习 23.1 到 23.4", "项目审计报告(docs/audit.md)", "学习单论文问题"],
)
D23.solutions = {"ans_23_1": 'int((tab["flags"] != "").sum())', "ans_23_2": "sorted(_c.idxmax())", "ans_23_3": "cronbach_alpha(wide.values)",
                 "ans_23_4": 'saturation_check({**scores, "model_3": (0.90, 0.86, 0.94)})["saturated"]'}

# ============================================================== day 24
D24 = Lesson(
    day=24, slug="reproducibility", title="可复现性与版本治理:托管模型会悄悄改变", track="eval",
    goals=["列出每次评测必须记录的信息,并写出 run manifest", "用锚定集和控制图监测托管裁判的漂移",
           "用哈希发现评测数据被改动", "把方法学要求写成其他团队能照做的规范"],
    slides=[
        ("同一个 API,不同的模型", P("Chen、Zaharia 和 Zou(Harvard Data Science Review, 2024)比较了同一模型名称在不同时间的行为,发现某些任务的表现在几个月内明显变化。")
         + WARN("托管模型和托管裁判可能在没有通知的情况下更新。今天的评测结果,明天不一定能复现。")),
        ("每次评测必须记录", TABLE(["类别", "内容"], [["被测模型", "名称、版本号或日期、采样参数"], ["裁判", "名称、版本、提示词全文、温度"],
              ["数据", "评测集版本和哈希"], ["代码", "git 提交号、依赖版本"], ["运行", "时间、随机种子、失败重试记录"]])),
        ("锚定集与控制图", FIG(fig_drift, "模拟:每月用同一批 60 条回复让裁判打分。")
         + UL(["固定一个锚定集,每次评测前先跑一遍。", "用前几次的结果设控制限(均值 ± 3 倍标准误)。", "超出控制限就暂停评测,查明原因。"])),
        ("规范要能执行", UL(["写成检查清单和模板,不写成长篇文档。", "能自动检查的就自动检查(例如 manifest 是否完整、哈希是否变化)。",
               "说明“为什么”,但篇幅放在附录。", "腾讯评测岗位要求的“把方法学要求转化为其他团队可执行的规范”,指的就是这一步。"])),
        ("今日作业", OL(["完成 notebook 练习 24.1 到 24.3。", "精读 Chen 等人 2024,回答学习单问题。",
                     "为你的项目写一页“评测复现规范”(docs/reproducibility.md)。"])),
    ],
    cells=[
        ("md", """## 1. 锚定集漂移监测

模拟 12 个月,每月让裁判给同一批 60 条回复打分。某个月起裁判被悄悄更新,整体打分偏宽松。"""),
        ("code", """from course.repro import anchor_drift, run_manifest, sha256_of
rng = np.random.default_rng(24)
CHANGE = int(rng.integers(5, 10))      # 变化发生的月份(从 0 开始),你不知道它
rows = []
for run in range(12):
    shift = 0.35 if run >= CHANGE else 0.0
    for i in range(60):
        rows.append({"run": run, "item": i, "score": 3.2 + shift + rng.normal(0, 0.7)})
hist = pd.DataFrame(rows)
mon = anchor_drift(hist, baseline_runs=4)
mon.round(3)"""),
        ("md", """**练习 24.1**:第一个被标记的月份(`run`,从 0 开始)是哪个?"""),
        ("code", """ans_24_1 = None
check("练习 24.1", ans_24_1, int(mon[mon.flag].run.min()) if mon.flag.any() else None)"""),
        ("md", """**练习 24.2**:把控制限从 3 倍改为 2 倍标准误(`z=2`),会多出几个被标记的月份?这对误报意味着什么?"""),
        ("code", """mon2 = anchor_drift(hist, baseline_runs=4, z=2.0)
ans_24_2 = None
check("练习 24.2", ans_24_2, int(mon2.flag.sum() - mon.flag.sum()))"""),
        ("md", """## 2. 用哈希发现数据被改动"""),
        ("code", """from course.data import load_prompts
p = load_prompts()
h1 = sha256_of(p.to_dict(orient="list"))
p2 = p.copy(); p2.loc[0, "prompt"] = p2.loc[0, "prompt"] + " "      # 只多了一个空格
h2 = sha256_of(p2.to_dict(orient="list"))
print(h1[:16], h2[:16], "相同" if h1 == h2 else "不同")"""),
        ("md", """## 3. 写一份可执行的规范

**练习 24.3**:把你的复现规范写成一个 Python 列表 `ans_24_3`,至少 6 条,每条是一句可以检查的要求(例如“每次评测保存 manifest.json,包含数据哈希”)。"""),
        ("code", """ans_24_3 = None
check_true("练习 24.3 至少 6 条可检查的要求", None if ans_24_3 is None else (isinstance(ans_24_3, list) and len(ans_24_3) >= 6))"""),
    ],
    paper=Paper("Chen L., Zaharia M., & Zou J. (2024). How Is ChatGPT's Behavior Changing over Time? Harvard Data Science Review, 6(2).",
                "这篇文章是“托管模型会变”的经典证据,也说明了为什么评测需要版本治理。",
                ["他们比较了哪些时间点、哪些任务?", "哪些任务的变化最大?", "如果你的裁判是托管模型,你会如何设计监测?"],
                "https://arxiv.org/abs/2307.09009"),
    research="心理健康场景中,托管模型更新可能改变危机处理行为。用一个固定的锚定集(例如 McBain 等人那样的分级风险问题)长期监测多个模型,是一项有公共价值的研究。",
    interview=[("托管裁判在无通知的情况下更新,你怎么发现?", "每次评测前跑固定锚定集,用历史数据设控制限,超限即告警;记录裁判版本和日期;必要时用本地固定版本的模型做对照裁判。")],
    homework=["notebook 练习 24.1 到 24.3", "docs/reproducibility.md", "学习单论文问题"],
)
D24.solutions = {"ans_24_1": "int(mon[mon.flag].run.min()) if mon.flag.any() else None", "ans_24_2": "int(mon2.flag.sum() - mon.flag.sum())",
                 "ans_24_3": '["每次评测保存 manifest.json,含数据哈希", "记录被测模型名称、版本与采样参数", "记录裁判名称、版本、提示词全文与温度", "记录 git 提交号与依赖版本", "评测前先跑锚定集,超出控制限则暂停", "所有结果表注明数据来源与生成脚本"]'}

# ============================================================== day 25
D25 = Lesson(
    day=25, slug="project_wrapup", title="项目收尾:仓库、README 与局限", track="project", hours="4 到 5 小时",
    goals=["把项目整理成一条命令可复现的仓库", "写出包含结果和区间的 README", "写出具体的局限,而不是泛泛的“样本量小”"],
    slides=[
        ("仓库结构", CODE("""your-repo/
  README.md        摘要、如何运行、结果概览、局限
  run_all.py       一条命令复现
  docs/            rubric、分析计划、协议、审计、复现规范
  data/            题目与去标识评分
  src/             代码
  results/         表、图、manifest
  tests/           至少覆盖核心计算""")),
        ("README 的结构", OL(["一句话说明研究问题。", "怎么运行(三行以内)。", "主要结果:一张表,每个数字带区间。", "一张图。",
                            "局限:具体到会影响哪个结论、往哪个方向偏。", "数据与伦理说明。"])),
        ("怎样写局限", COLS(UL(["**泛泛**:样本量较小。", "**泛泛**:未来研究可以扩大样本。"]),
                         UL(["**具体**:只有 3 位咨询师,α 的 95% 区间宽约 0.2,门槛判断在 D2 和 D5 上不稳定。", "**具体**:裁判与系统 B 同源,自我偏好估计可能与长度效应混在一起。"]))),
        ("引言怎么写", P("Stade 等人(2026, npj Digital Medicine)调查了 1871 位美国成人:约 24% 用过 LLM 处理心理健康问题,使用者中抑郁、焦虑和孤独的比例更高。")
         + NOTE("引言的逻辑:很多人在用(需求)→ 评价方法不统一(Hua 等人 2025)→ 自动裁判的效度很少被检验 → 本研究。", "结构")),
        ("今日作业", OL(["完成 notebook,自动生成 README 草稿。", "在新机器上(或新虚拟环境里)跑一次 run_all.py。", "写完局限部分。"])),
    ],
    cells=[
        ("md", """## 1. 用结果文件自动填 README

读取第 20 天生成的 `results/`(如果没有,就用模拟数据现算),生成 README 草稿,保证文中数字与结果文件一致。"""),
        ("code", """import pathlib
R20 = pathlib.Path("../day20_project_analysis/results")
if (R20 / "rq2_validity.csv").exists():
    rq1 = pd.read_csv(R20 / "rq1_reliability.csv", index_col=0)
    rq2 = pd.read_csv(R20 / "rq2_validity.csv", index_col=0)
    source = "第 20 天的结果文件"
else:
    from course.data import simulate_counselor_study, DIMENSIONS
    from course.agreement import krippendorff_alpha
    st = simulate_counselor_study(seed=20); h = st["human"]
    rq1 = pd.DataFrame({"alpha_counselors": [krippendorff_alpha(h[h.dim == d].pivot_table(index=["item", "system"], columns="rater", values="score").values.T, "ordinal") for d in DIMENSIONS]}, index=DIMENSIONS)
    rq2 = pd.DataFrame({"kappa": [np.nan] * 5, "lo": [np.nan] * 5, "hi": [np.nan] * 5}, index=DIMENSIONS)
    source = "模拟数据(请先运行第 20 天)"
print("来源:", source)"""),
        ("code", """def fmt_row(d):
    k = rq2.loc[d]
    ci = "未计算" if pd.isna(k["kappa"]) else f"{k['kappa']:.2f} [{k['lo']:.2f}, {k['hi']:.2f}]"
    return f"| {d} | {rq1.loc[d, 'alpha_counselors']:.2f} | {ci} |"

readme = "\\n".join([
    "# LLM 裁判评价 CBT 取向心理支持回复的效度",
    "",
    "**研究问题**:LLM 裁判能否替代咨询师评价心理支持回复?在哪些维度、哪些人群上不能?",
    "",
    "## 如何运行",
    "```bash", "pip install -r requirements.txt", "python run_all.py", "```",
    "",
    "## 主要结果",
    "",
    "| 维度 | 咨询师间 α | 裁判与咨询师 κ [95% CI] |",
    "|---|---|---|",
    *[fmt_row(d) for d in rq1.index],
    "",
    f"数据来源:{source}。",
    "",
    "## 局限",
    "",
    "- (具体写出:影响哪个结论,往哪个方向偏)",
    "",
    "## 数据与伦理",
    "",
    "所有题目为研究者编写,不含真实来访者文字。评分者姓名不公开。",
])
pathlib.Path("README_draft.md").write_text(readme, encoding="utf-8")
print(readme)"""),
        ("md", """**练习 25.1**:README 草稿里应当出现“95%”和“局限”。检查一下。"""),
        ("code", """ans_25_1 = None
check_true("练习 25.1", None if ans_25_1 is None else ("95%" in ans_25_1 and "局限" in ans_25_1))"""),
        ("md", """## 2. run_all.py 模板"""),
        ("code", """run_all = '''\"\"\"Reproduce every table and figure: python run_all.py\"\"\"
import json, pathlib, subprocess, sys

STEPS = ["src/generate_replies.py", "src/judge_scores.py", "src/analysis.py"]
for step in STEPS:
    print("running", step)
    subprocess.run([sys.executable, step], check=True)
print("done; see results/")
'''
pathlib.Path("run_all_template.py").write_text(run_all, encoding="utf-8")
print(run_all)"""),
        ("md", """**练习 25.2**:写出你的项目的三条具体局限(每条一句,说明影响哪个结论、往哪个方向偏),存成列表 `ans_25_2`。"""),
        ("code", """ans_25_2 = None
check_true("练习 25.2 三条具体局限", None if ans_25_2 is None else (isinstance(ans_25_2, list) and len(ans_25_2) >= 3))"""),
    ],
    paper=Paper("Stade E. C. et al. (2026). Real-world use of large language models for mental health in 2024. npj Digital Medicine, 9.",
                "这是一项具有全国代表性抽样设计的调查,适合作为你报告引言的第一段证据。",
                ["样本如何抽取?代表性如何保证?", "使用者与非使用者在哪些方面不同?", "这些发现对“评测应当覆盖哪些人群”有什么启示?"],
                "https://www.nature.com/articles/s41746-026-02842-9"),
    interview=[("你的项目最大的局限是什么?", "具体回答:例如评分者人数、题目数量、裁判同源、只用了英文题目。说明它影响哪个结论、往哪个方向偏,以及下一步怎么补。")],
    homework=["仓库定稿:run_all.py 能一条命令复现", "README 完成,包括结果表、图和具体局限", "学习单论文问题"],
)
D25.solutions = {"ans_25_1": "readme", "ans_25_2": '["只有 3 位咨询师,门槛判断在区间边缘的维度上不稳定", "裁判与系统 B 同源,自我偏好可能被高估或与长度混淆", "只用了英文题目,结论不能直接推广到中文老年人场景"]'}

# ============================================================== day 26
D26 = Lesson(
    day=26, slug="resume_pitch", title="简历与项目讲述:把研究翻译成岗位语言", track="career",
    goals=["把岗位描述拆成能力清单,并为每一项找到证据", "按“动作、方法、结果、意义”写简历条目",
           "录一段 2 分钟的项目介绍", "针对三个不同岗位调整侧重点"],
    slides=[
        ("先拆岗位描述", TABLE(["岗位要求(腾讯混元评测度量)", "你的证据"], [["一致性与信度效度分析", "课程项目:κ、α、ICC 与区间;博士阶段量表研究"],
              ["假设检验与置信区间", "RCT、三层元分析;项目中的配对比较与 Holm 校正"], ["项目反应理论", "课程第 6、8 天;项目题目分析"],
              ["LLM-as-judge 校准与偏差审计", "期末项目:七步协议、偏差检验"], ["数据污染检测", "课程第 13 天;项目污染说明"], ["因果推断基础", "课程第 22 天;RCT 设计经验"],
              ["Python 与统计工具", "统计硕士;Micron 数据科学工程师;课程代码库"]])),
        ("简历条目的写法", FORMULA("动作 + 方法 + 结果(带数字) + 意义")
         + QA("示例(期末项目,数字换成你自己的)", "设计并实施 LLM 裁判效度研究:3 位持证咨询师盲评 84 条回复,裁判与咨询师二次加权 κ 为 __(95% CI __),在 __ 个维度上未达预设门槛;识别出每 100 字 __ 分的长度偏差。")
         + WARN("只写真实做过的。数字必须和仓库里的结果文件一致。", "原则")),
        ("把已有经历翻译成岗位语言", TABLE(["原来的写法", "评测岗位的写法"], [["三层元分析", "对 60 项随机对照试验的 385 个效应量做三层元分析,处理效应量的依赖结构与异质性"],
              ["CBT 聊天机器人 RCT", "设计并完成生成式 CBT 聊天机器人的随机对照试验,以效应量和区间报告结局"], ["数据科学工程师", "在 Micron 三年,用机器学习与 BERT 类模型处理工业数据"]])
         + NOTE("数字以你 CV 上的为准,这里只示范写法。", "注意")),
        ("2 分钟项目介绍", OL(["**问题**(20 秒):心理健康 LLM 的评测依赖自动裁判,但裁判本身很少被检验。", "**方法**(40 秒):咨询师金标准、七步协议、偏差检验、Agent 评测。",
                               "**结果**(40 秒):两三个最重要的数字,包括否定性结论。", "**意义**(20 秒):这些发现对上线决策意味着什么。"])),
        ("三个岗位的不同侧重", TABLE(["岗位", "侧重讲什么"], [["腾讯混元评测度量", "测量理论、区间、门槛、否定性结论、规范落地"], ["微信用户人格建模", "状态与特质、证据门槛、画像的信效度、隐私"],
              ["字节 AI 数据与安全评测", "安全门槛、pass^k、轨迹检查、评分流程与质检"]])),
        ("今日作业", OL(["完成 notebook 的岗位匹配表。", "写出 3 条项目简历条目和 3 条已有经历条目。", "录 2 分钟项目介绍,回放并修改一次。"])),
    ],
    cells=[
        ("md", """## 1. 岗位匹配表

把岗位要求逐条列出,为每一条写“证据”(仓库文件、论文、经历)。证据为空的要求,就是你面试前最需要补的。"""),
        ("code", """jd = [
    "一致性与信度效度分析", "假设检验与置信区间估计", "项目反应理论与偏好聚合模型",
    "LLM-as-judge 校准与偏差审计", "数据污染检测", "因果推断基础", "精通 Python 与统计工具",
    "测量理论、实验设计、可复现性研究或众包质量控制背景", "能把方法学要求转化为可执行规范",
]
evidence = {
    "一致性与信度效度分析": "",
    "假设检验与置信区间估计": "",
    "项目反应理论与偏好聚合模型": "",
    "LLM-as-judge 校准与偏差审计": "",
    "数据污染检测": "",
    "因果推断基础": "",
    "精通 Python 与统计工具": "",
    "测量理论、实验设计、可复现性研究或众包质量控制背景": "",
    "能把方法学要求转化为可执行规范": "",
}
match = pd.DataFrame({"requirement": jd, "evidence": [evidence[r] for r in jd]})
match["covered"] = match.evidence.str.len() > 0
match"""),
        ("md", """**练习 26.1**:填写 `evidence`,让覆盖率达到至少 8/9。"""),
        ("code", """ans_26_1 = None if not match.covered.any() else int(match.covered.sum())
check_true("练习 26.1 覆盖至少 8 项", None if ans_26_1 is None else ans_26_1 >= 8)"""),
        ("md", """## 2. 用结果文件生成简历条目

条目里的数字直接从结果文件读取,避免简历和仓库对不上。"""),
        ("code", """import pathlib
R20 = pathlib.Path("../day20_project_analysis/results")
if (R20 / "rq2_validity.csv").exists():
    rq2 = pd.read_csv(R20 / "rq2_validity.csv", index_col=0)
    best = rq2.kappa.idxmax(); worst = rq2.kappa.idxmin()
    bullet = (f"设计并实施 LLM 裁判效度研究,以咨询师盲评为金标准:裁判与咨询师二次加权 κ 在 {worst} 最低 "
              f"({rq2.loc[worst, 'kappa']:.2f}, 95% CI {rq2.loc[worst, 'lo']:.2f} 到 {rq2.loc[worst, 'hi']:.2f}),"
              f"{int((~rq2['pass'].astype(bool)).sum())} 个维度未达预设门槛。")
else:
    bullet = "(请先运行第 20 天的分析)"
print(bullet)"""),
        ("md", """**练习 26.2**:写出你的 6 条简历条目(3 条项目,3 条已有经历),存成列表 `ans_26_2`。每条不超过 60 字,数字以你 CV 和仓库为准。"""),
        ("code", """ans_26_2 = None
check_true("练习 26.2 六条简历条目", None if ans_26_2 is None else (isinstance(ans_26_2, list) and len(ans_26_2) >= 6))"""),
    ],
    interview=[("你的心理学背景对这个岗位有什么用?", "具体讲:测量训练让我先检验工具;RCT 和元分析让我熟悉区间与异质性;我能组织咨询师做金标准评分;我了解心理健康场景的真实风险。然后用项目举一个例子。"),
               ("用两分钟介绍你的项目。", "问题、方法、结果(含否定性结论)、意义,各控制在规定时间内。")],
    homework=["岗位匹配表覆盖至少 8/9", "6 条简历条目", "2 分钟项目介绍录音"],
)
D26.solutions = {}
D26.replacements = [('    "一致性与信度效度分析": "",', '    "一致性与信度效度分析": "期末项目 RQ1/RQ2",'),
                    ('    "假设检验与置信区间估计": "",', '    "假设检验与置信区间估计": "RCT;元分析;项目 RQ4",'),
                    ('    "项目反应理论与偏好聚合模型": "",', '    "项目反应理论与偏好聚合模型": "课程第 6、8、9 天",'),
                    ('    "LLM-as-judge 校准与偏差审计": "",', '    "LLM-as-judge 校准与偏差审计": "期末项目 RQ3",'),
                    ('    "数据污染检测": "",', '    "数据污染检测": "课程第 13 天",'),
                    ('    "因果推断基础": "",', '    "因果推断基础": "课程第 22 天;RCT 设计",'),
                    ('    "精通 Python 与统计工具": "",', '    "精通 Python 与统计工具": "统计硕士;Micron",'),
                    ('    "测量理论、实验设计、可复现性研究或众包质量控制背景": "",', '    "测量理论、实验设计、可复现性研究或众包质量控制背景": "博士量表研究;Dawid-Skene",'),
                    ('ans_26_2 = None', 'ans_26_2 = ["a", "b", "c", "d", "e", "f"]')]

# ============================================================== day 27 (question bank from all lessons)
def _bank():
    import importlib
    qs = []
    for w in ["w1", "w2", "w3"]:
        for l in importlib.import_module(w).LESSONS:
            for q, a in l.interview:
                qs.append((l.day, q, a))
    return qs


BANK = _bank()

D27 = Lesson(
    day=27, slug="mock_interview", title="全真模拟面试", track="career", hours="3 小时",
    goals=["完成一次 45 分钟的全真模拟面试", "用评分表给自己的每个回答打分", "找出最弱的三个问题并重写答案"],
    slides=[
        ("面试结构", TABLE(["环节", "时长", "内容"], [["自我介绍", "2 分钟", "背景、为什么转向这个方向、核心项目一句话"], ["项目深挖", "15 分钟", "每个数字、每个设计决定"],
              ["方法学问题", "15 分钟", "题库随机抽题,每题追问两层"], ["场景题", "10 分钟", "给一个真实情境,说你会怎么做"], ["反问", "3 分钟", "问团队的评测流程与决策方式"]])),
        ("场景题示例", UL(["新模型在 10 个维度里 3 个变好、7 个没变,你怎么写结论?", "业务方坚持用原始一致率汇报,你怎么说服?",
               "给你两周,为“老年人心理支持”能力搭建评测,你怎么做?", "训练团队说他们的新奖励模型让共情分数提高了,你先查什么?"])),
        ("自评表", TABLE(["维度", "1 分", "3 分", "5 分"], [["结论先行", "绕了很久才说结论", "结论在中间", "第一句就是结论"], ["有证据", "没有数字", "有数字没区间", "有数字、区间和来源"],
              ["说局限", "没提", "泛泛而谈", "具体到影响哪个结论"], ["时间", "超时一倍", "略超", "在 90 秒内"]])),
        ("反问什么", UL(["团队目前最信不过的评测结论是哪一个?", "评测结论被推翻时,通常是什么原因?", "人工评分和自动裁判各占多大比重?"])),
    ],
    cells=[
        ("md", f"""## 题库

共 {len(BANK)} 道题,来自前三周每一课。参考要点在折叠区里,**先自己答,再展开看**。"""),
        ("md", "\n\n".join(f"**D{d:02d}.** {q}\n\n<details><summary>参考要点</summary>{a}</details>" for d, q, a in BANK)),
        ("md", """## 随机抽题

运行下一格,抽 8 道题。每题限时 90 秒,录音。"""),
        ("code", f"""BANK = {[q for _, q, _ in BANK]!r}
rng = np.random.default_rng()      # 每次运行抽到的题不同
picked = list(rng.choice(BANK, 8, replace=False))
for i, q in enumerate(picked, 1):
    print(i, q)"""),
        ("md", """## 自评记录

回放录音后,按幻灯片上的自评表给每题四个维度打 1 到 5 分。"""),
        ("code", """log = pd.DataFrame({"question": picked, "结论先行": None, "有证据": None, "说局限": None, "时间": None})
log"""),
        ("md", """**练习 27.1**:填完 `log` 的分数后运行下一格,找出总分最低的三题。把这三题的重写答案写在最后的 Markdown 单元格里。"""),
        ("code", """filled = log.drop(columns="question").notna().all(axis=1).all()
if filled:
    log["total"] = log.drop(columns="question").sum(axis=1)
    print(log.nsmallest(3, "total")[["question", "total"]])
check_true("练习 27.1 自评表已填写", True if filled else None)"""),
        ("md", """*最弱三题的重写答案:*"""),
    ],
    homework=["45 分钟模拟面试录音", "自评表", "最弱三题的重写答案"],
)
D27.replacements = [('log = pd.DataFrame({"question": picked, "结论先行": None, "有证据": None, "说局限": None, "时间": None})',
                     'log = pd.DataFrame({"question": picked, "结论先行": 4, "有证据": 3, "说局限": 3, "时间": 4})')]

# ============================================================== day 28
D28 = Lesson(
    day=28, slug="final_review", title="查漏补缺与提交", track="review", hours="3 到 4 小时",
    goals=["用自动检查确认期末作业的交付物齐全", "按评分标准给自己的期末作业打分", "完成结业小测", "规划下一步:投递、研究或发表"],
    slides=[
        ("提交清单", UL(["仓库:README、run_all.py、docs/(rubric、分析计划、协议、审计、复现规范)、results/。", "报告:正文不超过 6 页,每个估计带区间。",
               "汇报幻灯片:约 10 页。", "简历条目和 2 分钟项目介绍录音。", "习题集 1 到 3。"])),
        ("按评分标准自评", P("打开 `COURSEWORK.md` 的评分表,逐项给自己打分,并写一句理由。分数最低的一项,就是下一周最值得改的地方。")),
        ("下一步", TABLE(["方向", "具体动作"], [["投递", "用第 26 天的简历条目更新简历;优先投评测、数据与安全、用户建模类岗位;找内推"],
              ["研究", "与导师讨论把期末作业扩展成方法学研究;补伦理程序;增加评分者和中文题目"], ["学习", "第 15 到 18 天的模型与 Agent 内容可以继续深入,但先把评测方法用熟"]])),
        ("一句话总结这门课", LEAD("先检验测量工具,再相信结论。这句话对心理学研究成立,对大模型评测同样成立。")),
    ],
    cells=[
        ("md", """## 1. 交付物自动检查

把 `REPO` 改成你的项目仓库路径,运行检查。"""),
        ("code", """import pathlib
REPO = pathlib.Path("../../my-project")     # 改成你的仓库路径
REQUIRED = ["README.md", "run_all.py", "docs/rubric.md", "docs/analysis_plan.md", "docs/judge_protocol.md",
            "docs/audit.md", "docs/reproducibility.md", "results"]
def check_submission(repo):
    rows = []
    for r in REQUIRED:
        rows.append({"item": r, "exists": (repo / r).exists()})
    readme = (repo / "README.md").read_text(encoding="utf-8") if (repo / "README.md").exists() else ""
    rows.append({"item": "README 含 95% 区间", "exists": "95%" in readme})
    rows.append({"item": "README 含局限", "exists": "局限" in readme or "Limitation" in readme})
    return pd.DataFrame(rows)
if REPO.exists():
    print(check_submission(REPO))
else:
    print("没有找到", REPO.resolve(), ",请修改 REPO 路径。")"""),
        ("md", """## 2. 结业小测(8 题)

每题填选项字母。"""),
        ("md", """1. 两位评分者原始一致率 96%,风险消息只占 3%。最应该补充报告的是:A. Pearson 相关 B. kappa、AC1 和混淆矩阵 C. 平均分
2. 多轮对话的评分区间应当按什么重抽样:A. 轮 B. 对话 C. 评分者
3. 两个系统在同一批题上的通过率比较,用:A. 卡方检验 B. 精确 McNemar 检验 C. 独立 t 检验
4. 只有 6 个模型时挑题,应当:A. 拟合 2PL B. 用描述性题目分析或 Rasch C. 不需要分析
5. 位置互换翻转率应当和什么比较:A. 0 B. 无偏差裁判的基线翻转率 C. 1
6. 危机处理的可靠性应当报告:A. pass@k B. pass^k C. 平均步数
7. 观察数据中使用者孤独感更高,最可能的解释是:A. 聊天机器人有害 B. 自我选择与混杂 C. 测量误差
8. 托管裁判悄悄更新,最直接的发现方法是:A. 看新闻 B. 锚定集加控制图 C. 换一个裁判"""),
        ("code", """ans_28 = None      # 例如 "BBBBBBBB"
check("结业小测", ans_28, "BBBBBBBB")"""),
        ("md", """## 3. 自评

按 `COURSEWORK.md` 的评分表给自己打分,写在下面。"""),
        ("md", """| 部分 | 分值 | 自评 | 理由 |
|---|---|---|---|
| 研究问题与设计 | 15 | | |
| 信度与一致性分析 | 20 | | |
| 偏差检验 | 15 | | |
| Agent 实现与评测 | 15 | | |
| 统计推断 | 15 | | |
| 可复现性 | 10 | | |
| 写作与局限 | 10 | | |"""),
    ],
    homework=["提交期末作业全部交付物", "按评分标准完成自评", "结业小测"],
)
D28.solutions = {"ans_28": '"BBBBBBBB"'}

LESSONS = [D22, D23, D24, D25, D26, D27, D28]
