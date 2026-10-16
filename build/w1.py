"""Week 1: measurement foundations (days 1 to 7)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framework import (CODE, COLS, FIG, FORMULA, LEAD, NOTE, OL, P, QA, TABLE, UL, WARN, Lesson, Paper)

from course import agreement, irt, stats
from course.plotting import INK, OCHRE, ROSE, SAND, TEAL

# ============================================================== figures
def fig_stable_but_wrong():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    q = rng.uniform(1.5, 4.5, 120)
    words = rng.normal(120, 50, 120).clip(20, 300)
    a = q + rng.normal(0, 0.8, 120)
    b = 1.2 + 0.25 * q + 0.012 * words + rng.normal(0, 0.15, 120)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True)
    for k, (s, name, c) in enumerate([(a, "Judge A: noisy, unbiased", TEAL), (b, "Judge B: stable, length-driven", OCHRE)]):
        ax[k].scatter(q, s, s=12, color=c, alpha=.75)
        ax[k].plot([1.5, 4.5], [1.5, 4.5], color=SAND, lw=1)
        ax[k].set_title(f"{name}\ncorr with true quality = {np.corrcoef(q, s)[0,1]:.2f}", fontsize=10)
        ax[k].set_xlabel("true reply quality")
    ax[0].set_ylabel("judge score")
    fig.tight_layout()
    return fig


def fig_kappa_paradox():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    ps = [0.5, 0.3, 0.15, 0.08, 0.04, 0.02]
    raw, kap, ac1 = [], [], []
    for p in ps:
        t = rng.random(6000) < p
        a = np.where(rng.random(6000) < .06, ~t, t).astype(int)
        b = np.where(rng.random(6000) < .06, ~t, t).astype(int)
        raw.append(agreement.percent_agreement(a, b)); kap.append(agreement.cohen_kappa(a, b)); ac1.append(agreement.gwet_ac1(a, b))
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.plot(ps, raw, "o-", color=INK, label="raw agreement")
    ax.plot(ps, kap, "o-", color=OCHRE, label="Cohen's kappa")
    ax.plot(ps, ac1, "o--", color=TEAL, label="Gwet's AC1")
    ax.set_xscale("log"); ax.set_xticks(ps); ax.set_xticklabels([f"{p:.0%}" for p in ps])
    ax.set_xlabel("share of messages that are truly high-risk"); ax.set_ylabel("agreement")
    ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def fig_cluster():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1)
    widths_n, widths_c = [], []
    iccs = [0.0, 0.2, 0.4, 0.6]
    for icc in iccs:
        conv = np.repeat(rng.normal(0, np.sqrt(icc), 40), 6)
        y = 3 + conv + rng.normal(0, np.sqrt(1 - icc), 240)
        cl = np.repeat(np.arange(40), 6)
        _, l1, h1 = stats.bootstrap_ci(y, n_boot=800)
        _, l2, h2 = stats.cluster_bootstrap_ci(y, cl, n_boot=800)
        widths_n.append(h1 - l1); widths_c.append(h2 - l2)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(iccs))
    ax.bar(x - .18, widths_n, .36, color=SAND, label="resample turns (wrong)")
    ax.bar(x + .18, widths_c, .36, color=TEAL, label="resample conversations")
    ax.set_xticks(x); ax.set_xticklabels([f"ICC = {v}" for v in iccs])
    ax.set_ylabel("width of 95% CI"); ax.legend()
    ax.set_title("40 conversations x 6 rated turns", fontsize=10)
    fig.tight_layout()
    return fig


def fig_paired():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(4)
    diff = rng.normal(0, 0.9, 42)
    a = 3.4 + diff + rng.normal(0, 0.35, 42)
    b = 3.1 + diff + rng.normal(0, 0.35, 42)
    pc = stats.paired_compare(a, b)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.errorbar([pc["diff"]], [1], xerr=[[pc["diff"] - pc["ci_low"]], [pc["ci_high"] - pc["diff"]]], fmt="o", color=TEAL, capsize=4)
    lo, hi = pc["unpaired_ci"]
    ax.errorbar([pc["diff"]], [0], xerr=[[pc["diff"] - lo], [hi - pc["diff"]]], fmt="o", color=SAND, capsize=4)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["treated as\nindependent", "paired by item"])
    ax.set_xlabel("difference in mean score (system A minus B), 95% CI")
    ax.set_ylim(-.6, 1.6)
    fig.tight_layout()
    return fig


def fig_power():
    import matplotlib.pyplot as plt
    ns = [20, 40, 80, 160, 320]
    pw = [stats.simulate_power(n, 0.83, 0.90, rho=0.5, n_sim=400, seed=1) for n in ns]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(ns, pw, "o-", color=OCHRE)
    ax.axhline(.8, color=SAND, lw=1)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("number of test items"); ax.set_ylabel("power (exact McNemar)")
    ax.set_ylim(0, 1); ax.set_title("Safety pass rate 0.83 vs 0.90, correlated systems", fontsize=10)
    fig.tight_layout()
    return fig


def fig_icc_curves():
    import matplotlib.pyplot as plt
    th = np.linspace(-3, 3, 200)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for (a, b, lab, c) in [(0.6, -1.5, "easy, flat (a=0.6, b=-1.5)", SAND), (1.8, 0.0, "steep (a=1.8, b=0)", TEAL), (1.2, 1.5, "hard (a=1.2, b=1.5)", OCHRE)]:
        p = 1 / (1 + np.exp(-a * (th - b)))
        ax[0].plot(th, p, color=c, label=lab)
        ax[1].plot(th, a * a * p * (1 - p), color=c)
    ax[0].set_title("Item characteristic curves"); ax[0].set_xlabel("ability of the model (theta)"); ax[0].set_ylabel("P(meets criterion)")
    ax[1].set_title("Item information"); ax[1].set_xlabel("ability of the model (theta)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_recovery():
    import matplotlib.pyplot as plt
    ns = [40, 100, 300, 1000]
    ra = []
    for n in ns:
        r = []
        for rep in range(3):
            u, th, a, b = irt.simulate_2pl(n, 30, seed=50 + rep)
            f = irt.fit_2pl(u, max_iter=60)
            r.append(np.corrcoef(f["a"], a)[0, 1])
        ra.append(np.mean(r))
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.plot(ns, ra, "o-", color=OCHRE)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("number of models evaluated"); ax.set_ylabel("corr(estimated a, true a)")
    ax.set_ylim(0, 1); ax.set_title("Discrimination needs many respondents", fontsize=10)
    fig.tight_layout()
    return fig


# ============================================================== day 1
D1 = Lesson(
    day=1, slug="reliability_validity", title="信度与效度:先检验测量工具", track="eval",
    goals=["用经典测量理论解释信度,并说明信度是效度的必要不充分条件",
           "区分重测信度、内部一致性和评分者间信度",
           "把这些概念对应到两类问题:LLM 当裁判,以及让 LLM 填心理量表",
           "用模拟数据展示一个稳定但不准确的裁判"],
    slides=[
        ("为什么从测量开始", LEAD("评测结论被用来决定模型能不能上线,但给出结论的测量工具本身很少被检验。心理测量学处理同一个问题已经一百多年。")
         + UL(["**被测对象一:来访者**。PHQ-9 这样的量表,你最熟悉。",
               "**被测对象二:聊天机器人的回复**。由咨询师或 LLM 裁判按 rubric 打分。",
               "**被测对象三:模型本身**。让模型回答人格或心理量表,把它当作被试。",
               "三类问题用的是同一套概念:信度、效度、测量误差。"])),
        ("经典测量理论", FORMULA("X = T + E", "观测分 = 真分 + 误差")
         + FORMULA("reliability = Var(T) / Var(X)", "真分方差占观测分方差的比例")
         + UL(["假设误差均值为 0,且与真分无关。", "两次平行测量的相关系数等于信度。",
               "信度是**某个人群中分数**的性质,不是工具单独的属性。换一个人群(比如从大学生换到老年人),信度要重新估计。"])),
        ("信度的三种证据", TABLE(["类型", "要回答的问题", "心理学中的例子", "LLM 评测中的对应"],
              [["重测信度", "隔一段时间再测,结果一致吗", "PHQ-9 两周重测", "同一条回复,裁判打分三次是否一致"],
               ["内部一致性", "这些题目测的是同一个构念吗", "Cronbach's α", "rubric 的五个维度是否在测同一件事"],
               ["评分者间信度", "不同评分者一致吗", "两位督导给同一段会谈打 CTS-R", "LLM 裁判与咨询师是否一致"]])),
        ("效度:测到的是想测的东西吗", UL(["**内容效度**:题目是否覆盖了要测的范围,例如测试集是否覆盖老年人常见的求助场景。",
               "**结构效度**:分数的内部结构是否符合理论,例如共情维度是否与技术维度分开。",
               "**效标效度**:与外部标准的关系。同时效度看与专家评分的相关,预测效度看能否预测用户结局。",
               "**后果效度**:使用这些分数做决策会带来什么后果。"])
         + NOTE("现代测量理论(Messick, 1989;美国教育与心理测验标准, 2014)把效度看作对“分数如何解释和使用”的论证,不是工具本身的属性。", "概念")),
        ("信度是效度的必要不充分条件", FIG(fig_stable_but_wrong, "模拟数据。裁判 B 每次打分几乎相同,但它主要在看回复长度。")
         + P("裁判 B 的重测信度很高,和真实质量的相关却比裁判 A 低。**稳定只排除了随机误差,不排除系统偏差。**")),
        ("映射到大模型评测", TABLE(["心理测量概念", "在 LLM 评测中怎么检验"],
              [["重测信度", "同一输入、同一裁判,重复打分的一致性"], ["评分者间信度", "裁判与咨询师的加权 kappa 或 ICC"],
               ["内容效度", "测试集是否覆盖真实使用场景,请专家评审"], ["效标效度", "裁判分数与专家评分、用户结局的关系"],
               ["后果效度", "依据这个分数上线的模型,是否真的更安全"]])),
        ("例一:让 LLM 填人格量表", P("Serapio-García 等人(2023, arXiv:2307.00184)把人格测验用于大模型,检验测得的人格分数是否可靠、有效。")
         + UL(["在特定提示设置下,部分模型的人格分数显示出信度和效度。", "较大的、经过指令微调的模型证据更强。",
               "模型输出的人格可以被定向塑造。"])
         + WARN("模型对量表的回答随提示词变化,题目顺序、措辞都会改变分数。把人类量表直接用在模型上,需要先检验这些量表在模型上的测量性质。", "注意")),
        ("例二:心理健康 LLM 研究怎么评估", P("Hua 等人(2025, npj Digital Medicine)的范围综述纳入 16 项研究,发现多数使用临时编制、未经标准化的评价量表,结果难以比较。")
         + NOTE("这正是心理学背景的人能补上的地方:把评价量表当作一个需要做信效度研究的新工具。", "你的机会")),
        ("今日作业", OL(["完成 notebook 的 4 个练习。", "精读 Hua 等人 2025,回答学习单上的 3 个问题。",
                     "写一段 150 字以内的文字:如果你要给“老年人心理支持回复”设计评价量表,怎样收集内容效度证据?",
                     "录音回答 2 道口述题。"])),
    ],
    cells=[
        ("md", """## 1. 经典测量理论:用模拟来理解

我们模拟 1000 位来访者的抑郁真分 T,再测量两次,每次加入独立误差。两次测量的相关,就是信度的估计。"""),
        ("code", """rng = np.random.default_rng(0)
n = 1000
T = rng.normal(10, 4, n)          # 真分,标准差 4
E1 = rng.normal(0, 2, n)          # 第一次测量误差,标准差 2
E2 = rng.normal(0, 2, n)          # 第二次测量误差
X1, X2 = T + E1, T + E2

theoretical = 4**2 / (4**2 + 2**2)
print(f"理论信度 Var(T)/Var(X) = {theoretical:.3f}")
print(f"两次测量的相关         = {np.corrcoef(X1, X2)[0,1]:.3f}")"""),
        ("md", """**练习 1.1**:如果误差标准差从 2 增加到 4(例如来访者在嘈杂环境中用手机作答),理论信度是多少?把答案填到 `ans_1_1`。"""),
        ("code", """ans_1_1 = None   # 例如 0.5
check("练习 1.1 误差 SD=4 时的理论信度", ans_1_1, 16 / (16 + 16), tol=0.01)"""),
        ("md", """## 2. 心理量表的信度:PHQ-9 风格的模拟数据

`simulate_phq9` 生成 9 个题目(0 到 3 分),由同一个潜在抑郁严重度驱动,并在两周后重测。下面计算内部一致性(Cronbach's α)。"""),
        ("code", """from course.data import simulate_phq9
from course.audit import cronbach_alpha

t1, t2, latent = simulate_phq9(n=800, retest_r=0.85, seed=1)
print("第一次测量的 α =", round(cronbach_alpha(t1), 3))
print("总分范围:", t1.sum(1).min(), "到", t1.sum(1).max())"""),
        ("md", """**练习 1.2**:计算两次测量总分的重测相关(Pearson r),填到 `ans_1_2`。提示:`t1.sum(1)` 是每个人的总分。"""),
        ("code", """ans_1_2 = None
check("练习 1.2 PHQ 总分的重测相关", ans_1_2, np.corrcoef(t1.sum(1), t2.sum(1))[0, 1], tol=0.005)"""),
        ("md", """## 3. 稳定但不准确的裁判

模拟 120 条聊天机器人回复,每条有真实质量(1 到 5)和长度。
- 裁判 A:围绕真实质量随机波动,噪声大,没有系统偏差。
- 裁判 B:噪声小,但主要受回复长度影响。

每个裁判对同一批回复打分两次。"""),
        ("code", """rng = np.random.default_rng(3)
q = rng.uniform(1.5, 4.5, 120)                 # 真实质量
words = rng.normal(120, 50, 120).clip(20, 300) # 回复字数
def judge_A(): return q + rng.normal(0, 0.8, 120)
def judge_B(): return 1.2 + 0.25 * q + 0.012 * words + rng.normal(0, 0.15, 120)
A1, A2, B1, B2 = judge_A(), judge_A(), judge_B(), judge_B()

table = pd.DataFrame({
    "重测信度 r(run1, run2)": [np.corrcoef(A1, A2)[0,1], np.corrcoef(B1, B2)[0,1]],
    "与真实质量的相关": [np.corrcoef(A1, q)[0,1], np.corrcoef(B1, q)[0,1]],
    "与长度的相关": [np.corrcoef(A1, words)[0,1], np.corrcoef(B1, words)[0,1]],
}, index=["裁判 A", "裁判 B"])
table"""),
        ("md", """**练习 1.3**:哪个裁判的重测信度更高?哪个裁判的效度(与真实质量的相关)更高?分别填 `"A"` 或 `"B"`。"""),
        ("code", """ans_1_3_reliability = None
ans_1_3_validity = None
check("练习 1.3a 重测信度更高的裁判", ans_1_3_reliability, "B")
check("练习 1.3b 效度更高的裁判", ans_1_3_validity, "A")"""),
        ("md", """## 4. 衰减校正:一致性的上限

裁判和咨询师都有测量误差。两者分数的相关,上限是 $\\sqrt{r_{xx} \\cdot r_{yy}}$(两者信度的几何平均)。Spearman 衰减校正公式估计“去掉误差后”的相关:

$$ r_{true} = \\frac{r_{xy}}{\\sqrt{r_{xx} \\, r_{yy}}} $$

这告诉我们:如果咨询师之间本身就只有中等一致,就不能要求裁判与咨询师高度一致。"""),
        ("md", """**练习 1.4**:裁判与咨询师评分相关 0.50,裁判重测信度 0.70,咨询师评分者间信度 0.80。校正后的相关是多少?"""),
        ("code", """ans_1_4 = None
check("练习 1.4 衰减校正后的相关", ans_1_4, 0.5 / np.sqrt(0.7 * 0.8), tol=0.01)"""),
        ("md", """## 5. 写下来(不批改,但要写)

用三句话回答:为什么“裁判打分很稳定”不能说明“裁判打分很准”?你会用什么证据证明一个裁判有效?"""),
        ("md", """*在这里写你的回答。*"""),
    ],
    paper=Paper("Hua Y. et al. (2025). A scoping review of large language models for generative tasks in mental health care. npj Digital Medicine, 8, 230.",
                "这篇综述总结了 LLM 在心理健康生成任务中的研究现状,并指出评价方法不统一的问题。它说明了为什么心理测量训练在这个领域有用。",
                ["综述纳入了多少项研究?这些研究主要评估了哪几类任务?", "作者认为现有评价方法的主要问题是什么?",
                 "如果要你提出一个标准化的评价方案,第一步你会做什么?为什么?"],
                "https://www.nature.com/articles/s41746-025-01611-4"),
    research="你在博士研究中做过量表测量(UTAUT2 问卷、RCT 结局量表)。可以把“咨询师加 rubric”当作一个新的评价工具,按量表开发的流程收集信效度证据:内容效度(专家评审)、评分者间信度、与外部标准的相关。这本身可以构成一项方法学研究。",
    interview=[("裁判打分很稳定,能说明它准吗?", "不能。稳定是信度,只排除随机误差。系统偏差(偏好长回复、偏好同源模型)可以同时很稳定。要用专家评分做效标,并做位置互换、长度控制等偏差检验。"),
               ("信度和效度是什么关系?举一个 LLM 评测的例子。", "信度是效度的必要不充分条件。例子:一个按长度打分的裁判,重测一致性很高,但与咨询师的评分相关低。")],
    homework=["notebook 练习 1.1 到 1.4 全部显示“正确”", "150 字:老年人心理支持回复的内容效度证据如何收集", "学习单上的 3 个论文问题"],
)

# ============================================================== day 2
D2 = Lesson(
    day=2, slug="agreement", title="一致性指标:kappa、alpha 与 ICC", track="eval",
    goals=["手算 Cohen's kappa,并解释为什么要扣除偶然一致", "解释 kappa 悖论,知道类别不平衡时该报告什么",
           "按数据类型和评分者数量选择指标:加权 kappa、Krippendorff's α、ICC", "给一致性指标报告 bootstrap 区间"],
    slides=[
        ("场景:危机消息筛查", LEAD("400 条求助消息中约 5% 有自伤风险。一位咨询师和一个 LLM 筛查器分别标注“有风险/无风险”。")
         + P("原始一致率可能超过 90%,但这几乎全部来自双方都说“无风险”的大量消息。真正重要的是那 20 条高风险消息上是否一致。")),
        ("Cohen's kappa", FORMULA("κ = (Po − Pe) / (1 − Pe)")
         + UL(["**Po**:观测一致率,混淆矩阵对角线之和除以总数。", "**Pe**:偶然一致率,按两位评分者各自的边际比例计算:Σ pA(k)·pB(k)。",
               "κ = 1 表示完全一致,κ = 0 表示与偶然一致没有区别,κ < 0 表示比偶然还差。"])
         + TABLE(["", "LLM: 有风险", "LLM: 无风险"], [["咨询师: 有风险", "15", "6"], ["咨询师: 无风险", "4", "375"]])
         + P("Po = (15 + 375) / 400 = 0.975;Pe = (21/400)(19/400) + (379/400)(381/400) ≈ 0.905;κ ≈ 0.74。")),
        ("kappa 悖论", FIG(fig_kappa_paradox, "模拟:两位评分者的出错率始终是 6%,只改变高风险消息的比例。")
         + P("原始一致率几乎不变,kappa 随类别不平衡加剧而下降。**kappa 同时反映评分者和数据分布**,不能脱离患病率解读。")),
        ("有序评分:加权 kappa", P("共情评分常用 0 / 1 / 2 三级(参考 Sharma 等人 2020 年的 EPITOME 框架)。把 2 判成 1,比把 2 判成 0 错得轻。")
         + TABLE(["权重", "差 1 级的惩罚", "差 2 级的惩罚", "适用"], [["不加权", "1", "1", "名义类别"], ["线性", "0.5", "1", "有序,等距不确定"], ["二次", "0.25", "1", "有序,近似等距"]])
         + NOTE("二次加权 kappa 在一定条件下与 ICC 数值接近。", "关系")),
        ("多位评分者与缺失:Krippendorff's α", FORMULA("α = 1 − Do / De", "观测分歧 / 期望分歧")
         + UL(["支持任意数量的评分者,允许缺失(不是每位咨询师都评了每条回复)。",
               "支持名义、有序、等距等测量水平,由距离函数决定。", "多评分者研究的首选报告指标之一。"])),
        ("连续分数:ICC", TABLE(["形式", "模型", "回答的问题"],
              [["ICC(1,1)", "单向随机", "每条回复由不同评分者评"], ["ICC(2,1)", "双向随机,绝对一致", "换一位同类评分者,分数是否相同"],
               ["ICC(3,1)", "双向混合,一致性", "评分者固定,只看排序是否一致"], ["ICC(2,k)", "k 位评分者的平均分", "平均分的信度"]])
         + WARN("评分者有系统性的宽严差异时,ICC(3,1) 会高于 ICC(2,1)。报告 ICC 必须写明形式(Shrout & Fleiss, 1979;Koo & Li, 2016)。")),
        ("怎么选", TABLE(["数据", "两位评分者", "三位及以上或有缺失"],
              [["名义类别(风险有/无)", "Cohen's κ,同时报告 AC1 和混淆矩阵", "Krippendorff's α(名义)"],
               ["有序等级(0/1/2 共情)", "加权 κ(线性或二次)", "Krippendorff's α(有序)"],
               ["连续或近似连续(1 到 5 分平均)", "ICC", "ICC 或 α(等距)"]])),
        ("报告规范", UL(["给混淆矩阵,让读者看到分歧在哪里。", "报告不止一个指标,并给出 95% 区间。",
               "分层报告:高风险消息和普通消息分开算。", "Landis 和 Koch(1977)的分级(如 0.61 到 0.80 为“高度一致”)只是约定,不是理论标准。"])),
        ("今日论文:LLM 与临床专家的风险判断", P("McBain 等人(2025, Psychiatric Services):13 位临床专家把 30 个自杀相关问题分成 5 个风险等级;ChatGPT、Claude、Gemini 每题各回答 100 次。")
         + UL(["在风险最低和最高两端,模型的回应方式与专家一致。", "在中间风险等级上,模型的表现不能可靠地区分。",
               "启示:总体一致可以掩盖关键子集上的不一致,所以要**分层报告**。"])),
        ("今日作业", OL(["完成 notebook 练习 2.1 到 2.5。", "精读 McBain 等人 2025,回答学习单问题。",
                     "用一张表总结:你的项目中五个 rubric 维度和安全门槛,各自该用哪个一致性指标。"])),
    ],
    cells=[
        ("md", """## 1. 危机消息筛查:混淆矩阵与 kappa"""),
        ("code", """from course.data import simulate_risk_labels
from course.agreement import percent_agreement, cohen_kappa, gwet_ac1, confusion_table, krippendorff_alpha, icc, bootstrap_ci

df = simulate_risk_labels(n=400, prevalence=0.05, seed=0)
confusion_table(df.counselor, df.llm, categories=[1, 0])"""),
        ("md", """**练习 2.1**:根据上面的混淆矩阵,**手算** Po、Pe 和 kappa(可以用 Python 做算术,但不要调用 `cohen_kappa`)。"""),
        ("code", """ct = confusion_table(df.counselor, df.llm, categories=[1, 0]).values
n = ct.sum()
Po = None   # 对角线之和 / n
Pe = None   # 两位评分者边际比例乘积之和
kappa_by_hand = None
check("练习 2.1a Po", Po, percent_agreement(df.counselor, df.llm), tol=1e-6)
check("练习 2.1b kappa", kappa_by_hand, cohen_kappa(df.counselor, df.llm), tol=1e-6)"""),
        ("code", """print("原始一致率:", round(percent_agreement(df.counselor, df.llm), 3))
print("Cohen's kappa:", round(cohen_kappa(df.counselor, df.llm), 3))
print("Gwet's AC1:", round(gwet_ac1(df.counselor, df.llm), 3))"""),
        ("md", """## 2. kappa 悖论:改变患病率

固定两位评分者的灵敏度和特异度,只改变高风险消息的比例。"""),
        ("code", """rows = []
for prev in [0.5, 0.2, 0.1, 0.05, 0.02]:
    d = simulate_risk_labels(n=4000, prevalence=prev, seed=1)
    rows.append({"prevalence": prev, "raw": percent_agreement(d.counselor, d.llm),
                 "kappa": cohen_kappa(d.counselor, d.llm), "AC1": gwet_ac1(d.counselor, d.llm)})
sweep = pd.DataFrame(rows); sweep"""),
        ("code", """fig, ax = plt.subplots(figsize=(6, 3))
for col, c in [("raw", INK), ("kappa", OCHRE), ("AC1", TEAL)]:
    ax.plot(sweep.prevalence, sweep[col], "o-", color=c, label=col)
ax.set_xscale("log"); ax.set_xlabel("prevalence of high-risk messages"); ax.set_ylabel("agreement"); ax.legend(); plt.show()"""),
        ("md", """**练习 2.2**:在患病率 2% 时,raw 和 kappa 的差是多少?(`sweep` 最后一行的 raw 减 kappa)"""),
        ("code", """ans_2_2 = None
check("练习 2.2 raw - kappa @ 2%", ans_2_2, float(sweep.iloc[-1].raw - sweep.iloc[-1].kappa), tol=0.001)"""),
        ("md", """## 3. 有序评分:共情等级 0/1/2"""),
        ("code", """from course.data import simulate_empathy_levels
emp = simulate_empathy_levels(n=300, agree=0.7, seed=2)
for w in [None, "linear", "quadratic"]:
    print(f"{str(w):>9}: {cohen_kappa(emp.rater1, emp.rater2, weights=w, categories=[0,1,2]):.3f}")"""),
        ("md", """**练习 2.3**:为什么二次加权 kappa 最高?在 `ans_2_3` 里选择正确的解释编号:
1. 二次加权对相差 1 级的分歧惩罚最小,而这里多数分歧只差 1 级
2. 二次加权忽略了偶然一致
3. 二次加权只看对角线"""),
        ("code", """ans_2_3 = None
check("练习 2.3", ans_2_3, 1)"""),
        ("md", """## 4. 三位咨询师:Krippendorff's α 与 ICC

用课程的模拟研究数据:三位咨询师给两个聊天机器人系统的回复打分(1 到 5)。c2 偏宽松或偏严格,c3 噪声较大。"""),
        ("code", """from course.data import simulate_counselor_study
study = simulate_counselor_study(seed=0)
h = study["human"]
d1 = h[h.dim == "D1"].pivot_table(index=["item", "system"], columns="rater", values="score")
d1.head()"""),
        ("code", """print("Krippendorff α (ordinal):", round(krippendorff_alpha(d1.values.T, "ordinal"), 3))
for k in ["ICC(1,1)", "ICC(2,1)", "ICC(3,1)", "ICC(2,k)"]:
    print(f"{k}: {icc(d1.values, k):.3f}")
print("各评分者平均分:", d1.mean().round(2).to_dict())"""),
        ("md", """**练习 2.4**:咨询师之间有系统的宽严差异。ICC(3,1) 是否高于 ICC(2,1)?填 `True` 或 `False`。然后在下面写一句话解释原因。"""),
        ("code", """ans_2_4 = None
check("练习 2.4", ans_2_4, bool(icc(d1.values, "ICC(3,1)") > icc(d1.values, "ICC(2,1)")))"""),
        ("md", """*解释:*"""),
        ("md", """## 5. 给 kappa 加区间"""),
        ("code", """est, lo, hi = bootstrap_ci(cohen_kappa, df.counselor.values, df.llm.values, n_boot=2000, categories=[0, 1])
print(f"kappa = {est:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")"""),
        ("md", """**练习 2.5**:只有 20 条左右的高风险消息时,区间宽度大约是多少?把 `hi - lo` 填进来,并想一想:这样的区间能支持“LLM 筛查与咨询师一致”的结论吗?"""),
        ("code", """ans_2_5 = None
check("练习 2.5 区间宽度", ans_2_5, hi - lo, tol=0.01)"""),
    ],
    paper=Paper("McBain R. K. et al. (2025). Evaluation of alignment between large language models and expert clinicians in suicide risk assessment. Psychiatric Services. doi:10.1176/appi.ps.20250086",
                "这项研究把临床专家的风险分级当作参照,检验三个聊天机器人的回应方式是否随风险等级变化。它展示了“总体一致”和“关键子集一致”的区别。",
                ["专家是如何给 30 个问题分级的?专家之间的一致性有没有报告?", "为什么每个问题要让模型回答 100 次?这和今天学的哪个概念有关?",
                 "如果你重新设计这项研究,你会报告哪些一致性指标?"],
                "https://doi.org/10.1176/appi.ps.20250086"),
    research="你的项目里,安全门槛是低患病率的二元判断,正好会遇到 kappa 悖论。一个有价值的研究设计是:按风险等级分层,分别报告 LLM 裁判与咨询师的一致性,并比较老年人场景和大学生场景。",
    interview=[("业务方说人机一致率 90%,你怎么回应?", "先问类别分布。原始一致率没有扣除偶然一致,类别不平衡时会虚高。我会看混淆矩阵、kappa 和 AC1,给出区间,并单独报告少数类上的一致性。"),
               ("三位咨询师有人偏宽松,你报告哪种 ICC?", "说明研究问题。如果关心换一位评分者分数是否相同,用 ICC(2,1) 绝对一致;如果只关心排序,用 ICC(3,1)。两者差距本身说明存在系统宽严差异。")],
    homework=["notebook 练习 2.1 到 2.5", "一张表:每个 rubric 维度和安全门槛该用的一致性指标及理由", "学习单论文问题"],
)

# ============================================================== day 3
D3 = Lesson(
    day=3, slug="intervals", title="区间估计:一个分数没有区间就不是结论", track="eval",
    goals=["计算比例的标准误,并在接近 0 或 1 时使用 Wilson 区间", "用 bootstrap 给任意评测指标加区间",
           "识别多轮对话中的聚类结构,用整段对话重抽样", "计算 Hedges' g 及其区间,连接 RCT 与评测"],
    slides=[
        ("一个分数没有区间就不是结论", LEAD("某个 CBT 聊天机器人在 30 道题上的安全通过率是 90%(27/30)。")
         + P("Wilson 95% 区间大约是 74% 到 97%。换一批 30 道题,结果可能差十几个百分点。**没有区间的“90%”,不能用来比较或决策。**")),
        ("不确定性从哪里来", UL(["**题目抽样**:测试集只是所有可能求助消息中的一个样本。",
               "**模型采样**:temperature 大于 0 时,同一提示每次回复不同。", "**裁判采样**:裁判本身也有随机性。",
               "三者叠加。报告时至少要说明区间考虑了哪一种。"])),
        ("比例的标准误", FORMULA("SE = √( p(1 − p) / n )")
         + TABLE(["题目数 n", "p = 0.7 时 95% 区间半宽"], [["30", "±16.4 个百分点"], ["100", "±9.0"], ["400", "±4.5"], ["1000", "±2.8"]])
         + P("很多“提升 2 个点”的报告,落在这个误差范围内。")),
        ("接近 0 或 1 时用 Wilson 区间", P("安全通过率常常接近 1。Wald 区间(p ± 1.96·SE)在 p 接近 0 或 1 时会越界或过窄。")
         + TABLE(["结果", "Wald 95% 区间", "Wilson 95% 区间"], [["30/30 通过", "[1.00, 1.00]", "[0.89, 1.00]"], ["27/30 通过", "[0.79, 1.01]", "[0.74, 0.97]"]])
         + NOTE("30 道题全部通过,并不说明失败率是 0。Wilson 区间提醒你:失败率仍可能高达约 11%。", "解读")),
        ("Bootstrap", OL(["从 n 道题中有放回地抽 n 道。", "在抽出的样本上重算指标(平均分、kappa、通过率之差都可以)。",
                         "重复 2000 次以上。", "取 2.5% 和 97.5% 分位数作为 95% 区间。"])
         + P("优点:任何指标都适用,不需要推导公式。前提:重抽样的单位要相互独立。")),
        ("多轮对话的聚类", FIG(fig_cluster, "模拟:40 段对话,每段 6 轮被评分。")
         + FORMULA("设计效应 DEFF = 1 + (m − 1)·ICC", "m = 每段对话的被评分轮数")
         + WARN("同一段对话里的各轮回复不独立。按“轮”重抽样会低估区间宽度,应该按“对话”重抽样。")),
        ("效应量与区间:连接 RCT", FORMULA("g = J · (M₁ − M₂) / SD_pooled", "J = 1 − 3 / (4(n₁ + n₂) − 9)")
         + P("你在 RCT 和元分析里用的就是这一套。评测里比较两个系统,同样应该报告差值和区间,而不只是 p 值。")),
        ("今日论文:Therabot 随机对照试验", P("Heinz 等人(2025, NEJM AI):210 名成人,诊断为抑郁、广泛性焦虑或进食障碍高风险,随机分到 Therabot 或等待组,干预 4 周,随访到 8 周。")
         + UL(["三个诊断组的症状下降都显著优于对照。", "研究人员因安全问题干预 15 次,纠正不恰当回复 13 次。",
               "这些安全事件是**回复层面**的数据,和结局层面的效应量回答的是不同问题。"])),
        ("今日作业", OL(["完成 notebook 练习 3.1 到 3.4。", "精读 Heinz 等人 2025,回答学习单问题。",
                     "写一段话:如果要评估 Therabot 的回复质量,而不只是结局,你会怎么设计抽样单位?"])),
    ],
    cells=[
        ("md", """## 1. 比例的区间:Wald 与 Wilson"""),
        ("code", """from course.stats import se_proportion, wald_ci, wilson_ci
for k, n in [(30, 30), (27, 30), (88, 100)]:
    p = k / n
    print(f"{k}/{n}: Wald {tuple(round(x,3) for x in wald_ci(p, n))}  Wilson {tuple(round(x,3) for x in wilson_ci(k, n))}")"""),
        ("md", """**练习 3.1**:我们的项目有 42 道题,其中 8 道是安全相关题。若某系统在 8 道安全题中通过 6 道,Wilson 95% 区间的下限是多少?"""),
        ("code", """ans_3_1 = None
check("练习 3.1 Wilson 下限 (6/8)", ans_3_1, wilson_ci(6, 8)[0], tol=0.005)"""),
        ("md", """## 2. Bootstrap:给裁判平均分加区间"""),
        ("code", """from course.data import simulate_counselor_study
from course.stats import bootstrap_ci, cluster_bootstrap_ci
study = simulate_counselor_study(seed=0)
j = study["judge"]
sysA = j[(j.system == "A") & (j.dim == "D1") & (j.run == 0)].score.values
print("系统 A 在 D1 的裁判平均分与 95% CI:", tuple(round(x, 3) for x in bootstrap_ci(sysA)))"""),
        ("md", """**练习 3.2**:用 `bootstrap_ci` 计算系统 B 在 D3(引导发现)上的平均分区间,填入区间下限。"""),
        ("code", """sysB_D3 = j[(j.system == "B") & (j.dim == "D3") & (j.run == 0)].score.values
ans_3_2 = None
check("练习 3.2 系统 B D3 区间下限", ans_3_2, bootstrap_ci(sysB_D3)[1], tol=0.02)"""),
        ("md", """## 3. 多轮对话:为什么要按对话重抽样

模拟 40 段对话,每段 6 轮回复被评分。同一段对话的分数共享一个“对话效应”(例如来访者话题难度)。"""),
        ("code", """rng = np.random.default_rng(1)
icc_true = 0.4
conv_effect = np.repeat(rng.normal(0, np.sqrt(icc_true), 40), 6)
scores = 3 + conv_effect + rng.normal(0, np.sqrt(1 - icc_true), 240)
clusters = np.repeat(np.arange(40), 6)
_, lo_n, hi_n = bootstrap_ci(scores)
_, lo_c, hi_c = cluster_bootstrap_ci(scores, clusters)
print(f"按轮重抽样:   宽度 {hi_n - lo_n:.3f}")
print(f"按对话重抽样: 宽度 {hi_c - lo_c:.3f}")"""),
        ("md", """**练习 3.3**:用设计效应公式 `1 + (m - 1) * ICC` 计算 m = 6、ICC = 0.4 时的设计效应。它近似等于两种方法区间宽度之比的平方。"""),
        ("code", """ans_3_3 = None
check("练习 3.3 设计效应", ans_3_3, 1 + 5 * 0.4, tol=0.01)
print("宽度之比的平方 =", round(((hi_c - lo_c) / (hi_n - lo_n)) ** 2, 2))"""),
        ("md", """## 4. 效应量:模拟一个聊天机器人 RCT

模拟 120 名大学生,随机分到 CBT 聊天机器人组或等待组,结局是 PHQ-9 总分的下降量。"""),
        ("code", """from course.stats import hedges_g
rng = np.random.default_rng(7)
chatbot = rng.normal(4.0, 4.5, 60)    # 下降量,越大越好
waitlist = rng.normal(1.5, 4.5, 60)
g = hedges_g(chatbot, waitlist)
boots = []
for _ in range(2000):
    boots.append(hedges_g(rng.choice(chatbot, 60), rng.choice(waitlist, 60)))
print(f"Hedges g = {g:.3f}, bootstrap 95% CI [{np.quantile(boots, .025):.3f}, {np.quantile(boots, .975):.3f}]")"""),
        ("md", """**练习 3.4**:不调用 `hedges_g`,用公式手算 g(合并标准差用 ddof=1 的方差),填入 `ans_3_4`。"""),
        ("code", """ans_3_4 = None
check("练习 3.4 Hedges g", ans_3_4, g, tol=1e-6)"""),
    ],
    paper=Paper("Heinz M. V. et al. (2025). Randomized Trial of a Generative AI Chatbot for Mental Health Treatment. NEJM AI, 2(4). doi:10.1056/AIoa2400802",
                "这是生成式 AI 心理聊天机器人较早的随机对照试验之一。阅读时注意:结局层面的效应量,和回复层面的安全事件,是两种不同的证据。",
                ["试验的对照组是什么?这种对照方式会如何影响效应量的解释?", "安全干预和回复纠正的次数,是按什么单位统计的?能否计算一个带区间的比率?",
                 "如果要把这项研究的回复数据做成一个评测集,你会如何抽样?"],
                "https://doi.org/10.1056/AIoa2400802"),
    research="你的 JMIR mHealth 论文是一项 CBT 聊天机器人 RCT。一个自然的延伸是:把结局层面的效应(症状变化)和回复层面的质量评分联系起来,检验“回复质量更高的对话,是否对应更好的结局”。这需要多层模型,也需要处理今天讲的聚类问题。",
    interview=[("为什么多轮对话要按对话而不是按轮重抽样?", "同一对话的各轮不独立,共享来访者和话题。按轮重抽样把相关的观测当成独立的,会低估方差,区间过窄。设计效应 1 + (m − 1)·ICC 给出低估的程度。"),
               ("30 道安全题全部通过,能说模型安全吗?", "不能。Wilson 区间下限约 0.89,失败率仍可能达到约 11%。需要更多题,或者针对高风险场景加测。")],
    homework=["notebook 练习 3.1 到 3.4", "一段话:评估 Therabot 回复质量时的抽样单位", "学习单论文问题"],
)

# ============================================================== day 4
D4 = Lesson(
    day=4, slug="compare_models", title="比较两个模型:配对、McNemar 与多重比较", track="eval",
    goals=["用配对设计比较同一批题上的两个系统,并解释它为什么更有效",
           "对二元结局使用精确 McNemar 检验", "在多个维度同时比较时控制多重比较",
           "区分统计显著和实际重要,写出可执行的决策规则"],
    slides=[
        ("场景:新版本真的更好吗", LEAD("训练团队报告:新版 CBT 模型的安全通过率从 83% 升到 88%,共情得分提高 0.2 分。")
         + P("你要回答三个问题:差异是否超出随机误差?是否同时检验了太多维度?差异大到值得上线吗?")),
        ("配对设计", FIG(fig_paired, "模拟:42 道题,两个系统在同一批题上作答。")
         + P("两个系统做同一批题时,题目难度对两者的影响相同。逐题求差 dᵢ = Aᵢ − Bᵢ,题目难度的变异被消掉,区间明显变窄。")),
        ("二元结局:McNemar 检验", TABLE(["", "B 通过", "B 失败"], [["A 通过", "a(两者都过)", "b10"], ["A 失败", "b01", "d(两者都失败)"]])
         + P("只有两个系统结果不同的题(b10 和 b01)提供信息。精确检验:在 b10 + b01 道不一致的题里,b10 服从 p = 0.5 的二项分布。")
         + WARN("把两个通过率当作独立样本用卡方检验,会忽略配对信息,结论可能相反。")),
        ("多重比较", P("同时比较 10 个维度,每个用 α = 0.05,至少出现一个假阳性的概率约为 1 − 0.95¹⁰ ≈ 40%。")
         + TABLE(["方法", "控制什么", "特点"], [["Bonferroni", "族错误率", "最保守:p × m"], ["Holm", "族错误率", "逐步,比 Bonferroni 有效"], ["Benjamini-Hochberg", "错误发现率", "探索性分析常用"]])),
        ("统计显著与实际重要", UL(["样本足够大时,很小的差异也会显著。", "临床研究用“最小临床重要差异”(MCID)判断变化是否有意义。",
               "评测同样需要事先定义:多大的差异值得改变上线决策?", "这个阈值要在看数据之前写下来。"])),
        ("一条可执行的决策规则", OL(["事先确定主要指标(例如安全通过率)和最小重要差异。", "配对比较,报告差值和 95% 区间。",
                                   "次要维度用 Holm 校正。", "主要指标区间下限高于 0 且超过最小重要差异,才写“提升”。",
                                   "否则写“未发现可靠差异”,并给出可检测的最小差异。"])),
        ("今日论文:CBT-Bench", P("Zhang 等人(NAACL 2025)把 CBT 能力分成三级:CBT 基础知识选择题、认知模型理解(识别认知扭曲、分类核心信念)、治疗性回复生成。")
         + UL(["模型在知识题上表现好,在复杂真实场景里分析来访者认知结构、生成有效回复时表现差。",
               "阅读时注意:三级任务的分数能不能直接比较?模型之间的差异有没有报告区间?"])),
        ("今日作业", OL(["完成 notebook 练习 4.1 到 4.4。", "精读 CBT-Bench,回答学习单问题。",
                     "为你的项目写下一条决策规则(主要指标、最小重要差异、次要指标的校正方法)。"])),
    ],
    cells=[
        ("md", """## 1. 用模拟研究数据比较系统 A 与系统 B"""),
        ("code", """from course.data import simulate_counselor_study, DIMENSIONS
from course.stats import paired_compare, mcnemar_exact, p_adjust
study = simulate_counselor_study(seed=0)
h = study["human"]
# 三位咨询师的平均分,作为每条回复的分数
avg = h[h.dim != "gate"].groupby(["item", "system", "dim"]).score.mean().unstack("system")
avg.head()"""),
        ("code", """res = []
for d in DIMENSIONS:
    sub = avg.xs(d, level="dim")
    pc = paired_compare(sub["A"].values, sub["B"].values)
    res.append({"dim": d, "diff A-B": pc["diff"], "ci_low": pc["ci_low"], "ci_high": pc["ci_high"], "p": pc["p_paired_t"]})
res = pd.DataFrame(res); res"""),
        ("md", """**练习 4.1**:用 Holm 方法校正上表的 5 个 p 值(`p_adjust(..., "holm")`),有几个维度在校正后仍然 p < 0.05?"""),
        ("code", """ans_4_1 = None
check("练习 4.1 Holm 校正后显著的维度数", ans_4_1, int((p_adjust(res.p.values, "holm") < 0.05).sum()))"""),
        ("md", """## 2. 配对与独立:区间宽度对比"""),
        ("code", """sub = avg.xs("D4", level="dim")
pc = paired_compare(sub["A"].values, sub["B"].values)
print("配对区间宽度:", round(pc["ci_high"] - pc["ci_low"], 3))
print("当作独立样本的区间宽度:", round(pc["unpaired_ci"][1] - pc["unpaired_ci"][0], 3))"""),
        ("md", """**练习 4.2**:独立区间宽度是配对区间宽度的几倍?(保留两位小数)"""),
        ("code", """ans_4_2 = None
check("练习 4.2 宽度之比", ans_4_2, (pc["unpaired_ci"][1] - pc["unpaired_ci"][0]) / (pc["ci_high"] - pc["ci_low"]), tol=0.05)"""),
        ("md", """## 3. 安全门槛:McNemar 检验

用咨询师 c1 的安全门槛判断(1 = 通过)。"""),
        ("code", """gate = h[(h.dim == "gate") & (h.rater == "c1")].pivot_table(index="item", columns="system", values="score")
print("通过率:", gate.mean().round(3).to_dict())
print(mcnemar_exact(gate["A"], gate["B"]))"""),
        ("md", """**练习 4.3**:只看风险题(`risk_flag == 1`)。重算两个系统的通过率和 McNemar p 值。填入 p 值,并思考:题数这么少时,这个检验能发现多大的差异?"""),
        ("code", """risky_items = study["replies"].query("risk_flag == 1")["item"].unique()
g_r = gate.loc[gate.index.isin(risky_items)]
ans_4_3 = None
check("练习 4.3 风险题 McNemar p", ans_4_3, mcnemar_exact(g_r["A"], g_r["B"])["p"], tol=1e-6)"""),
        ("md", """## 4. 决策规则

**练习 4.4**:写一个函数 `decide(diff, ci_low, ci_high, mid)`,规则如下:
- 区间下限 > `mid`:返回 `"improved"`
- 区间上限 < `-mid`:返回 `"worse"`
- 区间完全落在 `(-mid, mid)` 内:返回 `"equivalent"`
- 其他:返回 `"inconclusive"`"""),
        ("code", """def decide(diff, ci_low, ci_high, mid):
    return None   # 在这里实现

tests = [((0.5, 0.3, 0.7, 0.2), "improved"), ((-0.5, -0.8, -0.3, 0.2), "worse"),
         ((0.0, -0.1, 0.1, 0.2), "equivalent"), ((0.1, -0.2, 0.4, 0.2), "inconclusive")]
check("练习 4.4 决策函数", [decide(*a) for a, _ in tests], [e for _, e in tests])"""),
    ],
    paper=Paper("Zhang M. et al. (2025). CBT-Bench: Evaluating Large Language Models on Assisting Cognitive Behavior Therapy. Proceedings of NAACL 2025.",
                "这是专门针对 CBT 能力的基准,和你的研究方向最接近。阅读时把注意力放在测量设计上:任务怎么分级、怎么计分、结论有多确定。",
                ["三级任务分别是什么?哪一级最接近真实咨询?", "模型之间的比较是否报告了区间或显著性检验?",
                 "如果把 Level III 的回复交给咨询师评分,你会用今天学的哪种方法比较两个模型?"],
                "https://aclanthology.org/2025.naacl-long.196"),
    research="CBT-Bench 以英文和一般成人场景为主。把类似的分级框架用于中文和老年人场景(例如老年人常见的认知扭曲和核心信念),并用配对设计比较模型,是一个与你的研究直接衔接的方向。",
    interview=[("训练团队报告新模型提升 2 分,你怎么判断?", "配对比较给出差值和区间;多次运行检查采样方差;多维度用 Holm 校正;确认裁判版本没变、测试集没被污染;最后对照事先定好的最小重要差异。"),
               ("为什么二元通过率要用 McNemar 而不是卡方?", "两个系统做的是同一批题,结果配对。只有不一致的题提供差异信息。卡方把两组当独立样本,忽略了配对。")],
    homework=["notebook 练习 4.1 到 4.4", "你的项目的决策规则(写入课程仓库 docs/decision_rule.md)", "学习单论文问题"],
)

# ============================================================== day 5
D5 = Lesson(
    day=5, slug="power", title="题量与统计功效:需要多少道题", track="eval",
    goals=["用公式估算检测给定差异所需的题量", "反过来计算现有题量下的最小可检测差异(MDE)",
           "用模拟估计二元结局的功效", "列出不增加题目也能提高功效的方法"],
    slides=[
        ("问题", LEAD("我们的项目有 42 道题。它能分辨多大的差异?要分辨安全通过率 83% 和 90%,需要多少道题?")
         + P("这与你做 RCT 时的样本量计算是同一件事,只是“样本”从参与者变成了题目。")),
        ("连续分数的题量公式", FORMULA("n ≈ ( (z₁₋α/₂ + z₁₋β) · s / δ )²", "s = 逐题差值的标准差,δ = 想检测的差")
         + P("例:s = 0.8,δ = 0.2,α = 0.05,功效 0.8:n ≈ (2.80 × 0.8 / 0.2)² ≈ 126 道题。")),
        ("反过来:最小可检测差异", FORMULA("MDE = (z₁₋α/₂ + z₁₋β) · s / √n")
         + TABLE(["题目数 n", "MDE(s = 0.8)"], [["42", "0.35 分"], ["100", "0.22 分"], ["400", "0.11 分"]])
         + NOTE("报告“未发现显著差异”时,同时报告 MDE。否则读者无法区分“没有差异”和“题太少看不出来”。", "规范")),
        ("二元结局:用模拟估计功效", FIG(fig_power, "模拟:两个相关的系统,用精确 McNemar 检验。")
         + P("两个系统在同一批题上的结果相关,相关越高,不一致的题越少,功效越依赖题量。")),
        ("与 RCT 样本量计算的对应", TABLE(["RCT", "模型评测"], [["参与者", "题目(求助消息、临床情景)"], ["组间差异", "系统间差异"],
              ["前测协变量提高功效", "配对设计提高功效"], ["测量误差降低功效", "裁判噪声降低功效"], ["事先注册主要结局", "事先确定主要指标"]])),
        ("不加题也能提高功效", UL(["**配对设计**:两个系统做同一批题。", "**重复采样**:每题生成或评分多次取平均,降低噪声。",
               "**降低裁判噪声**:更清楚的 rubric,温度设为 0,多裁判平均。", "**挑高信息量的题**:第 6、8 天的 IRT 方法。",
               "**聚焦主要指标**:少做比较,减少多重校正的代价。"])),
        ("今日论文:MentalHealthBench", P("OpenAI(2026)发布的心理健康对话基准,由 80 多位持照心理学家和精神科医生参与编写评分标准,结果带 95% 置信区间,可拆成十个行为维度。")
         + UL(["每条评分标准有 −10 到 +10 的权重。", "阅读时估算:按维度拆开后,每个维度大约有多少条对话?能检测多大的差异?"])),
        ("今日作业", OL(["完成 notebook 练习 5.1 到 5.4。", "阅读 MentalHealthBench 公开说明,回答学习单问题。",
                     "为你的项目写一段“功效说明”:现有题量下各维度的 MDE。"])),
    ],
    cells=[
        ("md", """## 1. 公式计算题量"""),
        ("code", """from course.stats import required_items, simulate_power
from scipy import stats as st
print("s=0.5, delta=0.05 ->", required_items(0.5, 0.05), "道题")
print("s=0.8, delta=0.2  ->", required_items(0.8, 0.2), "道题")"""),
        ("md", """**练习 5.1**:如果你希望检测 0.3 分的差异,逐题差值的标准差是 1.0,α = 0.05、功效 0.9,需要多少道题?(用 `required_items`)"""),
        ("code", """ans_5_1 = None
check("练习 5.1 题量", ans_5_1, required_items(1.0, 0.3, power=0.9))"""),
        ("md", """## 2. 最小可检测差异

**练习 5.2**:写函数 `mde(s, n, alpha=0.05, power=0.8)`,并计算 s = 0.8、n = 42 时的 MDE。"""),
        ("code", """def mde(s, n, alpha=0.05, power=0.8):
    return None   # 用 st.norm.ppf

ans_5_2 = mde(0.8, 42)
check("练习 5.2 MDE (s=0.8, n=42)", ans_5_2, (st.norm.ppf(0.975) + st.norm.ppf(0.8)) * 0.8 / np.sqrt(42), tol=0.005)"""),
        ("md", """## 3. 用项目数据估计 s

从模拟研究里取三位咨询师的平均分,计算每个维度上 A − B 逐题差值的标准差,再算 MDE。"""),
        ("code", """from course.data import simulate_counselor_study, DIMENSIONS
study = simulate_counselor_study(seed=0)
h = study["human"]
avg = h[h.dim != "gate"].groupby(["item", "system", "dim"]).score.mean().unstack("system")
rows = []
for d in DIMENSIONS:
    diff = avg.xs(d, level="dim").eval("A - B")
    s = diff.std(ddof=1)
    rows.append({"dim": d, "sd_diff": s, "n_items": len(diff),
                 "MDE": (st.norm.ppf(0.975) + st.norm.ppf(0.8)) * s / np.sqrt(len(diff))})
pd.DataFrame(rows)"""),
        ("md", """## 4. 二元结局的功效模拟

安全通过率 83% 对 90%,两个系统的潜在相关 0.5。"""),
        ("code", """ns = [20, 40, 80, 160, 320]
power = [simulate_power(n, 0.83, 0.90, rho=0.5, n_sim=500, seed=1) for n in ns]
pd.DataFrame({"n_items": ns, "power": power})"""),
        ("md", """**练习 5.3**:根据上表,大约需要多少道题功效才能达到 0.8?在给定的选项中选一个:`80`、`160`、`320`、`超过 320`。"""),
        ("code", """ans_5_3 = None
first = next((n for n, p in zip(ns, power) if p >= 0.8), "超过 320")
check("练习 5.3", ans_5_3, first)"""),
        ("md", """**练习 5.4**:把潜在相关从 0.5 改为 0.8,重新跑 n = 160 的功效。功效变高还是变低?填 `"higher"` 或 `"lower"`,再写一句解释。"""),
        ("code", """p_rho8 = simulate_power(160, 0.83, 0.90, rho=0.8, n_sim=500, seed=1)
ans_5_4 = None
check("练习 5.4", ans_5_4, "higher" if p_rho8 > power[ns.index(160)] else "lower")"""),
        ("md", """*解释:*"""),
    ],
    paper=Paper("OpenAI (2026). Introducing MentalHealthBench. openai.com/index/introducing-mentalhealthbench",
                "这是大厂发布的心理健康对话基准,方法上有很多可以学习和质疑的地方。今天只看一个问题:它的结论有多确定。",
                ["基准由多少位临床专家参与?评分标准如何筛选保留?", "结果的置信区间是怎么报告的?按十个维度拆开后,你估计每个维度的确定性会怎样变化?",
                 "页面是否报告了自动评分裁判与临床专家的一致性?如果没有,你会怎么补?"],
                "https://openai.com/index/introducing-mentalhealthbench/"),
    research="很多心理健康 LLM 研究只用几十条对话就得出结论。一项有价值的方法学工作是:系统地报告已发表研究的 MDE,类似元分析中的功效审查。你做过三层元分析,这个方向很顺手。",
    interview=[("评测集只有 200 题,怎么办?", "先报告 MDE,说明能检测多大差异;用配对设计;每题重复采样降低噪声;用 IRT 挑高信息量的题;必要时补题,并优先补在关心的能力区间。"),
               ("结果不显著,能说两个模型一样吗?", "不能。要看 MDE 或做等效性检验。只有区间完全落在事先定义的等效范围内,才能说差异可以忽略。")],
    homework=["notebook 练习 5.1 到 5.4", "项目功效说明(docs/power.md)", "学习单论文问题"],
)

# ============================================================== day 6
D6 = Lesson(
    day=6, slug="irt_1", title="项目反应理论(一):题目并不等价", track="eval",
    goals=["写出 Rasch、2PL、3PL 模型并解释参数", "画出项目特征曲线和信息曲线,并解释峰值位置",
           "说明边际极大似然加 EM 的思路", "把 IRT 用到“模型 × 临床情景”数据上,并认识样本量的限制"],
    slides=[
        ("从你熟悉的心理测量说起", LEAD("PHQ-9 的 9 道题并不等价。“情绪低落”和“自杀意念”在抑郁严重度的不同位置上提供信息。")
         + P("计算机自适应测验(CAT)利用这一点,用更少的题达到同样的精度,抑郁筛查的 CAT 就是这样开发的(Gibbons 等人, 2012)。评测大模型可以借用同样的思路。")),
        ("三个模型", FORMULA("Rasch:P(θ) = 1 / (1 + e^−(θ − b))") + FORMULA("2PL:P(θ) = 1 / (1 + e^−a(θ − b))")
         + FORMULA("3PL:P(θ) = c + (1 − c) / (1 + e^−a(θ − b))")
         + UL(["**θ**:能力(模型的心理支持能力)", "**b**:难度,P = 0.5 时的能力位置", "**a**:区分度,曲线越陡越能区分强弱", "**c**:猜测参数,用于选择题"])),
        ("曲线与信息量", FIG(fig_icc_curves, "左:项目特征曲线。右:项目信息曲线。")
         + FORMULA("I(θ) = a² P(θ)(1 − P(θ))", "在 θ = b 处最大,最大值 a²/4")
         + FORMULA("SE(θ) = 1 / √I_test(θ)", "测验信息是各题信息之和")),
        ("估计:边际极大似然 + EM", OL(["假设能力服从标准正态分布,用一组积分点近似。", "**E 步**:给定当前题目参数,算每个被试在各积分点上的后验概率。",
                                   "**M 步**:给定后验,对每道题最大化期望对数似然,更新 a 和 b。", "重复直到对数似然不再变化。"])
         + NOTE("课程代码库 `course.irt.fit_2pl` 就是这样实现的,不到 80 行,值得读一遍。", "代码")),
        ("映射到大模型评测", TABLE(["心理测量", "模型评测"], [["被试", "被评测的模型(或模型加系统提示)"], ["题目", "临床情景题、求助消息"],
              ["答对", "回复满足该题的评分标准"], ["难度高的题", "多数模型都做不好的情景,例如间接表达的自杀风险"], ["区分度低的题", "强模型和弱模型都一样,不提供排序信息"]])),
        ("样本量的限制", FIG(fig_recovery, "模拟:30 道题,区分度估计与真值的相关。")
         + WARN("2PL 的区分度需要大量被试才能估稳。只有几个模型时,不能用 2PL;可以用 Rasch 或描述性题目分析(通过率、题总相关)。这一点在面试中值得主动说明。")),
        ("今日论文", P("Rodriguez 等人(ACL 2021),Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards? 他们把 IRT 用在问答排行榜上,估计每道题的难度和区分度。")
         + UL(["一些题区分度很低,甚至是负的(常常是标注有误)。", "用 IRT 可以更好地预测模型在新题上的表现,并帮助挑选题目。"])),
        ("今日作业", OL(["完成 notebook 练习 6.1 到 6.4。", "读完 `src/course/irt.py` 中的 `fit_2pl`,用注释标出 E 步和 M 步。",
                     "精读 Rodriguez 等人 2021,回答学习单问题。"])),
    ],
    cells=[
        ("md", """## 1. 模拟“模型 × 临床情景”数据

60 个模型(或模型配置),40 道临床情景题。1 表示回复满足该题的评分标准。真实参数已知,方便检验估计是否准确。"""),
        ("code", """from course.data import simulate_vignette_matrix
from course.irt import fit_2pl, item_information, total_information
u, theta, a_true, b_true, items = simulate_vignette_matrix(n_models=60, n_items=40, seed=0)
items.head(8)"""),
        ("code", """fit = fit_2pl(u)
print("收敛:", fit["converged"], " 迭代:", fit["n_iter"])
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].scatter(b_true, fit["b"], color=OCHRE, s=14); ax[0].set_title("difficulty b"); ax[0].set_xlabel("true"); ax[0].set_ylabel("estimated")
ax[1].scatter(a_true, fit["a"], color=TEAL, s=14); ax[1].set_title("discrimination a"); ax[1].set_xlabel("true")
plt.tight_layout(); plt.show()
print("corr b:", round(np.corrcoef(fit['b'], b_true)[0,1], 3), " corr a:", round(np.corrcoef(fit['a'], a_true)[0,1], 3))"""),
        ("md", """**练习 6.1**:区分度 a = 1.6 的题,在 θ = b 处的信息量是多少?(提示:P = 0.5)"""),
        ("code", """ans_6_1 = None
check("练习 6.1 最大信息量", ans_6_1, 1.6**2 / 4, tol=1e-6)"""),
        ("md", """**练习 6.2**:用估计出的参数,找出在 θ = 1(强模型区间)上信息量最大的题,填它的下标(0 开始)。"""),
        ("code", """ans_6_2 = None
check("练习 6.2 θ=1 处信息量最大的题", ans_6_2, int(item_information(1.0, fit["a"], fit["b"])[0].argmax()))"""),
        ("md", """## 2. 用在人类量表上:PHQ-9 风格数据

把每题“≥ 2 分”(一半以上天数)视为认可该症状,拟合 2PL。模拟中第 2 题(情绪低落)的载荷最高。"""),
        ("code", """from course.data import simulate_phq9
t1, _, latent = simulate_phq9(n=1500, seed=2)
endorse = (t1 >= 2).astype(float)
phq_fit = fit_2pl(endorse)
pd.DataFrame({"item": [f"PHQ{j+1}" for j in range(9)], "a": phq_fit["a"], "b": phq_fit["b"],
              "endorse_rate": endorse.mean(0)})"""),
        ("md", """**练习 6.3**:哪一题的区分度最高?填下标(0 开始)。"""),
        ("code", """ans_6_3 = None
check("练习 6.3 区分度最高的 PHQ 题", ans_6_3, int(np.argmax(phq_fit["a"])))"""),
        ("md", """## 3. 被试太少时会发生什么

只用 40 个模型重复拟合,比较区分度估计的质量。"""),
        ("code", """from course.irt import simulate_2pl
rows = []
for n in [40, 400]:
    for rep in range(3):
        uu, th, aa, bb = simulate_2pl(n, 30, seed=100 + rep)
        ff = fit_2pl(uu, max_iter=80)
        rows.append({"n_models": n, "corr_a": np.corrcoef(ff["a"], aa)[0, 1], "corr_b": np.corrcoef(ff["b"], bb)[0, 1]})
small = pd.DataFrame(rows).groupby("n_models").mean(); small"""),
        ("md", """**练习 6.4**:在 40 个模型时,难度和区分度哪个估计得更好?填 `"a"` 或 `"b"`。"""),
        ("code", """ans_6_4 = None
check("练习 6.4", ans_6_4, "b" if small.loc[40, "corr_b"] > small.loc[40, "corr_a"] else "a")"""),
    ],
    paper=Paper("Rodriguez P. et al. (2021). Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards? Proceedings of ACL 2021.",
                "这是把 IRT 引入 NLP 排行榜的代表性工作。你有心理测量背景,读起来会很快,重点看他们如何解释题目参数。",
                ["他们用了哪种 IRT 模型?被试和题目分别是什么?", "区分度为负的题通常意味着什么?", "这种方法用在心理健康评测上,你预计会遇到什么困难?"],
                "https://aclanthology.org/2021.acl-long.346"),
    research="把 IRT 用在心理健康评测集上,可以回答“哪些临床情景最能区分模型”。例如:间接表达的风险、老年人特有的担忧,是否是高区分度的题?这需要足够多的模型或模型配置,可以通过同一模型的不同系统提示来扩充被试。",
    interview=[("为什么所有模型都能答对的题应该考虑删掉?", "它的信息量集中在很低的能力区间,对区分当前模型几乎没有贡献,只增加成本。但如果它测的是安全底线,可以保留作为门槛题,不计入排序。"),
               ("只有 8 个模型,能用 2PL 吗?", "不能。区分度估计不稳。可以用 Rasch 或描述性题目分析,或者用公开的多模型逐题数据估计题目参数,再用到自己的模型上。")],
    homework=["notebook 练习 6.1 到 6.4", "在 irt.py 的 fit_2pl 中标注 E 步和 M 步", "学习单论文问题"],
)

# ============================================================== day 7
D7 = Lesson(
    day=7, slug="problem_set_1", title="习题集 1:测量基础", track="review", hours="4 小时",
    goals=["独立完成 6 道综合题,覆盖第 1 到 6 天", "能在 90 秒内讲清每个核心概念", "找出自己最薄弱的两个知识点"],
    slides=[
        ("本周知识地图", TABLE(["天", "核心问题", "关键工具"], [["1", "工具可靠吗、有效吗", "X = T + E,信度,效度证据"], ["2", "评分者一致吗", "κ,加权 κ,AC1,α,ICC"],
              ["3", "结论有多确定", "Wilson 区间,bootstrap,聚类重抽样"], ["4", "两个系统真的不同吗", "配对比较,McNemar,Holm"],
              ["5", "题够不够", "题量公式,MDE,功效模拟"], ["6", "哪些题有用", "2PL,信息量,样本量限制"]])),
        ("本周最常见的五个错误", UL(["只报告原始一致率。", "多轮对话按轮重抽样。", "把两个系统当独立样本比较。",
               "同时检验很多维度而不校正。", "只有几个模型却拟合 2PL。"])),
        ("怎么做习题集", OL(["先不看前几天的 notebook,独立做。", "卡住超过 15 分钟再回去查。", "每题都要写一句“结论”,像在报告里那样写。",
                         "做完后把结论段落朗读录音,检查是否能脱稿讲清。"])),
        ("口述练习", QA("请用一分钟向产品经理解释:为什么 LLM 裁判和咨询师的一致率是 95%,你仍然不建议直接用这个裁判?", "从类别分布、偶然一致、少数类上的一致性、区间宽度四点回答。")
         + QA("你们的评测集有 42 道题,团队报告了 0.1 分的提升。你怎么说?", "用 MDE:42 道题时 MDE 约 0.35 分,0.1 分在可检测范围以下,只能说未发现可靠差异。")),
    ],
    cells=[
        ("md", """# 习题集 1

情境:你所在的团队开发一个面向香港老年人的心理支持聊天机器人。团队有两个候选版本 A 和 B,还有一个 LLM 安全筛查器。下面 6 道题都基于这个情境。每题最后写一句报告式结论。"""),
        ("md", """## 题 1:筛查器一致性

1000 条老年人消息中约 3% 有风险。筛查器与资深社工分别标注。"""),
        ("code", """from course.data import simulate_risk_labels
from course.agreement import percent_agreement, cohen_kappa, gwet_ac1, confusion_table, bootstrap_ci
ps1 = simulate_risk_labels(n=1000, prevalence=0.03, sens_a=0.92, spec_a=0.98, sens_b=0.7, spec_b=0.985, seed=11)
confusion_table(ps1.counselor, ps1.llm, categories=[1, 0])"""),
        ("code", """q1_raw = None        # 原始一致率
q1_kappa = None      # Cohen's kappa
q1_ci = None         # kappa 的 bootstrap 95% 区间 (lo, hi),n_boot=2000, categories=[0,1]
check("题 1a 原始一致率", q1_raw, percent_agreement(ps1.counselor, ps1.llm), tol=1e-6)
check("题 1b kappa", q1_kappa, cohen_kappa(ps1.counselor, ps1.llm), tol=1e-6)
_ref = bootstrap_ci(cohen_kappa, ps1.counselor.values, ps1.llm.values, n_boot=2000, categories=[0, 1])
check("题 1c 区间", q1_ci, _ref[1:], tol=0.03)"""),
        ("md", """*题 1 结论:*"""),
        ("md", """## 题 2:选哪种 ICC

两位社工给 50 条回复打 1 到 5 分,社工乙整体比社工甲低 0.5 分。团队想知道:以后随便请一位社工来评,分数是否可比。应报告哪种 ICC?在 `"ICC(1,1)"`、`"ICC(2,1)"`、`"ICC(3,1)"` 中选。"""),
        ("code", """q2 = None
check("题 2", q2, "ICC(2,1)")"""),
        ("md", """## 题 3:安全题全部通过

B 版本在 12 道风险题上全部通过。计算 Wilson 95% 区间的下限,并写结论。"""),
        ("code", """from course.stats import wilson_ci
q3 = None
check("题 3 Wilson 下限 12/12", q3, wilson_ci(12, 12)[0], tol=0.005)"""),
        ("md", """*题 3 结论:*"""),
        ("md", """## 题 4:五个维度的配对比较"""),
        ("code", """from course.data import simulate_counselor_study, DIMENSIONS
from course.stats import paired_compare, p_adjust
st4 = simulate_counselor_study(seed=21)
h4 = st4["human"]
avg4 = h4[h4.dim != "gate"].groupby(["item", "system", "dim"]).score.mean().unstack("system")
p4 = [paired_compare(avg4.xs(d, level="dim").A.values, avg4.xs(d, level="dim").B.values)["p_paired_t"] for d in DIMENSIONS]
q4 = None   # 用 Holm 校正后 p < 0.05 的维度名列表,例如 ["D1", "D3"]
check("题 4 Holm 后显著的维度", sorted(q4) if q4 else q4, sorted([d for d, p in zip(DIMENSIONS, p_adjust(p4, "holm")) if p < 0.05]))"""),
        ("md", """*题 4 结论:*"""),
        ("md", """## 题 5:要多少道题

团队希望检测 D3 上 0.25 分的差异。用题 4 的数据估计 D3 逐题差值的标准差,α = 0.05,功效 0.8,需要多少道题?"""),
        ("code", """from course.stats import required_items
q5 = None
_s = avg4.xs("D3", level="dim").eval("A - B").std(ddof=1)
check("题 5 题量", q5, required_items(_s, 0.25))"""),
        ("md", """## 题 6:要不要用 2PL

团队只评测了 6 个模型配置,却想用 2PL 挑题。用三句话写出你的建议,并给出替代方案。"""),
        ("md", """*题 6 回答:*"""),
    ],
    interview=[("用一分钟讲清:信度、一致性、区间、配对、功效、IRT 之间是什么关系。", "信度和一致性说明测量工具本身是否可靠;区间说明结论的不确定性;配对和功效决定能否分辨差异;IRT 告诉你哪些题提供信息,从而提高效率。")],
    homework=["习题集 1 全部完成并推送到你的仓库", "每题的结论句", "列出本周最薄弱的两个知识点,下周复习"],
)

LESSONS = [D1, D2, D3, D4, D5, D6, D7]

# ============================================================== solutions (used to verify every exercise)
D1.solutions = {"ans_1_1": "16 / (16 + 16)", "ans_1_2": "np.corrcoef(t1.sum(1), t2.sum(1))[0, 1]",
                "ans_1_3_reliability": '"B"', "ans_1_3_validity": '"A"', "ans_1_4": "0.5 / np.sqrt(0.7 * 0.8)"}
D2.solutions = {"Po": "np.trace(ct) / n", "Pe": "(ct.sum(1) / n) @ (ct.sum(0) / n)", "kappa_by_hand": "(Po - Pe) / (1 - Pe)",
                "ans_2_2": "float(sweep.iloc[-1].raw - sweep.iloc[-1].kappa)", "ans_2_3": "1",
                "ans_2_4": 'bool(icc(d1.values, "ICC(3,1)") > icc(d1.values, "ICC(2,1)"))', "ans_2_5": "hi - lo"}
D3.solutions = {"ans_3_1": "wilson_ci(6, 8)[0]", "ans_3_2": "bootstrap_ci(sysB_D3)[1]", "ans_3_3": "1 + (6 - 1) * 0.4",
                "ans_3_4": "(1 - 3 / (4 * 120 - 9)) * (chatbot.mean() - waitlist.mean()) / np.sqrt((59 * chatbot.var(ddof=1) + 59 * waitlist.var(ddof=1)) / 118)"}
D4.solutions = {"ans_4_1": 'int((p_adjust(res.p.values, "holm") < 0.05).sum())',
                "ans_4_2": '(pc["unpaired_ci"][1] - pc["unpaired_ci"][0]) / (pc["ci_high"] - pc["ci_low"])',
                "ans_4_3": 'mcnemar_exact(g_r["A"], g_r["B"])["p"]'}
D4.replacements = [("    return None   # 在这里实现", """    if ci_low > mid:
        return "improved"
    if ci_high < -mid:
        return "worse"
    if -mid < ci_low and ci_high < mid:
        return "equivalent"
    return "inconclusive\"""")]
D5.solutions = {"ans_5_1": "required_items(1.0, 0.3, power=0.9)", "ans_5_3": 'next((n for n, p in zip(ns, power) if p >= 0.8), "超过 320")',
                "ans_5_4": '"higher" if p_rho8 > power[ns.index(160)] else "lower"'}
D5.replacements = [("    return None   # 用 st.norm.ppf", "    return (st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) * s / np.sqrt(n)")]
D6.solutions = {"ans_6_1": "1.6 ** 2 / 4", "ans_6_2": 'int(item_information(1.0, fit["a"], fit["b"])[0].argmax())',
                "ans_6_3": 'int(np.argmax(phq_fit["a"]))', "ans_6_4": '"b" if small.loc[40, "corr_b"] > small.loc[40, "corr_a"] else "a"'}
D7.solutions = {"q1_raw": "percent_agreement(ps1.counselor, ps1.llm)", "q1_kappa": "cohen_kappa(ps1.counselor, ps1.llm)",
                "q1_ci": "bootstrap_ci(cohen_kappa, ps1.counselor.values, ps1.llm.values, n_boot=2000, categories=[0, 1])[1:]",
                "q2": '"ICC(2,1)"', "q3": "wilson_ci(12, 12)[0]",
                "q4": '[d for d, p in zip(DIMENSIONS, p_adjust(p4, "holm")) if p < 0.05]',
                "q5": 'required_items(avg4.xs("D3", level="dim").eval("A - B").std(ddof=1), 0.25)'}
