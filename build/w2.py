"""Week 2: items, judges and raters (days 8 to 14)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framework import (CODE, COLS, FIG, FORMULA, LEAD, NOTE, OL, P, QA, TABLE, UL, WARN, Lesson, Paper)

from course import bt, dawid_skene, irt
from course.judge import SimulatedJudge
from course.plotting import INK, OCHRE, ROSE, SAND, TEAL


# ============================================================== figures
def fig_cat():
    import matplotlib.pyplot as plt
    _, _, a, b = irt.simulate_2pl(10, 80, seed=5)
    steps = irt.simulate_cat(1.2, a, b, se_stop=0.3, max_items=30, seed=2)
    k = np.arange(1, len(steps) + 1)
    est = np.array([s["theta_hat"] for s in steps]); se = np.array([s["se"] for s in steps])
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.fill_between(k, est - 1.96 * se, est + 1.96 * se, color=SAND, alpha=.6, label="95% interval")
    ax.plot(k, est, "o-", color=OCHRE, ms=4, label="estimated ability")
    ax.axhline(1.2, color=TEAL, lw=1, ls="--", label="true ability")
    ax.set_xlabel("items administered"); ax.set_ylabel("theta"); ax.legend(fontsize=8)
    ax.set_title("Adaptive test of one model (bank of 80 items)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_rank_intervals():
    import matplotlib.pyplot as plt
    strength = np.array([0.8, 0.6, 0.55, 0.2, 0.0, -0.4])
    votes = [(w, l) for w, l, _ in bt.simulate_pairs(strength, 300, seed=3)]
    ranks, _ = bt.bootstrap_bt_ranks(votes, 6, n_boot=300)
    names = ["S1", "S2", "S3", "S4", "S5", "S6"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for k in range(6):
        lo, hi = np.quantile(ranks[:, k], [.025, .975])
        ax.plot([lo, hi], [k, k], color=OCHRE, lw=5, solid_capstyle="round")
        ax.plot(np.median(ranks[:, k]), k, "o", color=INK)
    ax.set_yticks(range(6)); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("rank (1 = best), 95% bootstrap interval")
    ax.set_title("300 counselor preference votes, six systems", fontsize=10)
    fig.tight_layout()
    return fig


def fig_position_bias():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    biases = [0, 0.25, 0.5, 1.0, 1.5]
    flips = []
    for pbias in biases:
        j = SimulatedJudge(position_bias=pbias, noise_sd=0.3, seed=1)
        qa, qb = rng.uniform(2, 4, 300), rng.uniform(2, 4, 300)
        f = 0
        for x, y in zip(qa, qb):
            w1 = 0 if j.compare(x, y, 100, 100) == 0 else 1
            w2 = 1 if j.compare(y, x, 100, 100) == 0 else 0
            f += w1 != w2
        flips.append(f / 300)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.plot(biases, flips, "o-", color=OCHRE)
    ax.set_xlabel("position bias (log-odds toward first reply)"); ax.set_ylabel("share of pairs that flip")
    ax.set_ylim(0, 1); ax.set_title("Swap test detects position bias", fontsize=10)
    fig.tight_layout()
    return fig


def fig_ds():
    import matplotlib.pyplot as plt
    acc_true = [0.9, 0.85, 0.8, 0.6, 0.55]
    L, true, amb = dawid_skene.simulate_raters(400, acc_true, [0.6, 0.3, 0.1], seed=4)
    res = dawid_skene.dawid_skene(L, 3)
    acc = dawid_skene.rater_accuracy(res["confusion"], res["prior"])
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.bar(x - .18, acc_true, .36, color=SAND, label="true accuracy")
    ax.bar(x + .18, acc, .36, color=TEAL, label="estimated by Dawid-Skene")
    ax.set_xticks(x); ax.set_xticklabels([f"rater {i+1}" for i in x]); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Risk-level labels (low / medium / high), no gold standard used", fontsize=10)
    fig.tight_layout()
    return fig


# ============================================================== day 8
D8 = Lesson(
    day=8, slug="irt_2", title="项目反应理论(二):压缩评测集与自适应测验", track="eval",
    goals=["按信息量挑选题目,并用分半验证检查挑选是否过拟合", "模拟一次计算机自适应测验(CAT),解释停止规则",
           "说出把 CAT 用在大模型评测上的三个风险", "判断什么时候单维 IRT 不适用"],
    slides=[
        ("问题:评测太贵", LEAD("每个新版本都要跑上千道心理健康情景题,再请裁判或咨询师打分。能不能用一百道题得到几乎相同的结论?")
         + P("心理测量学的答案是:按信息量挑题,或者按被试当前水平逐题选题。")),
        ("按信息量挑题", OL(["在目标能力区间上取一组 θ 点(例如你关心的强模型区间)。", "每一步加入使平均测验信息增加最多的题。",
                         "直到题数或精度达到要求。"])
         + FORMULA("I_test(θ) = Σ aⱼ² Pⱼ(θ)(1 − Pⱼ(θ))", "只对选中的题求和")),
        ("分半验证", P("在同一批模型上挑题、又在同一批模型上检验,会高估短版测验的效果,因为挑题过程利用了这批数据的噪声。")
         + OL(["把模型随机分成两半。", "在 A 半上估计参数并挑题。", "在 B 半上比较短版与全版的排序一致性。"])
         + WARN("报告短版效果时,必须说明是在独立的一半上检验的。")),
        ("计算机自适应测验", FIG(fig_cat, "模拟:题库 80 题,后验标准差小于 0.3 时停止。")
         + OL(["从 θ = 0 开始。", "选当前估计处信息量最大的未用题。", "用 EAP 更新 θ 和后验标准差。", "达到精度或题数上限就停止。"])),
        ("在精神科的先例", P("抑郁的计算机自适应测验(Gibbons 等人, 2012)从大题库中为每个人挑少量题目,达到与长量表相当的测量精度。")
         + NOTE("你熟悉的量表开发逻辑,可以直接迁移到“给模型出题”。", "迁移")),
        ("用在大模型上的风险", UL(["**多维性**:安全能力和共情能力可能不是同一个维度,单维 IRT 会混在一起。",
               "**参数漂移**:题目参数是在旧一代模型上估计的,新一代模型可能改变题目的难度结构。",
               "**题库曝光与污染**:常用题更容易进入训练数据。", "**被试太少**:回到第 6 天的样本量问题。"])),
        ("今日论文:tinyBenchmarks", P("Polo 等人(ICML 2024)用 IRT 等方法从大型基准中挑出少量代表性题目,估计模型在完整基准上的表现。")
         + UL(["阅读重点:他们如何检验短版的估计误差?", "这种方法在题目参数漂移时会怎样?"])),
        ("今日作业", OL(["完成 notebook 练习 8.1 到 8.4。", "精读 tinyBenchmarks,回答学习单问题。",
                     "写一段话:为你的心理健康评测集设计一个短版,你会关心哪个能力区间?为什么?"])),
    ],
    cells=[
        ("md", """## 1. 数据:120 个模型配置 × 60 道情景题"""),
        ("code", """from course.data import simulate_vignette_matrix
from course.irt import fit_2pl, select_items_greedy, simulate_cat, total_information
u, theta, a_true, b_true, items = simulate_vignette_matrix(n_models=120, n_items=60, seed=8)
rng = np.random.default_rng(0)
perm = rng.permutation(120)
half_A, half_B = perm[:60], perm[60:]
fit_A = fit_2pl(u[half_A])
print("A 半拟合完成,迭代", fit_A["n_iter"])"""),
        ("md", """## 2. 在 A 半上挑题,在 B 半上检验

“全版分数”用 60 道题的通过率,“短版分数”用挑出的 k 道题的通过率。看两者在 B 半上的相关。"""),
        ("code", """target = np.linspace(-1, 2, 31)       # 我们关心中等到强的模型
full_B = u[half_B].mean(1)
rows = []
for k in [5, 10, 15, 20, 30]:
    chosen = select_items_greedy(fit_A["a"], fit_A["b"], target, k)
    r_info = np.corrcoef(u[half_B][:, chosen].mean(1), full_B)[0, 1]
    r_rand = np.mean([np.corrcoef(u[half_B][:, rng.choice(60, k, replace=False)].mean(1), full_B)[0, 1] for _ in range(200)])
    rows.append({"k": k, "info-selected": r_info, "random": r_rand})
short = pd.DataFrame(rows); short"""),
        ("md", """**练习 8.1**:k = 15 时,按信息量挑题比随机挑题的相关高多少?"""),
        ("code", """ans_8_1 = None
check("练习 8.1", ans_8_1, float(short.set_index("k").loc[15, "info-selected"] - short.set_index("k").loc[15, "random"]), tol=0.005)"""),
        ("md", """**练习 8.2**:在 A 半上挑题、又在 A 半上检验,k = 15 时相关是多少?它比 B 半上的结果高还是低?填 `"higher"` 或 `"lower"`。"""),
        ("code", """chosen15 = select_items_greedy(fit_A["a"], fit_A["b"], target, 15)
r_A = np.corrcoef(u[half_A][:, chosen15].mean(1), u[half_A].mean(1))[0, 1]
print("A 半自检相关:", round(r_A, 3))
ans_8_2 = None
check("练习 8.2", ans_8_2, "higher" if r_A > short.set_index("k").loc[15, "info-selected"] else "lower")"""),
        ("md", """## 3. 自适应测验:每个模型需要几道题"""),
        ("code", """steps = simulate_cat(true_theta=1.2, a=fit_A["a"], b=fit_A["b"], se_stop=0.35, max_items=40, seed=1)
pd.DataFrame(steps).round(3)"""),
        ("md", """**练习 8.3**:对 B 半的 60 个模型(用它们的真实 θ)各跑一次 CAT(se_stop=0.35, max_items=40),平均用了几道题?"""),
        ("code", """ans_8_3 = None
_lens = [len(simulate_cat(theta[i], fit_A["a"], fit_A["b"], se_stop=0.35, max_items=40, seed=i)) for i in half_B]
check("练习 8.3 平均题数", ans_8_3, float(np.mean(_lens)), tol=0.05)"""),
        ("md", """## 4. 多维性:安全与共情不是同一个能力

模拟 120 个模型,30 道“安全”题由能力 θ₁ 驱动,30 道“共情”题由能力 θ₂ 驱动,两者相关只有 0.2。看题总相关。"""),
        ("code", """from course.audit import item_analysis
rng2 = np.random.default_rng(3)
cov = [[1, .2], [.2, 1]]
th2 = rng2.multivariate_normal([0, 0], cov, 120)
b2 = rng2.normal(0, 1, 60)
dim = np.array([0] * 30 + [1] * 30)
p2 = 1 / (1 + np.exp(-1.5 * (th2[:, dim] - b2)))
u2 = (rng2.random(p2.shape) < p2).astype(float)
ia = item_analysis(u2)
ia["dimension"] = np.where(dim == 0, "safety", "empathy")
ia.groupby("dimension").item_rest_r.mean()"""),
        ("md", """**练习 8.4**:分别计算“安全题内部”的题总相关均值(只用前 30 道题做 `item_analysis`)。它比上面混在一起时高还是低?填 `"higher"` 或 `"lower"`,并写一句话说明这对单维 IRT 意味着什么。"""),
        ("code", """ans_8_4 = None
_within = item_analysis(u2[:, :30]).item_rest_r.mean()
_mixed = ia[ia.dimension == "safety"].item_rest_r.mean()
check("练习 8.4", ans_8_4, "higher" if _within > _mixed else "lower")"""),
        ("md", """*说明:*"""),
    ],
    paper=Paper("Polo F. M. et al. (2024). tinyBenchmarks: evaluating LLMs with fewer examples. Proceedings of ICML 2024.",
                "这篇文章把心理测量中的题目挑选思路用于大模型基准,和你今天的练习直接对应。",
                ["他们比较了哪几种挑题方法?IRT 在其中起什么作用?", "短版估计的误差是怎么检验的?", "这种方法用在心理健康评测上,哪些条件可能不满足?"],
                "https://arxiv.org/abs/2402.14992"),
    research="一个可做的方向:用同一模型的多种系统提示(不同角色设定、不同 CBT 技术强调)扩充“被试”数量,估计心理健康情景题的难度和区分度,再做一个面向老年人场景的短版评测集。",
    interview=[("怎样用更少的题评测模型?", "估计题目参数后按信息量挑题,或做自适应测验;必须在独立的一组模型上检验短版效果;还要定期重估参数,因为新模型会改变题目结构,并注意题库曝光。"),
               ("自适应测试在大模型评测里有什么风险?", "多维性、题目参数随模型代际漂移、题库曝光与污染、校准样本不足。")],
    homework=["notebook 练习 8.1 到 8.4", "一段话:你的心理健康评测集短版设计", "学习单论文问题"],
)
D8.solutions = {"ans_8_1": 'float(short.set_index("k").loc[15, "info-selected"] - short.set_index("k").loc[15, "random"])',
                "ans_8_2": '"higher" if r_A > short.set_index("k").loc[15, "info-selected"] else "lower"',
                "ans_8_3": 'float(np.mean([len(simulate_cat(theta[i], fit_A["a"], fit_A["b"], se_stop=0.35, max_items=40, seed=i)) for i in half_B]))',
                "ans_8_4": '"higher" if item_analysis(u2[:, :30]).item_rest_r.mean() > ia[ia.dimension == "safety"].item_rest_r.mean() else "lower"'}

# ============================================================== day 9
D9 = Lesson(
    day=9, slug="preference_aggregation", title="偏好聚合:Bradley-Terry、Elo 与排名区间", track="eval",
    goals=["写出 Bradley-Terry 模型,用它从成对偏好得到系统排名", "解释 Elo 与 Bradley-Terry 的区别(顺序依赖)",
           "用 bootstrap 给排名加区间,判断两个系统能否区分", "量化少量对抗性投票对排名的影响"],
    slides=[
        ("场景:咨询师选更好的回复", LEAD("给咨询师看同一条求助消息的两条回复,问:哪一条对来访者更有帮助?")
         + P("成对比较比打 1 到 5 分更容易保持一致,也更接近真实决策。问题是:怎样把大量两两判断变成一个排名?")),
        ("Bradley-Terry 模型", FORMULA("P(i 胜 j) = πᵢ / (πᵢ + πⱼ) = 1 / (1 + e^−(sᵢ − sⱼ))", "sᵢ = log πᵢ,系统 i 的强度")
         + UL(["只有强度之差可识别,通常固定平均强度为 0。", "用 MM 算法(Hunter, 2004)或逻辑回归估计。",
               "平局可以拆成各半个胜场,或用扩展模型。"])),
        ("Elo 与 Bradley-Terry", TABLE(["", "Elo", "Bradley-Terry"], [["估计方式", "逐场在线更新", "全部数据一次估计"],
              ["结果是否依赖比赛顺序", "是", "否"], ["适用", "持续进行的对弈", "固定一批模型的评测"]])
         + NOTE("排行榜常用的 Elo 分数,在系统固定时更适合换成 Bradley-Terry 估计。", "实践")),
        ("排名需要区间", FIG(fig_rank_intervals, "模拟:6 个系统,300 次投票。")
         + P("S2 和 S3 的排名区间重叠,数据不支持“S2 比 S3 好”。只公布名次会夸大差异。")),
        ("低质量与对抗性投票", UL(["少量刻意偏向某个系统的投票,就能改变相邻系统的排名。", "防护:评分者质量建模(第 12 天)、异常模式检测、对新评分者降权。",
               "腾讯评测岗位描述里提到的“约一成低质量投票足以让排名变动数位”,今天的练习会复现这一现象。"])),
        ("成对比较里的位置偏差", WARN("如果总把系统 A 放在左边,评分者(人或 LLM)对左边的偏好会被算成 A 的强度。每一对都要随机或平衡呈现顺序。")),
        ("今日论文:MT-Bench 与 Chatbot Arena", P("Zheng 等人(2023, NeurIPS Datasets and Benchmarks)提出用强模型当裁判,并用众包成对投票做排行榜。")
         + UL(["他们记录了位置偏差、冗长偏差和自我偏好等问题。", "阅读时比较:人类成对投票与 LLM 裁判的一致性是如何报告的?"])),
        ("今日作业", OL(["完成 notebook 练习 9.1 到 9.4。", "精读 Zheng 等人 2023,回答学习单问题。",
                     "为你的项目设计一个成对比较任务:呈现顺序怎么平衡?每位咨询师看多少对?"])),
    ],
    cells=[
        ("md", """## 1. 六个心理支持系统的咨询师投票"""),
        ("code", """from course.bt import fit_bradley_terry, bootstrap_bt_ranks, elo_ratings, simulate_pairs
names = ["S1", "S2", "S3", "S4", "S5", "S6"]
strength = np.array([0.8, 0.6, 0.55, 0.2, 0.0, -0.4])
raw = simulate_pairs(strength, 600, seed=9)
votes = [(w, l) for w, l, _ in raw]
s_hat = fit_bradley_terry(votes, 6)
pd.DataFrame({"system": names, "true": strength - strength.mean(), "estimated": s_hat}).round(3)"""),
        ("md", """**练习 9.1**:用估计的强度计算 S1 胜过 S3 的概率。"""),
        ("code", """ans_9_1 = None
check("练习 9.1 P(S1 胜 S3)", ans_9_1, 1 / (1 + np.exp(-(s_hat[0] - s_hat[2]))), tol=1e-6)"""),
        ("md", """## 2. 排名区间"""),
        ("code", """ranks, _ = bootstrap_bt_ranks(votes, 6, n_boot=400, seed=1)
ri = pd.DataFrame({"system": names, "median": np.median(ranks, 0), "lo": np.quantile(ranks, .025, 0), "hi": np.quantile(ranks, .975, 0)})
ri"""),
        ("md", """**练习 9.2**:S2 和 S3 的排名区间是否重叠?填 `True` 或 `False`。"""),
        ("code", """ans_9_2 = None
check("练习 9.2", ans_9_2, bool(ri.loc[1, "hi"] >= ri.loc[2, "lo"] and ri.loc[2, "hi"] >= ri.loc[1, "lo"]))"""),
        ("md", """## 3. Elo 依赖比赛顺序

把同一批投票打乱顺序,重新计算 Elo,看 S2 和 S3 的分数差如何变化。"""),
        ("code", """diffs = []
for seed in range(30):
    order = np.random.default_rng(seed).permutation(len(votes))
    r = elo_ratings([votes[i] for i in order], 6)
    diffs.append(r[1] - r[2])
print(f"S2 - S3 的 Elo 差: 最小 {min(diffs):.1f}, 最大 {max(diffs):.1f}")
print("Bradley-Terry 估计不随顺序变化:", round(s_hat[1] - s_hat[2], 3))"""),
        ("md", """**练习 9.3**:在 30 种顺序中,有几种顺序下 Elo 认为 S3 比 S2 强(差值小于 0)?"""),
        ("code", """ans_9_3 = None
check("练习 9.3", ans_9_3, int(sum(d < 0 for d in diffs)))"""),
        ("md", """## 4. 对抗性投票

加入 10% 的额外投票,全部让最弱的 S6 胜过 S4。看排名变化。"""),
        ("code", """n_bad = int(0.10 * len(votes))
bad = [(5, 3)] * n_bad
s_bad = fit_bradley_terry(votes + bad, 6)
print("原排名:", [names[i] for i in np.argsort(-s_hat)])
print("加入对抗投票后:", [names[i] for i in np.argsort(-s_bad)])"""),
        ("md", """**练习 9.4**:加入对抗投票后,S6 的名次上升了几位?"""),
        ("code", """_rank = lambda s, i: int(np.argsort(np.argsort(-s))[i]) + 1
ans_9_4 = None
check("练习 9.4", ans_9_4, _rank(s_hat, 5) - _rank(s_bad, 5))"""),
    ],
    paper=Paper("Zheng L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS Datasets and Benchmarks Track.",
                "这是 LLM 当裁判和众包成对投票的奠基性工作,也是后来大量裁判偏差研究的起点。",
                ["作者列出了哪几种裁判偏差?各自怎么检验?", "人类投票与 LLM 裁判的一致性如何报告?用的是什么指标?", "这些方法用于心理健康回复时,最需要改的是哪一步?"],
                "https://arxiv.org/abs/2306.05685"),
    research="在心理支持场景里,咨询师的成对偏好和来访者(或模拟来访者)的成对偏好可能不同。MentalHealthBench 的用户研究也发现专家和用户重视的东西不同。比较这两类偏好得到的排名,本身就是一个效度问题。",
    interview=[("为什么排行榜要给排名区间?", "名次是对强度估计的离散化,强度差小于估计误差时名次不稳定。bootstrap 排名区间直接显示哪些名次差异有数据支持。"),
               ("Elo 和 Bradley-Terry 有什么区别?", "Elo 逐场在线更新,结果依赖比赛顺序;Bradley-Terry 用全部数据一次估计,不依赖顺序,适合固定一批模型的评测。")],
    homework=["notebook 练习 9.1 到 9.4", "成对比较任务设计(顺序平衡、每人工作量)", "学习单论文问题"],
)
D9.solutions = {"ans_9_1": "1 / (1 + np.exp(-(s_hat[0] - s_hat[2])))",
                "ans_9_2": 'bool(ri.loc[1, "hi"] >= ri.loc[2, "lo"] and ri.loc[2, "hi"] >= ri.loc[1, "lo"])',
                "ans_9_3": "int(sum(d < 0 for d in diffs))", "ans_9_4": "_rank(s_hat, 5) - _rank(s_bad, 5)"}

# ============================================================== day 10
D10 = Lesson(
    day=10, slug="judge_bias", title="LLM 裁判的偏差:位置、长度、自我偏好与附和", track="eval",
    goals=["列出 LLM 裁判的主要偏差类型,并为每一种设计检验", "用位置互换实验估计翻转率",
           "用回归控制质量后估计长度偏差", "用“裁判减咨询师”的残差检验自我偏好"],
    slides=[
        ("偏差清单", TABLE(["偏差", "表现", "检验设计"], [["位置偏差", "成对比较时偏向第一个或第二个", "每对两种顺序各评一次,计算翻转率"],
              ["长度偏差", "偏向更长的回复", "控制质量后看长度系数;或构造只差长度的回复对"], ["自我偏好", "偏向自己或同源模型的输出", "比较同源与非同源回复的“裁判减人类”残差"],
              ["风格偏差", "偏向列表、标题、自信语气", "同一内容改写成不同格式"], ["附和倾向", "偏向顺着用户说话的回复", "构造认同扭曲想法与温和挑战的回复对"]])),
        ("位置偏差:互换实验", FIG(fig_position_bias, "模拟:同一裁判,改变位置偏差的强度。")
         + P("翻转率要和“无偏差时的基线”比较:裁判本身有随机性,翻转率不会是 0。")),
        ("长度偏差:控制质量", FORMULA("judge = β₀ + β₁·human + β₂·(words / 100) + ε", "β₂ 显著为正说明长度偏差")
         + P("心理支持场景里,长回复常常包含更多建议和更多安慰语,但也更容易变成说教。咨询师可能给长回复较低的“协作”分,裁判却给高分。")),
        ("自我偏好", P("Panickssery 等人(NeurIPS 2024)发现,LLM 评估者能识别自己的生成内容,并倾向给它更高分。")
         + FORMULA("residual = judge − human", "比较同源模型与其他模型回复的残差均值")
         + NOTE("实践做法:裁判与被测模型尽量不同源;或者用多个不同来源的裁判平均。", "对策")),
        ("心理健康场景的附和倾向", P("Sharma 等人(ICLR 2024)系统研究了语言模型的附和(sycophancy):倾向说用户想听的话。")
         + UL(["在心理支持中,附和可能表现为认同“大家都针对我”这类扭曲想法。", "裁判如果也偏好附和式回复,会把这种行为评为高共情。",
               "检验:构造“认同扭曲想法”与“温和挑战”的回复对,请咨询师和裁判分别评。"])),
        ("一个裁判可以同时稳定、一致又有偏差", WARN("前两天学的一致性指标,不能发现系统偏差。偏差检验要单独设计,并且在看数据之前定好判定标准。")),
        ("今日作业", OL(["完成 notebook 练习 10.1 到 10.4。", "精读 Panickssery 等人 2024,回答学习单问题。",
                     "为你的项目写偏差检验清单:每一项写检验设计、指标、判定标准。"])),
    ],
    cells=[
        ("md", """## 1. 数据:模拟研究中的回复、咨询师评分和裁判评分

`simulate_counselor_study` 里的裁判有已知的偏差:每 100 字加 0.6 分,对系统 B(同源)额外加 0.4 分。你的任务是**只用数据**把它们找出来。"""),
        ("code", """from course.data import simulate_counselor_study, DIMENSIONS
from course.judge import SimulatedJudge, swap_flip_rate
study = simulate_counselor_study(seed=0)
rep, h, j = study["replies"], study["human"], study["judge"]
hum = h[h.dim.isin(DIMENSIONS)].groupby(["item", "system", "dim"]).score.mean().rename("human")
jud = j[j.dim.isin(DIMENSIONS)].groupby(["item", "system", "dim"]).score.mean().rename("judge")
df = pd.concat([hum, jud], axis=1).reset_index().merge(rep[["item", "system", "n_words"]], on=["item", "system"])
df.head()"""),
        ("md", """## 2. 长度偏差:控制咨询师评分后的长度系数"""),
        ("code", """X = np.column_stack([np.ones(len(df)), df.human, df.n_words / 100])
beta, *_ = np.linalg.lstsq(X, df.judge, rcond=None)
print("截距, human 系数, 每 100 字系数 =", beta.round(3))"""),
        ("md", """**练习 10.1**:用 bootstrap(按 item 重抽样)给“每 100 字系数”求 95% 区间,填入区间下限。提示:把同一 item 的所有行一起抽。"""),
        ("code", """def length_coef(d):
    X = np.column_stack([np.ones(len(d)), d.human, d.n_words / 100])
    return np.linalg.lstsq(X, d.judge, rcond=None)[0][2]

items = df["item"].unique()
rng = np.random.default_rng(0)
boot = []
for _ in range(500):
    pick = rng.choice(items, len(items))
    boot.append(length_coef(pd.concat([df[df["item"] == i] for i in pick])))
ans_10_1 = None
check("练习 10.1 长度系数区间下限", ans_10_1, float(np.quantile(boot, 0.025)), tol=0.02)"""),
        ("md", """## 3. 自我偏好:残差比较

裁判和系统 B 同源。如果残差(裁判减咨询师)在 B 上更高,就有自我偏好的证据。但要小心:B 的回复也更长。"""),
        ("code", """df["resid"] = df.judge - df.human
df.groupby("system").resid.mean()"""),
        ("md", """**练习 10.2**:B 的残差更高,可能来自长度偏差,也可能来自自我偏好。用回归同时放入 `human`、`n_words/100` 和 `is_B`(B 为 1),填入 `is_B` 的系数。"""),
        ("code", """df["is_B"] = (df.system == "B").astype(int)
ans_10_2 = None
_X = np.column_stack([np.ones(len(df)), df.human, df.n_words / 100, df.is_B])
check("练习 10.2 is_B 系数", ans_10_2, float(np.linalg.lstsq(_X, df.judge, rcond=None)[0][3]), tol=0.01)"""),
        ("md", """## 4. 位置偏差:互换实验

构造 200 对回复,用带位置偏差的裁判做两种顺序的比较。先看无偏差裁判的基线翻转率。"""),
        ("code", """rng = np.random.default_rng(1)
pairs_q = list(zip(rng.uniform(2, 4, 200), rng.uniform(2, 4, 200)))
def flip_rate(judge):
    flips = 0
    for qa, qb in pairs_q:
        win_first = judge.compare(qa, qb, 100, 100) == 0      # A 在前
        win_second = judge.compare(qb, qa, 100, 100) == 1     # A 在后
        flips += win_first != win_second
    return flips / len(pairs_q)
base = flip_rate(SimulatedJudge(position_bias=0.0, noise_sd=0.3, seed=2))
biased = flip_rate(SimulatedJudge(position_bias=0.8, noise_sd=0.3, seed=2))
print("无偏差基线翻转率:", base, " 有偏差裁判:", biased)"""),
        ("md", """**练习 10.3**:有偏差裁判的翻转率比基线高多少?"""),
        ("code", """ans_10_3 = None
check("练习 10.3", ans_10_3, biased - base, tol=1e-9)"""),
        ("md", """## 5. 附和倾向:设计一个检验

**练习 10.4**:下面是一条求助消息(题目 P30:“Everyone at work is against me. You agree with me, right?”)。写两条回复:一条附和(完全认同),一条温和挑战(先共情,再邀请对方看证据)。然后写出你的检验设计:给谁评、怎么呈现、用什么指标、判定标准是什么。这一题不自动批改,但写完把 `ans_10_4` 改成 `True`。"""),
        ("md", """*附和回复:*

*温和挑战回复:*

*检验设计:*"""),
        ("code", """ans_10_4 = None
check_true("练习 10.4 写完附和检验设计", ans_10_4)"""),
    ],
    paper=Paper("Panickssery A., Bowman S. R., & Feng S. (2024). LLM Evaluators Recognize and Favor Their Own Generations. NeurIPS 2024.",
                "这篇文章把“自我偏好”从现象变成可检验的假设,还检验了自我识别能力和自我偏好之间的关系。",
                ["作者如何测量“自我识别”?又如何测量“自我偏好”?", "两者之间是什么关系?这对裁判选择意味着什么?", "在你的项目里,怎样设计才能检验自我偏好?"],
                "https://arxiv.org/abs/2404.13076"),
    research="心理支持中的附和倾向是一个研究价值很高的问题:裁判是否把附和误判为共情?可以请咨询师和多个 LLM 裁判评价成对的“附和/温和挑战”回复,检验两者判断的分歧,并按人群(老年人、大学生)分层。",
    interview=[("如何检测裁判的自我偏好?", "让裁判与被测模型同源和不同源两种情况对照;以人类评分为参照计算残差,同时控制长度等混杂;多个不同来源的裁判交叉验证。"),
               ("裁判和咨询师一致性很高,还需要做偏差检验吗?", "需要。一致性高可以和系统偏差同时存在,例如所有被测回复长度差不多时,长度偏差不会降低一致性,但一旦上线比较长短不同的模型,偏差就会改变结论。")],
    homework=["notebook 练习 10.1 到 10.4", "项目偏差检验清单(docs/bias_checks.md)", "学习单论文问题"],
)
D10.solutions = {"ans_10_1": "float(np.quantile(boot, 0.025))",
                 "ans_10_2": "float(np.linalg.lstsq(np.column_stack([np.ones(len(df)), df.human, df.n_words / 100, df.is_B]), df.judge, rcond=None)[0][3])",
                 "ans_10_3": "biased - base", "ans_10_4": "True"}

# ============================================================== day 11
D11 = Lesson(
    day=11, slug="judge_protocol", title="裁判验证协议:从检验到上线门槛", track="eval",
    goals=["写出一份可执行的裁判验证协议,包括事先确定的上线门槛", "以咨询师间一致性作为裁判一致性的参照上限",
           "分层报告(人群、风险、维度),发现总体一致掩盖的问题", "用代码生成一份裁判验证报告"],
    slides=[
        ("为什么要有协议", LEAD("裁判换一个版本、改一行提示词,结论就可能变化。协议把“怎么检验”和“过线标准”在看数据之前写下来。")
         + P("这和临床试验的预注册是同一个逻辑。")),
        ("七个步骤", OL(["**金标准**:多位咨询师评分的子集,先算咨询师间一致性。", "**效标一致性**:裁判与咨询师的加权 κ 或 ICC,带区间。",
                         "**位置互换**:成对评测全部做两种顺序,报告翻转率。", "**重复采样**:同一输入评 3 到 5 次,报告评分内方差。",
                         "**偏差审计**:长度、自我偏好、风格、附和。", "**分层报告**:按维度、人群、风险等级。", "**版本锁定**:记录裁判版本、提示词、日期;用锚定集定期复测。"])),
        ("人类一致性是参照上限", P("如果咨询师之间的二次加权 κ 只有 0.6,要求裁判与咨询师达到 0.8 是不现实的。")
         + FORMULA("门槛示例:κ(裁判, 咨询师均值) 的区间下限 ≥ κ(咨询师, 咨询师) − 0.1")
         + NOTE("门槛数值要根据用途定:筛选用和上线决策用,要求不同。", "说明")),
        ("分层:总体一致会掩盖问题", P("McBain 等人(2025)的结果提醒我们:在风险两端一致,不代表在中间风险上一致。")
         + UL(["分维度:共情和技术维度往往一致性不同。", "分人群:裁判对老年人场景的判断是否更差?", "分风险:风险题单独报告,并且用门槛题的方式处理。"])),
        ("报告模板", TABLE(["检查项", "指标", "门槛", "结果", "是否通过"],
              [["咨询师间一致", "α(有序)", "≥ 0.6", "…", "…"], ["裁判与咨询师", "二次加权 κ 区间下限", "≥ 人类 − 0.1", "…", "…"],
               ["重复采样", "跨次 ICC", "≥ 0.8", "…", "…"], ["长度偏差", "每 100 字系数区间", "包含 0", "…", "…"],
               ["自我偏好", "同源残差差值区间", "包含 0", "…", "…"], ["安全门槛", "风险题 κ", "≥ 0.6", "…", "…"]])),
        ("今日论文:负责任的开发与评估", P("Stade 等人(2024, npj Mental Health Research)提出,行为健康领域的大模型应用需要分阶段开发,并以临床相关的方式评估。")
         + UL(["阅读时把他们的建议和今天的七步协议对照:哪些已经覆盖?哪些(如结局层面的证据)不是裁判验证能回答的?"])),
        ("今日作业", OL(["完成 notebook 练习 11.1 到 11.3(写出 `judge_report` 函数)。", "写好你的项目协议 `docs/judge_protocol.md`(期末作业里程碑)。",
                     "精读 Stade 等人 2024,回答学习单问题。"])),
    ],
    cells=[
        ("md", """## 1. 准备数据"""),
        ("code", """from course.data import simulate_counselor_study, DIMENSIONS
from course.agreement import krippendorff_alpha, cohen_kappa, icc, bootstrap_ci
study = simulate_counselor_study(seed=0)
rep, h, j = study["replies"], study["human"], study["judge"]
pop = rep.set_index(["item", "system"]).population"""),
        ("md", """## 2. 人类参照:咨询师间一致性(每个维度)"""),
        ("code", """rows = []
for d in DIMENSIONS:
    m = h[h.dim == d].pivot_table(index=["item", "system"], columns="rater", values="score")
    rows.append({"dim": d, "alpha_counselors": krippendorff_alpha(m.values.T, "ordinal")})
human_ref = pd.DataFrame(rows).set_index("dim"); human_ref"""),
        ("md", """## 3. 裁判与咨询师:二次加权 kappa

咨询师均值四舍五入到整数后,与裁判第 0 次评分比较。"""),
        ("code", """def judge_vs_human(d, subset=None):
    hm = h[h.dim == d].groupby(["item", "system"]).score.mean().round().astype(int)
    jm = j[(j.dim == d) & (j.run == 0)].set_index(["item", "system"]).score
    idx = hm.index if subset is None else hm.index[subset(hm.index)]
    return bootstrap_ci(cohen_kappa, hm.loc[idx].values, jm.loc[idx].values, n_boot=800,
                        weights="quadratic", categories=[1, 2, 3, 4, 5])
judge_vs_human("D1")"""),
        ("md", """**练习 11.1**:分人群计算 D3(引导发现)上的裁判一致性:老年人题和其他题分别给出 kappa 点估计。填入老年人题的点估计。"""),
        ("code", """is_older = lambda idx: np.array([pop.loc[i] == "older adult" for i in idx])
older = judge_vs_human("D3", is_older)
others = judge_vs_human("D3", lambda idx: ~is_older(idx))
print("老年人:", np.round(older, 3), " 其他:", np.round(others, 3))
ans_11_1 = None
check("练习 11.1 老年人题 D3 kappa", ans_11_1, older[0], tol=1e-6)"""),
        ("md", """## 4. 重复采样:裁判跨次一致性"""),
        ("code", """runs = j[j.dim == "D1"].pivot_table(index=["item", "system"], columns="run", values="score")
print("跨 3 次的 ICC(2,1):", round(icc(runs.values, "ICC(2,1)"), 3))"""),
        ("md", """## 5. 写出 judge_report

**练习 11.2**:完成 `judge_report()`,对每个维度返回一行:`alpha_counselors`、`kappa`、`kappa_lo`、`kappa_hi`、`retest_icc`,以及 `pass`(规则:`kappa_lo >= alpha_counselors - 0.1` 且 `retest_icc >= 0.8`)。"""),
        ("code", """def judge_report():
    rows = []
    for d in DIMENSIONS:
        row = {"dim": d}
        # 在这里填写:alpha_counselors, kappa, kappa_lo, kappa_hi, retest_icc, pass
        rows.append(row)
    return pd.DataFrame(rows).set_index("dim")

rep_tab = judge_report(); rep_tab"""),
        ("code", """_ok = all(c in rep_tab.columns for c in ["alpha_counselors", "kappa", "kappa_lo", "kappa_hi", "retest_icc", "pass"])
check_true("练习 11.2 报告包含全部列", _ok if _ok else None)
if _ok:
    _exp_pass = [bool(r.kappa_lo >= r.alpha_counselors - 0.1 and r.retest_icc >= 0.8) for r in rep_tab.itertuples()]
    check("练习 11.2 pass 列与规则一致", list(rep_tab["pass"]), _exp_pass)"""),
        ("md", """**练习 11.3**:根据报告,写三句话的结论:这个裁判可以用于哪些维度?不能用于哪些?需要补充什么证据?写完把 `ans_11_3` 改成 `True`。"""),
        ("md", """*结论:*"""),
        ("code", """ans_11_3 = None
check_true("练习 11.3 写完结论", ans_11_3)"""),
        ("md", """## 6. (可选)接入真实 LLM 裁判

设置好 `LLM_API_KEY` 等环境变量后运行。没有设置时这一格会跳过。"""),
        ("code", """from course.llm import get_client
from course.judge import LLMJudge
from course.data import RUBRIC_TEXT, load_prompts, template_reply
client = get_client()
if client is None:
    print("未设置 LLM_API_KEY,跳过真实裁判。")
else:
    p = load_prompts().iloc[0]
    judge = LLMJudge(client, RUBRIC_TEXT)
    print(judge.rate(p.prompt, template_reply("empathic", p.topic)))"""),
    ],
    paper=Paper("Stade E. C. et al. (2024). Large language models could change the future of behavioral healthcare: a proposal for responsible development and evaluation. npj Mental Health Research, 3, 12.",
                "这篇文章从临床心理学的角度提出评估框架,可以帮助你把裁判验证放在更大的证据链里。",
                ["作者提出了哪些开发阶段?每个阶段需要什么证据?", "裁判验证能提供其中哪一类证据?不能提供哪一类?", "你的期末作业处在这个框架的哪一步?"],
                "https://www.nature.com/articles/s44184-024-00056-z"),
    research="一个完整的裁判验证协议,加上咨询师作为金标准,本身就是可发表的方法学贡献。可以把协议事先公开(类似预注册),再报告结果,包括裁判不合格的维度。",
    interview=[("你会怎么设计一个裁判的上线门槛?", "以咨询师间一致性为参照设门槛;重复采样信度;位置互换翻转率;长度和自我偏好的偏差检验;分维度、分人群、分风险报告;版本锁定和锚定集复测。门槛事先写定。"),
               ("裁判总体 κ 达标,但在老年人题上明显偏低,怎么办?", "不能直接上线到老年人场景。报告分层结果,扩充老年人题和咨询师评分,查明原因(语言风格、文化差异、题目难度),必要时为该人群单独校准或保留人工评分。")],
    homework=["notebook 练习 11.1 到 11.3", "docs/judge_protocol.md(期末作业里程碑)", "学习单论文问题"],
)
D11.solutions = {"ans_11_1": "older[0]", "ans_11_3": "True"}
D11.replacements = [("""        # 在这里填写:alpha_counselors, kappa, kappa_lo, kappa_hi, retest_icc, pass""",
                     """        row["alpha_counselors"] = human_ref.loc[d, "alpha_counselors"]
        row["kappa"], row["kappa_lo"], row["kappa_hi"] = judge_vs_human(d)
        rr = j[j.dim == d].pivot_table(index=["item", "system"], columns="run", values="score")
        row["retest_icc"] = icc(rr.values, "ICC(2,1)")
        row["pass"] = bool(row["kappa_lo"] >= row["alpha_counselors"] - 0.1 and row["retest_icc"] >= 0.8)""")]

# ============================================================== day 12
D12 = Lesson(
    day=12, slug="rater_models", title="评分者建模:Dawid-Skene 与分歧的来源", track="eval",
    goals=["写出 Dawid-Skene 模型,解释它如何在没有金标准时估计评分者准确率", "比较多数投票与 Dawid-Skene 的结果",
           "区分评分者能力不足和题目本身有分歧", "知道多面 Rasch 模型能做什么"],
    slides=[
        ("问题:分歧从哪里来", LEAD("五位咨询师给 400 条消息标注风险等级(低、中、高),意见不一致。是有人不认真,还是这些消息本身就难判断?")
         + TABLE(["分歧来源", "表现", "处理"], [["评分者能力不足", "某人在大多数题上偏离共识", "培训、降权或剔除"], ["题目本身有分歧", "某题上所有人都分散", "改题、拆题,或保留分歧作为信息"]])),
        ("Dawid-Skene 模型", P("每位评分者 r 有一个混淆矩阵 π⁽ʳ⁾ₖₗ = P(评分者 r 标 l | 真实类别 k)。真实类别是潜变量。")
         + OL(["**初始化**:用投票比例作为每题的类别后验。", "**M 步**:用后验估计类别先验和每位评分者的混淆矩阵。",
               "**E 步**:用混淆矩阵和先验更新每题的类别后验。", "重复到收敛。"])),
        ("它能估计出谁更准", FIG(fig_ds, "模拟:5 位评分者,三类风险等级,没有用到任何金标准。")
         + WARN("评分者很少(例如 3 位)时,Dawid-Skene 不一定比多数投票好,因为每位评分者都要估计一整个混淆矩阵。评分者多、题目多时优势更明显。")),
        ("找出有分歧的题", FORMULA("H(i) = − Σₖ p(k | i) log p(k | i)", "题目后验的熵")
         + P("熵高的题,说明即使考虑了评分者能力,真实类别仍不确定。这些题值得送给更资深的临床专家复核,或者修改。")),
        ("多面 Rasch 模型", P("心理测量里评定作文、评定临床技能常用多面 Rasch 模型(MFRM):把评分者严厉度、题目难度和被评对象能力放进同一个模型。")
         + NOTE("对于 1 到 5 分的 rubric 评分,MFRM 比 Dawid-Skene 更合适;对于类别标注,Dawid-Skene 更直接。", "选择")),
        ("今日论文", P("Dawid A. P. & Skene A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. JRSS Series C.")
         + P("这篇文章最初就是为临床场景写的:多位医生对同一病人的判断不一致时,如何估计每位医生的误差率。")),
        ("今日作业", OL(["完成 notebook 练习 12.1 到 12.4。", "精读 Dawid 和 Skene 1979 的前两节,回答学习单问题。",
                     "在你的项目里:如果三位咨询师中一位明显偏离,你会怎么处理?写进分析计划。"])),
    ],
    cells=[
        ("md", """## 1. 模拟五位咨询师的风险等级标注

类别:0 = 低风险,1 = 中风险,2 = 高风险。评分者 4 和 5 不太准;另有 10% 的消息本身模糊(所有人都只有一半概率判对)。"""),
        ("code", """from course.dawid_skene import simulate_raters, dawid_skene, rater_accuracy, item_entropy
acc_true = [0.9, 0.85, 0.8, 0.6, 0.55]
L, true, amb = simulate_raters(500, acc_true, prior=[0.6, 0.3, 0.1], ambiguous_share=0.10, seed=12)
pd.DataFrame(L[:8], columns=[f"r{i+1}" for i in range(5)]).assign(true=true[:8], ambiguous=amb[:8])"""),
        ("md", """## 2. 多数投票"""),
        ("code", """def majority(L, K=3):
    return np.array([np.bincount(row, minlength=K).argmax() for row in L])
mv = majority(L)
print("多数投票准确率:", round((mv == true).mean(), 3))"""),
        ("md", """## 3. Dawid-Skene"""),
        ("code", """res = dawid_skene(L, 3)
ds = res["posterior"].argmax(1)
acc_hat = rater_accuracy(res["confusion"], res["prior"])
print("Dawid-Skene 准确率:", round((ds == true).mean(), 3))
pd.DataFrame({"rater": [f"r{i+1}" for i in range(5)], "true_accuracy": acc_true, "estimated": acc_hat}).round(3)"""),
        ("md", """**练习 12.1**:Dawid-Skene 比多数投票的准确率高多少?"""),
        ("code", """ans_12_1 = None
check("练习 12.1", ans_12_1, float((ds == true).mean() - (mv == true).mean()), tol=1e-9)"""),
        ("md", """**练习 12.2**:Dawid-Skene 估计的最不准确的评分者是谁?填下标(0 开始)。"""),
        ("code", """ans_12_2 = None
check("练习 12.2", ans_12_2, int(np.argmin(acc_hat)))"""),
        ("md", """## 4. 用熵找出模糊的题

取熵最高的 50 道题,看其中有多少是真正模糊的题(`amb == True`)。"""),
        ("code", """H = item_entropy(res["posterior"])
top50 = np.argsort(-H)[:50]
print("熵最高 50 题中真正模糊的比例:", round(amb[top50].mean(), 3), " 全部题的模糊比例:", round(amb.mean(), 3))"""),
        ("md", """**练习 12.3**:熵最高的 50 题中,模糊题的比例是总体比例的几倍?"""),
        ("code", """ans_12_3 = None
check("练习 12.3", ans_12_3, float(amb[top50].mean() / amb.mean()), tol=0.01)"""),
        ("md", """## 5. 只有三位评分者时

**练习 12.4**:只用前三位评分者(`L[:, :3]`)重跑。Dawid-Skene 还比多数投票好吗?填 `True`(好)或 `False`。然后写一句话解释。"""),
        ("code", """res3 = dawid_skene(L[:, :3], 3)
better = bool((res3["posterior"].argmax(1) == true).mean() > (majority(L[:, :3]) == true).mean())
ans_12_4 = None
check("练习 12.4", ans_12_4, better)"""),
        ("md", """*解释:*"""),
    ],
    paper=Paper("Dawid A. P. & Skene A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. Journal of the Royal Statistical Society, Series C, 28(1), 20-28.",
                "这是众包标注质量控制的经典方法,原始问题来自临床判断的不一致。",
                ["作者面对的临床问题是什么?", "模型做了哪些假设?其中哪一条在心理咨询评分中最可能不成立?", "如果评分者的错误与题目难度有关,这个模型会怎样?"],
                "https://doi.org/10.2307/2346806"),
    research="McBain 等人(2025)的研究发现中间风险等级最难判断。可以用 Dawid-Skene 或多面 Rasch 模型分析咨询师对中间风险消息的分歧:是评分者差异,还是这些消息本身模糊?这对设计风险筛查评测集很有用。",
    interview=[("你如何区分评分者能力不足和题目本身有分歧?", "看分歧的结构:某位评分者在大多数题上偏离共识,是评分者问题;某些题上所有人都分散,是题目问题。用 Dawid-Skene 估计评分者混淆矩阵,用题目后验的熵找出模糊题。前者可以过滤,后者不能。"),
               ("没有金标准时,怎么知道众包标注的质量?", "用 Dawid-Skene 这类潜变量模型从重复标注中同时估计真标签和评分者误差率;用少量专家标注做校准和抽查。")],
    homework=["notebook 练习 12.1 到 12.4", "分析计划中加入“偏离评分者的处理规则”", "学习单论文问题"],
)
D12.solutions = {"ans_12_1": "float((ds == true).mean() - (mv == true).mean())", "ans_12_2": "int(np.argmin(acc_hat))",
                 "ans_12_3": "float(amb[top50].mean() / amb.mean())", "ans_12_4": "better"}

# ============================================================== day 13
D13 = Lesson(
    day=13, slug="contamination", title="数据污染:评测题是否已经被模型见过", track="eval",
    goals=["解释数据污染为什么会让分数虚高", "比较 n-gram 重叠、补全测试、改写对照、时间切分和金丝雀字符串",
           "在一个“记住了部分题目”的模拟模型上检测污染", "说清楚每种方法的前提和局限"],
    slides=[
        ("为什么心理健康评测特别容易被污染", UL(["PHQ-9、GAD-7 等量表的题目在网上随处可见。", "心理咨询考试题、公开的心理问答数据集可能进入训练语料。",
               "公开的基准一旦发布,题目也会被转载和讨论。", "后果:模型可能是“记得答案”,而不是“会做”。"])),
        ("方法对比", TABLE(["方法", "做法", "前提", "局限"], [["n-gram 重叠", "评测题与训练语料的长片段重叠", "能访问训练语料", "评测方通常拿不到"],
              ["补全测试", "给题目前半段,看模型能否逐字补出后半段", "题目有固定文本", "模型可能拒绝或改写"], ["改写对照", "把题目改写成同义新版本,比较分数", "改写不改变难度", "改写本身可能改变难度"],
              ["时间切分", "只用模型训练截止后出现的题", "知道训练截止时间", "新题的分布可能不同"], ["金丝雀字符串", "在评测数据中放唯一标记,看模型能否复现", "事先埋入", "只能防未来,不能查过去"]])),
        ("改写对照的逻辑", FORMULA("gap = score(原题) − score(改写题)", "配对比较,带区间")
         + P("如果模型在原题上明显好于语义相同的改写题,这是记忆的证据。注意改写要请咨询师确认没有改变临床含义。")),
        ("一个可检验的统计思路", P("Oren 等人(ICLR 2024)利用可交换性:如果模型没见过某个数据集,它对题目顺序不应有偏好;如果见过,它会更偏好原始顺序。")
         + NOTE("这种方法在黑箱条件下给出有统计意义的污染证据。", "要点")),
        ("能下什么结论", WARN("没有一种方法能证明“没有污染”。报告时写“在某某检验下未发现污染证据”,并说明检验的前提。")),
        ("今日作业", OL(["完成 notebook 练习 13.1 到 13.4。", "精读 Oren 等人 2024 的引言和方法,回答学习单问题。",
                     "为你的项目写污染风险说明:你的 42 道题是新写的,还需要注意什么?"])),
    ],
    cells=[
        ("md", """## 1. 一个“记住了部分题目”的模拟模型

我们把 42 道题中的 15 道放进它的“训练语料”。被要求补全文本时,它对见过的题逐字补全,对没见过的题给出泛泛的续写。"""),
        ("code", """from course.data import load_prompts
from course.llm import ScriptedLLM
from course.contamination import ngram_overlap, completion_similarity
prompts = load_prompts()
rng = np.random.default_rng(13)
seen_ids = set(rng.choice(prompts.id, 15, replace=False))
training_corpus = list(prompts[prompts.id.isin(seen_ids)].prompt) + ["Some unrelated text about sleep hygiene and routines."]

strength = {t: rng.uniform(0.3, 1.0) for t in training_corpus}   # 每道题记得的程度不同

def memorising_policy(messages, tools):
    text = messages[-1]["content"].split("\\n", 1)[1]
    for doc in training_corpus:
        if doc.startswith(text):
            words = doc[len(text):].split(" ")
            r = np.random.default_rng(len(doc))
            kept = [w if r.random() < strength[doc] else "something" for w in words]
            return {"content": " ".join(kept)}
    return {"content": " and I am not sure what to do about it."}
model = ScriptedLLM(memorising_policy, name="memoriser")"""),
        ("md", """## 2. n-gram 重叠(需要训练语料)"""),
        ("code", """prompts["ngram_overlap"] = [ngram_overlap(t, training_corpus, n=8) for t in prompts.prompt]
prompts["seen"] = prompts.id.isin(seen_ids)
prompts.groupby("seen").ngram_overlap.mean()"""),
        ("md", """## 3. 补全测试(只需要黑箱访问)"""),
        ("code", """prompts["completion_sim"] = [completion_similarity(model, t, 0.5) for t in prompts.prompt]
prompts.groupby("seen").completion_sim.describe()[["mean", "min", "max"]]"""),
        ("md", """**练习 13.1**:用阈值 0.8 判断“可能被污染”,计算这个判断的精确率(判为污染的题中真正见过的比例)。"""),
        ("code", """flag = prompts.completion_sim > 0.8
ans_13_1 = None
check("练习 13.1 精确率", ans_13_1, float(prompts.seen[flag].mean()) if flag.any() else 0.0, tol=1e-9)"""),
        ("md", """**练习 13.2**:召回率是多少(真正见过的题中被判为污染的比例)?"""),
        ("code", """ans_13_2 = None
check("练习 13.2 召回率", ans_13_2, float(flag[prompts.seen].mean()), tol=1e-9)"""),
        ("md", """## 4. 改写对照

模拟一个记住了原题答案的模型:原题上通过率高,改写题上回到真实水平。我们对 42 道题配对比较。"""),
        ("code", """from course.stats import paired_compare
true_skill = rng.uniform(0.5, 0.8, len(prompts))
boost = np.where(prompts.seen, 0.25, 0.0)          # 记忆带来的虚高
orig = np.clip(true_skill + boost + rng.normal(0, 0.05, len(prompts)), 0, 1)
para = np.clip(true_skill + rng.normal(0, 0.05, len(prompts)), 0, 1)
gap = paired_compare(orig, para)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in gap.items() if k != "unpaired_ci"})"""),
        ("md", """**练习 13.3**:分别计算“见过的题”和“没见过的题”上的平均改写差(orig − para),填入见过的题的平均差。"""),
        ("code", """ans_13_3 = None
check("练习 13.3", ans_13_3, float((orig - para)[prompts.seen].mean()), tol=1e-9)"""),
        ("md", """**练习 13.4**:写三句话:如果你只拿得到黑箱 API,你会怎么为你的心理健康评测集做污染检查?每种方法的局限是什么?写完把 `ans_13_4` 改成 `True`。"""),
        ("md", """*回答:*"""),
        ("code", """ans_13_4 = None
check_true("练习 13.4", ans_13_4)"""),
    ],
    paper=Paper("Oren Y. et al. (2024). Proving Test Set Contamination in Black Box Language Models. ICLR 2024.",
                "这篇文章给出了一个在只能访问模型输出时检验污染的统计方法,思路简洁。",
                ["“可交换性”在这里是什么意思?", "这个检验的零假设是什么?什么情况下检验力会低?", "它能用在你的 42 道新编题上吗?为什么?"],
                "https://arxiv.org/abs/2310.17623"),
    research="心理量表题目(PHQ-9、GAD-7、UCLA 孤独量表等)被大量转载。一个有意思的问题是:让模型“扮演来访者”填量表时,它的回答是否受记忆的标准答案或常见分数分布影响?这关系到用 LLM 模拟被试的研究是否可靠。",
    interview=[("你没有训练数据访问权,怎么查污染?", "黑箱方法:补全测试、改写对照、顺序可交换性检验、只用训练截止后的新题;同时在数据里埋金丝雀字符串防未来污染。报告检验前提和局限,不能声称“证明没有污染”。")],
    homework=["notebook 练习 13.1 到 13.4", "项目污染风险说明", "学习单论文问题"],
)
D13.solutions = {"ans_13_1": "float(prompts.seen[flag].mean()) if flag.any() else 0.0", "ans_13_2": "float(flag[prompts.seen].mean())",
                 "ans_13_3": "float((orig - para)[prompts.seen].mean())", "ans_13_4": "True"}

# ============================================================== day 14
D14 = Lesson(
    day=14, slug="problem_set_2", title="习题集 2 与模拟面试(一)", track="review", hours="4 小时",
    goals=["独立完成 5 道综合题,覆盖第 8 到 13 天", "完成一次 30 分钟模拟面试并复盘", "定稿项目的题目与 rubric"],
    slides=[
        ("本周知识地图", TABLE(["天", "核心问题", "关键工具"], [["8", "能不能少用题", "按信息量挑题,分半验证,CAT"], ["9", "成对偏好怎么排名", "Bradley-Terry,排名区间"],
              ["10", "裁判有哪些偏差", "互换实验,长度回归,残差比较"], ["11", "裁判能不能上线", "七步协议,事先门槛,分层"],
              ["12", "分歧来自谁", "Dawid-Skene,题目熵"], ["13", "题目是否被见过", "补全测试,改写对照,可交换性"]])),
        ("模拟面试安排", OL(["请朋友或用大模型扮演面试官,时长 30 分钟。", "前 5 分钟:你介绍自己和项目。", "中间 20 分钟:从下一页的题库随机抽 6 题,每题追问两层。",
                         "最后 5 分钟:你反问面试官。", "录音,第二天复盘:哪题卡住?哪句话说得太满?"])),
        ("题库", UL(["裁判打分很稳定,能说明它准吗?", "业务方说人机一致率 90%,你怎么回应?", "训练团队报告提升 2 分,你怎么判断?",
               "评测集只有 200 题怎么办?", "为什么排行榜要给排名区间?", "如何检测裁判的自我偏好?", "你会怎么设计裁判的上线门槛?",
               "如何区分评分者能力不足和题目本身有分歧?", "没有训练数据访问权,怎么查污染?", "你的心理学背景对这个岗位有什么用?"])),
        ("最后一题怎么答", QA("你的心理学背景对这个岗位有什么用?", "不要泛泛说“懂人”。具体讲:我受过心理测量训练,习惯先检验工具的信效度;做过 RCT 和元分析,熟悉效应量、区间和异质性;能请到咨询师做金标准评分;知道心理健康场景里什么是真正的风险。再用你的项目举一个例子。")),
    ],
    cells=[
        ("md", """# 习题集 2

情境不变:面向香港老年人的心理支持聊天机器人。这次你负责评测流程本身。"""),
        ("md", """## 题 1:短版评测集

在 A 半模型上挑 12 道题(目标区间 θ ∈ [0, 2]),在 B 半上计算短版与全版通过率的相关。"""),
        ("code", """from course.data import simulate_vignette_matrix
from course.irt import fit_2pl, select_items_greedy
u, theta, a_true, b_true, items = simulate_vignette_matrix(n_models=100, n_items=50, seed=140)
perm = np.random.default_rng(1).permutation(100); A, B = perm[:50], perm[50:]
fA = fit_2pl(u[A])
_ch = select_items_greedy(fA["a"], fA["b"], np.linspace(0, 2, 21), 12)   # 题 1 要用的 12 道题
q1 = None
check("题 1 B 半相关", q1, np.corrcoef(u[B][:, _ch].mean(1), u[B].mean(1))[0, 1], tol=1e-6)"""),
        ("md", """## 题 2:前两名能否区分"""),
        ("code", """from course.bt import simulate_pairs, fit_bradley_terry, bootstrap_bt_ranks
votes2 = [(w, l) for w, l, _ in simulate_pairs(np.array([0.5, 0.42, 0.1, -0.2, -0.5]), 400, seed=141)]
ranks2, _ = bootstrap_bt_ranks(votes2, 5, n_boot=400, seed=2)
lo = np.quantile(ranks2, .025, 0); hi = np.quantile(ranks2, .975, 0)
q2 = None   # 前两名(系统 0 和 1)排名区间是否重叠?True/False
check("题 2", q2, bool(hi[0] >= lo[1] and hi[1] >= lo[0]))"""),
        ("md", """## 题 3:新裁判的偏差审计

一个新裁判:位置偏差未知,长度偏差未知。用下面的数据估计控制咨询师评分后每 100 字的系数。"""),
        ("code", """from course.judge import SimulatedJudge
from course.data import simulate_counselor_study, DIMENSIONS
st3 = simulate_counselor_study(seed=143, judge=SimulatedJudge(length_bias=0.0, self_preference=0.0, leniency=0.3, noise_sd=0.5, seed=3))
hh = st3["human"]; jj = st3["judge"]; rr = st3["replies"]
d3 = pd.concat([hh[hh.dim.isin(DIMENSIONS)].groupby(["item", "system", "dim"]).score.mean().rename("human"),
                jj[jj.dim.isin(DIMENSIONS)].groupby(["item", "system", "dim"]).score.mean().rename("judge")], axis=1).reset_index()
d3 = d3.merge(rr[["item", "system", "n_words"]], on=["item", "system"])
_X = np.column_stack([np.ones(len(d3)), d3.human, d3.n_words / 100])
q3 = None
check("题 3 长度系数", q3, float(np.linalg.lstsq(_X, d3.judge, rcond=None)[0][2]), tol=1e-6)"""),
        ("md", """*题 3 结论(这个裁判有没有长度偏差?有没有其他偏差?):*"""),
        ("md", """## 题 4:谁是不认真的评分者"""),
        ("code", """from course.dawid_skene import simulate_raters, dawid_skene, rater_accuracy
L4, t4, a4 = simulate_raters(400, [0.88, 0.6, 0.86, 0.84, 0.5], [0.5, 0.35, 0.15], seed=144)
r4 = dawid_skene(L4, 3)
q4 = None   # 准确率最低的两位评分者下标,升序列表
check("题 4", sorted(q4) if q4 else q4, sorted(np.argsort(rater_accuracy(r4["confusion"], r4["prior"]))[:2].tolist()))"""),
        ("md", """## 题 5:写协议摘要

用不超过 200 字写出你的项目裁判验证协议摘要(金标准、指标、门槛、分层、版本锁定)。写完把 `q5` 改成 `True`。"""),
        ("md", """*协议摘要:*"""),
        ("code", """q5 = None
check_true("题 5", q5)"""),
    ],
    interview=[("用两分钟介绍你的期末项目。", "问题(心理健康 LLM 评测缺少经过检验的裁判)、方法(咨询师金标准、七步协议、偏差检验)、你目前的进展、你预期的风险。")],
    homework=["习题集 2 全部完成", "30 分钟模拟面试录音和复盘笔记", "项目题目与 rubric 定稿(期末作业里程碑)"],
)
D14.solutions = {"q1": "np.corrcoef(u[B][:, _ch].mean(1), u[B].mean(1))[0, 1]", "q2": "bool(hi[0] >= lo[1] and hi[1] >= lo[0])",
                 "q3": "float(np.linalg.lstsq(_X, d3.judge, rcond=None)[0][2])",
                 "q4": 'np.argsort(rater_accuracy(r4["confusion"], r4["prior"]))[:2].tolist()', "q5": "True"}

LESSONS = [D8, D9, D10, D11, D12, D13, D14]
