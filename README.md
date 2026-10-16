# 心理大模型评测与 Agent:28 天课程

面向心理学背景的研究者：用心理测量和实验设计的方法评测心理健康大模型，并动手搭建一个有评测的 CBT 支持 Agent。

- 每天一份**幻灯片**（`lessons/dayXX_*/slides.html`，浏览器打开，可逐页放映）
- 每天一份 **notebook**（讲解、可运行代码、带自动批改的练习）
- 每天一份**学习单**（`README.md`：学习顺序、论文精读问题、作业、口述题）
- 第 7、14、21 天是**习题集**，第 28 天提交**期末作业**（见 [`COURSEWORK.md`](COURSEWORK.md))
- [`solutions/`](solutions) 里有参考答案，请先独立完成再对照
- [`workshop/`](workshop) 是带学习小组或工作坊用的带领者指南和分周教案

所有例子来自心理健康场景；所有数据都是课程编写或模拟的，不含真实来访者数据。

## 开始之前（约 15 分钟）

```bash
git clone <本仓库地址>
cd <仓库文件夹>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                    # 课程代码库自带测试,应全部通过
jupyter lab               # 打开 lessons/day01_*/notebook.ipynb
```

双击 `index.html` 打开课程首页，可以勾选每天的完成情况。

## 课程安排

| 天 | 内容 | 类型 |
|---|---|---|
| | **第 1 周 测量基础** | |
| 1 | [信度与效度：先检验测量工具](lessons/day01_reliability_validity/README.md) | 评测方法学 |
| 2 | [一致性指标：kappa、alpha 与 ICC](lessons/day02_agreement/README.md) | 评测方法学 |
| 3 | [区间估计：一个分数没有区间就不是结论](lessons/day03_intervals/README.md) | 评测方法学 |
| 4 | [比较两个模型：配对、McNemar 与多重比较](lessons/day04_compare_models/README.md) | 评测方法学 |
| 5 | [题量与统计功效：需要多少道题](lessons/day05_power/README.md) | 评测方法学 |
| 6 | [项目反应理论（一）：题目并不等价](lessons/day06_irt_1/README.md) | 评测方法学 |
| 7 | [习题集 1：测量基础](lessons/day07_problem_set_1/README.md) | 复习与习题 |
| | **第 2 周 题目、裁判与评分者** | |
| 8 | [项目反应理论（二）：压缩评测集与自适应测验](lessons/day08_irt_2/README.md) | 评测方法学 |
| 9 | [偏好聚合：Bradley-Terry、Elo 与排名区间](lessons/day09_preference_aggregation/README.md) | 评测方法学 |
| 10 | [LLM 裁判的偏差：位置、长度、自我偏好与附和](lessons/day10_judge_bias/README.md) | 评测方法学 |
| 11 | [裁判验证协议：从检验到上线门槛](lessons/day11_judge_protocol/README.md) | 评测方法学 |
| 12 | [评分者建模：Dawid-Skene 与分歧的来源](lessons/day12_rater_models/README.md) | 评测方法学 |
| 13 | [数据污染：评测题是否已经被模型见过](lessons/day13_contamination/README.md) | 评测方法学 |
| 14 | [习题集 2 与模拟面试（一）](lessons/day14_problem_set_2/README.md) | 复习与习题 |
| | **第 3 周 大模型、Agent 与项目实战** | |
| 15 | [大模型基础：采样、对齐训练与奖励过优化](lessons/day15_llm_basics/README.md) | Agent 与模型 |
| 16 | [Agent 基础：工具调用与 ReAct 循环](lessons/day16_agent_basics/README.md) | Agent 与模型 |
| 17 | [记忆与用户建模：状态、特质与隐私](lessons/day17_memory_user_model/README.md) | Agent 与模型 |
| 18 | [Agent 评测：pass@k、pass^k 与轨迹检查](lessons/day18_agent_evaluation/README.md) | Agent 与模型 |
| 19 | [项目实战（一）：生成回复、裁判评分与评分包](lessons/day19_project_pipeline/README.md) | 项目实战 |
| 20 | [项目实战（二）：按计划分析并写结果](lessons/day20_project_analysis/README.md) | 项目实战 |
| 21 | [习题集 3 与模拟面试（二）：项目深挖](lessons/day21_problem_set_3/README.md) | 复习与习题 |
| | **第 4 周 研究级评测与求职** | |
| 22 | [因果推断：聊天机器人使用能减轻老年人的孤独吗](lessons/day22_causal_inference/README.md) | 评测方法学 |
| 23 | [评测集审计：入库检查、饱和与退役](lessons/day23_validity_audit/README.md) | 评测方法学 |
| 24 | [可复现性与版本治理：托管模型会悄悄改变](lessons/day24_reproducibility/README.md) | 评测方法学 |
| 25 | [项目收尾：仓库、README 与局限](lessons/day25_project_wrapup/README.md) | 项目实战 |
| 26 | [简历与项目讲述：把研究翻译成岗位语言](lessons/day26_resume_pitch/README.md) | 求职准备 |
| 27 | [全真模拟面试](lessons/day27_mock_interview/README.md) | 求职准备 |
| 28 | [查漏补缺与提交](lessons/day28_final_review/README.md) | 复习与习题 |

## 接入真实模型（可选）

课程默认全程离线运行。要接真实模型，在终端设置环境变量，不要把密钥写进 notebook:

```bash
export LLM_API_KEY=你的密钥
export LLM_BASE_URL=https://api.openai.com/v1     # 或其他兼容 OpenAI 格式的接口
export LLM_MODEL=模型名
```

然后在 notebook 里用 `from course.llm import get_client; client = get_client()`。

## 目录

```
index.html            课程首页
COURSEWORK.md         期末作业说明与评分标准
coursework/           报告模板与汇报提纲
lessons/dayXX_*/      每天的 slides.html、notebook.ipynb、README.md
solutions/            参考答案 notebook
workshop/             带领者指南、分周教案、参与者清单、反馈问题
src/course/           课程代码库(一致性、区间、IRT、裁判、Dawid-Skene、Agent、记忆、因果……)
data/                 题目与情景(课程编写)
docs/rubric.md        评分标准初稿
tests/                代码库测试
build/                生成与核对课程材料的脚本(修改课程时用,见 CONTRIBUTING.md)
```

## 每天怎么学

1. 看幻灯片（30 到 45 分钟）。
2. 运行 notebook，完成练习（约 2 小时）。
3. 精读学习单上的论文，回答问题（约 40 分钟）。
4. 录音回答口述题（15 分钟）。讲不出来，就说明还没学会。

## 成绩构成（自我检查用）

| 部分 | 占比 |
|---|---|
| 期末作业 | 60% |
| 习题集 1、2、3 | 30% |
| 每日作业与口述题 | 10% |

## 用这套材料开学习小组或工作坊

见 [`workshop/README.md`](workshop/README.md)：四周学习小组、两天集中工作坊、三小时体验课三种形式，每种都有分时段的教案。

## 引用与版权

引用信息见 [`CITATION.cff`](CITATION.cff)。版权所有 © 2026 王雅慧。公开发布前会补充开源许可证。

发现错误或想改进内容，见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
