# Day 18 · Agent 评测：pass@k、pass^k 与轨迹检查

Agent 与模型 · 建议用时 3 到 4 小时

## 今天的目标

- 写出 pass@k 和 pass^k 的估计公式，并说明各自适用的场景
- 对每个任务重复运行多次，估计 Agent 的稳定性
- 用轨迹检查发现“结果看起来对、过程不对”的情况
- 量化关键词风险筛查的漏报

## 学习顺序

1. 看 `slides.html`（约 30 到 45 分钟）。
2. 运行并完成 `notebook.ipynb` 里的练习（约 2 小时）。
3. 精读今日论文，回答下面的问题（约 40 分钟）。
4. 录音回答口述题（约 15 分钟）。

## 今日论文

**Moore J. et al. (2025). Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers. ACM FAccT 2025. arXiv:2504.18412**  
https://arxiv.org/abs/2504.18412

这篇文章从临床实践指南出发设计检验，很适合作为你项目里安全检验的参照。

阅读问题：

1. 作者如何从治疗指南中提炼检验内容？
2. 污名是如何测量的？用了什么刺激材料？
3. 哪些发现可以转化为你的 Agent 的轨迹检查或评测题？

## 今日作业（提交到你的课程仓库）

- [ ] notebook 练习 18.1 到 18.4
- [ ] 项目 Agent 评测方案（docs/agent_eval_plan.md)
- [ ] 学习单论文问题

## 口述题

1. pass@k 和 k 次都成功有什么区别？各适合什么场景？
2. Agent 的最终回复看起来没问题，为什么还要看轨迹？

## 研究延伸

危机处理的可靠性应该用 pass^k 而不是单次通过率报告，这一点在心理健康 LLM 文献中很少被讨论。McBain 等人每题运行 100 次的设计已经提供了这类数据。可以提出并示范一种面向安全的可靠性报告规范。
