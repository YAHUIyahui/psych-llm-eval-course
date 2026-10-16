# Day 16 · Agent 基础：工具调用与 ReAct 循环

Agent 与模型 · 建议用时 3 到 4 小时

## 今天的目标

- 画出一次工具调用的完整消息流
- 读懂并修改一个不依赖框架的 Agent 循环
- 为 CBT 支持 Agent 增加一个工具，并让风险消息优先走危机处理
- 理解循环的保护措施：最大步数、错误返回、日志

## 学习顺序

1. 看 `slides.html`（约 30 到 45 分钟）。
2. 运行并完成 `notebook.ipynb` 里的练习（约 2 小时）。
3. 精读今日论文，回答下面的问题（约 40 分钟）。
4. 录音回答口述题（约 15 分钟）。

## 今日论文

**Yao S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023；以及 Habicht J. et al. (2024). Closing the accessibility gap to mental health treatment with a personalized self-referral chatbot. Nature Medicine, 30, 595-602.**  
https://doi.org/10.1038/s41591-023-02766-x

前者是 Agent 循环的基本思路，后者是心理健康领域一个真实部署、以结局评估的任务型 Agent。

阅读问题：

1. ReAct 和只做推理、只做行动相比，优势在哪里？
2. Habicht 等人用什么设计比较有无聊天机器人的服务？这种设计能支持因果结论吗（第 22 天会再讨论）？
3. 如果评估你的 CBT 支持 Agent，哪些结局指标是合理的？

## 今日作业（提交到你的课程仓库）

- [ ] notebook 练习 16.1 到 16.3
- [ ] 仓库新建 agent/ 并放入工具与策略
- [ ] 学习单论文问题

## 口述题

1. 不用框架，你能写出 Agent 循环吗？框架替你做了什么？
2. Agent 陷入循环怎么办？

## 研究延伸

大多数心理支持 Agent 研究只评估对话质量。Habicht 等人的工作提示了另一个方向：评估 Agent 对服务可及性的影响（例如老年人是否更愿意寻求帮助）。这与你的老年人数字心理健康研究直接相关。
