# XunBlog Knowledge Content Architecture

这份文件定义 XunBlog 的内容边界与长期维护规则。它服务于作者和审阅者，不作为面向读者的教程页面。

## 1. 当前工程基线

- 内容由 `docs/` 下的 Markdown 文件维护，VitePress 配置位于 `docs/.vitepress/config.ts`。
- 导航和侧边栏目前手工配置；在内容规模较小的阶段继续沿用，避免引入生成器。
- 站点启用 `cleanUrls`，部署时由 `BASE_PATH` 注入 GitHub Pages 的 `/XunBlog/` 前缀。
- Markdown 内部链接统一写站内绝对路径，例如 `/robot-learning/act/`；不写 `/XunBlog/`，也不写 `.html`。
- `NoteMeta` 已可优先读取显式 props，并在未传 props 时读取当前页面 frontmatter。
- 现有未标为 `reviewed` 或 `stable` 的页面均视为工作笔记，不作为后续文章的事实来源。

## 2. 页面类型与 canonical 规则

每个重要概念只有一个 canonical page。创建页面前必须先按标题、同义词和 slug 搜索仓库。

| `pageType` | 回答的问题 | 示例 |
| --- | --- | --- |
| `concept` | 这个知识本身是什么？ | `/deep-learning/transformer` |
| `paper` | 一篇论文提出了什么问题与方法？ | `/robot-learning/act/why-action-chunking` |
| `application` | 某概念在特定模型中如何使用？ | `/robot-learning/act/transformer` |
| `topic-index` | 一个专题应按什么顺序阅读？ | `/robot-learning/act/` |
| `domain-index` | 一个知识域包含哪些主题？ | `/robot-learning/` |

规则：

1. canonical page 的 `canonical` 指向自身稳定路径。
2. application page 不重复教授完整基础理论，而是链接 canonical page 后只解释论文中的具体用法。
3. 同义词通过术语说明或未来的 redirect 处理，不创建重复页面。
4. 文件名使用小写 kebab-case；只有专题和知识域入口使用 `index.md`。
5. 已发布 slug 原则上不改。必须改名时，同时配置 redirect 并验证旧链接。

## 3. 目录策略

目录按“知识归属”而非按某一篇文章临时需要来划分：

```text
docs/
├── mathematics/          # 数学定义与推导
├── deep-learning/        # 通用网络结构与训练概念
├── generative-models/    # 需要首个正式页面时再创建
├── robot-learning/       # 模仿学习、策略学习与控制概念
│   └── act/              # ACT 论文与 ACT-specific application pages
├── embodied-ai/
├── llm/
└── projects/
```

“未来可能有内容”不是创建空目录或占位页面的理由。新知识域在首篇经过审阅的内容出现时再接入导航。

## 4. Frontmatter 最小契约

正式页面采用下面的最小、可机器读取的字段。数组只记录真实存在或已明确规划的关系，不为了完整而填充。

```yaml
---
title: ACT：它到底解决什么问题？
description: ...
status: reviewed
pageType: paper
canonical: /robot-learning/act/why-action-chunking
updated: "2026-09-14"
prerequisites:
  - /robot-learning/imitation-learning
related:
  - /deep-learning/transformer
primarySources:
  - title: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
    authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
    venue: "Robotics: Science and Systems 2023"
    url: https://www.roboticsproceedings.org/rss19/p016.pdf
---
```

字段约定：

- `status`: `seed`、`learning`、`draft`、`reviewed`、`stable`。
- `pageType`: 使用上一节的五种类型。
- `canonical`: 无末尾 `/` 的页面路径；index 页面保留末尾 `/`。
- `prerequisites`: 不理解就会阻碍当前页面的最小前置集合。
- `related`: 有帮助但不阻碍当前阅读的横向关系。
- `primarySources`: 模型或算法结论的最高等级来源。数学页改用权威教材或标准定义。

`usedBy` 暂不手工维护，避免双向关系漂移；需要时由反向链接索引自动生成。

## 5. 内部链接与前置知识

链接优先级：

1. 当前论证不可缺少的 prerequisite；
2. 文中实际使用、且已有 canonical page 的概念；
3. 文章末尾一到三个自然的 next steps。

不为普通术语加链接，不链接尚不存在的页面，也不把“深入理解实现时才需要”的概念伪装成 overview 的必修前置。

ACT overview 的最小 prerequisite 只有模仿学习与 Behavior Cloning 的基本问题。Transformer、CVAE、KL Divergence 等属于理解实现细节的分支，不是理解“ACT 为什么出现”的前提。

## 6. 来源与事实追踪

- 模型/算法页：先读原始论文，再建立 claim ledger，最后写正文。
- 数学页：以标准定义、权威教材或经典资料为主。
- 正文明确区分“论文事实”“数学推论”“直觉”；解释可以原创，事实必须可追溯。
- 每篇正式模型页末尾展示 `Primary Source`。二手资料不能与原论文处于同一证据等级。
- 论文图优先自行重绘数据流；若直接引用，必须标明来源并单独核对使用许可。
- 研究记录放在 `research/<topic>/`，记录 section、equation、figure、table、appendix；原论文 PDF 不提交到仓库。

## 7. 写作与审阅流程

```text
确定 canonical topic
→ 获取并阅读 primary source
→ 建立 claim ledger
→ 裁剪真实 prerequisites
→ 设计认知阶梯
→ 写作并标明事实层级
→ 添加内部链接与来源
→ 检查数学和 terminology
→ VitePress build（含 dead-link 检查）
→ desktop / mobile / dark mode 预览
→ 更新成熟度
```

`reviewed` 表示已经逐项对照来源并通过本地构建；`stable` 还要求独立复核或复现实验支持。正式文章也可以继续修正，状态不是“永不更改”的承诺。

## 8. 当前 ACT 内容边界

- `/robot-learning/act/`：专题阅读地图和裁剪后的依赖图。
- `/robot-learning/act/why-action-chunking`：第一篇正式 paper page，解释 ACT 解决的问题与设计逻辑。
- `/robot-learning/act/transformer`、`cvae`、`temporal-ensemble`：现阶段仍是 `seed/learning` application pages，不能反向作为正式页面的证据来源。
- `/deep-learning/transformer`：Transformer 的 canonical page；正式重写前仍视为工作笔记。
- `/robot-learning/imitation-learning`：Imitation Learning 的 canonical page；目前包含 Behavior Cloning 的最小定义，后续是否拆页由内容边界而非目录对称性决定。
