# XunBlog Content Design Standard v4

本文件是当前 XunBlog 内容创作与维护的强制标准。它同时约束 **知识归属（ownership）** 与 **正文组织（exposition）**。

---

# 1. 两种独立必须同时成立

一个 topic 拥有独立 URL，不代表它已经真正独立。

必须同时满足：

1. **Canonical Ownership 独立**：它被放在真实的知识来源与学术谱系中；
2. **Canonical Exposition 独立**：正文首先服务于标题知识本身，而不是服务于 ACT、Transformer、π0 或其他阅读路径。

例如 Dot Product 不能写成：

```text
公式
→ QK^T
→ sqrt(d_k)
→ Attention
```

而应先完整建立：

```text
algebraic definition
→ geometric meaning
→ norm / angle
→ orthogonality
→ projection / similarity
→ matrix relation
→ applications
```

Attention 只是最后的 application。

---

# 2. Historical Ownership / Provenance Rule

知识树的归属首先看 **这个具体知识以什么身份进入文献**，而不是看后来哪些模型使用了它。

## 2.1 模型内部提出的机制

如果某个具体 mechanism / component 是在一个模型或算法中被正式提出、命名并定义为该模型的一部分，它保留在该模型下面。

例如：

```text
Transformer
├── Scaled Dot-Product Attention
├── Multi-Head Attention
└── Position-Wise Feed-Forward Network
```

```text
BERT
└── CLS Token
```

```text
DETR
└── Object Query
```

```text
ResNet
└── Residual Connection
```

```text
ACT
├── Action Chunking
└── Temporal Ensemble
```

```text
π0
├── Action Expert
└── Blockwise Causal Attention Mask in π0
```

后来被其他模型复用，不会自动改变它的 canonical home。

## 2.2 在目标模型之前已经独立存在的知识

如果概念在该模型之前已经作为独立理论、算法或机制存在，则不能因为某个后来的模型大量使用它，就把它归到后来的模型下面。

例如：

- Attention 早于 Transformer；
- Self-Attention 早于 Transformer；
- encoder–decoder attention / cross-source attention 早于 Transformer；
- Q / K / V 式 memory reading 有 Transformer 之前的 key-value memory lineage；
- Layer Normalization 早于 Transformer；
- positional representations 早于 Transformer；
- causal dependency masking 早于 Transformer；
- Flow Matching 早于 π0；
- CVAE 是独立论文提出的 conditional generative model，不属于 ACT；
- DAgger 是独立 imitation-learning algorithm，不属于 Behavior Cloning。

## 2.3 “相似思想更早存在”与“具体 formulation 来源”要区分

不要轻率声称绝对历史首创。

例如 ACT 论文明确说它受 neuroscience 的 action chunking 概念启发。因此不能写“ACT 首次发明了任何意义上的 action chunking”。

XunBlog 的 `ACT / Action Chunking` 页面范围是：

> ACT 中把 robot policy 输出改成 future action sequence、以降低 effective horizon 的具体 imitation-learning formulation。

这一具体 formulation 属于 ACT；更广泛的认知科学 chunking 概念不是本文主题。

同样，skip connection 早于 ResNet 并不妨碍 `Residual Connection` 讨论 ResNet 的 residual formulation：

\[
y=x+F(x).
\]

关键是把 scope 写清楚。

---

# 3. Navigation Graph 不迁移 Ownership

一个知识被多个模型使用，只增加 graph edges：

```text
π0 → Action Chunking
ACT → Transformer
ACT → Object Query
Transformer → Residual Connection
```

它不会因为使用次数增加而搬家。

Tree 决定 home；Graph 决定跳转。

---

# 4. 独立 Topic 判定

一个概念是否值得拥有自己的 canonical page，检查：

- 是否有稳定、明确、脱离当前模型仍成立的定义；
- 是否有值得独立建立的机制、数学性质、算法流程或历史身份；
- 是否本身就是公认模型 / 算法 / 数学对象；
- 独立后是否能减少其他页面承担不属于自己的教学责任。

不要因为出现一个专业词就机械拆页。

同样，一个模型内部 component 也可以拥有完整独立页面；“独立页面”不等于“离开父模型”。

---

# 5. Model / Architecture 页面

总览页首先建立 whole-system mental model：

```text
是什么
→ 解决什么问题
→ 输入 / 输出
→ 最小完整数据流
→ 核心设计变化
→ 主要组件
→ mathematics / training / inference
→ limitations / sources
```

读者在进入局部公式前，必须已经知道自己正在看什么系统。

Paper grounding 负责事实正确，不负责决定文章叙述顺序。

---

# 6. Mechanism / Component 页面

例如 Attention、QKV、Temporal Ensemble、Object Query：

```text
它处理什么对象
→ 输入 / 输出
→ 一次完整计算或数据流
→ mechanism intuition
→ formal mathematics
→ boundary / assumptions
→ history / provenance（需要时）
→ connections / applications
```

不要只写到“够理解父模型”为止。

---

# 7. Mathematics 页面

先建立数学对象，而不是 AI 使用场景：

```text
对象是什么
→ formal definition
→ geometric / probabilistic meaning
→ core properties
→ concrete examples
→ connections
→ AI applications
```

数学页不能以“为了看懂 Transformer / VAE”起笔。

---

# 8. Algorithms 页面

算法必须拥有自己的 problem setting、procedure 与 limitations。

例如 DAgger 必须完整讲：

- learner-induced distribution；
- expert query；
- dataset aggregation；
- iterative procedure；
- no-regret motivation；
- practical expert cost。

不能只在 Behavior Cloning 的一个 section 中带过。

---

# 9. Model-Specific 页面

`CVAE in ACT`、`ACT Training`、`Flow Matching in π0` 等可以明确依赖父模型。

它们的职责是：

> 解释父模型如何使用一个已有知识，或解释父模型自己的数据流。

它们不能重新拥有通用 CVAE、Flow Matching、Transformer 等理论。

---

# 10. 文章内部不要讨论网站维护规则

Ownership 规则用于组织知识库，不应成为正文里的产品说明。

正文可以写：

> Attention 早于 Transformer。

不要写：

> XunBlog 因此把 Attention 的 canonical home 放到……

正文应该永远像技术教材，而不是网站维护文档。

---

# 11. 问题型标题

问题可以在知识自然发展到某一步时出现，但不作为默认 H2/H3 组织方式。

优先：

```text
Posterior Collapse
Distribution Shift
KV Cache 的内存成本
```

而不是连续：

```text
为什么会……
那怎么办……
为什么还要……
```

唯一例外是页面本身就是一个 model-specific question，例如：

`为什么 ACT 推理时令 z = 0？`

---

# 12. 数学写作

每个重要公式至少要让读者知道：

1. variables 是什么；
2. 公式描述什么关系；
3. 为什么此处需要它；
4. 与前文什么概念相连；
5. 它对模型 / 算法意味着什么。

禁止“公式 → 显然 → 下一个公式”。

直觉与正式定义分开。

---

# 13. Paper / Code Grounding

- 模型、算法、架构：原论文是主要依据；
- released implementation detail：官方代码是主要依据；
- paper / code 冲突：明确区分 `Paper description` 与 `Released implementation`；
- 不用博客、README、二手文章替代原始依据；
- 不为了叙述整齐隐藏冲突。

---

# 14. 页面长度

独立不等于统一长度。

Vector Norm 可以短于 Variational Inference；Causal Mask 可以短于 Transformer。

标准是：

> 删除 application 段落后，标题知识是否仍然被完整建立。

不能用字符数替代完整性判断。

---

# 15. Final Self-Test

每篇 canonical page 发布前检查：

1. 标题是否就是知识名称；
2. ownership 是否有来源依据；
3. breadcrumb 是否反映真实 Tree；
4. 首段能否说明“这是什么”；
5. 是否有清楚的输入 / 输出或数学对象；
6. 是否有自然理解层级；
7. 是否把其他独立知识抢进来重复讲；
8. 删除所有 ACT / Transformer / π0 application 后，通用页面是否仍完整；
9. 是否存在 paper / code discrepancy 未披露；
10. 是否有断链、重复 canonical、多个 H1、损坏公式或控制字符；
11. 是否在正文里出现不必要的 XunBlog maintenance language；
12. 是否为了“新版”而重写已经合格的内容。

最后一条同样重要：

> **不为了重新而重新。只有内容、归属或准确性存在实质问题时才修改。**
