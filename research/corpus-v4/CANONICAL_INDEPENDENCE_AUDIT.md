# XunBlog v4 — Canonical Independence & Provenance Audit

本审计解决两个不同问题：

1. **这个 topic 是否值得独立存在？**
2. **如果独立存在，它应该属于哪里？**

v4 不再使用“被多个模型使用 → 自动提升为公共 topic”的规则。  
最终判断依据是：**知识本身的独立性 + 具体 formulation 的历史来源。**

---

# 1. Audit Result Summary

最终内容页：

- 82 个 content pages；
- 82 个唯一 canonical URLs；
- 0 个重复 canonical；
- 0 个内部断链；
- 0 个多 H1 页面；
- 0 个问题型 H2/H3；
- 0 个非法控制字符；
- 仅保留 1 个合理的问题型页面标题：`为什么 ACT 推理时令 z = 0？`。

本轮没有为了“新版”机械推翻 ACT、Transformer、π0 主页面。三者都重新审计后：

- `ACT`：整体结构保留，局部准确性和 provenance 修正；
- `Transformer`：整体 mental model 保留，ownership 与少量措辞修正；
- `π0`：整体结构保留，heading、KV Cache、mask scope、数据规模和 paper/code 边界修正。

---

# 2. Transformer Ownership Audit

## Attention — 独立

**判定：移出 Transformer。**

理由：

Bahdanau et al. 2014 已经在 encoder–decoder NMT 中使用动态 attention；Transformer 2017 不是 Attention 的起点。

最终：

```text
Deep Learning
└── Attention
    └── Attention
```

Transformer 通过链接使用它。

---

## Query / Key / Value — 独立于 Transformer

**判定：放在 Attention 下。**

理由：

Transformer 让 Q/K/V 成为现代 attention 的标准表达，但 Transformer 之前的 Key-Value Memory Networks 已经存在 query-addressed key/value reading structure。

页面重点：

- query 的检索角色；
- key 的匹配角色；
- value 的内容角色；
- learned projections；
- matrix form；
- self / cross attention 中的具体来源。

---

## Self-Attention — 独立于 Transformer

**判定：移出 Transformer。**

理由：

2016–2017 已经存在 sequence internal / self-attentive mechanisms。Transformer 把 self-attention 提升为主干 computation，但不是概念最早来源。

---

## Cross-Attention — 独立于 Transformer

**判定：移出 Transformer。**

理由：

source–target encoder–decoder attention 明显早于 Transformer。

页面不把 “cross-attention” 简化成 Transformer decoder 专属组件。

---

## Scaled Dot-Product Attention — Transformer

**判定：保留在 Transformer。**

Transformer 论文正式定义：

\[
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

这是 Transformer architecture 的具体 attention formulation。

---

## Multi-Head Attention — Transformer

**判定：保留在 Transformer。**

Multi-Head Attention 作为并行 learned projections + head concatenation 的具体 design 由 Transformer paper 系统定义。

---

## Position-Wise Feed-Forward Network — Transformer

**判定：重命名并保留在 Transformer。**

旧页面标题 `Feed-Forward Network` 太宽，而正文实际解释 Transformer position-wise FFN。

v4：

```text
Deep Learning / Core
└── Multilayer Perceptron

Transformer
└── Position-Wise Feed-Forward Network
```

这样 generic MLP 与 Transformer-specific sublayer 不再混用。

---

## Positional Encoding — 独立

**判定：移到 Sequence Modeling。**

理由：

位置表示不是 Transformer 首创；Transformer 之前的 sequence models 已经使用 learned positional embeddings。

Transformer 的 sinusoidal encoding 是其中一种具体 implementation。

---

## Causal Mask — 独立

**判定：移到 Sequence Modeling。**

autoregressive dependency restrictions 与 masked computation 早于 Transformer。Transformer decoder 是重要应用，而不是概念来源。

---

## Layer Normalization — 独立

**判定：Deep Learning Core。**

Ba et al. 2016 独立提出 Layer Normalization，早于 Transformer。

---

## Residual Connection — ResNet lineage

**判定：从 Deep Learning Core 移到 ResNet。**

需要区分：

- skip / shortcut ideas 不是 ResNet 历史上绝对第一次出现；
- ResNet 2015 系统提出 residual learning formulation：

\[
y=x+F(x).
\]

当前页面讨论的是这一 residual-connection lineage，因此：

```text
ResNet
└── Residual Connection
```

Transformer 只是后续复用者。

---

## CLS Token — BERT

**判定：从 Transformer 移到 BERT。**

`[CLS]` 是 BERT-specific input design，不是 2017 Transformer 的标准组件。

为避免让 CLS 成为“无父模型的背景页”，v4 新增完整 `BERT` canonical page。

---

## Learnable Query Embedding → Object Query — DETR

**判定：旧 generic 页面改造成 DETR 的 Object Query。**

ACT 的 learned action queries 更直接继承 DETR object-query / output-slot idea，而不是原始 Transformer 的通用标准组件。

v4 新增：

```text
DETR
└── Object Query
```

并让 ACT Architecture 直接链接该来源。

---

## KV Cache — Transformer lineage

**判定：新增独立页面，放在 Transformer 下。**

KV Cache 不是 2017 Transformer paper 的主要贡献，但它是 Q/K/V attention inference 的特定计算复用机制，不能作为 π0 私有知识。

π0 只解释它如何缓存 observation prefix。

---

# 3. Generative / Probability Ownership Audit

## Latent Variable — Probability

**判定：从 Generative Models 移到 Probability。**

latent variable 是 probabilistic modeling 的一般概念，远早于 VAE。

---

## Variational Inference — Probability

**判定：从 Generative Models 移到 Probability。**

它是 approximate Bayesian inference framework，不是 VAE 前置注释。

---

## Evidence Lower Bound — Variational Inference

**判定：放在 Variational Inference 下。**

ELBO 先从：

\[
\log p(x)
=
\mathcal L(q)
+
D_{KL}(q(z)\|p(z\mid x))
\]

这一类关系理解，再进入 VAE decomposition。

不能把 ELBO 定义成固定的 “reconstruction + KL”。

---

## Reparameterization Trick — Variational Inference

**判定：从 VAE 移到 Variational Inference。**

现代 VAE 让 reparameterization trick 广为人知，但 pathwise-gradient idea 更一般；Kingma & Welling / Rezende 等把它带入 modern deep variational inference。

因此：

```text
Probability
└── Variational Inference
    └── Reparameterization Trick
```

VAE 是重要 application，不是 ownership parent。

---

## Autoencoder — Representation Learning

**判定：从 Generative Models 移到 Deep Learning / Representation Learning。**

deterministic autoencoder 的核心目标是 encode–decode reconstruction / representation learning。它不能因为 VAE 名字中有 “autoencoder” 就成为 VAE background。

---

## Variational Autoencoder — 独立 Generative Model

**判定：保留 Generative Models 顶层。**

正文建立：

- latent generative model；
- approximate posterior；
- ELBO；
- reparameterized training；
- generation path；
- inference path。

---

## Conditional Variational Autoencoder — 独立 Generative Model

**判定：恢复为 Generative Models 顶层独立 topic。**

Sohn et al. 2015 单独提出 deep conditional generative model / CVAE。

它虽然是 VAE family extension，但不是 ACT 的 component，也不需要作为 VAE 私有 subpage。

ACT 只保留：

`CVAE in ACT`

用于解释 ACT 的具体 mapping。

---

## Posterior Collapse — 独立 failure mode

**判定：Generative Models 顶层。**

它与 VAE 强相关，但不是 VAE architecture 中“被发明的一个组件”。

页面独立解释：

- posterior 与 prior 接近；
- decoder bypass；
- information usage；
- KL / optimization dynamics；
- mitigation categories。

---

## Flow Matching — 独立 Generative Model / Training Framework

**判定：Generative Models 顶层。**

Lipman et al. 2022 明显早于 π0。

π0 只拥有：

`Flow Matching in π0`

---

# 4. CNN / Mathematical Ownership Audit

## Convolution — Mathematics / Analysis

**判定：从 CNN 移出。**

Convolution 是独立数学运算。页面完整覆盖：

- continuous convolution；
- discrete convolution；
- kernel flipping；
- 2D；
- algebraic properties；
- filtering；
- deep-learning cross-correlation convention。

CNN 是 application。

---

## Convolutional Neural Network — Deep Learning

**判定：保留。**

它是独立 architecture family，不把 Convolution 的数学定义据为己有。

---

## ResNet — Deep Learning / CNN

**判定：保留。**

主页面完整建立 degradation problem、residual learning、block structure、stage hierarchy。

---

## Residual Connection — ResNet

详见 Transformer audit。它是 ResNet residual-learning lineage 中的独立 subtopic。

---

# 5. Robot Learning Ownership Audit

## Imitation Learning — 独立领域

**判定：保留。**

正文不再把 ACT 当作终点，完整建立：

- expert demonstrations；
- sequential distribution shift；
- offline / interactive imitation；
- multimodality；
- partial observability；
- policy output forms。

---

## Behavior Cloning — 独立 algorithm / training paradigm

**判定：保留。**

完整建立 supervised objective、closed-loop deployment、covariate shift、compounding errors 与 multimodal outputs。

ACT 只是 application。

---

## DAgger — 独立 algorithm

**判定：独立页面。**

从旧 BC section 中拆出，依据 Ross et al. 2011。

---

## Vision-Language-Action Model — 独立 model family

**判定：保留。**

不使用 π0 定义全部 VLA，完整区分：

- VLM semantics；
- action interface；
- discrete / continuous actions；
- closed-loop control；
- generalist training。

---

## Cross-Embodiment Learning — 独立 topic

**判定：保留。**

不作为 π0 的数据背景页。

---

# 6. ACT Audit

## ACT main page

**判定：保留整体结构，局部修改。**

主页面已经符合：

```text
what it is
→ input / output
→ action chunk mental model
→ overlapping prediction
→ temporal ensemble
→ latent variation
→ architecture
→ training / inference
```

本轮只修正：

- “普通 Behavior Cloning” → “最简单的 single-step BC policy”；
- 问题型内部 heading 改成陈述式；
- links 指向真实 source pages；
- provenance 更清楚。

没有为了版本号全文重写。

---

## Action Chunking — ACT

**判定：保留在 ACT。**

重要 nuance：

ACT 论文明确说其受到 neuroscience action chunking 概念启发，因此 v4 不写“ACT 发明了历史上一切 action chunking”。

本页 scope 是 ACT 的 robot imitation-learning formulation：

\[
\pi(a_{t:t+k-1}\mid o_t).
\]

因此它仍然属于 ACT 板块。

---

## Temporal Ensemble — ACT

**判定：保留。**

v4 加强 released implementation data flow：

- temporal aggregation 下每步 query；
- `all_time_actions[t, t:t+k]`；
- 对当前 target timestep 的历史预测做指数加权。

---

## ACT Architecture

**判定：保留并修正 provenance links。**

现在明确链接：

- ResNet；
- Transformer；
- BERT / CLS Token；
- DETR / Object Query；
- CVAE。

ACT 页面只解释这些已有组件如何组合成 ACT。

---

## ACT paper / code discrepancy

**判定：保留。**

不隐藏 paper / released-code 之间的 reconstruction loss、optimizer、implementation behavior 等差异。

---

# 7. π0 Audit

## π0 main page

**判定：保留整体结构。**

已经先建立：

```text
Images + Language + Robot State
             ↓
             π0
             ↓
Continuous Action Chunk
```

再进入 VLM、Action Expert、Flow Matching、attention layout、training / inference。

无需为了 v4 全文重写。

---

## Action Expert — π0

**判定：保留在 π0。**

它是 π0 明确定义的 robotics-specific parameter branch。

---

## Blockwise Causal Attention Mask in π0 — π0

**判定：标题与 scope 修正后保留。**

旧标题过于通用，但正文实际上解释 π0 的 3-block layout。

v4 直接命名：

`Blockwise Causal Attention Mask in π0`

通用 Causal Mask 另有独立页面。

---

## Flow Matching in π0 — π0-specific usage

**判定：保留。**

通用 Flow Matching 独立存在；这里只解释 π0 如何定义 noisy action path、vector field target 与 sampling。

---

## KV Cache — link outward

π0 Inference 不再承担完整 KV Cache 理论，只解释：

- observation prefix 先 prefill；
- cache K/V；
- 每个 flow integration step 只重算 action suffix。

---

## Complete Data Flow

**判定：修复 Markdown hierarchy。**

旧版本一个页面出现多个 H1；v4 已统一：

```text
H1 Complete Data Flow
H2 Training Data Flow
H2 Inference Data Flow
H2 Final Diagram
```

---

## Pre-training / Post-training

**判定：保留并补充规模。**

补充论文报告的：

- 10,000+ robot-data hours；
- 7 robot configurations；
- 68 tasks；
- Open X-Embodiment mixture。

数字用于让 “broad pre-training” 具体化，不作为 architecture 定义。

---

## Paper / code flow-time convention

**判定：保留差异。**

paper 与 official openpi 使用相反的 time direction；两种 convention 均明确标注。

---

# 8. New Canonical Pages Introduced in v4

为修复旧页面“把来源吞进 background”的问题，v4 新增或正式建立：

- Multilayer Perceptron
- BERT
- DETR
- KV Cache
- Object Query
- CLS Token（重新归属）
- Vector Norm
- Linear Transformation
- Bayes' Theorem
- Covariance
- Entropy
- Cross-Entropy
- Activation Function
- Autoencoder
- DAgger

其中部分在上一轮草案已出现，本轮最终版重新按 provenance rule 校正 home 与 links。

---

# 9. Final Principle

v4 最终采用：

\[
\boxed{
\text{Independent exposition}
+
\text{historically grounded ownership}
}
\]

一个页面既不能因为“属于某模型”就写成两段 background，也不能因为“后来很多模型用了”就抹掉它真正的来源。

页面是否完整，与页面属于哪里，是两个都必须回答的问题。
