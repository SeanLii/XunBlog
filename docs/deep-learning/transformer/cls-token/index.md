---
title: "CLS Token"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/cls-token/"
prerequisites:
  - "/deep-learning/core/embedding/"
  - "/deep-learning/transformer/transformer-encoder/"
related:
  - "/robot-learning/act/cvae-in-act/"
---

# CLS Token

CLS Token 是人为加入序列中的一个特殊 learnable token，用来通过 self-attention 聚合整组输入信息，并把它的最终 hidden state 作为 sequence-level representation。

它并不是 Transformer 原始论文的必备组件，而是在 BERT 等后续架构中被广泛采用的一种设计。

## 聚合发生在哪里

把输入写成：

```text
[CLS], x1, x2, x3, ..., xn
```

经过 self-attention 后，CLS 位置也像普通 token 一样拥有 Query，可以读取其他 tokens 的 Keys/Values：

```text
x1 ─┐
x2 ─┤
x3 ─┼→ [CLS] reads them through attention
... │
xn ─┘
```

因此最后的

\[
h_{CLS}
\]

可以包含整段 sequence 的信息。

## CLS Token 不是自动平均

它不是把其他 token vectors 直接求平均。CLS representation 经过多层 self-attention 和 FFN，具体读哪些位置、读多少都由模型学习。

## ACT 中的 CLS Token

ACT training-only latent encoder 构造：

```text
[CLS], current qpos, action_1, ..., action_k
```

Transformer encoder 输出后，只取 CLS position 的 hidden state，再映射成

\[
\mu,\\log\sigma^2.
\]

所以在 ACT 中，CLS token 的职责非常具体：**把 current qpos 与整段 demonstration action chunk 汇总成一个 fixed-size representation，用来参数化 latent posterior。**

它不属于 ACT 独创机制；ACT 只是把这种 sequence summarization design 用在 CVAE encoder 中。

## Sources

- Devlin et al., **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, 2018. https://arxiv.org/abs/1810.04805
- Zhao et al., **ACT**, 2023. https://arxiv.org/abs/2304.13705
