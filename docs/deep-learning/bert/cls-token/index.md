---
title: "CLS Token"
kind: "canonical"
domain: "Deep Learning / BERT"
parent: "BERT"
canonical: "/deep-learning/bert/cls-token/"
prerequisites:
  - "/deep-learning/bert/"
related:
  - "/robot-learning/act/cvae-in-act/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# CLS Token

> **知识边界**：本文的 canonical 对象是 **CLS Token**。依赖机制由 [BERT](/deep-learning/bert/) 的 canonical page 定义；本文只在当前语境中调用其接口。


`[CLS]` 是 BERT 在输入 sequence 开头加入的 special token。它没有天然携带“整句话摘要”，而是提供一个可以在多层 self-attention 中不断读取其他 token、并被下游 loss 直接使用的专门位置。

初始输入可以写成：

```text
[CLS] x1 x2 x3 ... xN
```

经过 BERT Encoder 后得到：

```text
h_CLS h1 h2 h3 ... hN
```

sequence-level classification 通常读取：

\[
h_{CLS}.
\]

## 初始 CLS 只是一个 Learned Embedding

一开始：

\[
e_{CLS}\in\mathbb R^d
\]

只是一个可训练向量。

它之所以最后能承载全局信息，不是因为名字叫 CLS，而是因为它参与每一层 self-attention，可以与其他 positions 交换信息，并且训练 objective 会要求它对最终任务有用。

## 多层信息聚合

在某层 attention 中，CLS position 会形成自己的 query：

\[
q_{CLS}=h_{CLS}W_Q.
\]

它可以对 sequence 中各 key 计算 attention weights，再聚合 values。

经过多层之后：

\[
h_{CLS}^{(L)}
\]

已经依赖整个 sequence 的 contextual interactions。

因此 CLS 更接近一个 **learned global readout slot**，而不是固定 pooling 公式。

## 与 Mean Pooling 的区别

Mean pooling：

\[
h_{mean}=\frac1N\sum_i h_i
\]

以固定相等权重汇总所有 positions。

CLS 则让“怎样汇总”参与模型学习：

- 每层可以动态读取不同 positions；
- 读取过程与上下文有关；
- 下游 objective 可以直接塑造 CLS representation。

两者都是获得 sequence-level representation 的方法，但机制不同。

## CLS 不是 Transformer 组件

原始 Transformer 没有 `[CLS]`。

`[CLS]` 是 BERT 的 model-specific input design，后来 Vision Transformer 等模型借鉴了“额外 class token / readout token”的思路。

因此 `[CLS]` 应理解为 BERT 的具体输入设计，而不是 Transformer architecture 自带的标准 token。

## ACT 中的 CLS-like Token

ACT 的 CVAE training encoder 在 sequence 前加入一个 learnable CLS-like token：

```text
[CLS] , q_t , a_t , ... , a_t+k-1
```

然后读取该位置输出，预测 latent distribution 的 $\mu$ 与 $\log\sigma^2$。

ACT 借用了 summary-token 机制，但没有使用 BERT 的 MLM、segment embedding 或完整 BERT 模型。

## Sources

- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018.
- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023.
