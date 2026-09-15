---
title: "Attention"
kind: "canonical"
domain: "Deep Learning / Attention"
parent: "Deep Learning"
canonical: "/deep-learning/attention/"
prerequisites:
  - "/mathematics/linear-algebra/dot-product/"
  - "/deep-learning/core/softmax/"
related:
  - "/deep-learning/transformer/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Attention

> **知识边界**：本文的 canonical 对象是 **Attention**。依赖机制由 [Dot Product](/mathematics/linear-algebra/dot-product/)、[Softmax](/deep-learning/core/softmax/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Attention 是一种根据当前 query 与候选信息之间的匹配关系，动态计算读取权重并聚合 values 的机制。

Attention 在 Transformer 之前已经用于 neural machine translation；Transformer 后来把 attention 提升为主要网络计算，并提出 Scaled Dot-Product Attention 与 Multi-Head Attention 等具体形式。

一般地，给定 query $q$、keys $k_i$ 与对应 values $v_i$，attention 可以写成

\[
s_i=s(q,k_i),
\]

\[
\alpha_i=\operatorname{Normalize}(s_1,\ldots,s_N)_i,
\]

\[
o=\sum_{i=1}^{N}\alpha_i v_i.
\]

这里 score function 决定匹配方式，normalization 决定如何把 scores 转换为权重，weighted aggregation 决定最终读取结果。

## Weighted Aggregation

若权重已知，

\[
\alpha_i\ge0,
\qquad
\sum_i\alpha_i=1,
\]

则

\[
o=\sum_i\alpha_i v_i
\]

是 values 的加权组合。

Attention 的关键不在 weighted sum 本身，而在于 $\alpha_i$ 由当前输入动态产生。不同 query 面对同一组 memory 可以得到不同权重，因此读取模式随内容变化。

## Query, Key, and Value Roles

在常见 key-value formulation 中：

- Query 表示当前读取请求；
- Key 表示候选信息用于匹配的表示；
- Value 表示匹配后被聚合的内容。

这三者是计算角色，而不是固定语义类别。完整结构见 [Query / Key / Value](/deep-learning/attention/qkv/)。

## Score Functions

不同 attention mechanisms 使用不同 score functions。

Bahdanau additive attention 使用 learned nonlinear scoring function；dot-product attention 使用

\[
s(q,k)=q^\top k.
\]

Transformer 的 [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) 则使用

\[
s(q,k)=\frac{q^\top k}{\sqrt{d_k}}.
\]

因此 Attention 是更一般的机制，具体 score function 属于特定实现。

## Normalization

Transformer-style attention 通常对 scores 使用 Softmax：

\[
\alpha_i=\frac{e^{s_i}}{\sum_j e^{s_j}}.
\]

这使权重为正并总和为 1。其他 attention formulations 也可能使用不同 normalization 或 sparse weighting mechanism，因此 Softmax 不是 Attention 的逻辑定义所必需，但它是现代 Transformer attention 的标准组成。

## Matrix Form

对多个 queries，设

\[
Q\in\mathbb R^{N_q\times d_k},
\quad
K\in\mathbb R^{N_k\times d_k},
\quad
V\in\mathbb R^{N_k\times d_v}.
\]

Transformer-style attention 为

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

其中

\[
QK^\top\in\mathbb R^{N_q\times N_k}
\]

包含每个 query 与每个 key 的 pairwise scores。最终输出

\[
O\in\mathbb R^{N_q\times d_v},
\]

因此 output slot 数由 query 数量 $N_q$ 决定。

## Self-Attention and Cross-Attention

若 queries、keys、values 都由同一组 representations 产生，则得到 [Self-Attention](/deep-learning/attention/self-attention/)。

若 queries 来自一组 representations，而 keys / values 来自另一组，则得到 [Cross-Attention](/deep-learning/attention/cross-attention/)。

二者的核心 readout mechanism 相同，区别在于 information source 与 connectivity。

## Masking

Attention scores 可以在 normalization 前加入 mask。被屏蔽位置通常被赋予极大的负值，使 Softmax 后对应权重接近 0。

Mask 可以表达不同约束，例如：

- padding positions 不参与读取；
- [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) 禁止读取未来位置；
- blockwise mask 规定不同 token groups 之间的可见性。

因此 attention connectivity 不只由 Q/K/V 决定，也受到 mask structure 约束。

## Position Information

标准 self-attention 对输入 permutation 具有对应的 permutation-equivariance。若没有额外 position signal，机制本身无法区分“第一个”“前一个”或“相距多远”。

Sequence models 通常通过 [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) 或 relative-position mechanism 提供顺序信息。

## Multi-Head Attention

[Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) 并行建立多个 learned projection spaces，在每个 head 中独立计算 attention，再将结果合并。

这允许同一层同时学习多组 interaction patterns，但具体 head 并不保证对应可人工命名的唯一语义。

## Computational Complexity

Full self-attention 对长度为 $N$ 的 sequence 需要构造 $N\times N$ score matrix，因此 pairwise attention 的主要时间与显存成本通常随 $N^2$ 增长。

Efficient attention、sparse attention、linear attention 与 block attention 等方法，主要针对这类 pairwise interaction cost 进行结构化或近似化处理。

## Sources

- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. 2014.
- Luong, Pham, Manning. *Effective Approaches to Attention-based Neural Machine Translation*. 2015.
- Vaswani et al. *Attention Is All You Need*. 2017.
