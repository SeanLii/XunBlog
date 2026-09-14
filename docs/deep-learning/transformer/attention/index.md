---
title: "Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Transformer / Attention"
canonical: "/deep-learning/transformer/attention/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/deep-learning/transformer/attention/qkv/"
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
---

# Attention

Attention 是一种“让一个表示根据相关性，从一组表示中读取信息”的计算机制。

先看一个最小场景。我们有一个当前位置 $x_i$，以及一组候选信息 $x_1,\ldots,x_n$。Attention 不会把所有候选一视同仁，而是先得到每个候选的重要程度，再按这些权重汇总信息：

```text
candidate 1 ── weight 0.05 ─┐
candidate 2 ── weight 0.70 ─┤
candidate 3 ── weight 0.20 ─┤→ weighted sum → output
candidate 4 ── weight 0.05 ─┘
```

因此 attention 的核心有两步：

1. **决定看谁**；
2. **把被看的信息按权重汇总**。

## 从加权平均开始

如果候选信息向量为 $v_1,\ldots,v_n$，attention weights 为 $\alpha_1,\ldots,\alpha_n$，并满足

\[
\alpha_j\ge 0,
\qquad
\sum_j\alpha_j=1,
\]

那么输出就是

\[
y=\sum_{j=1}^{n}\alpha_j v_j.
\]

单看这条式子，attention 只是 weighted average。真正关键的是权重 $\alpha_j$ 不是固定的，而是根据当前 query 与候选 key 动态计算。

## Query、Key、Value 的分工

标准 Transformer attention 把表示投影成三种角色：

- [Query](/deep-learning/transformer/attention/qkv/)：当前读取者提出的“查找条件”；
- Key：候选信息用于匹配 query 的部分；
- Value：真正被加权汇总的内容。

对一个 query $q_i$ 与每个 key $k_j$，先计算 score：

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}.
\]

再通过 [Softmax](/deep-learning/core/softmax/) 变成权重：

\[
\alpha_{ij}
=
\frac{e^{s_{ij}}}{\sum_l e^{s_{il}}}.
\]

最后：

\[
y_i=\sum_j\alpha_{ij}v_j.
\]

这就是 [Scaled Dot-Product Attention](/deep-learning/transformer/attention/scaled-dot-product-attention/) 的完整骨架。

## Attention 改变的是信息流

普通 position-wise linear layer 只能用当前位置自己的向量：

\[
y_i=f(x_i).
\]

Attention 则可以让 $y_i$ 直接依赖很多位置：

\[
y_i=f(x_i,x_1,\ldots,x_n).
\]

而且每个输入样本、每个 query 的读取权重都可以不同。

因此它最重要的能力不是“计算相似度”本身，而是建立**动态的信息连接**。

## Self-Attention 与 Cross-Attention

如果 Query、Key、Value 都来自同一组表示，就是 [Self-Attention](/deep-learning/transformer/attention/self-attention/)：

```text
X ──→ Q
X ──→ K
X ──→ V
```

如果 Query 来自一组表示，而 Key/Value 来自另一组 memory，就是 [Cross-Attention](/deep-learning/transformer/attention/cross-attention/)：

```text
queries ──→ Q
memory  ──→ K,V
```

两者使用同一类 attention 公式，区别在于信息从哪里来、流向哪里。

## Multi-Head Attention

单次 attention 只有一套 Q/K/V projections。[Multi-Head Attention](/deep-learning/transformer/attention/multi-head-attention/) 会并行学习多套 projections，让不同 heads 可以形成不同的信息读取模式，再把结果合并。

所以 Multi-Head 不是“把同一个 attention 算很多次取平均”，而是让多个子空间独立学习关系。

## 在 Transformer 中的位置

Transformer layer 中，attention 负责位置之间的信息交换，而 [Feed-Forward Network](/deep-learning/transformer/feed-forward-network/) 负责每个位置自己的表示变换。

```text
X
│
↓
Attention     ← 跨位置读取
│
↓
FFN           ← 每个位置独立变换
│
↓
H
```

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
