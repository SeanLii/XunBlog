---
title: "Self-Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/self-attention/"
prerequisites:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
related:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/transformer/attention/cross-attention/"
---

# Self-Attention

Self-Attention 是“同一组表示内部彼此读取信息”的 attention。

给定

\[
X=[x_1,x_2,\ldots,x_n],
\]

Q、K、V 都从同一个 $X$ 得到：

\[
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
\]

因此每个位置既可以作为读取者，也可以作为被读取的 memory。

## 单个位置的信息读取

以第 3 个位置为例：

```text
x1 ──┐
x2 ──┤
x3 ──┼──→ x3 的 Query 与所有 Keys 比较
x4 ──┤              │
x5 ──┘              ↓
               attention weights
                      │
                      ↓
               weighted Values
                      │
                      ↓
                     y3
```

输出 $y_3$ 因此可以包含来自 $x_1,x_2,x_4,x_5$ 的信息。

## Self-Attention 的 shape

若

\[
X\in\mathbb R^{n\times d},
\]

通常输出仍有 $n$ 个位置：

\[
Y\in\mathbb R^{n\times d}.
\]

数量没变，但每个位置从“局部表示”变成“contextual representation”。

## Self-Attention 与顺序

标准 self-attention 公式本身主要依赖内容关系。若没有 [Positional Encoding](/deep-learning/transformer/positional-encoding/) 或其他位置机制，交换输入位置会相应交换输出，并不会凭空得到序列顺序。

所以“Attention 能看全局”与“Attention 自动知道先后顺序”不是同一件事。

## 双向与因果 Self-Attention

如果所有位置都可以互相读取，就是 full / bidirectional self-attention。

如果位置 $i$ 只能读取 $j\le i$ 的内容，则通过 [Causal Mask](/deep-learning/transformer/causal-mask/) 得到 causal self-attention。

两者使用的 QKV 公式相同，区别是允许建立哪些连接。

## 在 Transformer Encoder 中

原始 Transformer encoder layer 使用 full self-attention，让每个 input position 读取整个输入序列。

在多层堆叠后，表示会反复更新：

```text
X
↓ self-attention
H1
↓ self-attention
H2
↓ ...
context-rich representations
```

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
