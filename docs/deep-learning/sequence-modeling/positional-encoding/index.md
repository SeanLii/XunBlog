---
title: "Positional Encoding"
kind: "canonical"
domain: "Deep Learning / Sequence Modeling"
parent: "Deep Learning"
canonical: "/deep-learning/sequence-modeling/positional-encoding/"
prerequisites:
  - "/deep-learning/core/embedding/"
related:
  - "/deep-learning/transformer/"
---

# Positional Encoding

Self-Attention 根据 content vectors 建立 pairwise interactions，但 vanilla self-attention 本身不包含 sequence order。

因此 Transformer 需要额外把 position information 注入 representation。

Positional Encoding 是这一类机制的统称。

显式 position representation 早于 Transformer；Transformer 采用了 sinusoidal positional encoding，并同时比较了 learned positional embeddings。

## Position Information 的必要性

考虑 tokens：

```text
A B C
```

和：

```text
C B A
```

在缺少 position information 时，self-attention 只能接收到同一组 content vectors 的不同 permutation。

模型无法凭空知道：

- 谁在第 1 位；
- 谁在第 3 位；
- 两个 tokens 相距几步。

因此需要 representation 同时包含：

```text
content information
+
position information
```

## Sinusoidal Positional Encoding

原始 Transformer 使用固定 sinusoidal encoding：

\[
PE(pos,2i)
=\sin\left(
\frac{pos}{10000^{2i/d_{model}}}
\right),
\]

\[
PE(pos,2i+1)
=\cos\left(
\frac{pos}{10000^{2i/d_{model}}}
\right).
\]

不同 dimensions 使用不同 frequencies。

然后：

\[
x_{pos}
=e_{token}+PE(pos).
\]

所以 attention 看到的 input vector 已经同时带有 content 与 position。

## Multi-Frequency Representation

低频 dimensions 随 position 缓慢变化，适合表达较大尺度位置；高频 dimensions 变化更快，提供更细粒度区分。

多组 sinusoidal signals 合在一起，使不同 positions 得到独特 pattern。

原论文还指出，固定 sinusoidal representation 允许模型通过线性关系访问某些 relative offsets。

## Learned Positional Embedding

另一种简单方法是为每个 position 直接学习一个 embedding table：

\[
P\in\mathbb R^{N_{max}\times d}.
\]

position $i$ 使用：

\[
p_i=P_{i,:}.
\]

再：

\[
x_i=e_i+p_i.
\]

优点是灵活；限制是通常绑定 maximum trained positions，并且 extrapolation behavior 取决于具体方法。

## Absolute 与 Relative Position

Absolute position 告诉模型：

> “这个 token 位于位置 17。”

Relative position 更关注：

> “这个 key 相对 query 在前面 3 个位置。”

后续 Transformer family 中出现 relative position bias、rotary position embedding（RoPE）等方法，它们把 relative geometry 更直接地放入 attention computation。

因此 positional encoding 不是只有 sin/cos 一种实现。

## 2D Position

Vision Transformer 中，一个 token 对应 image patch。

除了 sequence index，真实结构还有 row / column spatial position。

可以：

- 把 patches flatten 后学习一维 position table；
- 使用二维 position decomposition；
- 使用 relative spatial bias。

关键仍然是让模型区分“内容相同但位置不同”的 patches。

## Position 与 Attention 是两层职责

Attention 负责：

> 根据 representations 计算谁应该读取谁。

Position mechanism 负责：

> 让 representation / score computation 知道这些 items 在结构中的位置关系。

两者互相配合，但不是同一个概念。

## Sources

- Gehring et al. *Convolutional Sequence to Sequence Learning*. 2017.
- Vaswani et al. *Attention Is All You Need*. 2017.
- Dosovitskiy et al. *An Image Is Worth 16×16 Words*. 2021.
