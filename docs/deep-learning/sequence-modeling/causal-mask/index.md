---
title: "Causal Mask"
kind: "canonical"
domain: "Deep Learning / Sequence Modeling"
parent: "Deep Learning"
canonical: "/deep-learning/sequence-modeling/causal-mask/"
prerequisites:
  - "/deep-learning/attention/"
related:
  - "/deep-learning/transformer/transformer-decoder/"
---

# Causal Mask

Causal Mask 限制 sequence 中 position $i$ 只能读取自己以及过去位置，不能读取未来位置。

如果 sequence positions 为：

\[
1,2,3,4,
\]

则允许关系：

```text
1 → 1
2 → 1,2
3 → 1,2,3
4 → 1,2,3,4
```

它的目的不是“让 attention 更稀疏”，而是维持 autoregressive factorization：预测当前位置时不能使用未来 ground-truth information。

因果约束早于 Transformer；Transformer Decoder 只是把同样的 autoregressive requirement 实现成对 query–key pairs 的 attention mask。

## Mask Matrix

可定义：

\[
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i.
\end{cases}
\]

例如四 positions：

\[
M=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}.
\]

Attention logits：

\[
S=\frac{QK^\top}{\sqrt{d_k}}.
\]

加入 mask：

\[
S'=S+M.
\]

再 Softmax。

被 mask 位置：

\[
e^{-\infty}=0,
\]

所以最终 reading weight 为 0。

## Causal Constraint 下的 Parallel Training

Autoregressive generation 看起来必须一个 token 一个 token 地生成。

但 training 时完整 target sequence 已知，可以一次性构造整个 causal mask：

\[
N\times N.
\]

所有 positions 的 attention 可以并行计算，只是每个 row 的 allowed keys 不同。

因此 causal dependency 不等于 training 计算必须 serial。

## Causal Mask 与 Padding Mask

Padding mask 解决的是：

> 不要读取 padding positions。

Causal mask 解决的是：

> 不要读取未来 positions。

两种 mask 可以同时存在，并在 attention logits 上组合。

## Prefix / Blockwise Masks

更复杂模型可能允许某一段 tokens 彼此双向可见，而另一段保持 causal。

例如 prefix-LM 或 multimodal blockwise attention。

这些都可以看成同一思想的推广：

> 用 mask 显式定义哪些 query-key connections 合法。

Causal Mask 因此是 attention connectivity constraint，而不是 Decoder 的同义词。

## Sources

- van den Oord et al. *Pixel Recurrent Neural Networks*. 2016.
- van den Oord et al. *WaveNet: A Generative Model for Raw Audio*. 2016.
- Vaswani et al. *Attention Is All You Need*. 2017.
