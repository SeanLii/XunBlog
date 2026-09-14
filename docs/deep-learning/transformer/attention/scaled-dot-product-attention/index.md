---
title: "Scaled Dot-Product Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/scaled-dot-product-attention/"
prerequisites:
  - "/deep-learning/transformer/attention/qkv/"
  - "/mathematics/linear-algebra/dot-product/"
  - "/deep-learning/core/softmax/"
related:
  - "/deep-learning/transformer/attention/multi-head-attention/"
---

# Scaled Dot-Product Attention

Scaled Dot-Product Attention 是 Transformer 中最基本的一次 attention 计算：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

理解这条式子时，最好按数据流从左到右走，而不是一次记住整条公式。

## 第一步：QKᵀ 产生匹配分数

设

\[
Q\in\mathbb R^{n_q\times d_k},
\qquad
K\in\mathbb R^{n_k\times d_k}.
\]

矩阵乘法得到

\[
QK^\top\in\mathbb R^{n_q\times n_k}.
\]

其中元素

\[
(QK^\top)_{ij}=q_i^\top k_j
\]

表示 query $i$ 与 key $j$ 的 dot-product score。

每一行对应“一个 query 看所有 keys”。

## 第二步：除以 √d_k

如果 $q_i$ 与 $k_j$ 的各维分量近似独立、均值为 0、方差为 1，那么 dot product

\[
q_i^\top k_j=\sum_{r=1}^{d_k}q_{ir}k_{jr}
\]

的方差会随 $d_k$ 增大到大约 $d_k$ 的量级。

因此维度越高，score 的绝对值更容易变大。进入 softmax 后，过大的 logits 会让分布非常尖锐，gradient 变小。

Transformer 用

\[
\frac{1}{\sqrt{d_k}}
\]

把 score 的尺度拉回更稳定的范围。

这就是 “scaled” 的来源。

## 第三步：Softmax 变成读取权重

对每个 query 的整行 scores 做 softmax：

\[
A_{ij}
=
\frac{
\exp(S_{ij})
}{
\sum_l\exp(S_{il})
}.
\]

于是每一行满足

\[
\sum_j A_{ij}=1.
\]

可以把 $A_{ij}$ 看成 query $i$ 从 value $j$ 读取多少比例的信息。

## 第四步：AV 汇总 Values

\[
Y=AV.
\]

第 $i$ 行为

\[
y_i=\sum_jA_{ij}v_j.
\]

这一步才真正把信息拿回来。

所以完整计算可以压缩成：

```text
Q × K^T
  ↓
matching scores
  ↓ divide √d_k
scaled scores
  ↓ softmax
attention weights
  ↓ × V
weighted information
```

## Mask 加在哪里

某些位置不允许被读取时，可以在 softmax 前对对应 score 加上一个极大的负数：

\[
S'_{ij}=S_{ij}+M_{ij}.
\]

若 $M_{ij}=-\infty$，softmax 后该位置权重变成 0。

[Causal Mask](/deep-learning/transformer/causal-mask/) 与 padding mask 都可以通过这个机制实现，但它们表达的语义不同。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
