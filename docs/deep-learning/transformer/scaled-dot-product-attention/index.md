---
title: "Scaled Dot-Product Attention"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/scaled-dot-product-attention/"
prerequisites:
  - "/deep-learning/attention/qkv/"
  - "/mathematics/linear-algebra/dot-product/"
  - "/deep-learning/core/softmax/"
related:
  - "/deep-learning/transformer/multi-head-attention/"
---

# Scaled Dot-Product Attention

Scaled Dot-Product Attention 是 Transformer 中使用的具体 attention computation：

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
\]

这个公式可以拆成四步，每一步都有明确作用。

## 1. Pairwise Matching：$QK^\top$

设：

\[
Q\in\mathbb R^{N_q\times d_k},
\]

\[
K\in\mathbb R^{N_k\times d_k}.
\]

则：

\[
S=QK^\top
\in\mathbb R^{N_q\times N_k}.
\]

其中：

\[
S_{ij}=q_i^\top k_j.
\]

也就是说，一次 matrix multiplication 计算所有 query-key pairs 的 dot-product scores。

## 2. Scaling：除以 $\sqrt{d_k}$

Transformer 不直接把 $QK^\top$ 送入 Softmax，而是：

\[
\tilde S=\frac{S}{\sqrt{d_k}}.
\]

原因来自高维 dot product 的尺度。

假设 q/k 各维近似独立、均值 0、variance 1，那么：

\[
q^\top k
=\sum_{r=1}^{d_k}q_rk_r.
\]

在简化假设下，其 variance 随 $d_k$ 增长，大约为：

\[
\operatorname{Var}(q^\top k)\propto d_k.
\]

除以 $\sqrt{d_k}$ 后，variance 回到近似常数量级。

这样可以避免 dimension 大时 logits 绝对值过大，使 Softmax 过于尖锐、gradient 进入不理想区域。

## 3. Softmax：从 Scores 到 Reading Weights

对每个 query 的 row 做 Softmax：

\[
A_{i,:}
=
\operatorname{softmax}(\tilde S_{i,:}).
\]

于是：

\[
A_{ij}>0,
\qquad
\sum_j A_{ij}=1.
\]

每个 query 都得到一套对 keys 的 normalized weights。

注意 Softmax 是沿 key dimension 做，而不是把整个 matrix 一次归一化。

## 4. Weighted Value Aggregation

若：

\[
V\in\mathbb R^{N_k\times d_v},
\]

则：

\[
O=AV
\in\mathbb R^{N_q\times d_v}.
\]

第 $i$ 个 output：

\[
o_i=
\sum_{j=1}^{N_k}A_{ij}v_j.
\]

因此 scores 决定读取权重，被加权汇总的内容来自 Values。

## Mask 加在哪里

如果某些 query-key pairs 不允许连接，常在 Softmax 前给对应 logit 加一个极大负数：

\[
\tilde S'=\tilde S+M.
\]

对 masked 位置：

\[
M_{ij}=-\infty.
\]

于是：

\[
\operatorname{softmax}(-\infty)=0.
\]

[Causal Mask](/deep-learning/sequence-modeling/causal-mask/) 与 padding masks 都可通过这个机制进入 attention。

## Shape 总结

```text
Q: [Nq, dk]
K: [Nk, dk]
V: [Nk, dv]

QKᵀ            → [Nq, Nk]
÷ sqrt(dk)     → [Nq, Nk]
softmax        → [Nq, Nk]
× V            → [Nq, dv]
```

这个 shape flow 是理解 Self-Attention、Cross-Attention 和 Multi-Head Attention 的共同基础。

## Sources

- Vaswani et al. *Attention Is All You Need*. 2017.
