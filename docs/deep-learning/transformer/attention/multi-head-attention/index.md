---
title: "Multi-Head Attention"
kind: "canonical"
domain: "Deep Learning / Transformer / Attention"
parent: "Attention"
canonical: "/deep-learning/transformer/attention/multi-head-attention/"
prerequisites:
  - "/deep-learning/transformer/attention/scaled-dot-product-attention/"
related:
  - "/deep-learning/transformer/"
---

# Multi-Head Attention

Multi-Head Attention 把模型维度分成多个 attention heads。每个 head 拥有自己的 $Q,K,V$ projections，可以在不同的投影子空间中独立计算 attention，再把结果拼接起来。

## 定义

第 $h$ 个 head 为

\[
\operatorname{head}_h
=
\operatorname{Attention}
(QW_h^Q,KW_h^K,VW_h^V).
\]

所有 heads 的输出沿特征维拼接：

\[
H=\operatorname{Concat}
(\operatorname{head}_1,\ldots,\operatorname{head}_H),
\]

再通过输出投影

\[
\operatorname{MultiHead}(Q,K,V)=HW^O.
\]

原始 Transformer 中通常令每个 head 的维度约为

\[
d_k=d_v=\frac{d_{model}}{H}.
\]

因此增加 head 数并不必然把总 hidden dimension 乘上 $H$；常见做法是把固定的 $d_{model}$ 分给多个 heads。

## 多个 Head 带来的结构

如果只有一个 head，所有匹配关系都必须在同一个投影空间中完成。多个 heads 允许模型学习多组不同的 compatibility functions 与 value projections。

不能预先断言“head 1 一定负责位置、head 2 一定负责语义”。某些 head 可能在训练后表现出可解释模式，但 head 的功能是数据和目标共同学习出来的。

## Shape

设 batch size 为 $B$、sequence length 为 $N$、model dimension 为 $d_{model}$、head 数为 $H$。常见实现会把

\[
(B,N,d_{model})
\]

reshape 成

\[
(B,H,N,d_k),
\]

分别计算各 head 的 attention，然后再合并回

\[
(B,N,d_{model}).
\]

ACT 官方配置使用 8 个 attention heads。这个数值属于具体实现超参数，不是 Multi-Head Attention 定义的一部分。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
