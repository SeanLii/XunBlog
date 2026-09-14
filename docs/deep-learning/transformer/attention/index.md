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

Attention 是一种从一组候选信息中按当前需求计算权重并汇总信息的机制。它不是某一个固定网络结构，而是一类计算方式；Transformer 使用的是其中的 scaled dot-product attention。

## 定义

可以把 attention 写成三个角色：一个 query，以及一组 key-value pairs。对第 $i$ 个候选项，先由 query $q$ 与 key $k_i$ 计算 compatibility score：

\[
s_i=f(q,k_i).
\]

再把 scores 转换成归一化权重 $\alpha_i$，最后对 values 做加权和：

\[
\operatorname{Attention}(q,K,V)
=\sum_i\alpha_i v_i.
\]

权重满足

\[
\alpha_i\ge0,
\qquad
\sum_i\alpha_i=1
\]

时，输出可以看成 value vectors 的加权组合。

## Query、Key 与 Value 的分工

query 表示当前正在寻找什么；key 表示每个候选项用来参与匹配的特征；value 是匹配完成后真正被聚合的信息。三者通常由模型学习得到，而不是人工赋予语义。

因此 attention 的核心不是“计算向量相似度”本身，而是两步：

1. 用 query-key compatibility 决定该看哪些位置；
2. 用这些权重汇总对应的 value。

[Query / Key / Value](/deep-learning/transformer/attention/qkv/) 会进一步说明为什么需要把“匹配依据”和“被取出的内容”分开。

## Attention 与固定加权

如果权重是预先固定的，那么所有输入都会以同样方式组合。Attention 的权重由当前输入动态计算，因此同一个 value 在不同 query 下可以得到完全不同的权重。

## Transformer 中的形式

Transformer 把多个 queries、keys、values 堆成矩阵，并采用

\[
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

这个公式同时完成大量 query-key 匹配，因此非常适合并行矩阵运算。它的完整数学结构属于 [Scaled Dot-Product Attention](/deep-learning/transformer/attention/scaled-dot-product-attention/)。

## Sources

- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
