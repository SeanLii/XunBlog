---
title: "Query / Key / Value"
kind: "canonical"
domain: "Deep Learning / Attention"
parent: "Attention"
canonical: "/deep-learning/attention/qkv/"
prerequisites:
  - "/deep-learning/attention/"
related:
  - "/deep-learning/transformer/scaled-dot-product-attention/"
---

# Query / Key / Value

Query、Key、Value 是 Attention 中同一份信息承担的三种**计算角色**。

它们不是三类固定语义向量，也不是“Query 就是问题句、Key 就是关键词、Value 就是答案”。更准确地说：

- Query 决定当前输出想读取什么；
- Key 决定每个候选信息怎样被匹配；
- Value 是匹配完成后真正被汇总的内容。

Query–key–value 的检索结构早于 Transformer；Transformer 将这套角色与 learned projections、dot-product attention 和 multi-head computation 组合成了今天最常见的形式。

## 从一个 Retrieval System 理解

假设 memory 中有三项：

```text
Key k1 → Value v1
Key k2 → Value v2
Key k3 → Value v3
```

给一个 query $q$，先计算：

\[
s_i=q^\top k_i.
\]

scores 越大，表示这个 query 与对应 key 在 learned matching space 中越匹配。

再转换为 weights：

\[
\alpha_i=\operatorname{softmax}(s)_i.
\]

最后读取：

\[
o=\sum_i\alpha_i v_i.
\]

所以 Key 决定“是否应该被读”，Value 决定“被读以后提供什么”。

## Q、K、V 往往来自 Learned Projections

对 input representation：

\[
x_i\in\mathbb R^{d_{model}},
\]

常通过三个不同 Linear Layers：

\[
q_i=x_iW_Q,
\]

\[
k_i=x_iW_K,
\]

\[
v_i=x_iW_V.
\]

矩阵形式：

\[
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
\]

虽然都来自同一个 $X$，parameters 不同，所以它们会学习成适合不同角色的 representation spaces。

## Q、K、V 的独立 Learned Roles

理论上可以构造不带独立 projections 的 attention，但独立 Q/K/V projections 给模型更多自由度。

一个 token 可以：

- 通过 Query 表达“当前需要寻找的关系”；
- 通过 Key 表达“我在什么条件下应该被别人读取”；
- 通过 Value 表达“如果被读取，我应该贡献什么内容”。

这三个角色不要求编码同样的 features。

## 一个 Query 怎样读取所有 Keys

单个 query：

\[
q\in\mathbb R^{d_k}.
\]

keys：

\[
K=
\begin{bmatrix}
k_1^\top\\
k_2^\top\\
\vdots\\
k_N^\top
\end{bmatrix}
\in\mathbb R^{N\times d_k}.
\]

scores：

\[
s=qK^\top
\in\mathbb R^N.
\]

得到 N 个 matching scores 后，softmax 得到 N 个 weights，再：

\[
o=\alpha V.
\]

因此一个 query 产生一个 aggregated output vector。

## 多个 Queries 的 Matrix Form

如果：

\[
Q\in\mathbb R^{N_q\times d_k},
\quad
K\in\mathbb R^{N_k\times d_k},
\]

则：

\[
QK^\top
\in\mathbb R^{N_q\times N_k}.
\]

第 $(i,j)$ 个元素：

\[
q_i^\top k_j.
\]

每一 row 对应一个 query 对所有 keys 的 matching scores。

## Key–Value Pairing

每个 key 对应一个 value：

```text
k1 ↔ v1
k2 ↔ v2
...
kN ↔ vN
```

score 是针对 key 算出来的，但 weight 最终作用到对应 value。

因此通常：

\[
N_k=N_v.
\]

但 query 数量可以不同：

\[
N_q\ne N_k.
\]

这在 Cross-Attention 中尤其重要。

## Self-Attention 与 Cross-Attention 中的 QKV

Self-Attention：

\[
Q,K,V
\]

都来自同一 sequence。

Cross-Attention：

- Q 来自 target / decoder / query set；
- K,V 来自 source / encoder memory。

因此 QKV 是统一的计算角色，而不是绑定某一种 architecture 的词汇。

## QKV 与 Learnable Query Embedding 的区别

[Object Query](/deep-learning/detr/object-query/) 是某个 learnable input vector / slot。

而 Q 矩阵是 attention computation 中经过 projection 后承担 query role 的 vectors。

一个 learnable query embedding 进入 decoder 后可以被投影成 Q，但两者不能直接当作同义词。

## Sources

- Miller et al. *Key-Value Memory Networks for Directly Reading Documents*. 2016.
- Vaswani et al. *Attention Is All You Need*. 2017.
