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

Multi-Head Attention 把一次 attention 拆成多个并行 heads。每个 head 有自己的一组 Q/K/V projections，因此可以在不同 learned subspaces 中建立不同关系。

## 从单头到多头

单头 attention：

\[
\operatorname{Attn}(Q,K,V).
\]

多头结构为每个 head $h$ 使用不同参数：

\[
Q_h=XW_Q^{(h)},
\quad
K_h=XW_K^{(h)},
\quad
V_h=XW_V^{(h)}.
\]

然后：

\[
head_h=
\operatorname{Attention}(Q_h,K_h,V_h).
\]

所有 heads 拼接：

\[
H=\operatorname{Concat}(head_1,\ldots,head_H),
\]

最后再做输出 projection：

\[
Y=HW_O.
\]

## Dimension 怎样分配

原始 Transformer 常把 model dimension $d_{model}$ 分给 $H$ 个 heads：

\[
d_k=d_v=\frac{d_{model}}{H}.
\]

例如

\[
d_{model}=512,
\quad H=8,
\]

则每个 head 的 key/query/value dimension 通常为 64。

拼接 8 个 heads 后又回到 512 维。

## 多个 heads 带来的不是简单重复

如果所有 heads 使用完全相同 projections，它们没有增加有意义的多样性。真正的作用来自每个 head 有独立 learned matrices。

因此某些 heads 可以更关注局部关系，另一些关注远距离依赖；在视觉或机器人场景中，也可能形成不同空间区域或状态变量之间的读取模式。

这些语义不是人为固定的，只是模型可能通过训练形成的功能分工。

## Multi-Head 与并行输出不同

“100 个 action queries”与“8 个 attention heads”是两个不同维度。

- action queries：决定有多少 output slots；
- attention heads：决定每个 slot 在一次 attention 中有多少套并行关系建模。

ACT 可以同时有 100 queries 和 8 heads，两者不能互相替代。

## Sources

- Vaswani et al., **Attention Is All You Need**, 2017. https://arxiv.org/abs/1706.03762
