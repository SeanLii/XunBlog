---
title: "Posterior Collapse"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/posterior-collapse/"
prerequisites:
  - "/generative-models/variational-autoencoder/"
  - "/mathematics/information-theory/kl-divergence/"
related:
  - "/robot-learning/act/cvae-in-act/"
---

# Posterior Collapse

Posterior Collapse 指 VAE 类模型训练后，approximate posterior 退化得接近 prior，使 latent $z$ 对 decoder 几乎不再携带输入相关信息。

## 典型状态

理想情况下，不同数据 $x$ 可以产生不同的

\[
q_\phi(z|x).
\]

Posterior collapse 时可能出现

\[
q_\phi(z|x)\approx p(z)
\]

对大量 $x$ 都成立。于是 $z$ 的分布几乎不再依赖输入。

对应的 KL term 会接近 0：

\[
D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))\approx0.
\]

## 形成条件

ELBO 同时包含 reconstruction/log-likelihood 与 KL regularization。若 decoder 足够强，即使忽略 $z$ 也能很好预测数据，那么降低 KL 最简单的方式之一就是让 posterior 靠近 prior。

因此问题不是“KL 本身错误”，而是模型可能找到一种高 ELBO 解：decoder 承担大部分建模工作，latent channel 被闲置。

## 信息角度

当 $q_\phi(z|x)$ 对不同 $x$ 几乎相同，看到 $z$ 就很难判断它来自哪个输入。换句话说，$z$ 携带的 input-specific information 很少。

## 与 Beta 的关系

把 KL term 乘以更大的权重会更强地推动 posterior 接近 prior，通常会减少通过 latent channel 传递的信息。但实际是否发生 posterior collapse 还取决于 decoder capacity、训练过程、数据和其他设计，不能只由一个 $\beta$ 数值单独判断。

ACT 论文说明更高的 $\beta$ 会使 $z$ 传递更少信息，但并没有把 ACT 的设计问题直接等同于 posterior collapse。因此在 ACT 页面中只把这里作为相关生成模型概念，而不把它当作论文明确报告的失败模式。

## Sources

- [Generating Sentences from a Continuous Space — Bowman et al., 2015](https://arxiv.org/abs/1511.06349)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
