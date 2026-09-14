---
title: "Conditional Variational Autoencoder"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/conditional-variational-autoencoder/"
prerequisites:
  - "/generative-models/variational-autoencoder/"
related:
  - "/robot-learning/act/cvae-in-act/"
---

# Conditional Variational Autoencoder

Conditional Variational Autoencoder（CVAE）把 VAE 从“建模 $x$ 的分布”扩展到“在给定 condition $c$ 时，建模 output $y$ 的多种可能结果”。

目标是：

\[
p(y\mid c).
\]

如果同一个 condition 可能对应多个合理 outputs，只做 deterministic regression：

\[
y=f(c)
\]

往往会压成单一答案。

CVAE 引入 latent variable $z$：

\[
p(y,z\mid c)
=p(z\mid c)p(y\mid c,z).
\]

让 latent 表示 condition 没有完全决定的随机变化。

## 一个 Mental Model

```text
Condition c
    │
    ├───────────────┐
    │               │
    ↓               │
latent z            │
    │               │
    └──────┬────────┘
           ↓
     Decoder
           ↓
        output y
```

同一个 $c$，改变 sampled $z$，可以产生不同 plausible $y$。

## Training-Time Recognition Path

训练数据提供 pairs：

\[
(c,y).
\]

为了知道这个具体 output $y$ 对应什么 latent variation，recognition model 可以使用：

\[
q_\phi(z\mid c,y).
\]

也就是：已经知道 condition 和真实 output 后，推断这次 example 的 latent distribution。

然后 sample：

\[
z\sim q_\phi(z\mid c,y),
\]

decoder 学：

\[
p_\theta(y\mid c,z).
\]

## Generation 时没有真实 y

真正生成时，目标 $y$ 还不存在，所以不能使用：

\[
q(z\mid c,y).
\]

必须从 conditional prior：

\[
p_\theta(z\mid c)
\]

采样，或者某些模型使用固定 prior：

\[
p(z)=\mathcal N(0,I).
\]

然后：

\[
y\sim p_\theta(y\mid c,z).
\]

这就是 CVAE training 与 generation 最重要的结构差异。

## Conditional ELBO

目标 log-likelihood：

\[
\log p_\theta(y\mid c).
\]

其 ELBO：

\[
\log p_\theta(y\mid c)
\ge
\mathbb E_{q_\phi(z\mid c,y)}
[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
\left(
q_\phi(z\mid c,y)
\|p_\theta(z\mid c)
\right).
\]

第一项要求 latent + condition 能解释 target；第二项让 training-time recognition posterior 与 generation-time prior 对齐。

## Conditional Prior

通用 CVAE 并不要求：

\[
p(z)=\mathcal N(0,I).
\]

完全可以学习：

\[
p_\theta(z\mid c).
\]

这意味着不同 condition 可以拥有不同 latent distributions。

所以把某个具体模型使用 standard normal prior 的做法写成“CVAE 定义”是不准确的。

## Multimodality

设一个 condition $c$ 对应两种明显不同 outputs：

\[
y_A,
\qquad y_B.
\]

普通 MSE regression 可能产生：

\[
\hat y\approx\frac{y_A+y_B}{2},
\]

而这个平均结果本身可能并不合理。

CVAE 可以让不同 $z$ regions 对应不同 output modes，从而保留 one-to-many structure。

但是否真的学出清楚 modes，仍取决于 latent usage、decoder capacity 和 training dynamics。

## ACT 是一种特殊使用方式

ACT 用 CVAE-style latent 处理 demonstration style variation，但它的 condition、recognition input、prior choice 与 inference convention 都是 ACT-specific design choices。

因此完整机制应在 [CVAE in ACT](/robot-learning/act/cvae-in-act/) 解释；这些选择不能反过来定义一般 CVAE。

## Sources

- Sohn, Lee, Yan. *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS, 2015.
