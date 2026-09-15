---
title: "Posterior Collapse"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/posterior-collapse/"
prerequisites:
  - "/generative-models/variational-autoencoder/"
related:
---

# Posterior Collapse

Posterior Collapse 是 latent-variable models，尤其 VAE 中的一种 failure mode：模型学会几乎不使用 latent variable $z$。

典型现象：

\[
q_\phi(z\mid x)
\approx p(z).
\]

如果 approximate posterior 几乎与 prior 一样，那么看到不同 $x$ 后，encoder 给出的 latent distribution 也几乎不变。

于是 $z$ 不再携带多少 input-specific information。

## ELBO 中的 Collapse Pressure

VAE ELBO：

\[
\mathcal L
=
\mathbb E_{q(z\mid x)}
[
\log p(x\mid z)
]
-
D_{KL}(q(z\mid x)\|p(z)).
\]

如果：

\[
q(z\mid x)=p(z),
\]

那么：

\[
D_{KL}=0.
\]

这对 objective 的 KL term 来说是最省代价的状态。

如果 decoder 足够强，即使忽略 $z$ 也能很好预测 $x$，模型就可能没有动力让 latent 承担 information。

## Decoder Bypass

例如 autoregressive text decoder 可以利用：

\[
p(x_t\mid x_{<t},z).
\]

如果过去 tokens 已经足以预测下一个 token，decoder 可以让：

\[
p(x_t\mid x_{<t},z)
\approx p(x_t\mid x_{<t}),
\]

从而几乎不依赖 $z$。

这时 KL 还能降到接近 0，所以 collapse 成为 objective 的可行 solution。

## Collapse 不等于“KL 小就是坏”

如果某些 data points 本来不需要大量 latent information，较小 KL 不一定有问题。

直接的诊断包括：

- $q(z\mid x)$ 是否随 $x$ 有 meaningful variation；
- decoder output 是否显著依赖 $z$；
- latent 是否携带 task 需要的 information。

所以不能只看一个 scalar KL 数值就判定 collapse。

## 与 Mutual Information 的联系

在 aggregate sense 上，如果 $z$ 与 $x$ 几乎 independent：

\[
I(X;Z)\approx0,
\]

说明 latent 没有保留 observation information。

这提供了比“KL 是否为零”更概念化的理解。

## 常见缓解方向

不同工作采用不同策略，例如：

- KL annealing：训练早期降低 KL pressure；
- free bits / minimum rate；
- 限制 decoder capacity；
- 更强 inference optimization；
- 修改 objective；
- 改善 encoder / decoder training balance。

这些方法针对的机制不同，没有一个适用于所有 VAE 的通用 fix。

## Lagging Inference Network 视角

He 等 2019 分析了 inference network 在训练早期跟不上 changing model posterior 的问题：如果 approximate posterior 学得太慢，generator 会被推向一个更容易忽略 latent 的 solution。

这说明 posterior collapse 不只是“KL coefficient 太大”，还与 optimization dynamics 有关。

## Conditional Models

CVAE 也会 collapse。

如果 condition $c$ 已经足以预测 $y$，decoder 可能忽略 $z$：

\[
p(y\mid c,z)
\approx p(y\mid c).
\]

所以“加入 latent 就会自动学到多样性”并不成立。

## Sources

- Bowman et al. *Generating Sentences from a Continuous Space*. 2016.
- He et al. *Lagging Inference Networks and Posterior Collapse in Variational Autoencoders*. ICLR, 2019.
