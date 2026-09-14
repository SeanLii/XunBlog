---
title: "Variational Inference"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/variational-inference/"
prerequisites:
  - "/mathematics/probability/bayes-theorem/"
  - "/mathematics/information-theory/kl-divergence/"
  - "/mathematics/probability/latent-variable/"
related:
  - "/mathematics/probability/variational-inference/evidence-lower-bound/"
  - "/generative-models/variational-autoencoder/"
---

# Variational Inference

Variational Inference（VI）是一类 approximate inference 方法：当真实 posterior 难以直接计算时，用一个更容易处理的 distribution $q$ 去近似它，并把“求 posterior”转化成 optimization problem。

目标 posterior：

\[
p(z\mid x).
\]

选择 tractable family：

\[
q_\phi(z).
\]

然后调整 $\phi$，让 $q_\phi$ 尽量接近 posterior。

## Inference Problem 从哪里来

Latent-variable model：

\[
p(x,z)=p(z)p(x\mid z).
\]

Bayes' theorem：

\[
p(z\mid x)
=\frac{p(x,z)}{p(x)}.
\]

而：

\[
p(x)=\int p(x,z)\,dz.
\]

如果这个 integral 高维、nonlinear 或没有 closed form，真实 posterior 就难以直接计算。

VI 的核心不是改变 generative model，而是构造一个可计算 approximation。

## Variational Family

先规定 $q$ 来自某个 family：

\[
\mathcal Q=\{q_\phi(z)\}.
\]

例如：

- diagonal Gaussian；
- full-covariance Gaussian；
- mean-field factorization；
- normalizing flow transformed distribution。

然后求：

\[
q^*(z)
=
\arg\min_{q\in\mathcal Q}
D_{KL}(q(z)\|p(z\mid x)).
\]

所以 approximation quality 受到 variational family capacity 限制。

## Mean-Field Approximation

经典 VI 常假设 latent dimensions / groups factorize：

\[
q(z)=\prod_{j=1}^{m}q_j(z_j).
\]

这叫 mean-field approximation。

它让 optimization 更 tractable，但也可能无法表达真实 posterior 中复杂 dependencies。

因此 variational inference 的误差不仅来自 optimization 没做好，也可能来自 family 本身太受限。

## Posterior KL 不能直接算

目标：

\[
D_{KL}(q(z)\|p(z\mid x))
\]

里面仍然包含 posterior：

\[
p(z\mid x)=\frac{p(x,z)}{p(x)}.
\]

而 $p(x)$ 正是我们难算的 evidence。

所以 VI 需要进一步把 objective 改写成可以优化的 [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/)。

关键恒等式：

\[
\log p(x)
=
\operatorname{ELBO}(q)
+
D_{KL}(q(z)\|p(z\mid x)).
\]

因为左边对 $q$ 固定，maximizing ELBO 等价于 minimizing posterior KL。

## Optimization View

这一步非常重要：VI 把 probabilistic inference 转成 numerical optimization。

原问题：

> 算出整个 posterior function。

变成：

> 在可处理的 distribution family 里找一组 parameters，使 ELBO 最大。

于是可以使用 gradient-based optimization、coordinate ascent 等方法。

## Amortized Variational Inference

经典 VI 对每个 observation $x_n$ 单独优化一组 variational parameters：

\[
\phi_n.
\]

如果 dataset 很大，每来一个新 sample 都重新 optimization 很昂贵。

Amortized inference 使用一个 neural network：

\[
\phi=f_\psi(x).
\]

一次前向就产生 approximate posterior parameters：

\[
q_\psi(z\mid x).
\]

训练成本被 amortize 到整个 dataset。

VAE encoder 就是典型 amortized inference network。

## VI 与 VAE 的关系

[Variational Autoencoder](/generative-models/variational-autoencoder/) 把三件事组合起来：

1. deep latent-variable generative model；
2. amortized variational inference network；
3. reparameterized stochastic gradient optimization of ELBO。

所以 VI 不是“VAE 的一个 loss trick”。VAE 是 VI 在 deep latent-variable modeling 中非常重要的一种实现框架。

## Sources

- Blei, Kucukelbir, McAuliffe. *Variational Inference: A Review for Statisticians*. JASA, 2017.
- Kingma & Welling. *An Introduction to Variational Autoencoders*. 2019.
