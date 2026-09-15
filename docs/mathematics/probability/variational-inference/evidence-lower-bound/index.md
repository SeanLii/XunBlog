---
title: "Evidence Lower Bound"
kind: "canonical"
domain: "Mathematics / Probability / Variational Inference"
parent: "Variational Inference"
canonical: "/mathematics/probability/variational-inference/evidence-lower-bound/"
prerequisites:
  - "/mathematics/probability/variational-inference/"
  - "/mathematics/information-theory/kl-divergence/"
related:
  - "/generative-models/variational-autoencoder/"
---

# Evidence Lower Bound

Evidence Lower Bound（ELBO）是 marginal log-likelihood：

\[
\log p_\theta(x)
\]

的一个 lower bound，也是 Variational Inference 中最常用的 optimization objective。

设 latent-variable model：

\[
p_\theta(x,z),
\]

以及 variational distribution：

\[
q_\phi(z\mid x).
\]

ELBO 定义为：

\[
\mathcal L(\theta,\phi;x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right].
\]

## Jensen Derivation

从 evidence 开始：

\[
\log p_\theta(x)
=
\log
\int p_\theta(x,z)\,dz.
\]

乘除 $q_\phi(z\mid x)$：

\[
\log p_\theta(x)
=
\log
\int
q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\,dz.
\]

写成 expectation：

\[
\log p_\theta(x)
=
\log
\mathbb E_{q_\phi}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\]

由于 $\log$ concave，Jensen inequality 给出：

\[
\log p_\theta(x)
\ge
\mathbb E_{q_\phi}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right].
\]

右侧就是 ELBO。

## Posterior-KL Identity

ELBO 与 true posterior 的精确关系是：

\[
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{KL}(
q_\phi(z\mid x)
\|
p_\theta(z\mid x)
).
\]

因此：

\[
\mathcal L\le\log p_\theta(x).
\]

Bound 的 gap 恰好是 posterior KL。

当：

\[
q_\phi(z\mid x)=p_\theta(z\mid x),
\]

KL 为 0，ELBO 与 log evidence 相等。

## Likelihood–Prior Decomposition

若 generative model factorizes：

\[
p_\theta(x,z)
=
p_\theta(x\mid z)p(z),
\]

则 ELBO 可以写成：

\[
\boxed{
\mathcal L
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{KL}(
q_\phi(z\mid x)
\|
p(z)
)
}
\]

第一项是 expected log-likelihood；第二项约束 approximate posterior 与 prior 的差异。

在某些 VAE likelihood parameterization 下，第一项可以表现为 MSE 或 BCE-like reconstruction term，但“ELBO = reconstruction loss + KL”不是 ELBO 的一般定义。

## ELBO as Model-Learning Objective

若 $\theta$ 也参与优化：

\[
\max_{\theta,\phi}
\mathcal L(\theta,\phi;x),
\]

则：

- $\phi$ 改善 approximate posterior；
- $\theta$ 提高 model 对 data 的 likelihood。

因此 ELBO 同时服务 inference 与 generative-model learning。

## Conditional ELBO

若模型有 observed condition $c$：

\[
p_\theta(y,z\mid c)
=
p_\theta(y\mid c,z)p(z\mid c),
\]

则 conditional ELBO 为：

\[
\log p_\theta(y\mid c)
\ge
\mathbb E_{q_\phi(z\mid c,y)}
[
\log p_\theta(y\mid c,z)
]
-
D_{KL}
(q_\phi(z\mid c,y)\|p(z\mid c)).
\]

这是 CVAE 的基础 objective。

## Tightness

ELBO tightness 取决于：

\[
D_{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x)).
\]

若 variational family 过于简单，即使 optimization 完美，bound 也可能较松。

因此低 ELBO gap 需要：

1. variational family 有足够表达能力；
2. inference optimization / amortized encoder 能找到好的 approximation。

## Monte Carlo Estimation

Expected log-likelihood：

\[
\mathbb E_q[
\log p_\theta(x\mid z)
]
\]

通常使用 samples：

\[
z^{(l)}\sim q_\phi(z\mid x)
\]

估计：

\[
\frac1L
\sum_{l=1}^{L}
\log p_\theta(x\mid z^{(l)}).
\]

若 KL term 有 analytic form，可以精确计算；否则也可能需要 sampling estimator。

## Importance-Weighted Bounds

使用多个 importance samples 可以构造 tighter lower bound，例如 IWAE objective：

\[
\mathcal L_K
=
\mathbb E
\left[
\log
\frac1K
\sum_{k=1}^{K}
\frac{p(x,z_k)}{q(z_k\mid x)}
\right].
\]

随着 $K$ 增大，bound 通常可以更接近 $\log p(x)$。这说明 ELBO 并不是唯一 possible variational lower bound。

## Connections

- [Variational Inference](/mathematics/probability/variational-inference/)：ELBO 的 inference interpretation。
- [KL Divergence](/mathematics/information-theory/kl-divergence/)：ELBO gap 与 regularization term 的核心 quantity。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：用 neural networks 优化 ELBO。
