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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Variational Inference

Variational Inference（VI）是一类 approximate inference 方法。它把难以直接计算的 posterior inference 转化为 optimization problem：选择一个 tractable distribution family $q_\phi(z)$，并寻找其中最接近目标 posterior 的成员。

在 latent-variable model：

\[
p_\theta(x,z)=p_\theta(x\mid z)p(z)
\]

中，目标 posterior 为：

\[
p_\theta(z\mid x)
=
\frac{p_\theta(x,z)}{p_\theta(x)}.
\]

困难通常来自 marginal likelihood：

\[
p_\theta(x)
=
\int p_\theta(x,z)\,dz.
\]

当这个积分难以精确计算时，posterior 也难以直接归一化。

## Variational Family

VI 先选择一族可处理 distributions：

\[
\mathcal Q
=
\{q_\phi(z):\phi\in\Phi\}.
\]

然后寻找：

\[
q_{\phi^*}
=
\arg\min_{q_\phi\in\mathcal Q}
D_{KL}(
q_\phi(z)
\|
p_\theta(z\mid x)
).
\]

这里同时存在两类 approximation error：

1. variational family 可能无法表示 true posterior；
2. optimization 可能没有找到 family 中的最佳 solution。

## ELBO Derivation

Posterior KL：

\[
D_{KL}(q_\phi(z)\|p_\theta(z\mid x))
\]

可以展开为：

\[
\mathbb E_q
\left[
\log q_\phi(z)
-
\log p_\theta(z,x)
+
\log p_\theta(x)
\right].
\]

由于 $\log p_\theta(x)$ 与 $z$ 无关：

\[
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{KL}(
q_\phi(z)
\|
p_\theta(z\mid x)
),
\]

其中：

\[
\mathcal L(\theta,\phi;x)
=
\mathbb E_{q_\phi(z)}
[
\log p_\theta(x,z)-
\log q_\phi(z)
]
\]

称为 [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/)。

因为 KL non-negative：

\[
\mathcal L(\theta,\phi;x)
\le
\log p_\theta(x).
\]

Maximum ELBO 等价于在当前 variational family 中 minimum posterior KL。

## Mean-Field Approximation

经典 VI 常采用 factorized family：

\[
q(z_1,\ldots,z_m)
=
\prod_{j=1}^{m}q_j(z_j).
\]

这称为 mean-field approximation。

它降低计算复杂度，但会忽略 posterior variables 之间某些 dependencies。

在 conjugate exponential-family models 中，可以推导 coordinate-ascent updates：

\[
\log q_j^*(z_j)
=
\mathbb E_{q_{-j}}
[
\log p(x,z)
]
+
\text{const}.
\]

## Stochastic Variational Inference

大数据下，全 dataset ELBO 每次都精确计算成本很高。Stochastic Variational Inference 使用 minibatch 估计 global objective gradient，使 VI 可以扩展到大规模 datasets。

现代 neural variational inference 进一步使用 stochastic gradient optimization 直接训练 neural parameters。

## Amortized Inference

传统 VI 对每个 observation $x_i$ 单独优化一组 variational parameters：

\[
\phi_i.
\]

Amortized inference 改为学习 inference network：

\[
\phi(x)=f_\psi(x),
\]

直接输出 approximate posterior parameters：

\[
q_\psi(z\mid x).
\]

这样新 observation 不需要从头执行 optimization。

VAE 的 encoder 就是 amortized inference network。

## Amortization Gap

Amortized inference 的 approximate posterior 可能比“对当前 sample 单独优化到最好”的 variational solution 更差。

可将 inference error 区分为：

- **approximation gap**：variational family 本身的限制；
- **amortization gap**：共享 inference network 没有达到该 family 对当前 sample 的最优解。

这一区分有助于分析 VAE encoder 的 inference quality。

## Gradient Estimation

VI objective 中常包含：

\[
\mathbb E_{q_\phi(z)}[f(z)].
\]

当 $z$ 是 stochastic sample 时，如何对 $\phi$ 求 gradient 是核心问题。

常见方法包括：

- score-function estimator；
- pathwise / reparameterization gradient；
- discrete relaxations；
- analytic expectations（如果可得）。

[Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) 是 continuous latent neural models 中最常用的方法之一。

## Reverse KL Behavior

VI 常优化：

\[
D_{KL}(q\|p).
\]

当 target posterior multimodal、但 variational family 很受限时，reverse KL 可能让 $q$ 集中在某个 mode，而不是平均覆盖所有 modes。

这种 behavior 来自 KL direction 与 variational family 的共同作用，而不是 VI 的普遍“缺陷定义”。

## Joint Learning of Model and Inference

在 generative modeling 中，往往同时优化：

- generative parameters $\theta$；
- variational parameters $\phi$。

Objective：

\[
\max_{\theta,\phi}
\mathcal L(\theta,\phi;x).
\]

因此 VI 不只用于“在固定 model 中做 inference”，也可以与 model learning 联合进行。VAE 正是这种 setting。

## 由机制产生的边界

VI 的主要限制包括：

- variational family misspecification；
- posterior dependency 被过度简化；
- local optima / optimization difficulty；
- amortization gap；
- KL direction 带来的 approximation bias；
- ELBO 与 downstream posterior quality 并不完全等价。

VI 的优势是把 inference 转成可用 optimization 与 automatic differentiation 处理的形式，因此特别适合 large-scale probabilistic learning。

## Connections

- [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/)：VI 的主要优化形式。
- [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/)：continuous stochastic nodes 的低方差 gradient estimator。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：amortized neural variational inference 的代表性模型。
