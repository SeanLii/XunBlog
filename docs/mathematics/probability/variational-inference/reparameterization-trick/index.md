---
title: "Reparameterization Trick"
kind: "canonical"
domain: "Mathematics / Probability / Variational Inference"
parent: "Variational Inference"
canonical: "/mathematics/probability/variational-inference/reparameterization-trick/"
prerequisites:
  - "/mathematics/probability/variational-inference/"
related:
  - "/mathematics/probability/variational-inference/evidence-lower-bound/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Reparameterization Trick

> **知识边界**：本文的 canonical 对象是 **Reparameterization Trick**。依赖机制由 [Variational Inference](/mathematics/probability/variational-inference/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Reparameterization Trick 把 parameter-dependent random sampling 写成“parameter-free noise + differentiable deterministic transformation”，从而使 Monte Carlo expectation 可以使用 pathwise gradients。

设：

\[
z\sim q_\phi(z).
\]

若可以表示为：

\[
\epsilon\sim p(\epsilon),
\qquad
z=g_\phi(\epsilon),
\]

其中 base noise distribution $p(\epsilon)$ 不依赖 $\phi$，则：

\[
\mathbb E_{z\sim q_\phi}[f(z)]
=
\mathbb E_{\epsilon\sim p(\epsilon)}
[f(g_\phi(\epsilon))].
\]

随机性因此移到 $\epsilon$，而 $\phi$ 只出现在 differentiable computation graph 中。

## Gaussian Reparameterization

最经典情况：

\[
z\sim\mathcal N(\mu,\sigma^2).
\]

可以写成：

\[
\epsilon\sim\mathcal N(0,1),
\]

\[
z=\mu+\sigma\epsilon.
\]

多维 diagonal Gaussian：

\[
q_\phi(z\mid x)
=
\mathcal N(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
),
\]

则：

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon.
\]

## Pathwise Gradient

考虑 objective：

\[
J(\phi)
=
\mathbb E_{z\sim q_\phi}[f(z)].
\]

reparameterize 后：

\[
J(\phi)
=
\mathbb E_{\epsilon\sim p(\epsilon)}
[f(g_\phi(\epsilon))].
\]

于是：

\[
\nabla_\phi J
=
\mathbb E_\epsilon
\left[
\nabla_z f(z)
\frac{\partial g_\phi(\epsilon)}{\partial\phi}
\right].
\]

这允许 gradient 直接沿 deterministic path 从 $f$ 回传到 distribution parameters。

## Why Direct Sampling Is Awkward

若 computation graph 只有：

```text
φ → distribution qφ → random sample z → f(z)
```

普通 automatic differentiation 不能把 discrete “sample operation” 当成普通 deterministic function 对 $\phi$ 求 pathwise derivative。

Reparameterization 把它改成：

```text
ε ~ fixed noise
φ ───────────┐
             ↓
        z = gφ(ε)
             ↓
            f(z)
```

这样对固定 sample $\epsilon$，整个后半部分是 differentiable computation。

## Log-Variance Parameterization

Neural implementations 常预测：

\[
\log\sigma^2
\]

而不是直接预测 $\sigma$。

可以恢复：

\[
\sigma
=
\exp\left(
\frac12\log\sigma^2
\right).
\]

这样 $\sigma>0$ 自动成立，并提高数值稳定性。

## Score-Function Estimator

另一类 gradient estimator 是：

\[
\nabla_\phi
\mathbb E_{q_\phi(z)}[f(z)]
=
\mathbb E_{q_\phi(z)}
[
f(z)\nabla_\phi\log q_\phi(z)
].
\]

它不要求 reparameterizable distribution，因此适用范围更广，但 variance 往往较高。

Pathwise gradient 与 score-function estimator 是两种不同 stochastic-gradient strategies。

## Beyond Diagonal Gaussian

Reparameterization 不只适用于 diagonal Gaussian。

例如 full-covariance Gaussian：

\[
z=\mu+L\epsilon,
\qquad
LL^\top=\Sigma,
\]

其中 $L$ 可以取 Cholesky factor。

其他 continuous distributions 也可能通过 inverse CDF、location-scale transformation 或 implicit reparameterization 构造 pathwise gradient。

Discrete random variables 则通常不能直接使用普通 pathwise reparameterization，需要 relaxations 或其他 gradient estimators。

## Role in Variational Inference

Neural VI / VAE 中，ELBO 含有：

\[
\mathbb E_{q_\phi(z\mid x)}
[
\log p_\theta(x\mid z)
].
\]

Reparameterization 使这项可以对 encoder parameters $\phi$ 使用低方差 Monte Carlo pathwise gradient。

因此它是 stochastic variational inference 的 gradient-estimation technique，而不是 VAE 的专属理论。

## Connections

- [Variational Inference](/mathematics/probability/variational-inference/)：reparameterization 解决 stochastic expectation 的 gradient problem。
- [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/)：Gaussian sampling transformation。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：使用 amortized Gaussian posterior 与 reparameterized training。
