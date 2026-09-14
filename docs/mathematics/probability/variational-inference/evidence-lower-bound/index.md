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

Evidence Lower Bound（ELBO）是对 log evidence：

\[
\log p(x)
\]

的一个可优化 lower bound。

它在 Variational Inference 中承担关键角色：我们原本想让 approximate posterior $q(z)$ 靠近真实 posterior $p(z\mid x)$，但真实 posterior 中包含难算的 evidence。ELBO 把这个目标改写成只依赖 joint model 与 $q$ 的形式。

## 从一个恒等式开始

考虑：

\[
D_{KL}(q(z)\|p(z\mid x)).
\]

展开：

\[
D_{KL}
=
\mathbb E_q
\left[
\log q(z)-\log p(z\mid x)
\right].
\]

使用 Bayes：

\[
\log p(z\mid x)
=
\log p(x,z)-\log p(x).
\]

代入：

\[
D_{KL}
=
\mathbb E_q[
\log q(z)-\log p(x,z)
]
+
\log p(x).
\]

整理：

\[
\log p(x)
=
\underbrace{
\mathbb E_q[
\log p(x,z)-\log q(z)
]
}_{\operatorname{ELBO}}
+
D_{KL}(q(z)\|p(z\mid x)).
\]

## Lower-Bound Property

因为：

\[
D_{KL}(q\|p)\ge0,
\]

所以：

\[
\operatorname{ELBO}
\le
\log p(x).
\]

当：

\[
q(z)=p(z\mid x),
\]

KL 为 0，ELBO 恰好等于 log evidence。

因此 bound 的 gap 就是 approximate posterior 与 true posterior 的 KL divergence。

## Latent Model 中的另一种写法

若：

\[
p(x,z)=p(z)p(x\mid z),
\]

则：

\[
\operatorname{ELBO}
=
\mathbb E_{q(z)}[
\log p(x\mid z)
]
-
D_{KL}(q(z)\|p(z)).
\]

这分成两部分。

第一项：

\[
\mathbb E_q[\log p(x\mid z)]
\]

要求 sampled latent 能让 generative model 对 observed $x$ 给出高 likelihood。

第二项：

\[
-D_{KL}(q(z)\|p(z))
\]

限制 approximate posterior 不要任意偏离 prior。

## ELBO 不是“Reconstruction Loss + KL”的定义

在 VAE 中，如果 decoder likelihood 选 Gaussian，negative expected log-likelihood 常可以对应到 MSE-like reconstruction term；若选 Bernoulli，则会出现 binary cross-entropy-like term。

所以“reconstruction + KL”是特定 likelihood parameterization 下的训练形式。

更根本的定义仍然是：

\[
\mathbb E_q[
\log p(x,z)-\log q(z)
].
\]

## Conditional ELBO

如果建模：

\[
p(y,z\mid x)
=p(z\mid x)p(y\mid x,z),
\]

使用 variational posterior：

\[
q(z\mid x,y),
\]

则 conditional ELBO：

\[
\log p(y\mid x)
\ge
\mathbb E_{q(z\mid x,y)}
[
\log p(y\mid x,z)
]
-
D_{KL}
(q(z\mid x,y)\|p(z\mid x)).
\]

这就是 Conditional VAE 等模型的数学基础。

## ELBO 的两个角色

ELBO 同时在做：

1. **model learning**：提高 observed data likelihood 的 lower bound；
2. **inference learning**：缩小 $q$ 与 true posterior 的 gap。

所以它不是单纯 regularizer，也不是为了让 latent “长得好看”而加的 penalty。

## Sources

- Blei, Kucukelbir, McAuliffe. *Variational Inference: A Review for Statisticians*. 2017.
- Kingma & Welling. *Auto-Encoding Variational Bayes*. 2013/2014.
