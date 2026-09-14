---
title: "Evidence Lower Bound"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/evidence-lower-bound/"
prerequisites:
  - "/generative-models/variational-inference/"
  - "/mathematics/information-theory/kl-divergence/"
  - "/mathematics/probability/expectation/"
related:
  - "/generative-models/variational-autoencoder/"
---

# Evidence Lower Bound

Evidence Lower Bound（ELBO）是对数据 log-likelihood $\log p(x)$ 的一个下界，也是 VAE 训练目标的数学来源。

它最重要的关系是：

\[
\log p(x)
=
\mathcal L_{ELBO}(x)
+
D_{KL}(q(z\mid x)\|p(z\mid x)).
\]

因为 KL divergence 非负：

\[
D_{KL}\ge0,
\]

所以

\[
\mathcal L_{ELBO}(x)\le\log p(x).
\]

这就是 “lower bound”。

## 从 log p(x) 开始

引入任意 approximate posterior $q(z\mid x)$：

\[
\log p(x)
=
\log\int p(x,z)\,dz.
\]

乘除 $q(z\mid x)$：

\[
\log p(x)
=
\log\int q(z\mid x)
\frac{p(x,z)}{q(z\mid x)}\,dz.
\]

写成 expectation：

\[
\log p(x)
=
\log
\mathbb E_{q(z\mid x)}
\left[
\frac{p(x,z)}{q(z\mid x)}
\right].
\]

由于 $\log$ 是 concave function，Jensen inequality 给出：

\[
\log\mathbb E[X]
\ge
\mathbb E[\log X].
\]

因此

\[
\log p(x)
\ge
\mathbb E_q
\left[
\log p(x,z)-\log q(z\mid x)
\right].
\]

右边定义为 ELBO。

## VAE 中的常用形式

把 joint distribution 分解：

\[
p(x,z)=p(z)p_\theta(x\mid z).
\]

得到

\[
\mathcal L_{ELBO}
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{KL}(q_\phi(z\mid x)\|p(z)).
\]

这就是常见的 “reconstruction term - KL term”。

## 两个目标项的作用

第一项：

\[
\mathbb E_q[\log p_\theta(x\mid z)]
\]

要求 decoder 在 sampled latent 下给真实 $x$ 较高 likelihood。

第二项：

\[
D_{KL}(q_\phi(z\mid x)\|p(z))
\]

限制 approximate posterior 不要任意偏离 prior。

所以训练不是简单追求“重建越精确越好”，而是在数据拟合与 latent distribution regularization 之间共同优化。

## 下界什么时候等于真实 log-likelihood

由

\[
\log p(x)-\mathcal L_{ELBO}
=
D_{KL}(q(z\mid x)\|p(z\mid x))
\]

可知，只有当

\[
q(z\mid x)=p(z\mid x)
\]

时 gap 为 0。

所以 ELBO 的松紧程度直接由 approximate posterior 和 true posterior 的差异决定。

## Conditional ELBO

CVAE 中目标变成 conditional likelihood $\log p(y\mid x)$。对应下界为

\[
\mathbb E_{q(z\mid x,y)}[\log p(y\mid x,z)]
-
D_{KL}(q(z\mid x,y)\|p(z\mid x)).
\]

结构没有变，只是所有分布都放在 condition $x$ 下。

## Sources

- Kingma & Welling, **Auto-Encoding Variational Bayes**, 2013/2014. https://arxiv.org/abs/1312.6114
- Sohn, Lee & Yan, **Conditional Variational Autoencoder**, 2015. https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
