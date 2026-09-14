---
title: "Variational Inference"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/variational-inference/"
prerequisites:
  - "/mathematics/probability/conditional-probability/"
  - "/mathematics/information-theory/kl-divergence/"
related:
  - "/generative-models/evidence-lower-bound/"
  - "/generative-models/variational-autoencoder/"
---

# Variational Inference

Variational Inference 是用一个容易计算的 distribution $q(z)$ 去近似一个难以直接计算的 posterior distribution $p(z\mid x)$ 的方法。

问题起点是 Bayes rule：

\[
p(z\mid x)=\frac{p(x,z)}{p(x)}.
\]

其中

\[
p(x)=\int p(x,z)\,dz.
\]

如果这个积分难以计算，那么 posterior 的归一化常数也难以得到。

## 把推断变成优化

Variational inference 选择一族容易处理的 distributions：

\[
q_\phi(z).
\]

然后调整参数 $\phi$，让它尽量接近真实 posterior：

\[
q_\phi(z)\approx p(z\mid x).
\]

“接近”通常通过 [KL Divergence](/mathematics/information-theory/kl-divergence/) 衡量：

\[
D_{KL}(q_\phi(z)\|p(z\mid x)).
\]

于是原本的积分推断问题变成了 optimization problem。

## 难点仍然存在

直接优化上面的 KL 看起来仍需要知道 $p(z\mid x)$。展开：

\[
D_{KL}(q\|p(z\mid x))
=
\mathbb E_q[
\log q(z)-\log p(z\mid x)
].
\]

代入

\[
\log p(z\mid x)
=
\log p(x,z)-\log p(x),
\]

得到

\[
D_{KL}
=
\log p(x)
-
\left(
\mathbb E_q[\log p(x,z)]
-
\mathbb E_q[\log q(z)]
\right).
\]

括号中的量就是 [Evidence Lower Bound](/generative-models/evidence-lower-bound/)：

\[
\mathcal L_{ELBO}.
\]

因此

\[
\log p(x)
=
\mathcal L_{ELBO}
+
D_{KL}(q(z)\|p(z\mid x)).
\]

因为 KL 非负，最大化 ELBO 会推动近似 posterior 靠近真实 posterior，同时提高对数据的解释能力。

## Amortized Variational Inference

传统 variational inference 可以为每个样本单独优化一组 $q$ 参数。VAE 进一步训练一个 neural network encoder：

\[
q_\phi(z\mid x).
\]

输入任何 $x$，一次 forward pass 就得到该样本的 approximate posterior parameters。

这叫 amortized inference：很多样本共享同一个 inference network 参数 $\phi$。

## Variational Inference 与 VAE 的关系

VAE 不是“variational inference 的同义词”。

Variational inference 是更广泛的 approximate inference 思想；VAE 把它与 neural generative model、amortized encoder 和 reparameterized stochastic optimization 组合起来。

## Sources

- Kingma & Welling, **Auto-Encoding Variational Bayes**, 2013/2014. https://arxiv.org/abs/1312.6114
