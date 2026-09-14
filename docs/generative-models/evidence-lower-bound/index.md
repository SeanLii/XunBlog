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

Evidence Lower Bound（ELBO）是对 log evidence $\log p_\theta(x)$ 的一个可优化下界。VAE 通过最大化 ELBO，在无法直接计算 marginal likelihood 的情况下同时训练 generative model 和 approximate posterior。

## 从 Log Evidence 开始

引入任意满足条件的 approximate posterior $q_\phi(z|x)$：

\[
\log p_\theta(x)
=
\mathbb E_{q_\phi(z|x)}
\left[
\log p_\theta(x)
\right].
\]

利用 Bayes 关系

\[
p_\theta(z|x)=\frac{p_\theta(x,z)}{p_\theta(x)},
\]

可以得到分解

\[
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{\mathrm{KL}}
\left(
q_\phi(z|x)\|p_\theta(z|x)
\right).
\]

因为 KL divergence 非负，

\[
\mathcal L(\theta,\phi;x)
\le \log p_\theta(x).
\]

因此 $\mathcal L$ 就是 evidence lower bound。

## ELBO 的常用形式

把联合分布分解为

\[
p_\theta(x,z)=p_\theta(x|z)p(z),
\]

ELBO 可以写为

\[
\boxed{
\mathcal L
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{\mathrm{KL}}
(q_\phi(z|x)\|p(z))
}
\]

第一项鼓励 decoder 在 sampled $z$ 下给真实 $x$ 较高概率；第二项约束 approximate posterior 不要任意偏离 prior。

## 两个目标的关系

Reconstruction/log-likelihood term 希望 $z$ 保留对解释 $x$ 有帮助的信息。KL term 则会把 $q_\phi(z|x)$ 拉向共同 prior。如果 KL 权重过强，encoder 可能减少通过 $z$ 传递的信息；如果完全没有这项，latent distributions 又可能失去统一 prior 所提供的可采样结构。

这不是把两个目标粗略说成“一个对一个错”，而是同一个 probabilistic objective 中两个必要部分承担不同约束。

## Negative ELBO

训练代码通常最小化 loss，因此会使用 ELBO 的负数：

\[
\mathcal J
=
-\mathbb E_q[\log p_\theta(x|z)]
+
D_{\mathrm{KL}}(q\|p).
\]

当 decoder likelihood 选择不同分布时，第一项会对应不同 reconstruction loss。比如 Gaussian likelihood 在固定方差假设下可导出与 squared error 相关的目标；实际工程代码也可能直接采用 L1 等 surrogate loss。ACT 的 paper/code 差异就在这里出现。

## Sources

- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
