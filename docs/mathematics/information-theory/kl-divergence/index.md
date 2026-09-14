---
title: "KL Divergence"
kind: "canonical"
domain: "Mathematics / Information Theory"
parent: "Information Theory"
canonical: "/mathematics/information-theory/kl-divergence/"
prerequisites:
  - "/mathematics/information-theory/entropy/"
  - "/mathematics/information-theory/cross-entropy/"
related:
  - "/mathematics/probability/variational-inference/"
  - "/generative-models/variational-autoencoder/"
---

# KL Divergence

KL Divergence（Kullback–Leibler Divergence）衡量两个 probability distributions 之间的差异，但它不是 ordinary geometric distance。

对于 discrete distributions $p,q$：

\[
D_{KL}(p\|q)
=\sum_x p(x)
\log\frac{p(x)}{q(x)}.
\]

continuous 情况：

\[
D_{KL}(p\|q)
=\int p(x)
\log\frac{p(x)}{q(x)}\,dx.
\]

Expectation form：

\[
D_{KL}(p\|q)
=\mathbb E_{x\sim p}
\left[
\log p(x)-\log q(x)
\right].
\]

## 它比较的是“用 q 描述 p”的代价

因为 expectation 是在 $p$ 下取的，$D_{KL}(p\|q)$ 的方向很重要。

可以把它理解为：

> 数据实际上来自 $p$，但我们用 $q$ 去描述它，相比直接使用 $p$ 自己，多付出了多少平均 log-loss / coding cost。

由：

\[
H(p,q)=H(p)+D_{KL}(p\|q)
\]

可见：

\[
D_{KL}(p\|q)=H(p,q)-H(p).
\]

所以 KL 正好是 cross-entropy 超过真实 entropy 的部分。

## 非负性

KL divergence 满足 Gibbs' inequality：

\[
D_{KL}(p\|q)\ge0.
\]

并且在适当条件下：

\[
D_{KL}(p\|q)=0
\iff p=q\quad\text{almost everywhere}.
\]

因此如果把 $q$ 优化到尽可能接近 $p$，minimizing KL 是一种自然目标。

## KL 不是 Metric

一个 metric 通常要求 symmetry：

\[
d(p,q)=d(q,p).
\]

但 KL 一般不满足：

\[
D_{KL}(p\|q)
\ne
D_{KL}(q\|p).
\]

它也不满足一般意义上的 triangle inequality。

所以更准确叫 divergence，而不是 distance。

## KL 的 Directionality

考虑某些区域：

- $p(x)>0$；
- $q(x)$ 非常小。

在 $D_{KL}(p\|q)$ 中，这些区域会受到很强惩罚，因为模型 $q$ 漏掉了真实 $p$ 的 mass。

反过来 $D_{KL}(q\|p)$ 是对 $q$ 采样区域取 expectation，优化行为会不同。

这种 directionality 是 variational inference 中 “forward / reverse KL” 行为差异的基础之一。

## Gaussian 之间的 KL

如果：

\[
q(z)=\mathcal N(\mu_q,\Sigma_q),
\qquad
p(z)=\mathcal N(\mu_p,\Sigma_p),
\]

Gaussian KL 有 closed form。

VAE 常见的特殊情况是：

\[
q(z\mid x)=
\mathcal N(\mu,\operatorname{diag}(\sigma^2)),
\]

\[
p(z)=\mathcal N(0,I).
\]

此时：

\[
D_{KL}(q\|p)
=
\frac12
\sum_i
\left(
\mu_i^2+
\sigma_i^2-
\log\sigma_i^2-
1
\right).
\]

这个解析式很重要，但它只是 KL 的一个 Gaussian application，不是 KL Divergence 的全部内容。

## 在 Variational Inference 中

[Variational Inference](/mathematics/probability/variational-inference/) 常通过优化：

\[
D_{KL}(q(z)\|p(z\mid x))
\]

让 tractable approximation $q$ 靠近难以直接求解的 posterior。

VAE 又进一步把这一思路 amortize 到 inference network 中。

因此 KL 与 VAE 有很深的连接，但它首先是一个独立的信息论量，并不由 VAE 定义。
