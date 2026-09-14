---
title: "Reparameterization Trick"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/reparameterization-trick/"
prerequisites:
  - "/mathematics/probability/normal-distribution/"
  - "/mathematics/probability/multivariate-normal-distribution/"
related:
  - "/generative-models/variational-autoencoder/"
---

# Reparameterization Trick

Reparameterization Trick 把“从一个由网络参数决定的分布中采样”改写成“从固定噪声分布采样，再通过可微函数变换”。这样 Monte Carlo sample 仍可以用于普通反向传播。

## Gaussian 情形

设 encoder 给出

\[
q_\phi(z|x)
=
\mathcal N(\mu,\sigma^2).
\]

与其直接写

\[
z\sim\mathcal N(\mu,\sigma^2),
\]

改写为

\[
\epsilon\sim\mathcal N(0,1),
\]

\[
z=\mu+\sigma\epsilon.
\]

这两个过程产生相同的 $z$ 分布。

多维 diagonal Gaussian 对应

\[
\boldsymbol\epsilon\sim\mathcal N(0,I),
\qquad
\mathbf z=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\epsilon.
\]

## 梯度路径

随机节点 $\epsilon$ 与 encoder parameters $\phi$ 无关。对一次固定采样而言，

\[
z=g_\phi(x,\epsilon)
\]

就是普通 differentiable computation。Loss 对 $z$ 的梯度可以继续传到 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$。

这就是 reparameterization 真正解决的问题：它不是“让随机性消失”，而是把随机性移动到参数无关的外部噪声变量上。

## Log-Variance 参数化

网络实现常输出

\[
\log\sigma^2
\]

而不是直接输出 $\sigma$。如果记

\[
\ell=\log\sigma^2,
\]

则

\[
\sigma=\exp(\ell/2)>0.
\]

这样网络可以无约束地输出任意实数 $\ell$，再通过指数保证标准差为正。

ACT 官方实现的 `reparametrize(mu, logvar)` 正是先计算

\[
\sigma=\exp(\tfrac12\log\sigma^2)
\]

再采样 standard normal noise，并返回 $\mu+\sigma\epsilon$。

## Sources

- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
- [ACT official implementation](https://github.com/tonyzhaozh/act)
