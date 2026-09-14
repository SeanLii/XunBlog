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

Reparameterization Trick 把“从一个依赖模型参数的 distribution 中随机采样”改写成“先从固定噪声分布采样，再通过可微函数得到目标样本”。

VAE 中最常见的 Gaussian 情况是：

原本：

\[
z\sim\mathcal N(\mu,\sigma^2).
\]

改写为：

\[
\epsilon\sim\mathcal N(0,1),
\]

\[
z=\mu+\sigma\epsilon.
\]

多维情况：

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

## 随机性被移到了哪里

改写前，采样操作直接依赖 $\mu,\sigma$。

改写后：

```text
fixed random noise ε
       │
       ├──── μ
       ├──── σ
       ↓
z = μ + σ ⊙ ε
```

随机节点 $\epsilon$ 与 encoder parameters 无关；给定某次 sampled $\epsilon$ 后，$z$ 是 $\mu,\sigma$ 的普通可微函数。

## Gradient 可以怎样传播

若 downstream loss 为 $L(z)$，则

\[
\frac{\partial L}{\partial \mu}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial \mu}
=
\frac{\partial L}{\partial z},
\]

而

\[
\frac{\partial z}{\partial \sigma}=\epsilon.
\]

所以 reconstruction objective 的 gradient 可以通过 sampled $z$ 回到产生 $\mu,\sigma$ 的 encoder。

## Distribution 没有被改变

如果

\[
\epsilon\sim\mathcal N(0,1),
\]

则

\[
\mu+\sigma\epsilon
\sim
\mathcal N(\mu,\sigma^2).
\]

因此这不是近似另一个 distribution，而只是同一随机变量的另一种构造方式。

## Log Variance 参数化

Encoder 常输出

\[
\log\sigma^2
\]

而不是直接输出 $\sigma$。这样无需强制 raw network output 为正。

恢复标准差：

\[
\sigma
=
\exp\left(\frac12\log\sigma^2\right).
\]

然后：

\[
z=\mu+
\exp\left(\frac12\log\sigma^2\right)
\odot\epsilon.
\]

ACT released latent encoder 也采用 $\mu,\log variance$ 这种参数化。

## Sources

- Kingma & Welling, **Auto-Encoding Variational Bayes**, 2013/2014. https://arxiv.org/abs/1312.6114
