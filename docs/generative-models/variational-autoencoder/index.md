---
title: "Variational Autoencoder"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/variational-autoencoder/"
prerequisites:
  - "/generative-models/latent-variable/"
  - "/generative-models/variational-inference/"
  - "/generative-models/evidence-lower-bound/"
  - "/generative-models/reparameterization-trick/"
related:
  - "/generative-models/conditional-variational-autoencoder/"
  - "/generative-models/posterior-collapse/"
---

# Variational Autoencoder

Variational Autoencoder（VAE）是带有 neural-network encoder 与 decoder 的 latent-variable generative model。它用 variational inference 学习 approximate posterior，并通过 reparameterization 让随机 latent sampling 可以参与 gradient-based training。

## Generative Model

VAE 先定义 latent prior

\[
p(z),
\]

再定义 decoder likelihood

\[
p_\theta(x|z).
\]

联合分布为

\[
p_\theta(x,z)=p(z)p_\theta(x|z).
\]

生成时可以先采样

\[
z\sim p(z),
\]

再由 decoder 产生

\[
x\sim p_\theta(x|z).
\]

经典 VAE 常使用

\[
p(z)=\mathcal N(0,I).
\]

## Inference Model

真实 posterior

\[
p_\theta(z|x)
\]

通常难以精确求出，因此引入 encoder

\[
q_\phi(z|x)
\]

作为 approximate posterior。常见 diagonal Gaussian 参数化为

\[
q_\phi(z|x)
=
\mathcal N
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
\]

Encoder 的输出不是最终 reconstruction，也不是一个唯一的 latent code，而是决定 latent distribution 的参数。

## Training Objective

VAE 最大化 ELBO：

\[
\mathcal L(x)
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{\mathrm{KL}}
(q_\phi(z|x)\|p(z)).
\]

第一项要求 sampled latent 能让 decoder 解释数据；第二项把 approximate posterior 约束在 prior 附近。

对于 diagonal Gaussian posterior 和 standard normal prior，KL 有闭式解；因此 KL 部分不需要 Monte Carlo 估计。

## Reparameterization

直接写

\[
z\sim q_\phi(z|x)
\]

会把一个随机采样节点放在 encoder parameters 到 loss 的路径上。VAE 对 Gaussian 使用

\[
\epsilon\sim\mathcal N(0,I),
\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
\]

随机性现在来自与 $\phi$ 无关的 $\epsilon$；$z$ 对 $\mu,\sigma$ 是普通可微函数。详细推导属于 [Reparameterization Trick](/generative-models/reparameterization-trick/)。

## Training 与 Generation

训练时需要 encoder，是因为我们有数据 $x$，需要近似“哪些 $z$ 能解释这个 $x$”。生成时不需要先有 $x$，可以直接从 prior $p(z)$ 采样，再交给 decoder。

这一区别对理解 ACT 很重要：ACT 的 CVAE encoder 也只在训练时使用，但 ACT 推理时并不是随机采样 style，而是固定使用 prior mean $z=0$。这是 ACT-specific design，不属于 VAE 的一般定义。

## Autoencoder 名称容易造成的误解

VAE 虽然名字里有 Autoencoder，但它不是普通 autoencoder 加一点噪声。它建立了显式概率模型、prior、approximate posterior 和 variational objective。Encoder/decoder 的“编码—解码”结构只是表面相似，核心区别在 probabilistic formulation。

## Sources

- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
