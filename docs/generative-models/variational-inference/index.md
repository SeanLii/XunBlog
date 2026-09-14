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

Variational Inference 用一个较容易处理的分布族去近似难以直接计算的 posterior distribution。VAE 的 encoder 本质上就是一个 amortized variational inference model。

## Posterior Inference

对 latent-variable model

\[
p_\theta(x,z)=p_\theta(x|z)p(z),
\]

看到 $x$ 后，我们关心

\[
p_\theta(z|x)
=\frac{p_\theta(x,z)}{p_\theta(x)}.
\]

困难在分母

\[
p_\theta(x)=\int p_\theta(x,z)\,dz.
\]

高维神经生成模型中，这个积分通常无法直接精确计算。

## Approximate Posterior

Variational inference 引入一个可处理的分布

\[
q_\phi(z|x)
\]

去近似真实 posterior $p_\theta(z|x)$。理想目标可以写成最小化

\[
D_{\mathrm{KL}}
\left(q_\phi(z|x)\|p_\theta(z|x)\right).
\]

但这个 KL 中仍然包含难算的 $p_\theta(x)$。通过概率恒等式，可以把问题改写为最大化 [Evidence Lower Bound](/generative-models/evidence-lower-bound/)（ELBO），从而得到可训练目标。

## Amortized Inference

传统 variational inference 可以为每个 datapoint 单独优化 variational parameters。VAE 则用一个共享 neural network 根据 $x$ 直接输出 $q_\phi(z|x)$ 的参数。

例如 diagonal Gaussian：

\[
q_\phi(z|x)
=\mathcal N(\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))).
\]

所有 datapoints 共享网络参数 $\phi$，但不同输入会得到不同的 $\mu$ 与 $\sigma$。这就是 amortized inference：推断成本被“摊销”进 encoder 的训练中。

## 与普通 Encoder 的差别

普通 autoencoder encoder 输出一个确定向量。Variational encoder 输出的是一个分布的参数。训练目标也不只要求 reconstruction，还要求 approximate posterior 具有与 prior 兼容的概率结构。

## Sources

- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
