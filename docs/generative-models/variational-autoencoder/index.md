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

Variational Autoencoder（VAE）是一种带有 latent variable 的生成模型。它学习的不是单纯的

```text
x → 压缩向量 → 重建 x
```

而是一套概率模型：先假设数据背后存在一个没有直接观测到的变量 $z$，再学习如何从 $z$ 生成数据 $x$。

最核心的生成方向是：

```text
z ~ p(z)
   │
   ↓
decoder pθ(x|z)
   │
   ↓
   x
```

通常会选择一个简单 prior，例如

\[
p(z)=\mathcal N(0,I).
\]

训练完成后，可以从这个 prior 采样 $z$，再通过 decoder 生成新的样本。

## 真正困难的是反方向

如果训练数据已经给了一个 $x$，我们需要知道“什么样的 $z$ 可能生成它”。这对应真实 posterior：

\[
p_\theta(z\mid x).
\]

但这个 posterior 往往难以直接计算。VAE 因此引入一个 encoder，学习一个近似分布：

\[
q_\phi(z\mid x)\approx p_\theta(z\mid x).
\]

于是 VAE 有两条方向相反但彼此配合的路径：

```text
inference / encoder:
      x → qφ(z|x)

     generation / decoder:
      z → pθ(x|z)
```

这里 encoder 输出的不是“唯一 latent vector”，而是一个 distribution 的参数。最常见的 Gaussian 情况中，它会输出

\[
\mu(x),\qquad \log \sigma^2(x),
\]

从而定义

\[
q_\phi(z\mid x)=\mathcal N(\mu(x),\operatorname{diag}(\sigma^2(x))).
\]

## 一次训练样本怎样流过 VAE

假设输入是一张图像 $x$。

第一步，encoder 根据它得到 $\mu$ 和 $\sigma$：

```text
x
│
↓
Encoder
│
├── μ
└── σ
```

第二步，从这个 posterior approximation 中得到一个 latent sample：

\[
z\sim q_\phi(z\mid x).
\]

第三步，decoder 根据 $z$ 重建输入：

\[
\hat x\sim p_\theta(x\mid z).
\]

整体变成：

```text
x → Encoder → μ, σ → sample z → Decoder → x_hat
```

但如果只有 reconstruction loss，encoder 完全可以把每个训练样本编码到彼此孤立的位置，latent space 未必适合从一个统一 prior 中采样。

因此 VAE 还需要第二个目标：让每个 $q_\phi(z\mid x)$ 不要离 prior 太远。

## VAE 的训练目标

标准 VAE 最大化 [Evidence Lower Bound](/generative-models/evidence-lower-bound/)：

\[
\mathcal L_{\text{ELBO}}(x)
=
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{KL}\big(q_\phi(z\mid x)\|p(z)\big).
\]

这两个部分承担不同任务。

第一项要求 decoder 能根据 $z$ 解释或重建 $x$：

\[
\mathbb E_q[\log p_\theta(x\mid z)].
\]

第二项要求 approximate posterior 不要任意漂离 prior：

\[
D_{KL}(q_\phi(z\mid x)\|p(z)).
\]

所以可以把它理解为：

```text
保留关于 x 的信息
        ↑
        │ reconstruction
        │
        z
        │
        │ KL regularization
        ↓
保持 latent distribution 接近统一 prior
```

这两个要求不是同一件事。重建希望 $z$ 足够有信息，而 KL 又限制 posterior 不要为每个样本随意跑到完全不同的位置。

## Sampling 与反向传播

如果直接写

\[
z\sim\mathcal N(\mu,\sigma^2),
\]

随机采样这一步会让普通反向传播难以直接把 gradient 传回产生 $\mu,\sigma$ 的 encoder。

VAE 使用 [Reparameterization Trick](/generative-models/reparameterization-trick/)：

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z=\mu+\sigma\odot\epsilon.
\]

随机性现在集中在与参数无关的 $\epsilon$ 上，而 $z$ 对 $\mu,\sigma$ 是可微函数。

## VAE 不是普通 Autoencoder 加一点噪声

普通 autoencoder 关注的是 deterministic encoding：

\[
z=f_\phi(x),\qquad \hat x=g_\theta(z).
\]

VAE 则显式定义概率模型：

\[
p(z),\qquad p_\theta(x\mid z),\qquad q_\phi(z\mid x).
\]

它的训练目标来自对数据 log-likelihood 的 variational lower bound，而不是为了“让压缩向量更平滑”而随意加噪声。

## 从 VAE 到 CVAE

VAE 学习的是数据分布 $p(x)$ 或其 latent-variable approximation。如果我们还希望在某个条件 $c$ 下生成结果，例如“给定一张输入图像，生成可能的 segmentation”，则需要建模

\[
p(y\mid c).
\]

这会自然进入 [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)。ACT 使用的正是这种 conditional latent-variable 思路：当前 robot observation 是 condition，future action chunk 是要预测的 structured output。

## Sources

- Kingma & Welling, **Auto-Encoding Variational Bayes**, 2013/2014. https://arxiv.org/abs/1312.6114
