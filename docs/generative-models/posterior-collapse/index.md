---
title: "Posterior Collapse"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/posterior-collapse/"
prerequisites:
  - "/generative-models/variational-autoencoder/"
  - "/mathematics/information-theory/kl-divergence/"
related:
  - "/robot-learning/act/cvae-in-act/"
---

# Posterior Collapse

Posterior Collapse 是 latent-variable generative model 中的一种训练现象：approximate posterior 变得非常接近 prior，并且 decoder 基本不再使用 latent $z$ 中的信息。

典型表现是

\[
q_\phi(z\mid x)\approx p(z),
\]

于是

\[
D_{KL}(q_\phi(z\mid x)\|p(z))\approx0.
\]

这看起来像 KL objective 完成得很好，但如果同时 $z$ 与 $x$ 几乎无关，latent representation 就失去了原本要承担的信息作用。

## 从 VAE 目标看这个现象

VAE ELBO：

\[
\mathcal L
=
\mathbb E_q[\log p_\theta(x\mid z)]
-
D_{KL}(q_\phi(z\mid x)\|p(z)).
\]

KL 项偏好 posterior 接近 prior。

Reconstruction / likelihood 项则只有在 decoder **需要 $z$** 时，才会推动 encoder 往 $z$ 中放信息。

如果 decoder 本身已经非常强，能够主要依靠其他上下文预测 $x$，那么最省事的解可能是：

```text
q(z|x) ≈ p(z)
       ↓
z carries little information
       ↓
decoder mostly ignores z
```

## Collapse 不等于“所有 KL 小都是坏的”

KL 较小本身不是正式判定条件。我们真正关心的是 latent 是否仍然影响 reconstruction / generation，是否携带关于输入的有用信息。

因此需要结合 KL、latent usage、decoder sensitivity、mutual-information style measurements 或生成行为一起判断。

## 与条件模型的关系

CVAE 中 decoder 还拿到 condition $x$：

\[
p(y\mid x,z).
\]

如果 $x$ 已经足以很好预测 $y$，decoder 更容易忽略 $z$。因此 conditional models 同样可能出现 posterior collapse。

## 与 ACT 的连接

ACT 的 action predictor 有强 observation condition（视觉 + proprioception）。理论上 latent 也可能被弱化，因此 KL weight、model capacity 与 training dynamics 都会影响 $z$ 实际被使用多少。

但不能仅凭“ACT inference 使用 $z=0$”就宣称发生 posterior collapse。Inference 固定 prior mean 是设计选择；posterior collapse 是训练后 latent 是否失去信息作用的现象，两者不是同一个概念。
