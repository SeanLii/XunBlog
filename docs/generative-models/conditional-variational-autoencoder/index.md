---
title: "Conditional Variational Autoencoder"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/conditional-variational-autoencoder/"
prerequisites:
  - "/generative-models/variational-autoencoder/"
related:
  - "/robot-learning/act/cvae-in-act/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Conditional Variational Autoencoder

> **知识边界**：本文的 canonical 对象是 **Conditional Variational Autoencoder**。依赖机制由 [Variational Autoencoder](/generative-models/variational-autoencoder/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Conditional Variational Autoencoder（CVAE）是对 VAE 的条件化扩展。它建模的不是无条件 data distribution $p(x)$，而是给定 observed condition $c$ 后的 conditional distribution：

\[
p_\theta(y\mid c).
\]

当同一个 condition 可能对应多个合理 outputs 时，CVAE 使用 latent variable $z$ 表示 condition 没有完全决定的变化：

\[
p_\theta(y,z\mid c)
=
p_\theta(z\mid c)
p_\theta(y\mid c,z).
\]

其中：

- $c$ 是 observed condition；
- $y$ 是需要建模或生成的 output；
- $z$ 是 latent variable。

## Conditional Generative Model

CVAE 的生成方向为：

\[
c
\rightarrow
p_\theta(z\mid c)
\rightarrow
z
\rightarrow
p_\theta(y\mid c,z).
\]

因此 conditional likelihood 为：

\[
p_\theta(y\mid c)
=
\int
p_\theta(y\mid c,z)
p_\theta(z\mid c)
\,dz.
\]

如果 conditional prior 不依赖 $c$，可以使用固定 prior：

\[
p(z)=\mathcal N(0,I),
\]

但 fixed standard-normal prior 是一种模型选择，不是 CVAE 的定义。

## Recognition / Inference Model

训练数据提供 paired observations：

\[
(c,y).
\]

真实 posterior 为：

\[
p_\theta(z\mid c,y),
\]

通常难以直接计算，因此使用：

\[
q_\phi(z\mid c,y)
\]

进行 approximate inference。

常见 Gaussian parameterization 为：

\[
q_\phi(z\mid c,y)
=
\mathcal N
\left(
\mu_\phi(c,y),
\operatorname{diag}(\sigma_\phi^2(c,y))
\right).
\]

训练时 latent sample 来自 recognition posterior；生成时则不能使用真实 $y$，必须从 prior $p_\theta(z\mid c)$ 获得 $z$。

## Conditional ELBO

目标 conditional log-likelihood：

\[
\log p_\theta(y\mid c)
\]

通常不可直接计算。CVAE 优化 conditional ELBO：

\[
\boxed{
\mathcal L(c,y)
=
\mathbb E_{q_\phi(z\mid c,y)}
[\log p_\theta(y\mid c,z)]
-
D_{KL}
\left(
q_\phi(z\mid c,y)
\|p_\theta(z\mid c)
\right)
}
\]

第一项衡量给定 $c,z$ 后对真实 $y$ 的 likelihood；第二项使 training-time posterior 与 generation-time prior 对齐。

与 VAE 一样，所谓 reconstruction loss 的具体形式取决于 likelihood $p_\theta(y\mid c,z)$。

## Training and Generation

Training：

```text
condition c + target y
          │
          ↓
qφ(z | c, y)
          │
        sample z
          │
          ↓
pθ(y | c, z)
          │
          ├── expected log-likelihood
          └── KL to pθ(z | c)
```

Generation：

```text
condition c
    │
    ↓
pθ(z | c)
    │
  sample z
    │
    ↓
pθ(y | c, z)
    │
    ↓
output y
```

训练与生成的差异是 conditional latent-variable model 的基本结构，而不是实现细节。

## Conditional Multimodality

若同一 $c$ 对应多个合理 outputs：

\[
p(y\mid c)
\]

可能具有多个 modes。

Deterministic squared-error regression：

\[
\hat y=f(c)
\]

倾向于学习 conditional mean。在多模态问题中，该 mean 可能位于多个真实 modes 之间，并不对应合理 sample。

CVAE 通过 $z$ 允许：

\[
(c,z_1)\rightarrow y_1,
\qquad
(c,z_2)\rightarrow y_2,
\]

从而表示 one-to-many conditional relation。

但 latent variable 的存在不保证模型一定学到 distinct modes；Decoder 仍可能忽略 $z$，形成 conditional posterior collapse。

## Conditional Prior

CVAE 的重要设计维度之一是：

\[
p_\theta(z\mid c).
\]

### Fixed Prior

\[
p(z)=\mathcal N(0,I).
\]

该选择简单、稳定，但无法让 latent prior 随 condition 改变。

### Learned Conditional Prior

网络根据 $c$ 输出：

\[
\mu_p(c),
\qquad
\sigma_p(c),
\]

定义：

\[
p_\theta(z\mid c)
=
\mathcal N
(\mu_p(c),\operatorname{diag}(\sigma_p^2(c))).
\]

这样不同 conditions 可以拥有不同 latent distributions。

Posterior-to-prior KL 也相应变为两个 conditional distributions 之间的 KL。

## Condition Injection

Condition $c$ 可以进入模型多个位置，例如：

- recognition network：$q_\phi(z\mid c,y)$；
- prior network：$p_\theta(z\mid c)$；
- Decoder：$p_\theta(y\mid c,z)$。

具体 architecture 可以使用 concatenation、cross-attention、feature modulation、shared encoder 等方式。CVAE 的定义不要求某一种固定 condition-injection mechanism。

## Relationship to VAE

VAE 建模：

\[
p(x).
\]

CVAE 建模：

\[
p(y\mid c).
\]

相应的 inference distributions 为：

\[
q_\phi(z\mid x)
\]

与：

\[
q_\phi(z\mid c,y).
\]

因此 CVAE 不是简单把额外 feature 拼接到 VAE 输入；条件变量进入了整个概率模型的定义。

## 由机制产生的边界

CVAE 的常见限制包括：

- posterior collapse；
- learned prior 与 posterior mismatch；
- latent dimension / family 选择困难；
- likelihood 选择可能导致过度平滑；
- one-to-many structure 不一定自动分解成可解释 latent modes；
- generation diversity 与 conditional fidelity 之间可能存在权衡。

## ACT 中的使用

ACT 使用 CVAE-style latent variable 建模 demonstration variation，但其 condition、posterior input、fixed prior 与 deterministic inference convention 都属于 ACT-specific design。

这些细节见 [CVAE in ACT](/robot-learning/act/cvae-in-act/)，不属于通用 CVAE 的定义。

## Sources

- Sohn, Lee, Yan. *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS, 2015.
