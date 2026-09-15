---
title: "Variational Autoencoder"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/variational-autoencoder/"
prerequisites:
  - "/deep-learning/representation-learning/autoencoder/"
  - "/mathematics/probability/latent-variable/"
  - "/mathematics/probability/variational-inference/evidence-lower-bound/"
related:
  - "/mathematics/probability/variational-inference/reparameterization-trick/"
  - "/generative-models/conditional-variational-autoencoder/"
  - "/generative-models/posterior-collapse/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Variational Autoencoder

> **知识边界**：本文的 canonical 对象是 **Variational Autoencoder**。依赖机制由 [Autoencoder](/deep-learning/representation-learning/autoencoder/)、[Latent Variable](/mathematics/probability/latent-variable/)、[Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Variational Autoencoder（VAE）是将 **latent-variable generative model** 与 **amortized variational inference** 结合起来的概率生成模型框架。

它同时包含两个方向：

\[
\text{generative model:}
\qquad
z\rightarrow x,
\]

\[
\text{inference model:}
\qquad
x\rightarrow q_\phi(z\mid x).
\]

Generative model 定义：

\[
p_\theta(x,z)
=
p(z)p_\theta(x\mid z),
\]

Inference network 定义 approximate posterior：

\[
q_\phi(z\mid x)
\approx
p_\theta(z\mid x).
\]

VAE 的目标不是把输入压缩后再简单重建，而是学习一个可以通过 latent prior 采样的概率模型，同时训练一个神经网络近似 latent posterior。

## Generative Model

VAE 首先假设 latent variable：

\[
z\sim p(z).
\]

常见 prior 为：

\[
p(z)=\mathcal N(0,I).
\]

给定 $z$，Decoder 参数化 observation likelihood：

\[
p_\theta(x\mid z).
\]

因此 joint distribution 为：

\[
p_\theta(x,z)
=
p(z)p_\theta(x\mid z).
\]

Observed-data distribution 需要 marginalize latent variable：

\[
p_\theta(x)
=
\int p_\theta(x\mid z)p(z)\,dz.
\]

这一步通常没有可直接计算的 closed form，是 VAE 需要 approximate inference 的根本原因。

## Posterior Inference

观察到 $x$ 后，真实 latent posterior 为：

\[
p_\theta(z\mid x)
=
\frac{p_\theta(x\mid z)p(z)}{p_\theta(x)}.
\]

由于 denominator：

\[
p_\theta(x)
=
\int p_\theta(x\mid z)p(z)\,dz
\]

通常难以计算，VAE 引入 inference network：

\[
q_\phi(z\mid x).
\]

对于常见 diagonal Gaussian posterior：

\[
q_\phi(z\mid x)
=
\mathcal N
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
\]

Encoder 因此不直接输出一个 latent point，而是输出 distribution parameters，例如：

\[
\mu_\phi(x),
\qquad
\log\sigma_\phi^2(x).
\]

## Amortized Inference

传统 variational inference 可以为每个 observation 单独优化 variational parameters。VAE 改为用共享 neural network：

\[
x
\mapsto
\phi(x),
\]

直接得到该 observation 的 posterior approximation。

这种方法称为 amortized inference：训练阶段学习一个通用 inference function，新 observation 到来时不需要重新从头优化 posterior。

代价是可能出现 amortization gap：共享 Encoder 未必能对每个 sample 都达到 variational family 内的最优 posterior approximation。

## Evidence Lower Bound

VAE 希望最大化 marginal likelihood：

\[
\log p_\theta(x),
\]

但该量通常不可直接计算。于是优化 [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/)：

\[
\boxed{
\mathcal L(x)
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
D_{KL}
\left(
q_\phi(z\mid x)
\|p(z)
\right)
}
\]

并且：

\[
\log p_\theta(x)
=
\mathcal L(x)
+
D_{KL}
\left(
q_\phi(z\mid x)
\|p_\theta(z\mid x)
\right).
\]

因此 ELBO 与 log evidence 之间的 gap 恰好是 approximate posterior 与 true posterior 的 KL divergence。

## Expected Log-Likelihood

第一项：

\[
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
\]

要求从 approximate posterior 采样得到的 latent variable 能让 Decoder 对 observation 赋予较高 likelihood。

它经常被称为 reconstruction term，但其正式含义是 **expected log-likelihood**。

具体 loss 形式取决于 likelihood family。

### Gaussian Likelihood

若：

\[
p_\theta(x\mid z)
=
\mathcal N
(\mu_\theta(z),\sigma_x^2I),
\]

且 $\sigma_x$ 固定，则 negative log-likelihood 与 squared reconstruction error 只差常数和 scale。

### Bernoulli Likelihood

对于适合 Bernoulli observation model 的数据，negative log-likelihood 具有 binary cross-entropy 形式。

因此 VAE 的 reconstruction term 并不固定为 MSE；其形式由 likelihood $p_\theta(x\mid z)$ 决定。

## KL Term

第二项：

\[
D_{KL}
\left(
q_\phi(z\mid x)
\|p(z)
\right)
\]

约束每个 observation 的 approximate posterior 与 prior 之间的差异。

对：

\[
p(z)=\mathcal N(0,I),
\]

以及 diagonal Gaussian posterior，有 closed form：

\[
D_{KL}(q\|p)
=
\frac12
\sum_j
\left(
\mu_j^2
+
\sigma_j^2
-
1
-
\log\sigma_j^2
\right).
\]

这项约束使 latent codes 不能任意散布在没有统一概率结构的空间中，但它也限制 posterior 能携带的信息量。

## Reparameterized Sampling

ELBO 的 likelihood term 需要从：

\[
z\sim q_\phi(z\mid x)
\]

采样。为了对 $\phi$ 使用低方差 pathwise gradient，VAE 使用 [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/)：

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon.
\]

随机性被隔离到与 $\phi$ 无关的 $\epsilon$，而 $z$ 对 $\mu,\sigma$ 的变换保持可微。

## Complete Training Flow

一条训练样本的数据流为：

```text
x
│
↓
Inference Network / Encoder
│
├── μ(x)
└── log σ²(x)
       │
       ↓
qφ(z|x)
       │
ε ~ N(0,I)
       │
       ↓
z = μ + σ ⊙ ε
       │
       ↓
Generative Network / Decoder
       │
       ↓
parameters of pθ(x|z)
       │
       ├── expected log-likelihood
       └── KL(qφ(z|x) || p(z))
```

Encoder 与 Decoder 通过同一个 ELBO 联合优化。

## Generation

训练完成后，生成过程不需要真实 observation 作为 Encoder 输入。

直接从 prior 采样：

\[
z\sim p(z),
\]

再从 likelihood 生成：

\[
x\sim p_\theta(x\mid z).
\]

若 Decoder 输出 likelihood mean，也可以使用该 mean 作为 deterministic reconstruction / visualization；但这与从完整 likelihood 采样是不同操作。

## Reconstruction 与 Generation

Reconstruction 使用：

\[
x
\rightarrow
q_\phi(z\mid x)
\rightarrow
z
\rightarrow
p_\theta(x\mid z).
\]

Generation 使用：

\[
z\sim p(z)
\rightarrow
p_\theta(x\mid z).
\]

因此 reconstruction quality 主要测试 inference + decoding；prior generation 则测试 learned generative model 在 prior samples 上的行为。

## Latent Geometry

KL regularization 让 approximate posteriors 受到共同 prior 的约束，因此 latent space 通常比普通 deterministic Autoencoder 更适合 interpolation 与 prior sampling。

但以下性质都不是 VAE 自动保证的：

- disentanglement；
- semantic axis alignment；
- uniform perceptual interpolation；
- every prior point corresponds to equally realistic data。

这些性质受 prior、likelihood、architecture、ELBO weighting 与 data distribution 共同影响。

## Posterior Family

Diagonal Gaussian 是常见但受限的 variational family：

\[
q_\phi(z\mid x)
=
\prod_j q_\phi(z_j\mid x).
\]

它无法直接表达复杂 posterior correlations 或 multimodality。

更灵活的 approximate posterior 可以通过 full covariance、normalizing flows、hierarchical latent variables 等方式构造。更强 posterior family 可以缩小 approximation gap，但会增加 inference 和 optimization complexity。

## Posterior Collapse

如果 Decoder 能在很少使用 $z$ 的情况下获得较高 likelihood，模型可能出现：

\[
q_\phi(z\mid x)
\approx
p(z).
\]

此时 latent variable 与 observation 的依赖显著减弱。这称为 [Posterior Collapse](/generative-models/posterior-collapse/)。

Posterior collapse 与 Decoder capacity、optimization dynamics、KL pressure、data structure 等因素有关，并不是“VAE 必然发生”的现象。

## Variants and Objective Modifications

VAE framework 可以改变：

- prior；
- posterior family；
- likelihood family；
- latent hierarchy；
- ELBO weighting；
- Decoder / Encoder architecture。

例如 $\beta$-VAE 使用：

\[
\mathcal L_{\beta}
=
\mathbb E_q[\log p_\theta(x\mid z)]
-
\beta D_{KL}(q\|p),
\]

改变 reconstruction 与 latent regularization 的权衡。这样的 modified objectives 不再与标准 ELBO 完全相同，需要单独理解其建模目标。

## Relationship to Autoencoder

普通 Autoencoder 与 VAE 都包含 Encoder / Decoder，因此表面结构相似，但概率含义不同：

| | Autoencoder | VAE |
|---|---|---|
| Encoder output | deterministic code $h$ | distribution $q_\phi(z\mid x)$ |
| Latent prior | 不要求 | 明确定义 $p(z)$ |
| Decoder | reconstruction function | likelihood model $p_\theta(x\mid z)$ |
| Objective | reconstruction / regularized reconstruction | ELBO |
| prior sampling | 没有统一定义 | 是标准生成过程 |

VAE 因此不是“Autoencoder 加随机噪声”，而是一个 probabilistic latent-variable model 加 amortized inference network。

## Sources

- Kingma & Welling. *Auto-Encoding Variational Bayes*. ICLR, 2014.
- Rezende, Mohamed, Wierstra. *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. ICML, 2014.
- Kingma & Welling. *An Introduction to Variational Autoencoders*. 2019.
