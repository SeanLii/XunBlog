---
title: "Latent Variable"
kind: "canonical"
domain: "Mathematics / Probability"
parent: "Probability"
canonical: "/mathematics/probability/latent-variable/"
prerequisites:
  - "/mathematics/probability/random-variable/"
  - "/mathematics/probability/probability-distribution/"
related:
  - "/mathematics/probability/variational-inference/"
  - "/generative-models/variational-autoencoder/"
---

# Latent Variable

Latent Variable（潜变量）是概率模型中没有被直接观测、但参与数据生成或概率结构的随机变量。

设 observed variable 为 $x$，latent variable 为 $z$。一个基本 latent-variable model 可以写成：

\[
p(x,z)=p(z)p(x\mid z).
\]

其中：

- $p(z)$ 描述 latent variable 的先验分布；
- $p(x\mid z)$ 描述给定 $z$ 后 observation 的分布；
- $z$ 没有直接出现在观测数据中。

Latent variable 的定义来自概率模型中的“未观测随机变量”这一身份，而不是来自神经网络内部是否存在 hidden activation。

## Observed Variable 与 Latent Variable

概率模型中的变量可以按是否直接进入数据记录区分为 observed 与 latent。

例如 Gaussian mixture model 可以写成：

\[
p(x,z)=p(z)p(x\mid z),
\]

其中：

- $x$ 是实际观测到的数据点；
- $z$ 是 component identity。

训练数据只提供 $x$，而每个 sample 属于哪个 component 需要通过模型推断。

类似地，topic model 中 document 的 topic proportions、state-space model 中系统的 hidden state、VAE 中的 $z$ 都可以作为 latent variables。

## Marginal Distribution

因为 $z$ 没有直接观测到，observed distribution 需要通过 marginalization 得到。

连续 $z$：

\[
p(x)
=
\int p(x,z)\,dz
=
\int p(x\mid z)p(z)\,dz.
\]

离散 $z$：

\[
p(x)
=
\sum_z p(x,z)
=
\sum_z p(x\mid z)p(z).
\]

因此 latent-variable model 可以把复杂的 observed distribution 表示为多个 latent-conditioned distributions 的组合。

## Posterior Inference

观测到 $x$ 后，对 latent variable 的不确定性由 posterior 表示：

\[
p(z\mid x)
=
\frac{p(x\mid z)p(z)}{p(x)}.
\]

它描述在已经观察 $x$ 的条件下，不同 latent values 的相对可能性。

posterior inference 和 generative direction 是两个不同问题：

\[
z\rightarrow x
\]

描述生成模型；

\[
x\rightarrow z
\]

描述 inference。

在很多模型中，生成方向容易定义，但 posterior 因为 marginal likelihood

\[
p(x)=\int p(x\mid z)p(z)\,dz
\]

难以计算而变得 intractable。Variational Inference、MCMC 等方法都在处理这类 posterior inference 问题。

## Discrete 与 Continuous Latent Variables

Latent variable 可以是离散的，也可以是连续的。

### Discrete latent variable

例如 mixture component：

\[
z\in\{1,\ldots,K\}.
\]

它适合表示有限类别、模式或离散结构。

### Continuous latent variable

例如：

\[
z\in\mathbb R^d.
\]

它可以表示连续变化因素，并形成 latent space。VAE 常使用 continuous Gaussian latent variables。

两类 latent variables 对 inference 和 optimization 的要求不同。连续可重参数化变量可以使用 pathwise gradient；离散 latent variables 往往需要其他 estimator、relaxation 或 exact marginalization。

## Latent Space

当：

\[
z\in\mathbb R^d,
\]

所有可能的 latent vectors 构成 latent space。

如果 decoder / likelihood model 为：

\[
p_\theta(x\mid z),
\]

latent space 中的位置决定 observation distribution 的参数。

Latent space 的平滑性、cluster structure、disentanglement 和 semantic directions 都不是由“存在 $z$”自动保证的，而取决于 prior、likelihood、training objective、model capacity、data distribution 与 inductive bias。

## Identifiability

Latent-variable models 经常存在 identifiability 问题：不同 latent parameterizations 可能产生相同 observed distribution。

若某个变换 $g$ 可逆，可以用：

\[
z'=g(z)
\]

重新表示 latent variable，并相应修改 decoder，而保持 $p(x)$ 不变。

因此仅从 observations 出发，未必能够唯一恢复所谓“真实 latent coordinates”。某个 latent dimension 是否对应 pose、lighting 或 style，不只由数据决定，还受到 model assumptions 与 supervision 影响。

## Conditional Latent-Variable Model

若模型还包含 observed condition $c$，可以写成：

\[
p(y,z\mid c)
=
p(z\mid c)p(y\mid c,z).
\]

此时：

- $c$ 是已观测条件；
- $z$ 表示在相同 $c$ 下仍未被条件解释的随机变化；
- $y$ 是需要建模的 output。

这类结构适合 one-to-many conditional prediction。

[Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) 是其中一种常见实现。

## 与 Hidden Representation 的区别

神经网络 hidden representation 通常是输入与参数的确定性函数，例如：

\[
h=f_\theta(x).
\]

如果模型没有为 $h$ 定义随机变量及其概率分布，$h$ 通常不称为 probabilistic latent variable。

相比之下，latent-variable model 明确定义：

\[
z\sim p(z)
\]

或：

\[
z\sim q_\phi(z\mid x),
\]

并对 $z$ 的概率分布进行建模、积分或近似推断。

## 在生成模型中的作用

Latent variables 为生成模型提供分层概率结构：

\[
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z).
\]

这样可以通过潜在结构组织复杂的 observed distribution。

Latent variable 本身不是 VAE 专属概念。VAE 的特殊之处在于使用 neural generative model、amortized variational inference 和 reparameterized gradient 来学习 latent-variable model。

## Connections

- [Variational Inference](/mathematics/probability/variational-inference/)：近似 latent posterior。
- [Variational Autoencoder](/generative-models/variational-autoencoder/)：用 neural networks 学习连续 latent-variable model。
- [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)：在 observed condition 下引入 latent variation。
