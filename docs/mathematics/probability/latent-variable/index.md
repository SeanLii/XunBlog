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

Latent Variable 是 probabilistic model 中**没有被直接观测到的 random variable**。

如果 observed variable 是 $x$，latent variable 是 $z$，一个常见 generative model 写成：

\[
p(x,z)=p(z)p(x\mid z).
\]

这里：

- $z$ 先按照 prior $p(z)$ 产生；
- 然后 $x$ 根据 conditional distribution $p(x\mid z)$ 产生。

我们通常只能看到 $x$，看不到数据生成过程中真正的 $z$。

## Latent 不等于“隐藏层”

Neural network 的 hidden activation 也没有直接暴露给用户，但这并不自动让它成为 probabilistic latent variable。

Latent variable 的关键是：

> **它是概率模型中的随机变量，并且在 observation 中没有直接给出。**

例如 mixture model 中“这个 sample 属于哪个 cluster”的 cluster identity 可以是 latent variable；topic model 中 document topic proportions 可以是 latent；VAE 中 $z$ 也是 latent random variable。

## Latent Variable 的作用

很多 observed data 具有复杂变化，而我们希望把这些变化解释成一些未观测因素。

例如一个 image distribution 可能受到：

- object identity；
- pose；
- lighting；
- background；
- style；
- noise。

Latent-variable model 并不保证自动把每个 latent dimension 精确对应这些人类概念，但通过 $z$ 可以让 model 表达：

> 同一个 observed distribution 背后存在没有直接观测的随机因素。

## Marginalization

因为 $z$ 没有被直接观察，计算 observed data distribution 时要把它积分掉：

\[
p(x)=
\int p(x,z)\,dz
=
\int p(x\mid z)p(z)\,dz.
\]

离散 latent variable 则使用 sum：

\[
p(x)=
\sum_z p(x\mid z)p(z).
\]

这叫 marginalization。

一个 latent model 的生成过程可能简单，但这个积分可能非常难算。

## Posterior

观察到 $x$ 后，我们自然会问：

> 哪些 latent values $z$ 更可能产生这个 $x$？

这由 posterior 表示：

\[
p(z\mid x)
=\frac{p(x\mid z)p(z)}{p(x)}.
\]

由于 denominator：

\[
p(x)=\int p(x\mid z)p(z)\,dz
\]

可能难以计算，posterior 也常变得 intractable。

这就是 [Variational Inference](/mathematics/probability/variational-inference/) 等 approximate inference 方法出现的核心背景。

## Latent Space

当：

\[
z\in\mathbb R^d,
\]

所有可能 $z$ 构成 latent space。

在 learned generative model 中，不同 region 的 latent values 会通过 decoder / likelihood model 对应不同 output distributions。

但“latent space 是模型自动找到的语义地图”不是定义。它是否具有平滑、可解释、disentangled structure，取决于 model、objective、data 与 inductive biases。

## Generative Direction 与 Inference Direction

Generative model：

\[
z\rightarrow x
\]

描述“latent 怎样产生 observation”。

Inference：

\[
x\rightarrow z
\]

描述“看到 observation 后怎样反推 latent”。

这两个方向不要混在一起。

VAE 的 decoder / generative model 负责前者，encoder / recognition model 负责近似后者。

## Conditional Latent Variable

如果还有 observed condition $c$：

\[
p(y,z\mid c)
=p(z\mid c)p(y\mid c,z).
\]

这类 conditional latent-variable model 可以表示：同一个 condition 下仍然存在多种可能 outputs。

[Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) 就建立在这个结构上。
