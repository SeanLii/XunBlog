---
title: "Latent Variable"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/latent-variable/"
prerequisites:
  - "/mathematics/probability/random-variable/"
related:
  - "/generative-models/variational-autoencoder/"
  - "/generative-models/conditional-variational-autoencoder/"
---

# Latent Variable

Latent Variable 是概率模型中**没有被直接观测到，但被假设参与生成观测数据**的随机变量。

如果我们直接观察到 $x$，而模型认为背后还有一个隐藏因素 $z$，可以写成：

```text
latent z
   │
   ↓
observed x
```

概率上常写成

\[
p(x,z)=p(z)p(x\mid z).
\]

这里 $z$ 是 latent variable，$x$ 是 observed variable。

## “隐藏”不等于“神秘语义”

Latent variable 不一定天然对应“风格”“姿态”“情绪”这类人类可命名因素。它首先只是模型中的未观测随机变量。

训练后某些 latent dimensions 可能与可解释因素相关，也可能形成分布式 representation，无法给每一维一个简单名称。

所以正式定义应保持概率意义：

> $z$ 没有直接出现在观测数据中，但模型通过它描述数据的生成过程或隐藏结构。

## Marginalizing latent variable

如果只关心 observed data $x$，需要把所有可能的 $z$ 汇总掉：

离散情形：

\[
p(x)=\sum_z p(x,z).
\]

连续情形：

\[
p(x)=\int p(x,z)\,dz.
\]

这一步叫 marginalization。

Latent-variable models 的困难往往正来自这里：对高维 $z$ 的积分可能无法解析计算。

## Posterior 表示“看到 x 后 z 可能是什么”

生成方向是

\[
p(z)p(x\mid z).
\]

但当数据 $x$ 已经观测到时，我们更想知道

\[
p(z\mid x).
\]

这叫 posterior distribution。它回答：**在已经看到 $x$ 的条件下，哪些 latent values 更可能解释这个样本。**

根据 Bayes rule：

\[
p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)}.
\]

问题是分母 $p(x)$ 往往包含难算的积分，所以 posterior 也可能难以直接求得。

这会自然引出 [Variational Inference](/generative-models/variational-inference/)。

## 在 VAE 中的位置

VAE 假设：

\[
z\sim p(z),
\]

\[
x\sim p_\theta(x\mid z).
\]

因为真实 posterior $p_\theta(z\mid x)$ 难算，再训练一个近似 distribution：

\[
q_\phi(z\mid x).
\]

因此 VAE 的 encoder 不是在“生成 latent variable 的定义”，而是在近似观察到 $x$ 后的 latent posterior。

## 在 ACT 中的位置

ACT training 把 future action chunk 中未被当前 observation 完全决定的变化交给 latent $z$ 表示。它仍然是概率模型中的 latent variable；“style”只是帮助理解它可能承载什么信息的直觉。

ACT inference 固定 $z=0$，属于 ACT-specific deployment choice，不改变 latent variable 的通用定义。
