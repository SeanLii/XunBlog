---
title: "Autoencoder"
kind: "canonical"
domain: "Deep Learning / Representation Learning"
parent: "Deep Learning"
canonical: "/deep-learning/representation-learning/autoencoder/"
prerequisites:
  - "/deep-learning/core/multilayer-perceptron/"
related:
  - "/generative-models/variational-autoencoder/"
---

# Autoencoder

Autoencoder 是一个训练成“把输入压到 intermediate representation，再重建原输入”的 neural network structure。

最基本的数据流：

```text
x
↓
Encoder
↓
h
↓
Decoder
↓
x_hat
```

形式上：

\[
h=f_\phi(x),
\]

\[
\hat x=g_\theta(h).
\]

训练目标让：

\[
\hat x\approx x.
\]

例如 squared reconstruction loss：

\[
L(x,\hat x)=\|x-\hat x\|_2^2.
\]

## Autoencoder 学的不是“复制”这么简单

如果 encoder 和 decoder capacity 无限，而且 intermediate representation 没有限制，网络完全可能学到 trivial identity mapping。

所以 Autoencoder 真正有意义的关键在于给 representation path 加约束，例如：

- bottleneck dimension 较小；
- sparsity constraint；
- denoising objective；
- regularization；
- architecture constraints。

这些限制迫使模型保留对 reconstruction 有用的信息，而不是直接逐值复制。

## Bottleneck Autoencoder

如果：

\[
x\in\mathbb R^D,
\qquad
h\in\mathbb R^d,
\qquad d<D,
\]

encoder 把高维 input 压缩到低维 code。

```text
high-dimensional x
       ↓
     encoder
       ↓
 low-dimensional h
       ↓
     decoder
       ↓
 reconstructed x_hat
```

Hinton 与 Salakhutdinov 2006 的工作展示了 deep autoencoder 用于 nonlinear dimensionality reduction 的代表性思路。

## Representation 是 Deterministic 的

普通 Autoencoder 常写成：

\[
h=f_\phi(x).
\]

给定同一个 $x$，encoder 直接输出一个 deterministic code $h$。

这和 VAE 有根本区别。

VAE encoder 通常不是直接输出一个 latent point，而是输出 approximate posterior distribution 的 parameters：

\[
q_\phi(z\mid x).
\]

然后从 distribution 中 sample $z$。

## Reconstruction Objective 决定保留什么

Autoencoder 不会自动学习“最有语义的 features”。

它被直接监督的是 reconstruction：哪些 information 对恢复 input 有用，就有动力被保留。

如果 data 中 pixel-level details 很容易支配 loss，representation 也可能优先保存这些细节，而不是人类希望的 abstract semantics。

所以 learned code 的含义取决于 data、architecture 和 objective。

## Denoising Autoencoder

一种重要变体是先破坏 input：

\[
\tilde x\sim C(\tilde x\mid x),
\]

再让 network 从 $\tilde x$ 重建 clean $x$：

\[
\hat x=g(f(\tilde x)).
\]

这让模型不能只记住 exact identity，而需要学习 data structure 中更稳定的信息。

## Autoencoder 不是天然 Generative Model

普通 Autoencoder 的 latent codes $h$ 不一定服从一个已知、可采样的 distribution。

即使 training examples 在 latent space 中形成某些 clusters，我们也不知道从哪里随机 sample 才一定得到合理 decode result。

因此：

```text
sample random h
↓
decoder
```

不保证产生 data-like samples。

[Variational Autoencoder](/generative-models/variational-autoencoder/) 的一个关键改变，就是给 latent space 建立 explicit probabilistic model 与 prior，使 sampling 有清楚定义。

## Sources

- Hinton & Salakhutdinov. *Reducing the Dimensionality of Data with Neural Networks*. Science, 2006.
