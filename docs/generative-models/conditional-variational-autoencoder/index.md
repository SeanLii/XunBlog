---
title: "Conditional Variational Autoencoder"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/conditional-variational-autoencoder/"
prerequisites:
  - "/generative-models/variational-autoencoder/"
  - "/mathematics/probability/conditional-probability/"
related:
  - "/robot-learning/act/cvae-in-act/"
---

# Conditional Variational Autoencoder

Conditional Variational Autoencoder（CVAE）把 VAE 扩展到条件生成：模型不是只学习 $y$ 的整体分布，而是学习在给定条件 $x$ 后，输出 $y$ 的条件分布。

## Conditional Generative Model

一个一般的 CVAE 可以写成

\[
p_\theta(y,z|x)
=
p_\theta(y|x,z)p_\theta(z|x).
\]

这里：

- $x$ 是条件；
- $y$ 是希望建模或生成的输出；
- $z$ 是 latent variable；
- $p_\theta(z|x)$ 是 conditional prior；
- $p_\theta(y|x,z)$ 是 conditional decoder。

原始 CVAE 论文的核心目标是处理“同一个输入条件可以对应多个合理输出”的 structured prediction 问题。

## Recognition Model

训练时同时知道 $x$ 和真实 $y$，因此可以定义 approximate posterior / recognition model

\[
q_\phi(z|x,y).
\]

它利用目标 $y$ 帮助推断这一个训练样本可能对应怎样的 latent factor。

## Conditional ELBO

条件 log-likelihood 的下界为

\[
\log p_\theta(y|x)
\ge
\mathbb E_{q_\phi(z|x,y)}
[\log p_\theta(y|x,z)]
-
D_{\mathrm{KL}}
\left(
q_\phi(z|x,y)\|p_\theta(z|x)
\right).
\]

第一项要求 decoder 在给定条件 $x$ 和 latent $z$ 时能解释真实输出 $y$。第二项要求 training-time recognition distribution 与 inference-time prior 保持兼容。

## Prior 不一定固定

CVAE 的“conditional”并不意味着 prior 必须是 $\mathcal N(0,I)$。原始 CVAE formulation 可以让 prior 本身依赖 $x$：

\[
p_\theta(z|x).
\]

一些实际模型为了简化，会使用固定 standard normal prior。那是具体设计选择，不是 CVAE 定义本身。

## Inference

推理时没有真实 $y$ 可供 recognition model 使用，因此不能依赖

\[
q_\phi(z|x,y).
\]

应从 prior $p_\theta(z|x)$ 得到 $z$，再通过 $p_\theta(y|x,z)$ 生成输出。可以随机采样以得到多样结果，也可以在某些任务中选择 prior mean 得到 deterministic output。

ACT 属于后者：它使用 standard normal prior，并在推理时固定 $z=0$。这种用法属于 [CVAE in ACT](/robot-learning/act/cvae-in-act/)，不应反过来改写 CVAE 的通用定义。

## Sources

- [Learning Structured Output Representation using Deep Conditional Generative Models — Sohn, Lee, Yan, 2015](https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
