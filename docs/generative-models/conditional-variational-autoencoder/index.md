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

Conditional Variational Autoencoder（CVAE）可以先理解成：**在已知条件 $x$ 的情况下，对可能输出 $y$ 的分布进行 latent-variable modeling。**

普通确定性模型通常学习

\[
y=f(x).
\]

它隐含地把一个输入对应成一个主要答案。但很多任务天然存在“一对多”：同一个 $x$ 可能对应多个都合理的 $y$。

CVAE 希望学习的不是单一映射，而是

\[
p_\theta(y\mid x).
\]

它通过额外的 latent variable $z$ 表示那些没有被条件 $x$ 完全决定的变化：

```text
condition x ───────────────┐
                           ↓
latent z ─────────→ decoder pθ(y|x,z)
                           │
                           ↓
                           y
```

## Latent z 表示条件未决定的变化

假设 $x$ 只提供了部分信息。对于同一个 $x$，数据中可能出现 $y_1,y_2,y_3$ 三种不同但都合理的输出。

如果模型只有

\[
y=f(x),
\]

它必须把这些差异全部压进同一个确定性答案里。

加入 $z$ 后，模型可以写成

\[
y\sim p_\theta(y\mid x,z),
\qquad z\sim p_\theta(z\mid x).
\]

于是 $x$ 表示已经知道的条件，$z$ 表示在这个条件下仍然需要补充的潜在变化。

## Training：看到 x 和 y以后推断 z

训练时我们同时知道条件 $x$ 和真实输出 $y$。因此可以训练 recognition / inference model：

\[
q_\phi(z\mid x,y).
\]

它回答的是：

> 已经看到这组 $(x,y)$ 以后，什么样的 latent $z$ 可以解释这个输出为什么是这样？

数据流为：

```text
x ─────────────┐
               ├→ qφ(z|x,y) → z ──┐
y ─────────────┘                   │
                                   ↓
x ───────────────────────→ pθ(y|x,z)
                                   │
                                   ↓
                                  y_hat
```

然后模型要求 $\hat y$ 能解释真实 $y$。

## Generation：没有真实 y 时从 prior 得到 z

真正生成时，真实 $y$ 当然还不存在，因此不能再使用 $q_\phi(z\mid x,y)$。模型需要从 conditional prior

\[
p_\theta(z\mid x)
\]

得到 $z$，再通过

\[
p_\theta(y\mid x,z)
\]

生成输出。

因此 CVAE 的核心不是“encoder 和 decoder”这两个神经网络名词，而是三种概率关系：

\[
q_\phi(z\mid x,y),\qquad
p_\theta(z\mid x),\qquad
p_\theta(y\mid x,z).
\]

## Conditional ELBO

对 conditional likelihood $\log p_\theta(y\mid x)$，CVAE 使用 variational lower bound：

\[
\log p_\theta(y\mid x)
\ge
\mathbb E_{q_\phi(z\mid x,y)}
[\log p_\theta(y\mid x,z)]
-
D_{KL}\left(
q_\phi(z\mid x,y)
\|p_\theta(z\mid x)
\right).
\]

第一项要求 sampled latent 与 condition 一起能生成正确输出；第二项让 training-time posterior approximation 靠近 generation-time prior。

这样做的原因非常直接：训练时 encoder 看得到 $y$，推理时看不到。如果两边的 latent distribution 完全不相干，那么训练好的 decoder 到真正生成时就会接收到完全陌生的 $z$。

## 原始 CVAE 与 ACT 中 CVAE 的区别

通用 CVAE 并不要求 prior 一定是 $\mathcal N(0,I)$，也不要求推理一定取 $z=0$。原始 CVAE formulation 可以学习 conditional prior

\[
p_\theta(z\mid x).
\]

ACT 做了更具体的设计：它把 prior 固定为 standard normal，并在部署时取其均值 $z=0$。

因此：

- “CVAE 可以有 conditional prior”属于这个页面；
- “ACT 为什么固定 standard normal，以及为什么 inference 用 $z=0$”属于 [CVAE in ACT](/robot-learning/act/cvae-in-act/) 与 [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)。

## 一个最小 mental model

把 CVAE 压缩成一张图：

```text
TRAIN
x + y ──→ infer z ──→ x + z ──→ reconstruct y
             │
             └── kept close to prior p(z|x)

GENERATE
x ──→ prior z ──→ x + z ──→ generate y
```

只要这张图清楚，后面的 ELBO、reparameterization、Gaussian latent 才有落脚点。

## Sources

- Sohn, Lee & Yan, **Learning Structured Output Representation using Deep Conditional Generative Models**, NeurIPS 2015. https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
- Kingma & Welling, **Auto-Encoding Variational Bayes**. https://arxiv.org/abs/1312.6114
