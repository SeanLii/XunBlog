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

Latent Variable 是概率模型中没有被数据直接观测到、但被模型用来解释观测数据结构的随机变量。

## 定义

设观测变量为 $x$，latent variable 为 $z$。一个 latent-variable generative model 可以写成联合分布

\[
p_\theta(x,z)=p_\theta(x|z)p(z).
\]

这里 $z$ 没有直接出现在训练数据的观测列中，但模型假设数据生成过程受到 $z$ 的影响。

观测数据的概率需要把 latent variable 积分或求和掉：

\[
p_\theta(x)=\int p_\theta(x|z)p(z)\,dz.
\]

这就是“latent”的正式含义：$z$ 存在于模型中，却没有直接被观测。

## 直觉

可以把 $z$ 理解成模型内部用于表示未直接观测因素的变量。这只是理解方式，不是说某一维 $z_i$ 必然自动对应一个清晰的人类概念。

例如同一个机器人观测下，人类 demonstrator 可能采用略有不同的动作风格。一个生成模型可以让 latent variable 表示这些不能仅由当前 observation 唯一确定的变化因素。但是否真的形成可解释“风格维度”，取决于训练目标、数据和模型容量。

## Latent Code 与确定性 Feature

神经网络中的 hidden feature 不一定是 latent variable。关键区别在于概率建模：VAE 中 $z$ 被当作随机变量，有 prior、posterior/approximate posterior，并参与概率目标；普通 deterministic encoder 输出的 feature vector 可以没有这些概率结构。

## 在 VAE 中的位置

VAE 设定 prior $p(z)$，用 encoder 近似看到 $x$ 之后的 posterior $q_\phi(z|x)$，再用 decoder $p_\theta(x|z)$ 解释或生成数据。这样 latent variable 同时连接了生成过程和 inference process。

## Sources

- [Auto-Encoding Variational Bayes — Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
