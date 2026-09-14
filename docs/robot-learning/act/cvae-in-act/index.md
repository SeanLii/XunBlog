---
title: "CVAE in ACT"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/cvae-in-act/"
prerequisites:
  - "/generative-models/conditional-variational-autoencoder/"
related:
  - "/robot-learning/act/why-z-zero-at-inference/"
  - "/robot-learning/act/training/"
---

# CVAE in ACT

CVAE in ACT 只描述 ACT 怎样使用 Conditional Variational Autoencoder。CVAE 的一般概率理论、ELBO 与 reparameterization 分别属于它们自己的 canonical pages，这里只建立 ACT 中各变量的对应关系。

## Variable Mapping

通用 CVAE 可以写成：条件 $x$、目标输出 $y$、latent $z$。

在 ACT 中对应为：

\[
x\longleftrightarrow o_t,
\]

\[
y\longleftrightarrow A_t=a_{t:t+k-1},
\]

\[
z\longleftrightarrow \text{style variable}.
\]

Policy decoder 因此建模

\[
\pi_\theta(A_t|o_t,z).
\]

它要生成的不是图像 reconstruction，而是 future action chunk。

## Recognition Model

训练时 ground-truth action chunk 已知。ACT 定义 training-only encoder

\[
q_\phi(z|A_t,\bar o_t),
\]

其中 $\bar o_t$ 表示去掉 image observations 后的 observation。论文明确说明，为了加快训练，CVAE encoder 不使用图像，只使用 proprioceptive joint observation 和 action sequence。

Released implementation 对应输入是：

```text
[CLS], current qpos, action_1, ..., action_k
```

Transformer encoder 的 CLS output 再投影成 diagonal Gaussian 的

\[
\mu,
\qquad
\log\sigma^2.
\]

## Latent Distribution

ACT 使用

\[
q_\phi(z|A_t,\bar o_t)
=
\mathcal N(
\mu_\phi,
\operatorname{diag}(\sigma_\phi^2)
).
\]

Prior 为

\[
p(z)=\mathcal N(0,I).
\]

Released implementation 中 latent dimension 固定为 32。

Training sample 通过

\[
\epsilon\sim\mathcal N(0,I),
\]

\[
z=\mu+\sigma\odot\epsilon
\]

得到。这一步属于 [Reparameterization Trick](/generative-models/reparameterization-trick/)。

## Policy Decoder

得到 $z$ 后，ACT policy 使用完整 current observation：

\[
o_t=(\text{images}_t,q_t).
\]

因此 decoder 输入的条件比 recognition encoder 更完整：图像只在真正预测 action 的 policy 路径中使用。

模型学习

\[
(o_t,z)\mapsto \hat A_t.
\]

训练时 reconstruction objective 迫使 $z$ 与 observation 一起解释 demonstration action chunk；[KL Divergence](/mathematics/information-theory/kl-divergence/) 则推动 $q_\phi$ 与 standard normal prior 保持接近。

## Beta Weight

论文写成

\[
\mathcal L
=
\mathcal L_{reconst}
+
\beta D_{KL}(q_\phi\|\mathcal N(0,I)).
\]

因为训练代码最小化 loss，所以这里 KL 是正向加到 loss 上。论文指出更大的 $\beta$ 会减少通过 $z$ 传递的信息。默认实验使用

\[
\beta=10.
\]

这个数值是实验配置，不是 CVAE 的固定常数。

## Style Variable 的准确含义

论文把 $z$ 称为 style variable，因为它被用来吸收 human demonstrations 中同一 observation 下可能存在的不同 trajectory choices。

但不能进一步断言某一 latent dimension 一定对应“手快/手慢”“左偏/右偏”等具体人类概念。训练没有给这些语义标签。更准确的说法是：$z$ 为 decoder 提供一个受 KL regularization 的 latent channel，用来表示 observation 之外仍能帮助解释 action chunk 的变化。

## Inference Boundary

推理时真实 future action chunk 不存在，所以不能计算

\[
q_\phi(z|A_t,\bar o_t).
\]

ACT 直接丢弃 CVAE encoder，并固定

\[
z=0,
\]

即 prior mean。这样 decoder 变为 deterministic policy。这个选择的详细含义单独见 [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
