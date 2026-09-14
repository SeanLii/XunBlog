---
title: "为什么 ACT 推理时令 z = 0？"
kind: "question"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/why-z-zero-at-inference/"
prerequisites:
  - "/robot-learning/act/cvae-in-act/"
  - "/mathematics/probability/normal-distribution/"
related:
  - "/robot-learning/act/inference/"
---

# 为什么 ACT 推理时令 z = 0？

ACT 的训练阶段会从 demonstration action chunk 推断 latent $z$，但推理阶段直接令

\[
z=0.
\]

这看起来反直觉，是因为 training-time posterior 和 inference-time prior 很容易被混成同一个东西。

## 训练时 z 从哪里来

训练时知道真实 action chunk $A_t$。Latent encoder 学习

\[
q_\phi(z\mid q_t,A_t)
=
\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
\]

然后 sample

\[
z=\mu+\sigma\odot\epsilon.
\]

因此这个 $z$ 可以携带“这条 demonstration 具体采用了哪种细节”的信息。

## 但部署时没有 A_t

机器人在真正执行时只知道当前 observation。未来正确动作正是模型要预测的东西，所以不可能先把它送进 latent encoder。

因此

```text
q_t + true future actions → latent encoder → z
```

这整条路径只能属于 training。

## KL 把 posterior 拉向哪里

ACT 选择 prior

\[
p(z)=\mathcal N(0,I).
\]

训练时 KL loss 约束

\[
q_\phi(z\mid q_t,A_t)
\]

不要离这个 prior 太远。

Standard normal 的均值是

\[
\mathbb E[z]=0.
\]

因此 inference 直接使用 $z=0$，相当于选择 prior 的中心位置，而不是随机采一个 demonstration style。

## z=0 不是“latent 没有作用”

一个常见误解是：既然部署时永远是 0，那训练 latent branch 岂不是没用？

不是。训练期间，decoder 在不同 $z$ 下学习重建不同 demonstrations；与此同时 KL 让这些 latent codes 围绕 standard normal 形成受约束的空间。$z=0$ 是这个训练过程中的一个有意义输入位置，而不是一个“把 latent 删除掉”的操作。

而且 zero vector 进入 policy 前还会经过 learned linear projection：

\[
e_z=W_z 0+b_z=b_z.
\]

所以即使 raw latent 为 0，进入 Transformer 的 latent feature 也不一定是全零向量。

## 固定 prior mean 与随机采样的区别

从 CVAE 理论上完全可以：

\[
z\sim\mathcal N(0,I).
\]

这样每次 inference 可能产生不同 action chunk。

但机器人控制通常希望同一 observation 下的执行具有稳定性。ACT 论文选择 prior mean 来得到 deterministic policy，避免 deployment 时额外的 latent sampling variability。

所以“CVAE 是生成模型”与“ACT inference 是确定性的”并不矛盾。

## 一个直观例子

假设不同 demonstrations 在“抬起杯子”的微小姿态上有差异：

```text
style A: 稍微向左
style B: 几乎垂直
style C: 稍微向右
```

训练 latent 可以帮助模型不要被迫把这些差异全部解释成 observation 的确定函数。

部署时使用 $z=0$ 可以理解成选择 latent prior 的中心，而不是明确指定“style A/B/C”中的某一个极端样本。

这只是帮助理解的直觉；正式定义仍然是：released ACT inference 把 latent sample 设为 zero vector，而 training objective 用 KL 将 approximate posterior 约束到 standard normal prior。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official implementation: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
