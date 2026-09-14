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

ACT 训练时从 approximate posterior 中采样 latent $z$，推理时却直接令 $z=0$。这不是因为“训练时的 $z$ 没有用”，也不是因为 $z=0$ 等于“没有 latent”。它来自 ACT 对 CVAE prior 和 deterministic policy 的具体选择。

## Training-Time Distribution

训练时 ACT encoder 得到

\[
q_\phi(z|A_t,q_t)
=
\mathcal N
(\mu,\operatorname{diag}(\sigma^2)).
\]

并用

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
\]

产生 sample。

与此同时，KL term 推动这个 posterior 接近 prior：

\[
p(z)=\mathcal N(0,I).
\]

因此 decoder 训练时不能只适应任意散乱的 latent codes；它被鼓励在与 standard normal prior 兼容的 latent region 中工作。

## Inference-Time Constraint

推理时只有当前 observation $o_t$，没有真实 future action chunk $A_t$。因此 training-time encoder

\[
q_\phi(z|A_t,q_t)
\]

缺少必要输入，无法继续使用。

CVAE 的一般做法是在 inference 时从 prior 获得 latent：

\[
z\sim p(z).
\]

ACT 进一步选择不随机采样，而是取 prior 的均值：

\[
\mathbb E_{p(z)}[z]=0.
\]

所以

\[
\boxed{z=0}
\]

是“使用 prior mean 做 deterministic decoding”，不是“把 latent module 删除”。Zero vector 仍然经过 learned latent projection，并进入 policy network。

## Zero 不表示“所有训练因素被消除”

一种容易产生的直觉是：encoder 把睡眠、手抖等额外因素编码进 $z$，推理时令 $z=0$ 就把这些因素全部消除。这个说法可以帮助初步理解“选择中性 style”，但不能当作正式定义。

训练没有告诉模型某一维 latent 对应“睡眠好坏”或“手抖”。$z$ 只是在 loss 约束下学到的 hidden random variable。它可能编码 demonstration variation，也可能以不可解释的方式组合多个因素。

因此更准确的表述是：

> $z$ 为训练提供一个 latent channel，用来帮助解释在 current observation 之外仍存在的 action-sequence variation；推理时 ACT 选择 prior mean 作为统一 deterministic latent condition。

## Zero 的可用条件

有三个相互配合的条件。

第一，policy decoder 本身已经获得强条件信息：多视角 images 与 current joint positions。它不是只依赖 $z$ 决定动作。

第二，KL regularization 把 training posterior 拉向

\[
\mathcal N(0,I),
\]

使 latent codes 不应长期远离 prior structure。Zero 是这个 prior 的 mean，也是 density mode。

第三，ACT 的目标不是在推理时展示 demonstration style diversity，而是稳定执行 manipulation policy。固定 $z$ 去掉了每次 rollout 主动采样 latent 带来的随机 variation。

这些条件解释了这种设计为什么合理，但它不是数学定理：VAE objective 本身并不保证“令 $z=0$ 一定得到任务最优动作”。ACT 论文用实验验证了这一具体设计在其任务中的效果。

## High-Dimensional Gaussian 的一个细节

Released ACT latent dimension 为 32。对高维 standard normal，$z=0$ 虽然是 density 最大的点和均值，但典型随机 sample 的范数并不接近 0；其尺度大约在

\[
\sqrt{d_z}
\]

附近。

因此不能用“训练时经常采样到接近全零向量，所以推理用 0”来解释。精确的零向量在连续分布下概率为 0，而且高维 Gaussian samples 通常位于离原点有一定半径的区域。

更可靠的解释仍然是：prior mean 是一个确定、对称、任务无额外 style preference 的 reference latent；KL regularization 与 decoder continuity 使模型有机会在这一区域形成可用输出，而 ACT 实际选择并验证了这个 deterministic decode rule。

## 与随机 Prior Sampling 的区别

若改为

\[
z\sim\mathcal N(0,I)
\]

每次 inference，模型就可能从同一 observation 产生不同 action chunks。对于需要探索多种输出的生成任务，这种 diversity 可能有价值；对于 precise closed-loop robot control，额外随机 variation 可能降低重复性。

ACT 因而没有把 CVAE 的“可随机生成”能力直接用于 deployment，而主要利用 CVAE objective 改善对 human demonstration distribution 的训练建模。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
