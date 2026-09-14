---
title: "Flow Matching in π0"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/flow-matching-in-pi0/"
prerequisites:
  - "/generative-models/flow-matching/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/inference/"
  - "/mathematics/numerical-methods/euler-method/"
---

# Flow Matching in π0

π0 使用 Flow Matching 生成的对象不是图片，也不是一个 latent vector，而是**未来一整个 action chunk**。

目标条件分布写成：

\[
p(A_t\mid o_t),
\]

其中

\[
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}]
\]

是未来 $H$ 个 robot actions，$o_t$ 是当前 images、language 与 robot state。

## 从随机 action noise 到真实 action chunk

π0 把生成过程想成：

```text
A_t^0  ≈ random noise
   │
   ↓
A_t^0.1
   │
   ↓
A_t^0.2
   │
  ...
   │
   ↓
A_t^1  ≈ executable action chunk
```

论文 convention 中 $\tau=0$ 是 noise 端，$\tau=1$ 是 data 端。

每一步网络都预测：

\[
v_\theta(A_t^\tau,o_t),
\]

也就是当前 action chunk 应该向哪里移动。

## Training 中怎样得到中间 action

训练时真实 action chunk $A_t$ 是已知的。

先采样：

\[
\epsilon\sim\mathcal N(0,I).
\]

再选择 flow timestep $\tau$，并构造：

\[
A_t^\tau
=
\tau A_t+(1-\tau)\epsilon.
\]

这个式子非常值得直接看端点：

当 $\tau=0$：

\[
A_t^0=\epsilon,
\]

完全是 noise。

当 $\tau=1$：

\[
A_t^1=A_t,
\]

完全是真实 action。

中间值就是二者之间的线性 interpolation。

## 训练 target 是速度，不是 action

沿上面的线性 path 对 $\tau$ 求导：

\[
\frac{dA_t^\tau}{d\tau}
=
A_t-\epsilon.
\]

所以 target vector field 是

\[
u(A_t^\tau\mid A_t)=A_t-\epsilon.
\]

π0 网络预测

\[
v_\theta(A_t^\tau,o_t),
\]

并使用平方误差让它接近 target：

\[
\mathcal L^\tau(\theta)
=
\mathbb E
\left[
\|v_\theta(A_t^\tau,o_t)
-(A_t-\epsilon)\|_2^2
\right].
\]

因此训练时网络真正学的是：

> 已知当前 observation，而且 action chunk 现在被噪声污染到程度 $\tau$，接下来整块 action 应该朝什么方向移动。

## Observation 作为 Flow Matching Condition

如果没有 $o_t$，Flow Matching 只能学“所有动作数据整体长什么样”。

但 robot policy 需要的是条件分布：

\[
p(A_t\mid o_t).
\]

同样一团 noisy actions，在看到不同 camera image 或不同 instruction 后，应该走向完全不同的最终动作。

所以 π0 的 vector field 是 conditional vector field：

```text
noisy actions A^τ
        +
images + language + state
        │
        ↓
      π0
        │
        ↓
conditioned velocity vθ
```

## Chunk-Level Flow Generation

论文使用 $H=50$。整个 action chunk 一起加噪、一起进入 Action Expert、一起预测 vector field。

因此：

\[
A_t^\tau\in\mathbb R^{H\times d_a},
\]

网络输出也有同样 shape：

\[
v_\theta\in\mathbb R^{H\times d_a}.
\]

Flow Matching 并没有把 chunk 拆成 50 次独立生成；相反，action tokens 之间使用 bidirectional attention，可以共同形成连贯轨迹。

## 非均匀 Timestep Sampling

原始 Flow Matching 可以从 $[0,1]$ 均匀采样 timestep。π0 论文改用了 shifted beta distribution，让训练更多看到靠近 noise 端的 timesteps，并且不采样超过 cutoff $s=0.999$ 的区域。

论文给出的直觉是：对于 robot action prediction，哪怕在高 noise 情况下，仅仅根据 observation 预测一个合理的 conditional mean action 本身也不容易，因此更值得把训练量放在 noisy region。

这是 π0-specific training design，不属于 Flow Matching 的通用定义。

## 论文与当前 openpi 的时间方向相反

这是必须明确记录的实现差异。

论文 convention：

```text
τ = 0 : noise
τ = 1 : data
```

当前官方 openpi code 明确采用 diffusion literature 更常见的相反 convention：

```text
t = 1 : noise
t = 0 : data
```

因此代码中构造：

\[
x_t=t\epsilon+(1-t)A,
\]

并学习

\[
u_t=\epsilon-A.
\]

推理时从 $t=1$ 向 $t=0$ 积分。

两者描述的是同一条路径的相反参数方向，不应把符号直接混用。

完整差异见 [Paper and Released Implementation](/robot-learning/pi0/paper-and-released-implementation/)。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Section IV and Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi `pi0.py`. https://github.com/Physical-Intelligence/openpi
