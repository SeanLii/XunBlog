---
title: "Flow Matching"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/flow-matching/"
prerequisites:
  - "/mathematics/probability/normal-distribution/"
  - "/mathematics/calculus/ordinary-differential-equation/"
related:
  - "/mathematics/numerical-methods/euler-method/"
  - "/robot-learning/pi0/flow-matching-in-pi0/"
---

# Flow Matching

Flow Matching 是一种训练连续生成模型的方法。它的核心想法不是“让网络直接输出最终样本”，而是：**学习一个 vector field，告诉样本在每个中间状态应该往哪里移动，最终把简单的 noise distribution 搬运到 data distribution。**

先看最小数据流：

```text
noise x_0
   │
   │  沿 learned vector field 连续移动
   ↓
x_t
   │
   ↓
...
   │
   ↓
data-like x_1
```

这里的“flow”就是样本沿着连续轨迹流动。

## 生成模型要学的是一条运输过程

设容易采样的起始分布为

\[
p_0(x),
\]

目标数据分布为

\[
p_1(x).
\]

例如 $p_0$ 可以是 standard normal，而 $p_1$ 是真实图片、动作轨迹或其他连续数据。

Flow-based model 希望构造一个随时间变化的分布

\[
p_t(x),\qquad t\in[0,1],
\]

使得：

\[
p_{t=0}=p_0,
\qquad
p_{t=1}=p_1.
\]

也就是把“noise”与“data”之间补上一整条连续路径。

## Vector field 决定样本怎样移动

样本的运动由 ODE 描述：

\[
\frac{dx}{dt}=v_\theta(x,t).
\]

其中 $v_\theta(x,t)$ 是神经网络预测的 vector field。

它不是预测“最终应该到哪个样本”，而是在当前 $(x,t)$ 下预测一个局部速度：

```text
当前位置 x_t
    │
    ↓
vθ(x_t,t)
    │
    ↓
下一小步应该往哪里走
```

不断重复，就得到完整生成轨迹。

## 训练困难转化成速度回归

直接要求神经网络学习整个 marginal vector field 并不方便。Flow Matching 的关键做法是先定义容易处理的 **conditional probability path**，再对对应的 conditional vector field 做回归。

一个非常直观的例子是在线性路径上连接 noise $\epsilon$ 与数据 $x_1$：

\[
x_t=(1-t)\epsilon+t x_1.
\]

当 $t=0$：

\[
x_0=\epsilon,
\]

当 $t=1$：

\[
x_1=x_1.
\]

沿这条直线路径对 $t$ 求导：

\[
\frac{dx_t}{dt}=x_1-\epsilon.
\]

于是训练 target 变得非常简单：模型在中间点 $x_t$ 上预测速度

\[
v_\theta(x_t,t),
\]

并让它接近

\[
u_t=x_1-\epsilon.
\]

常见损失形式是

\[
\mathcal L(\theta)
=
\mathbb E
\left[
\|v_\theta(x_t,t)-u_t\|_2^2
\right].
\]

## Training 与 Sampling 是两件不同的事

训练时，我们有真实数据 $x_1$，也主动采样 noise $\epsilon$，因此可以构造任意中间点 $x_t$ 并知道对应 target velocity。

```text
TRAIN
真实数据 x1 + noise ε + 随机 t
            │
            ↓
      构造中间状态 x_t
            │
            ↓
      网络预测 vθ(x_t,t)
            │
            ↓
与已知 target velocity 比较
```

生成时没有真实 $x_1$。只能从 noise 开始，反复调用网络：

```text
SAMPLE
noise x0
   ↓
ODE step
   ↓
x1/N
   ↓
ODE step
   ↓
...
   ↓
generated sample
```

这就是为什么 Flow Matching 的训练可以只做一次随机 timestep 的监督，而推理却需要多个 integration steps。

## Flow Matching 与 Diffusion 的关系

Flow Matching 与 diffusion models 有紧密联系，但两者不是同一个定义。Flow Matching 是训练 continuous normalizing flow / vector field 的框架，可以使用 diffusion probability path，也可以使用其他路径，例如 optimal-transport-style interpolation。

因此更准确的说法是：

> Flow Matching 可以覆盖与 diffusion 密切相关的 probability paths，但它的核心训练对象是 vector field。

π0 论文为了帮助读者定位，把 flow matching 描述成 diffusion 的一种相关变体；真正进入 π0 数学时，直接抓住“noise → learned vector field → action”会更清楚。

## 与 π0 的连接

π0 不用 Flow Matching 生成图像，而是生成一整个连续机器人 action chunk。

把通用符号替换掉：

```text
普通 Flow Matching:
noise vector → data vector

π0:
noisy action chunk → executable action chunk
```

具体怎样构造 noisy action、怎样加入 observation condition、怎样进行 10 步积分，在 [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) 中展开。

## Sources

- Lipman et al., **Flow Matching for Generative Modeling**, 2022. https://arxiv.org/abs/2210.02747
- Liu, **Rectified Flow: A Marginal Preserving Approach to Optimal Transport**, 2022. https://arxiv.org/abs/2209.14577
