---
title: "Flow Matching"
kind: "canonical"
domain: "Generative Models"
parent: "Generative Models"
canonical: "/generative-models/flow-matching/"
prerequisites:
  - "/mathematics/calculus/ordinary-differential-equation/"
  - "/mathematics/probability/probability-distribution/"
related:
  - "/mathematics/numerical-methods/euler-method/"
  - "/robot-learning/pi0/flow-matching-in-pi0/"
---

# Flow Matching

Flow Matching 是训练 continuous normalizing flow 的一种方法。它的核心目标是学习一个 time-dependent vector field，使 samples 从一个简单 base distribution 连续运输到 data distribution。

Mental model：

```text
simple noise distribution
        ↓
continuous flow over time
        ↓
data distribution
```

如果 sample state 为 $x_t$，动态由 ODE 描述：

\[
\frac{dx_t}{dt}=v_\theta(t,x_t).
\]

模型学习的就是 velocity field $v_\theta$。

## Generative Modeling 变成 Transport Problem

设：

\[
p_0(x)
\]

是容易 sample 的 base distribution，

\[
p_1(x)
\]

是 target data distribution。

我们希望存在一族 intermediate distributions：

\[
p_t(x),\qquad t\in[0,1],
\]

满足：

\[
p_{t=0}=p_0,
\qquad
p_{t=1}=p_1.
\]

vector field 让 samples 沿 trajectory 移动，同时整体 probability distribution 随时间从 $p_0$ 演化到 $p_1$。

## Marginal Vector Field 的训练困难

理想 target 是一个能够生成整个 probability path 的 marginal vector field：

\[
u_t(x).
\]

但这个 field 通常依赖对 data distribution 的整体积分，不容易直接得到监督 target。

Flow Matching 的关键技巧是：改为构造**conditional probability paths**，在给定 data sample 时拥有简单 closed-form conditional vector field。

## Conditional Path

可以先 sample data：

\[
x_1\sim p_{data}.
\]

再定义一个从 base noise 到这个 sample 附近的 conditional path：

\[
p_t(x\mid x_1).
\]

对应 conditional vector field：

\[
u_t(x\mid x_1).
\]

这个 target 可以直接计算。

于是训练 network 回归：

\[
L_{FM}
=
\mathbb E_{t,x_1,x_t}
\left[
\|v_\theta(t,x_t)-u_t(x_t\mid x_1)\|^2
\right].
\]

Lipman 等证明，在适当构造下，这种 conditional regression 可以得到正确的 marginal flow training objective。

## 一个最直观的 Linear Interpolation Path

如果使用 base sample：

\[
x_0\sim p_0,
\]

和 data sample：

\[
x_1\sim p_{data},
\]

最简单 path：

\[
x_t=(1-t)x_0+t x_1.
\]

对时间求 derivative：

\[
\frac{dx_t}{dt}=x_1-x_0.
\]

于是 target velocity 非常直接。

实际 Flow Matching 可以使用更一般 Gaussian probability paths，包括 diffusion-related paths 与 optimal-transport-inspired paths。

## Training 与 Sampling 是两件事

Training：

```text
sample data / noise / t
       ↓
construct x_t
       ↓
known target velocity u_t
       ↓
network predicts vθ(t, x_t)
       ↓
regression loss
```

Sampling：

```text
x_0 ~ base distribution
       ↓
solve dx/dt = vθ(t,x)
       ↓
x_1 ≈ data sample
```

训练不需要在每个 step 完整跑 ODE trajectory，这就是原始 Flow Matching 方法强调的 simulation-free training 特点。

## ODE Solver

生成时需要 numerical integration。

最简单可以使用 [Euler Method](/mathematics/numerical-methods/euler-method/)：

\[
x_{k+1}
=x_k+h v_\theta(t_k,x_k).
\]

也可以使用 Runge–Kutta 等更高阶 solvers。

所以 sampling quality / compute 还受 solver 与 step count 影响。

## 与 Diffusion Models 的关系

Flow Matching 与 diffusion 并不是简单的对立关系。

原始 Flow Matching framework 可以使用 diffusion probability paths；同时也允许使用 non-diffusion paths，例如 optimal transport displacement interpolation。

两类模型都可以描述“从简单 distribution 到 data distribution”的连续变化，但 parameterization、training target 和 sampling interpretation 不同。

## π0 中的使用

π0 把 action chunk 当作要生成的 continuous data，并训练 flow-style action expert 从 noise action 逐步变成合理 action chunk。

那是 Flow Matching 在 robot action generation 中的一种特定 parameterization，详细应放在 π0 自己页面，而不是用 π0 来定义 Flow Matching。

## Sources

- Lipman et al. *Flow Matching for Generative Modeling*. 2022/ICLR 2023.
