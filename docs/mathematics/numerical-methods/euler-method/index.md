---
title: "Euler Method"
kind: "canonical"
domain: "Mathematics / Numerical Methods"
parent: "Numerical Methods"
canonical: "/mathematics/numerical-methods/euler-method/"
prerequisites:
  - "/mathematics/calculus/ordinary-differential-equation/"
related:
  - "/generative-models/flow-matching/"
  - "/robot-learning/pi0/inference/"
---

# Euler Method

Euler Method（欧拉法）是一种用很多个小步，近似求解常微分方程的方法。

假设有 ODE：

\[
\frac{dx}{dt}=v(x,t).
\]

模型知道当前位置 $x_t$ 和当前位置的变化速度 $v(x_t,t)$，但不知道完整轨迹。Euler Method 的做法很直接：假设在一个很短的时间 $\Delta t$ 内，速度近似不变，于是

\[
x_{t+\Delta t}
\approx
x_t+\Delta t\,v(x_t,t).
\]

这条式子就是 Euler step。

## 一步在做什么

把它拆开：

\[
\Delta x\approx \Delta t\,v(x_t,t).
\]

其中 $v$ 是“每单位时间变化多少”，乘上时间长度 $\Delta t$，得到这一小步应该移动多少。

所以：

```text
当前状态 x_t
      │
      ↓
计算 v(x_t,t)
      │
      ↓
乘 Δt 得到小位移
      │
      ↓
x_{t+Δt} = x_t + Δt · v
```

## 一个数值例子

设

\[
\frac{dx}{dt}=2x,
\qquad x(0)=1.
\]

取 $\Delta t=0.1$。第一步：

\[
v(x_0,0)=2\times1=2,
\]

因此

\[
x_{0.1}\approx1+0.1\times2=1.2.
\]

第二步重新计算当前位置的速度：

\[
v(1.2,0.1)=2.4,
\]

所以

\[
x_{0.2}\approx1.2+0.1\times2.4=1.44.
\]

每一步都使用**新的状态重新计算方向**。

## 步长与误差

$\Delta t$ 越小，一小步内“速度不变”的近似通常越合理，但需要更多步骤。

因此存在一个基本权衡：

```text
更小 step size
→ 通常更精确
→ 但需要更多次函数 / 神经网络计算
```

在神经生成模型中，这一点非常实际：如果每个 Euler step 都要运行一次大模型，那么积分步数直接影响推理速度。

## π0 中的 Euler Method

π0 的 flow matching inference 从 noisy action chunk 开始。论文使用 10 个积分步骤，并写成

\[
A_t^{\tau+\delta}
=
A_t^\tau+
\delta v_\theta(A_t^\tau,o_t),
\]

其中 $\delta=0.1$。

这里：

- $A_t^\tau$ 是当前 flow timestep 的 action chunk；
- $v_\theta$ 是 π0 预测的 vector field；
- 每次更新后，action chunk 都更接近真实动作分布。

所以“π0 需要 10 次 flow matching forward pass”并不是重复生成十次答案，而是在用 Euler Method 走完一条从 noise 到 action 的轨迹。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
