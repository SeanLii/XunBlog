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
---

# Euler Method

Euler Method 是求解 ordinary differential equation 的最简单 numerical integration 方法之一。

给定 initial value problem：

\[
\frac{dx}{dt}=f(t,x),
\qquad
x(t_0)=x_0,
\]

Euler Method 使用当前位置的 derivative，向前走一个小 step：

\[
\boxed{
x_{k+1}=x_k+h f(t_k,x_k)
}
\]

同时：

\[
t_{k+1}=t_k+h.
\]

其中 $h$ 是 step size。

## 从 Tangent Line 得到公式

对足够小的 $h$，Taylor expansion：

\[
x(t+h)
=x(t)+h x'(t)+O(h^2).
\]

因为：

\[
x'(t)=f(t,x(t)),
\]

忽略 higher-order terms：

\[
x(t+h)
\approx x(t)+h f(t,x(t)).
\]

这正是 Euler update。

所以 Euler Method 的本质是：

> **假设当前 tangent slope 在这一小步里近似不变。**

## 一个数值例子

考虑：

\[
\frac{dx}{dt}=x,
\qquad
x(0)=1.
\]

真实解：

\[
x(t)=e^t.
\]

取：

\[
h=0.1.
\]

第一步：

\[
x_1=1+0.1\times1=1.1.
\]

第二步：

\[
x_2=1.1+0.1\times1.1=1.21.
\]

继续迭代，在 $t=1$ 时会得到一个对 $e$ 的近似。

## Local Error 与 Global Error

Euler Method 每一步忽略了 Taylor expansion 中的二阶及以上项，因此 local truncation error 是：

\[
O(h^2).
\]

累积很多步后，global error 通常为：

\[
O(h).
\]

所以 step size 减半时，整体误差通常大约按一阶比例下降。

## Step Size 的取舍

更小的 $h$：

- 通常更精确；
- 需要更多 steps；
- 计算更慢。

更大的 $h$：

- 计算更快；
- approximation 更粗；
- 对某些 ODE 甚至可能 numerical instability。

因此 numerical integration 不只是“多跑几次 update”，而是在 accuracy、stability 与 compute 之间取舍。

## 高维状态

如果：

\[
x_k\in\mathbb R^d,
\]

公式完全相同：

\[
x_{k+1}
=x_k+h f(t_k,x_k).
\]

只是 $f$ 输出一个 $d$-dimensional velocity vector。

所以 Euler Method 可以直接用于 neural ODE、continuous normalizing flow 或 action-space flow。

## 与 Flow Matching 的关系

[Flow Matching](/generative-models/flow-matching/) 学习的是 vector field。Sampling 时需要把：

\[
\frac{dx_t}{dt}=v_\theta(x_t,t)
\]

沿时间积分。

Euler Method 是最直接的 solver：

\[
x_{k+1}=x_k+h v_\theta(x_k,t_k).
\]

一些模型会使用固定少量 Euler steps；另一些会使用更高阶 ODE solver。Euler 是其中一种数值求解方法，而不是 Flow Matching 自己定义出来的更新规则。
