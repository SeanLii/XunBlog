---
title: "Ordinary Differential Equation"
kind: "canonical"
domain: "Mathematics / Calculus"
parent: "Calculus"
canonical: "/mathematics/calculus/ordinary-differential-equation/"
prerequisites:
  - "/mathematics/linear-algebra/vector/"
related:
  - "/mathematics/numerical-methods/euler-method/"
  - "/generative-models/flow-matching/"
---

# Ordinary Differential Equation

Ordinary Differential Equation（ODE）不是直接告诉我们一个状态“在哪里”，而是规定这个状态**怎样随一个连续变量变化**。

最常见的形式：

\[
\frac{dx(t)}{dt}=f(t,x(t)).
\]

这里：

- $t$：独立变量，常被理解为 time；
- $x(t)$：随 $t$ 变化的 state；
- $f(t,x)$：给出当前 state 的变化率。

如果 $x$ 是位置，那么 $dx/dt$ 可以理解为速度；如果 $x$ 是一个高维 vector，那么 $f$ 给出 vector space 中的瞬时运动方向。

## ODE 描述的是 local rule

ODE 给的是局部变化：

\[
\frac{dx}{dt}=f(t,x).
\]

但我们真正想知道的通常是完整 trajectory：

\[
x(t_0),x(t_1),\ldots
\]

因此需要从一个 initial state 开始，把局部变化累积起来。

## Initial Value Problem

只有 differential equation 通常还不足以确定唯一 trajectory。

还需要 initial condition：

\[
x(t_0)=x_0.
\]

于是得到 initial value problem：

\[
\begin{cases}
\dfrac{dx}{dt}=f(t,x),\\
x(t_0)=x_0.
\end{cases}
\]

在满足适当 regularity 条件时，这可以确定一条唯一 solution trajectory。

## 一个简单解析例子

考虑：

\[
\frac{dx}{dt}=kx.
\]

解为：

\[
x(t)=x_0e^{k(t-t_0)}.
\]

这里每个时刻的变化率都与当前值成比例。

但大多数实际 ODE 不会有这么简单的 closed-form solution。

## 高维 ODE

如果：

\[
x(t)\in\mathbb R^d,
\]

则：

\[
\frac{dx(t)}{dt}=v_t(x(t))
\]

中的 $v_t$ 是 vector field：对 space 中每个位置与时间，指定一个 velocity vector。

可以把它想成一个高维“流场”：一个 particle 放在任意位置，都能查到它此刻应该往哪里移动。

## 解析解与数值解

如果找不到 closed-form solution，就需要 numerical integration。

最简单的方法之一是 [Euler Method](/mathematics/numerical-methods/euler-method/)：

\[
x_{k+1}
=x_k+h f(t_k,x_k).
\]

它用当前瞬时 slope 近似未来一小段时间内的运动。

更高阶方法如 Runge–Kutta 会更精确地估计一步中的变化。

## ODE 与 Flow

如果每个 initial point 都沿 vector field 随时间移动，整个 space 会产生一个 continuous flow。

这正是 Continuous Normalizing Flow 和 [Flow Matching](/generative-models/flow-matching/) 使用 ODE 的原因：模型学习一个 vector field，把 samples 从一个 distribution 连续运输到另一个 distribution。

Flow Matching 是 ODE 的应用；ODE 本身则是描述连续动态系统的基础数学语言。
