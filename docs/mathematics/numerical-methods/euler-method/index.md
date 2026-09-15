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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Euler Method

Euler Method 是求解 initial-value ODE 的最基本显式 numerical integration 方法。

给定：

\[
\dot x(t)=f(t,x(t)),
\qquad
x(t_0)=x_0,
\]

选 step size $h$，Euler update 为：

\[
\boxed{
x_{n+1}
=
x_n+h f(t_n,x_n)
}
\]

其中：

\[
t_{n+1}=t_n+h.
\]

## Derivation

Taylor expansion：

\[
x(t+h)
=
x(t)
+h\dot x(t)
+O(h^2).
\]

由 ODE：

\[
\dot x(t)=f(t,x(t)),
\]

得到：

\[
x(t+h)
=
x(t)
+h f(t,x(t))
+O(h^2).
\]

忽略 $O(h^2)$ term，就得到 forward Euler。

## Geometric Interpretation

在 $(t_n,x_n)$ 处，ODE 给出 local tangent / velocity：

\[
f(t_n,x_n).
\]

Euler method 假设在长度为 $h$ 的小区间内保持该 velocity：

\[
\Delta x
\approx
h f(t_n,x_n).
\]

然后在新位置重新计算 vector field。

## Local and Global Error

Forward Euler 的 local truncation error 为：

\[
O(h^2),
\]

在固定 total integration interval 上，其 global error 通常为：

\[
O(h).
\]

因此它是一阶 method。

减小 $h$ 通常提高 accuracy，但需要更多 function evaluations。

## Vector ODE

若：

\[
x\in\mathbb R^d,
\]

update 形式不变：

\[
x_{n+1}
=
x_n+h f(t_n,x_n).
\]

所有 dimensions 同时沿 vector field 更新。

## Stability Example

考虑 test equation：

\[
\dot x=\lambda x,
\qquad
\lambda<0.
\]

Euler update：

\[
x_{n+1}
=(1+h\lambda)x_n.
\]

为了 numerical decay，需要：

\[
|1+h\lambda|<1.
\]

因此即使真实 ODE stable，step size 太大也可能导致 numerical solution oscillate 或 diverge。

这说明 numerical stability 与 truncation error 是不同问题。

## Step Size Trade-Off

较小 $h$：

- 更高 accuracy；
- 通常更稳定；
- 更多 model / vector-field evaluations。

较大 $h$：

- compute 更少；
- discretization error 更大；
- 可能产生 instability。

实际 solver 选择需要同时考虑 accuracy、stability 与 computation cost。

## Use in Flow Matching

Flow-based generative model 可以定义：

\[
\frac{dx_t}{dt}
=v_\theta(t,x_t).
\]

Euler sampling：

\[
x_{t+h}
=
x_t+h v_\theta(t,x_t).
\]

[π0](/robot-learning/pi0/) 的 flow-matching action inference 使用有限次 Euler steps 生成 action chunk。

Euler Method 在这里仍然只是 ODE solver；它不属于 π0 或 Flow Matching 本身的定义。
