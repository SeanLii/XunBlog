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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Ordinary Differential Equation

> **知识边界**：本文的 canonical 对象是 **Ordinary Differential Equation**。依赖机制由 [Vector](/mathematics/linear-algebra/vector/) 的 canonical page 定义；本文只在当前语境中调用其接口。


Ordinary Differential Equation（ODE）描述未知函数与其一个或多个 derivatives 之间的关系。

一阶 ODE 的一般形式可以写成：

\[
\frac{dx}{dt}
=f(t,x(t)).
\]

其中：

- $t$ 是 independent variable；
- $x(t)$ 是未知 trajectory；
- $f$ 给出每个时刻与状态下的 instantaneous rate of change。

## Initial Value Problem

仅有 differential equation 通常不足以确定唯一 solution。还需要 initial condition：

\[
x(t_0)=x_0.
\]

于是形成 initial value problem：

\[
\begin{cases}
\dot x(t)=f(t,x(t)),\\
x(t_0)=x_0.
\end{cases}
\]

其解是一条满足 equation 与 initial condition 的 function $x(t)$。

## Autonomous ODE

若 vector field 不显式依赖时间：

\[
\dot x=f(x),
\]

则称为 autonomous system。

此时 dynamics 只由当前 state 决定。

若：

\[
\dot x=f(t,x),
\]

则称为 non-autonomous system。

## Vector-Valued ODE

对于：

\[
x(t)\in\mathbb R^d,
\]

可以写成：

\[
\frac{d x}{dt}
=f(t,x),
\qquad
f:\mathbb R\times\mathbb R^d\to\mathbb R^d.
\]

此时 $f$ 是 vector field，为 state space 中每个位置指定局部 velocity。

这类形式是 dynamical systems、control 与 flow-based generative models 的基础。

## Higher-Order ODE

例如二阶 equation：

\[
\frac{d^2x}{dt^2}
=g(t,x,\dot x)
\]

可以通过引入 state：

\[
y_1=x,
\qquad
y_2=\dot x
\]

转为一阶 system：

\[
\dot y_1=y_2,
\qquad
\dot y_2=g(t,y_1,y_2).
\]

因此很多理论与 numerical solvers 只需处理 first-order systems。

## Integral Form

若 $x$ 是 solution，则：

\[
x(t)
=
x(t_0)
+
\int_{t_0}^{t}
f(s,x(s))\,ds.
\]

这一 integral equation 与对应 ODE 在满足相应 regularity conditions 时描述同一 initial-value dynamics。

Numerical methods 通过离散化近似这一 accumulated integral。

## Analytical Example

Consider：

\[
\dot x=ax,
\qquad
x(0)=x_0.
\]

其 solution：

\[
x(t)=x_0e^{at}.
\]

若 $a<0$，state 指数衰减；若 $a>0$，指数增长。

这展示 local derivative rule 如何确定整条 trajectory。

## Existence and Uniqueness

并不是任意 $f$ 都保证 initial value problem 有唯一 solution。

Picard–Lindelöf theorem 的常见 sufficient condition 是：$f$ 对 $x$ 局部 Lipschitz，并对 $t$ 具有适当连续性。

Lipschitz condition 形式为：

\[
\|f(t,x)-f(t,y)\|
\le
L\|x-y\|.
\]

它限制 nearby states 的 velocities 变化过快，从而支持 local uniqueness。

## Equilibrium and Stability

Autonomous system：

\[
\dot x=f(x)
\]

的 equilibrium $x^*$ 满足：

\[
f(x^*)=0.
\]

若 nearby trajectories 随时间保持接近或收敛到 $x^*$，可以进一步讨论 Lyapunov stability、asymptotic stability 等性质。

## Linear ODE

Linear system：

\[
\dot x=Ax
\]

的 solution：

\[
x(t)=e^{At}x(0),
\]

其中：

\[
e^{At}
=
\sum_{k=0}^{\infty}
\frac{(At)^k}{k!}
\]

是 matrix exponential。

Eigenvalues of $A$ 与 system stability 密切相关。

## Numerical Solution

很多 ODE 没有可用 closed-form solution，需要 numerical integration。

最基本方法是 [Euler Method](/mathematics/numerical-methods/euler-method/)：

\[
x_{n+1}
=
x_n+h f(t_n,x_n).
\]

更高精度方法包括 Runge–Kutta families、adaptive-step solvers 与 implicit methods。

## Stiffness

某些 ODE 同时含有差异很大的 time scales。显式 solver 为保持 stability 可能需要极小 step size，这类 system 称为 stiff。

Stiff ODE 常使用 implicit solvers 或专门 numerical methods。

Stiffness 是 numerical property，与 equation 是否“看起来复杂”没有直接对应关系。

## ODE in Generative Modeling

Continuous Normalizing Flow / Flow Matching 使用：

\[
\frac{dx_t}{dt}
=v_\theta(t,x_t)
\]

定义 probability samples 的 continuous transport。

已学习的 vector field $v_\theta$ 规定从 base distribution 到 data distribution 的 trajectory，而 numerical solver 负责实际积分。

这只是 ODE 的一个应用；ODE 本身属于更广泛的 dynamical-systems 数学框架。

## Connections

- [Euler Method](/mathematics/numerical-methods/euler-method/)：一阶显式 ODE solver。
- [Flow Matching](/generative-models/flow-matching/)：用 ODE vector field 定义生成过程。
