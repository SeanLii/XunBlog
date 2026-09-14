---
title: "Ordinary Differential Equation"
kind: "canonical"
domain: "Mathematics / Calculus"
parent: "Calculus"
canonical: "/mathematics/calculus/ordinary-differential-equation/"
prerequisites: []
related:
  - "/mathematics/numerical-methods/euler-method/"
  - "/generative-models/flow-matching/"
---

# Ordinary Differential Equation

Ordinary Differential Equation（ODE，常微分方程）描述的不是“某个变量等于多少”，而是：**一个量在当前状态下应该怎样变化。**

最简单的形式是

\[
\frac{dx}{dt}=f(x,t).
\]

其中：

- $x(t)$ 是随时间变化的状态；
- $t$ 是连续时间；
- $\frac{dx}{dt}$ 表示状态此刻的变化速度；
- $f(x,t)$ 给出在状态 $x$、时间 $t$ 下应该朝哪个方向变化。

与普通函数

\[
y=f(x)
\]

不同，ODE 通常不会直接告诉你最终答案。它告诉你**每一个位置上的局部运动规则**，然后通过连续地沿着这个规则移动，得到完整轨迹。

## 从“位置”到“速度”

假设一个一维点的位置是 $x(t)$。如果

\[
\frac{dx}{dt}=2,
\]

表示它始终以速度 2 向正方向移动。

如果初始状态是

\[
x(0)=3,
\]

那么经过时间 $t$ 后：

\[
x(t)=3+2t.
\]

这里真正定义运动的是“速度场” $dx/dt=2$。初始条件 $x(0)=3$ 决定从哪里开始。

更一般地，变化速度可以依赖当前状态：

\[
\frac{dx}{dt}=-x.
\]

此时 $x$ 越大，向 0 移动得越快。轨迹不再是一条直线，而是由这个局部变化规律决定。

## 高维状态

在机器学习中，$x$ 往往不是一个数，而是向量：

\[
x(t)\in\mathbb R^d.
\]

于是

\[
\frac{dx}{dt}=v(x,t)
\]

中的 $v(x,t)\in\mathbb R^d$ 可以理解成：在 $d$ 维空间中的每一个位置，都给出一个移动方向和速度。

例如二维空间中：

```text
当前位置 x
    │
    │  查询 v(x,t)
    ↓
得到一个二维箭头
    │
    ↓
沿箭头移动一点
    │
    ↓
新的位置
```

所有位置上的这些箭头合在一起，就构成 vector field（向量场）。

## 初值问题

一个 ODE 要产生确定轨迹，通常还需要初始状态：

\[
\frac{dx}{dt}=v(x,t),
\qquad
x(0)=x_0.
\]

可以把它读成：

> 从 $x_0$ 出发，在每个时刻按照 $v$ 给出的方向移动。

这正是很多 continuous-time generative model 的基本结构。模型不直接一次输出最终样本，而是学习一个 vector field，让一个简单分布中的样本逐渐流向目标数据分布。

## 解析解与数值积分

少数简单 ODE 可以直接写出解析解，但神经网络定义的

\[
\frac{dx}{dt}=v_\theta(x,t)
\]

通常没有方便的闭式解。

因此计算机需要离散地近似连续运动：

```text
x0
 ↓  根据 v(x0,t0) 移动一点
x1
 ↓  根据 v(x1,t1) 再移动一点
x2
 ↓
...
 ↓
xN
```

最基本的数值方法是 [Euler Method](/mathematics/numerical-methods/euler-method/)。

## 与 Flow Matching 的连接

[Flow Matching](/generative-models/flow-matching/) 学习的就是一个参数化 vector field：

\[
\frac{dx}{dt}=v_\theta(x,t).
\]

训练阶段让 $v_\theta$ 学会正确的局部运动方向；生成阶段从 noise 出发，通过求解这个 ODE，把 noise 逐步移动成数据样本。

π0 把这个思想用于机器人动作：起点是一整个 noisy action chunk，终点是一整个可执行 action chunk。

## Sources

- Lipman et al., **Flow Matching for Generative Modeling**, 2022. https://arxiv.org/abs/2210.02747
