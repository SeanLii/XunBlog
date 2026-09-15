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

Flow Matching 是训练 Continuous Normalizing Flow（CNF）的一类方法。它学习一个 time-dependent vector field

\[
v_\theta(t,x),
\]

使样本在 Ordinary Differential Equation

\[
\frac{dx_t}{dt}=v_\theta(t,x_t)
\]

的驱动下，从容易采样的 base distribution $p_0$ 连续运输到目标 data distribution $p_1$。

Flow Matching 的关键特点是：**训练时直接回归一个已知的目标 vector field，不需要在每个 training sample 上完整模拟 ODE trajectory。**

## Probability Path

设

\[
p_t(x),\qquad t\in[0,1]
\]

是一族连续变化的 probability distributions，并满足 base endpoint

\[
p_{t=0}=p_0,
\]

以及目标 endpoint

\[
p_{t=1}\approx p_{data}.
\]

在理想 transport formulation 中可以令 $p_{t=1}=p_{data}$。实际 conditional Gaussian constructions 有时保留一个很小的 terminal variance，此时终点是 data distribution 的轻微平滑版本，因此写成 $\approx$ 更准确。

若 vector field $u_t(x)$ 生成这条 probability path，则 density 与 velocity 满足 continuity equation：

\[
\frac{\partial p_t(x)}{\partial t}
+
\nabla\cdot\bigl(p_t(x)u_t(x)\bigr)
=0.
\]

这条方程表达 probability mass conservation：density 的局部变化由 probability flow 的 divergence 决定。

因此 generative modeling 可以被表述为 transport problem：构造从 base distribution 到 data distribution 的 path，并学习产生这条 path 的 vector field。

## Continuous Normalizing Flow

给定初始样本

\[
x_0\sim p_0,
\]

求解 ODE

\[
\frac{dx_t}{dt}=v_\theta(t,x_t)
\]

得到

\[
x_1\sim p_1
\]

的近似样本。

CNF 还允许通过 instantaneous change-of-variables formula 跟踪 log density：

\[
\frac{d}{dt}\log p_t(x_t)
=
-\nabla\cdot v_\theta(t,x_t).
\]

因此 CNF 不只是 sampling mechanism，也可以在适当条件下定义显式 likelihood。但 Flow Matching training 本身主要关注如何高效学习 vector field。

## Direct Marginal Vector-Field Regression

理想情况下，可以直接回归生成 marginal path $p_t(x)$ 的 vector field $u_t(x)$：

\[
\mathcal L_{FM}
=
\mathbb E_{t,x_t\sim p_t}
\left[
\lVert v_\theta(t,x_t)-u_t(x_t)\rVert^2
\right].
\]

困难在于：对复杂 data distribution，marginal $u_t(x)$ 往往难以直接计算。

Flow Matching 的重要构造是把这个不可直接监督的问题转换成 conditional vector-field regression。

## Conditional Probability Paths

对 data sample $x_1\sim p_{data}$，构造条件 path

\[
p_t(x\mid x_1)
\]

以及可计算的 conditional vector field

\[
u_t(x\mid x_1).
\]

条件 path 对 data distribution 求边缘后得到

\[
p_t(x)
=
\int p_t(x\mid x_1)p_{data}(x_1)\,dx_1.
\]

若 $p_1(x\mid x_1)$ 在 $x_1$ 处退化成 point mass，则 marginal endpoint 正好是 $p_{data}$；若像原论文常用的 Gaussian path 一样保留 $\sigma_{min}>0$，则 $p_1$ 是围绕 data samples 的窄 Gaussian mixture，对 $p_{data}$ 作近似。

Conditional Flow Matching objective 为

\[
\mathcal L_{CFM}
=
\mathbb E_{t,x_1,x_t}
\left[
\lVert
v_\theta(t,x_t)-u_t(x_t\mid x_1)
\rVert^2
\right].
\]

Lipman 等人证明，在相应 regularity 条件下，这个 conditional objective 与目标 marginal vector-field regression 具有相同的 gradient，因此可以用容易生成的 conditional targets 训练正确的 marginal flow。

## Linear Interpolation Path

为了说明 linear path，可以先看一个 zero-terminal-noise 的简化情形。采样

\[
x_0\sim p_0,
\qquad
x_1\sim p_{data},
\]

并选择一条直线 sample trajectory

\[
X_t=(1-t)x_0+t x_1.
\]

对**这一条已经选定 endpoints 的 trajectory** 求导得到

\[
\dot X_t=x_1-x_0.
\]

这里必须区分三个不同对象。

### 1. Sample-level trajectory velocity

\[
x_1-x_0
\]

只是某一对 sampled endpoints 所定义直线轨迹的速度。它依赖这一次采样得到的 $x_0,x_1$。

### 2. Conditional target vector field

Conditional Flow Matching 训练时可以构造一个可计算的 target

\[
u_t(x\mid x_1)
\]

或在 paired-endpoint / rectified-style parameterization 中直接用相应 sample velocity 作为 stochastic regression target。这个 target 是在附加 conditioning information 下定义的，并不等于模型最终只需要学会“两个 endpoints 相减”。

### 3. Marginal vector field

真正用于生成 marginal probability path 的 field 是

\[
u_t(x).
\]

在标准 Conditional Flow Matching construction 中，它可写为 conditional fields 的 posterior-weighted average：

\[
u_t(x)
=
\int
u_t(x\mid x_1)
\frac{p_t(x\mid x_1)p_{data}(x_1)}{p_t(x)}\,dx_1.
\]

模型 $v_\theta(t,x)$ 在 inference 时只看到当前 $t,x$（以及模型本身允许的外部 condition），并不知道训练时那一对具体的 $x_0,x_1$。通过对大量 conditional / sample-level targets 做 squared-error regression，它学习的是与 marginal transport 一致的 vector field。

因此公式

\[
\dot X_t=x_1-x_0
\]

应该理解为“这个简化 linear path 下可直接得到的监督速度”，而不是 Flow Matching 对所有 probability paths 的统一定义。原始 Flow Matching 论文主要把 conditional paths 写成 Gaussian family。对其 OT-style conditional path，可以写成

\[
p_t(x\mid x_1)
=
\mathcal N\bigl(x\mid \mu_t(x_1),\sigma_t^2 I\bigr),
\]

其中

\[
\mu_t(x_1)=t x_1,\qquad
\sigma_t=1-(1-\sigma_{min})t.
\]

若 $x_0\sim\mathcal N(0,I)$，对应 sample path 是

\[
X_t=t x_1+\sigma_t x_0,
\]

所以 sample-level velocity 为

\[
\dot X_t
=
x_1-(1-\sigma_{min})x_0.
\]

当 $\sigma_{min}=0$ 时，它退化成前面的 $x_1-x_0$。而把 $x_0$ 用当前状态 $x$ 与 $x_1$ 消去后，conditional vector field 可写成

\[
u_t(x\mid x_1)
=
\frac{x_1-(1-\sigma_{min})x}
{1-(1-\sigma_{min})t}.
\]

这说明 sample velocity 与 conditional vector field 虽然来自同一条 path，但表示方式不同：前者显式依赖 sampled noise $x_0$，后者写成当前位置 $x$、时间 $t$ 与 condition $x_1$ 的函数。论文也允许 diffusion-style Gaussian paths，因此不同 path 会产生不同的 conditional vector fields。

## Path Choice

Flow Matching 将“模型 architecture”和“probability path”分离。不同 path 会改变训练时模型看到的 intermediate states 与 target velocities。

Path choice 会影响：

- trajectory curvature；
- vector-field complexity；
- numerical integration difficulty；
- sampling step efficiency。

因此两个模型即使都使用 Flow Matching，也可能因为 path parameterization 不同而具有明显不同的 training / sampling behavior。

## Training Procedure

一次 Flow Matching training sample 一般包含：

1. sample data endpoint $x_1$；
2. sample base noise / conditional randomness；
3. sample time $t$；
4. construct $x_t$；
5. compute known conditional target velocity $u_t$；
6. regress $v_\theta(t,x_t)$ toward $u_t$。

训练 objective 通常是 vector-field mean squared error。

这一过程不要求从 $t=0$ 数值积分到 $t=1$，因此常被称为 simulation-free training。

## Sampling

Generation 时没有 data endpoint。必须从

\[
x_0\sim p_0
\]

出发，求解 learned ODE：

\[
\frac{dx_t}{dt}=v_\theta(t,x_t).
\]

一种一阶 numerical solver 是 [Euler Method](/mathematics/numerical-methods/euler-method/)：

\[
x_{k+1}
=x_k+h\,v_\theta(t_k,x_k).
\]

也可以使用 Runge–Kutta 或 adaptive solvers。Sampling quality、latency 与 solver error 因此同时依赖 learned vector field 和 numerical integration scheme。

## Relationship to Diffusion Models

Flow Matching 与 diffusion models 不是互斥分类。

Diffusion models 通常通过 forward noising process 与 reverse-time dynamics 构造生成过程；Flow Matching 则从 vector-field regression 的角度训练 continuous flow。Flow Matching framework 可以选择 diffusion-style probability paths，也可以选择 non-diffusion paths。

因此更准确的关系是：

- diffusion 提供一类 stochastic / probability paths 与相应 score / reverse-process parameterizations；
- Flow Matching 提供直接回归 flow vector field 的训练框架；
- 某些具体模型在路径与生成 ODE 上可以产生紧密联系。

## Conditional Flow Matching

若生成目标还依赖条件 $c$，模型学习

\[
v_\theta(t,x;c).
\]

这对应 conditional distribution

\[
p(x\mid c).
\]

条件可以是 class label、text embedding、image context、robot observation 等。π0 就把当前 robot observation 作为 condition，把未来 action chunk 作为需要运输到 data distribution 的连续变量。

## Limitations

Flow Matching 并不消除生成模型的所有难点：

- path choice 会影响 vector field 的学习难度；
- sampling 仍需要 ODE integration；
- 少量 integration steps 会引入 discretization error；
- high-dimensional multimodal distributions 仍要求模型学习复杂 conditional field；
- likelihood computation 若需要 divergence 估计，成本可能很高。

其主要优势是提供了直接、稳定的 vector-field regression objective，并允许灵活选择 probability paths。

## π0 中的使用

π0 将整个 future action chunk 作为 continuous sample，并学习 observation-conditioned vector field。具体 linear action path、timestep sampling、Action Expert 与 paper/code time-direction difference 见 [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/)。

## Sources

- Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. https://arxiv.org/abs/2210.02747
- Chen et al. *Neural Ordinary Differential Equations*. 2018.
