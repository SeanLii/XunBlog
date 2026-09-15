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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Flow Matching in π0

[Flow Matching](/generative-models/flow-matching/) 给出从简单 base distribution 到数据分布的连续运输方法。π0 把其中的数据样本替换为整段机器人动作，并让速度场以图像、语言和状态为条件。它由此学习的不是单个平均动作，而是条件动作分布上的运输方向。

## 动作块是一个联合随机变量

令示范中的未来动作块为

\[
A=[a_t,\ldots,a_{t+H-1}]\in\mathbb R^{H\times d_a},
\]

噪声样本为 $\epsilon\sim\mathcal N(0,I)$，shape 与 $A$ 相同。π0 论文取 $H=50$。把整块视为一个随机变量意味着 flow 同时改变所有未来位置，动作间的时间协调进入联合分布，而不是被拆成 $H$ 个互不相关的回归目标。

## 线性 Probability Path

论文把 noise end 记为 $\tau=0$，data end 记为 $\tau=1$：

\[
A^\tau=\tau A+(1-\tau)\epsilon.
\]

沿 $\tau$ 求导得到常量 target velocity：

\[
u(A^\tau\mid A,\epsilon)=\frac{dA^\tau}{d\tau}=A-\epsilon.
\]

这一步只定义训练路径；它不声称真实动作分布沿直线形成。网络在许多数据、噪声与时间样本上学习条件期望后，得到的整体 vector field 可以把 Gaussian mass 弯曲成复杂、多峰的动作分布。

## Observation 把无条件运输变成策略

π0 的网络输入还包括

\[
o_t=(I_t^1,\ldots,I_t^n,\ell_t,q_t).
\]

[π0 Architecture](/robot-learning/pi0/architecture/) 将 observation 编码成固定 prefix，将 $A^\tau$ 与 $\tau$ 编码成 action suffix。模型预测

\[
v_\theta(A^\tau,\tau,o_t)\approx A-\epsilon.
\]

同一份噪声在“把杯子放到左侧”和“把杯子放到右侧”的指令下会获得不同速度，因此生成分布由当前视觉、语言与机器人构型共同决定。

## 回归目标为何能够训练速度场

目标函数为

\[
\mathcal L(\theta)=
\mathbb E_{A,o,\epsilon,\tau}
\left[
\left\|v_\theta(A^\tau,\tau,o)-(A-\epsilon)\right\|_2^2
\right].
\]

训练时已知 $A$ 与 $\epsilon$，所以可直接构造任意 $\tau$ 的中间点和监督速度，不需要先解 ODE。一次样本只需在随机时间上做一次 vector-field regression；多步积分只出现在 inference。

## Paper 与 openpi 的时间方向

released openpi 使用相反记号：

\[
x_t=t\epsilon+(1-t)A,\qquad u_t=\epsilon-A,
\]

此时 $t=1$ 是 noise，$t=0$ 是 data。两种写法描述同一条路径，只是积分方向与速度符号同时反转。阅读公式或实现时必须成对比较 path 和 target；只替换时间变量而不替换速度符号会让采样朝噪声端移动。

## 从速度场到动作的计算代价

推理从 Gaussian noise 出发，用数值积分反复调用 $v_\theta$。多次评估使 π0 能表达比单次 L2 regression 更丰富的条件分布，但也直接增加 policy latency。Action Expert 的较小规模、blockwise mask 与 prefix caching 都是对这项成本的结构性回应。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Section IV. https://arxiv.org/abs/2410.24164
- Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. https://arxiv.org/abs/2210.02747
- Physical Intelligence. *openpi*, `Pi0.compute_loss` and sampling code. https://github.com/Physical-Intelligence/openpi
