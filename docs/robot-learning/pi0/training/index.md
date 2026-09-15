---
title: "Training"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/training/"
prerequisites:
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/robot-learning/pi0/pretraining-and-posttraining/"
  - "/robot-learning/pi0/inference/"
  - "/robot-learning/pi0/complete-data-flow/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# π0 Training

π0 把一条机器人示范切成“当前条件—未来动作块”样本，再把动作块转换为 conditional Flow Matching 的速度回归目标。训练阶段可以直接使用真实未来动作构造任意中间状态，因此一次样本只评估一个随机 flow time；推理阶段没有真实动作，才需要沿速度场连续积分。

## 从轨迹截取监督样本

对轨迹中的时刻 $t$，条件为

\[
o_t=(I_t^1,\ldots,I_t^n,\ell_t,q_t),
\]

监督目标是未来 $H$ 步动作

\[
A_t=[a_t,\ldots,a_{t+H-1}]\in\mathbb R^{H\times d_a}.
\]

论文取 $H=50$。数据管线必须处理轨迹尾部 padding、无效动作 mask、不同 embodiment 的维度适配与数值归一化；否则相同 loss scale 会对应不同物理幅度。

## 采样 Noise 与 Flow Time

训练为每个 $A_t$ 采样同 shape 的 Gaussian noise：

\[
\epsilon\sim\mathcal N(0,I),
\]

再采样 $\tau\in[0,1]$。π0 使用 shifted beta distribution，使训练更频繁覆盖路径中特定噪声区域；这属于 π0 的优化选择，并非 [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) 成立的必要条件。

论文时间方向下，中间动作与监督速度为

\[
A_t^\tau=\tau A_t+(1-\tau)\epsilon,
\qquad
u_t=A_t-\epsilon.
\]

## 条件编码与动作编码汇合

图像、语言和机器人状态经 [π0 Architecture](/robot-learning/pi0/architecture/) 形成 observation prefix。每个 $a_i^\tau$ 与 $\tau$ 的 embedding 组成 action token。blockwise mask 允许 action 读取全部条件以及其他 action positions，使输出速度对场景、指令、当前构型和整段未来轨迹同时条件化。

## 速度场回归

模型在 $H$ 个动作位置输出

\[
v_\theta(A_t^\tau,\tau,o_t)\in\mathbb R^{H\times d_a},
\]

并最小化

\[
\mathcal L(\theta)=
\mathbb E\left[
\left\|
v_\theta(A_t^\tau,\tau,o_t)-(A_t-\epsilon)
\right\|_2^2
\right].
\]

真实 $A_t$ 只用于构造 noisy input 与 target velocity，不会作为网络条件泄漏到 prediction branch。一次 forward 即可得到无偏的随机时间训练目标，无需在每个 batch 内执行完整 ODE solver。

## Mask 与 Padding 如何进入 Loss

短轨迹末端或低维机器人接口产生的 padding 不能参与同等监督。设有效位置 mask 为 $m\in\{0,1\}^{H\times d_a}$，实际 reduction 应只覆盖有效元素：

\[
\mathcal L_{masked}
=
\frac{\sum m\odot\lVert v_\theta-u\rVert_2^2}
{\sum m}.
\]

否则模型会浪费容量学习输出 padding 常数，且不同 action dimension 的数据源会获得不合理的相对权重。

## openpi 的反向时间记号

released openpi 写作

\[
x_t=t\epsilon+(1-t)A,\qquad u_t=\epsilon-A,
\]

其中 $t=1$ 为 noise，$t=0$ 为 data。它与论文写法在路径方向和 target sign 上同时相反，平方回归问题等价。核对实现时应追踪端点、速度符号和 inference update 三者，而不能只比较单条公式。

## 训练分布决定策略边界

损失只要求模型拟合示范分布中的条件速度。数据未覆盖的相机视角、机器人构型或接触状态不会因 Flow Matching 自动得到正确控制。pre-training/post-training 的数据混合、归一化统计与任务采样权重，因此共同决定模型最终能在哪些条件下生成可靠动作。

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Section IV. https://arxiv.org/abs/2410.24164
- Physical Intelligence. *openpi*, `Pi0.compute_loss`. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
