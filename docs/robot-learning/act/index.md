---
title: "ACT"
kind: "canonical"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/"
prerequisites:
  - "/robot-learning/behavior-cloning/"
related:
  - "/robot-learning/act/action-chunking/"
  - "/robot-learning/act/temporal-ensemble/"
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/cvae-in-act/"
---

# ACT

ACT（Action Chunking with Transformers）是 Zhao 等人在 ALOHA 工作中提出的 imitation-learning policy。它从当前视觉与关节状态出发，一次预测未来一段连续 joint-position targets，并用 action chunking、temporal ensemble 和 CVAE training 处理 fine-grained manipulation 中的长时序误差与 human demonstration variability。

## Problem Setting

ACT 面向 demonstration-based robot control。原论文中的 observation 包含：

\[
o_t=(I_t^{1},I_t^{2},I_t^{3},I_t^{4},q_t),
\]

其中 $I_t^c$ 是第 $c$ 个相机的 RGB image，$q_t\in\mathbb R^{14}$ 是双臂 joint positions。一个 action 也是 14 维 absolute target joint positions：

\[
a_t\in\mathbb R^{14}.
\]

低层 PID controller 负责追踪这些 target positions。ACT 学习的是从 observation 到目标关节序列的 policy，而不是直接学习电机 torque controller。

## Action Sequence Prediction

普通 single-step behavior cloning 学习

\[
\pi_\theta(a_t|o_t).
\]

ACT 改为预测长度为 $k$ 的 future action chunk：

\[
\pi_\theta(a_{t:t+k-1}|o_t).
\]

原论文常用 $a_{t:t+k}$ 记号描述这个长度为 $k$ 的 chunk；为了避免端点记号歧义，这里显式写成 $a_t,\ldots,a_{t+k-1}$。

这一改变属于 [Action Chunking](/robot-learning/act/action-chunking/)。论文的核心动机是缩短高频长轨迹在 policy 决策层面的 effective horizon，并让一次预测直接表示一段连贯行为。

## Closed-Loop Execution

只每隔 $k$ 步观察一次会降低反馈频率。ACT 因此进一步提出 [Temporal Ensemble](/robot-learning/act/temporal-ensemble/)：推理时每个 timestep 都重新查询 policy，于是多个相邻 action chunks 会对同一个物理时刻产生重叠预测，再对这些“同一时刻的预测”做指数加权平均。

这保留了高频视觉反馈，同时避免每 $k$ 步硬切换 chunk 造成的突变。

## Modeling Human Demonstrations

同一个 observation 并不一定只有一种合理 human action trajectory。ACT 将 policy 训练成 [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/)：训练时用 demonstration action chunk 推断 latent style variable $z$，再让 decoder 根据当前 observation 与 $z$ 重建 action chunk。

ACT-specific 的对应关系属于 [CVAE in ACT](/robot-learning/act/cvae-in-act/)。推理时 training-only CVAE encoder 被移除，并令

\[
z=0,
\]

也就是 standard normal prior 的均值，从而得到 deterministic decoding。

## Architecture Boundary

ACT 使用多个已有模块，但这些模块不属于 ACT 自己的通用理论：

- [ResNet](/deep-learning/cnn/resnet/)：视觉 feature extraction；
- [Transformer Encoder](/deep-learning/transformer/transformer-encoder/)：融合 observation features；
- [Transformer Decoder](/deep-learning/transformer/transformer-decoder/)：用 learnable action queries 读取 encoder memory；
- [CLS Token](/deep-learning/transformer/cls-token/)：在 training-only CVAE encoder 中聚合 action sequence 与 proprioception。

ACT 自己决定的是这些模块如何围绕 action chunking、latent conditioning 和 temporal ensembling 组合起来。

## Default Paper Configuration

原论文 Table III 给出的 ACT 配置为：hidden dimension 512、4 encoder layers、7 decoder layers、8 attention heads、feed-forward dimension 3200、chunk size 100、$\beta=10$、dropout 0.1、batch size 8、learning rate $10^{-5}$。

这些数值是论文实验配置，不是 ACT 的定义。换机器人、数据频率或任务后，chunk size、model size 与 optimization settings 都可以改变。

## Empirical Scope

论文在 ALOHA 的 simulated 与 real-world bimanual manipulation tasks 上验证 ACT。它的结果支持 action chunking、temporal ensembling 与 CVAE training 在这些实验中的有效性，但不意味着 ACT 对任意机器人、任意数据分布都有相同保证。

尤其需要保留 [Behavior Cloning](/robot-learning/behavior-cloning/) 的基本限制：ACT 主要使用 offline demonstrations；action chunking 可以缓解 long-horizon compounding effect，却没有从原则上消除 policy-induced distribution shift 或 demonstration coverage 问题。

## Reading Map

ACT 自己的完整内容被拆成独立页面：

- [Action Chunking](/robot-learning/act/action-chunking/)
- [Temporal Ensemble](/robot-learning/act/temporal-ensemble/)
- [Architecture](/robot-learning/act/architecture/)
- [CVAE in ACT](/robot-learning/act/cvae-in-act/)
- [Vision Pipeline](/robot-learning/act/vision-pipeline/)
- [Training](/robot-learning/act/training/)
- [Inference](/robot-learning/act/inference/)
- [Complete Data Flow](/robot-learning/act/complete-data-flow/)
- [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)
- [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/)

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
