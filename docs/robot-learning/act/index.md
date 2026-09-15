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
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# ACT

> **知识边界**：本文的 canonical 对象是 **ACT**。依赖机制由 [Behavior Cloning](/robot-learning/behavior-cloning/) 的 canonical page 定义；本文只在当前语境中调用其接口。


ACT（Action Chunking with Transformers）是一种面向机器人 imitation learning 的 policy architecture。给定当前多路视觉观测与机器人 proprioception，ACT 一次预测未来一段连续动作，而不是只输出下一个 control action。

在时刻 $t$，可以抽象写成

\[
\hat A_t
=
(\hat a_t,\hat a_{t+1},\ldots,\hat a_{t+k-1})
=
\pi_\theta(o_t),
\]

其中 $k$ 是 action chunk length。

整体数据流为

```text
multi-camera images ─┐
                     │
current joint state ─┤
                     ↓
                    ACT
                     ↓
           future action chunk
[a_t, a_{t+1}, ..., a_{t+k-1}]
```

在 ALOHA 的双臂设置中，action 表示 absolute joint-position targets。若 action dimension 为 14、chunk size 为 100，则单次 policy output 的主要 action tensor 为

\[
100\times14.
\]

Transformer、CVAE 与 ResNet 都是 ACT architecture 的组成或依赖机制；ACT 的核心 policy design 是围绕 action-chunk prediction、latent-conditioned imitation learning 与 closed-loop chunk execution 组织这些组件。

## Action Chunk Prediction

一个 single-step [Behavior Cloning](/robot-learning/behavior-cloning/) policy 可以写成

\[
\hat a_t=\pi_\theta(o_t).
\]

ACT 改为同时预测连续 $k$ 个动作：

\[
\hat a_{t:t+k-1}=\pi_\theta(o_t).
\]

这就是 [Action Chunking](/robot-learning/act/action-chunking/)。一个 chunk 内的多个动作被作为联合输出结构建模，因此模型可以直接表示一段局部运动的时间相关性。

论文将这种设计用于降低 policy 层面的 effective horizon：较长任务不再完全依赖每个 control timestep 的独立 one-step prediction 逐步串联。

## Closed-Loop Execution

预测一个长度为 $k$ 的 action chunk 并不要求机器人在接下来 $k$ 步完全 open-loop 执行。

ACT 的 temporal aggregation 模式在每个 timestep 都可以根据最新 observation 再次查询 policy：

```text
t      : [a_t,   a_t+1, a_t+2, ...]
t + 1  :        [a_t+1, a_t+2, ...]
t + 2  :               [a_t+2, ...]
```

因此同一个实际执行时刻可能拥有多个在不同历史 observations 下产生的预测。[Temporal Ensemble](/robot-learning/act/temporal-ensemble/) 对这些 overlapping predictions 进行加权融合。

ACT 的执行策略由此同时具有：

- chunk-level temporal prediction；
- timestep-level observation feedback；
- overlapping predictions 的 temporal aggregation。

## Policy Architecture

部署时的 policy 主干可以概括为

```text
camera images
     ↓
ResNet backbone
     ↓
spatial visual features ───────┐
                               │
current joint state ───────────┤
                               ↓
                    Transformer Encoder
                               ↓
                            memory
                               ↑
                        action queries
                               ↓
                    Transformer Decoder
                               ↓
                       future action chunk
```

[ResNet](/deep-learning/cnn/resnet/) 把 RGB images 转为 spatial feature maps；current joint state 被投影到 Transformer hidden dimension；Transformer encoder 建立 observation-side memory；decoder 中的 learned action queries 对应 future chunk 的输出 slots。

ACT 的 query-based decoder pattern 继承自 [DETR](/deep-learning/detr/) 的 learned output queries，而具体输出语义由 object slots 改成 future action positions。

更详细的 tensor 与模块关系见 [Architecture](/robot-learning/act/architecture/) 和 [Vision Pipeline](/robot-learning/act/vision-pipeline/)。

## Training-Time Latent Variable

Human demonstrations 即使处于相似 observation，也可能存在动作速度、细微轨迹、操作风格等差异。ACT 在训练阶段使用 Conditional Variational Autoencoder 结构，引入 latent variable $z$。

Training-time recognition branch 使用 current qpos 与 ground-truth future action chunk 估计

\[
q_\phi(z\mid q_t,A_t),
\]

其中

\[
A_t=a_{t:t+k-1}.
\]

得到 $z$ 后，policy predictor 学习

\[
p_\theta(A_t\mid o_t,z).
\]

因此 ground-truth future action 在训练中同时具有两种角色：

1. reconstruction target；
2. recognition encoder 的输入。

这条 latent branch 只在训练时存在。其具体对应关系见 [CVAE in ACT](/robot-learning/act/cvae-in-act/)。

## Inference-Time Latent Choice

部署时没有 ground-truth future action，因此 recognition encoder 无法运行。ACT 使用 standard normal prior

\[
p(z)=\mathcal N(0,I)
\]

并在 released inference 中固定

\[
z=0.
\]

因此 inference graph 只保留 observation-side policy predictor：

```text
latest images + latest joint state + z=0
                  ↓
               ACT policy
                  ↓
          future action chunk
                  ↓
          Temporal Ensemble
                  ↓
         current executed action
```

这使 released ACT 的部署预测是确定性的；CVAE 的 stochastic posterior sampling 属于 training-time latent modeling，而不是 deployment 时必须保留的随机控制策略。

## Input and Output Variables

可将时刻 $t$ 的 observation 写为

\[
o_t=(I_t^1,\ldots,I_t^C,q_t),
\]

其中：

- $I_t^c$：第 $c$ 个 camera 的 RGB image；
- $q_t$：当前 proprioceptive state，在 ALOHA 实验中主要是 joint positions；
- $A_t$：未来 action chunk。

在论文和 released implementation 的主要 ALOHA 设置中，action 是 absolute joint-position target，而不是 torque command 或 end-effector pose。

## Training Objective

Released ACT implementation 的主要 objective 为

\[
\mathcal L
=
\mathcal L_{L1}
+
\beta\mathcal L_{KL},
\]

其中 $\mathcal L_{L1}$ 比较 predicted chunk 与 demonstration action chunk，$\mathcal L_{KL}$ 约束 approximate posterior 接近 standard normal prior。

论文 Algorithm 1、方法正文与 released code 在 reconstruction loss 描述上存在细节差异，见 [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/)。

## Component Boundaries

ACT 使用多项已有机制，但各组件承担不同职责：

| Component | Role in ACT |
|---|---|
| [Behavior Cloning](/robot-learning/behavior-cloning/) | 从 demonstrations 学习 policy 的监督学习基础 |
| [Action Chunking](/robot-learning/act/action-chunking/) | 把输出单位扩展为 future action sequence |
| [ResNet](/deep-learning/cnn/resnet/) | 提取 spatial visual features |
| [Transformer](/deep-learning/transformer/) | 建立 observation memory 与 query-based action decoding |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | 训练阶段表示 demonstration variation |
| [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) | 融合 overlapping chunks 对同一 timestep 的预测 |

这些组件共同构成 ACT，但它们各自的通用理论仍属于对应 canonical topics。

## 机制导出的能力边界

ACT 的 action chunk 能降低 sequential prediction burden，但 chunk 内较远动作仍然更依赖未来未观测信息；chunk size 因此存在 prediction horizon 与 temporal structure 之间的折中。

Temporal Ensemble 可以平滑 overlapping predictions，却不能单独解决严重 distribution shift 或 observation error。CVAE latent 也不保证自动学习出可解释的 human style dimensions。

ACT 的实验重点是 fine-grained bimanual manipulation 与低成本 teleoperation demonstrations，因此把其结果推广到不同 action spaces、control frequencies 或 robot embodiments 时需要重新验证 architecture 与 data assumptions。

## Internal Topics

- [Action Chunking](/robot-learning/act/action-chunking/)
- [Temporal Ensemble](/robot-learning/act/temporal-ensemble/)
- [Architecture](/robot-learning/act/architecture/)
- [CVAE in ACT](/robot-learning/act/cvae-in-act/)
- [Vision Pipeline](/robot-learning/act/vision-pipeline/)
- [Training](/robot-learning/act/training/)
- [Inference](/robot-learning/act/inference/)
- [Complete Data Flow](/robot-learning/act/complete-data-flow/)

## Sources

- Zhao et al. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023. https://arxiv.org/abs/2304.13705
- Official ACT implementation. https://github.com/tonyzhaozh/act
