---
title: "π0"
kind: "canonical"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/"
prerequisites:
  - "/robot-learning/vision-language-action-model/"
  - "/generative-models/flow-matching/"
  - "/robot-learning/act/action-chunking/"
related:
  - "/robot-learning/pi0/architecture/"
  - "/robot-learning/pi0/action-expert/"
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/inference/"
---

# π0

π0（pi-zero）是一种 **Vision-Language-Action robot policy**。它接收机器人当前看到的图像、用户给出的语言指令和机器人自身状态，然后生成未来一段连续机器人动作。

先把整个模型压缩成一张图：

```text
camera images ───────┐
language instruction ├──→ π0 ───→ future continuous action chunk
robot state ─────────┘
```

如果暂时不看内部结构，π0 做的事情和这个图一样直接：

> **看环境 + 读指令 + 知道自己现在的姿态 → 决定接下来一段时间机器人应该怎么动。**

理解 π0 的关键，不是先记 PaliGemma、SigLIP 或 Flow Matching 的公式，而是先看到它试图把两种能力合在一起：

1. VLM 的视觉与语言理解；
2. 高频、连续、精细的机器人控制。

## VLM 已经会“看懂”，但不会直接控制机器人

π0 从预训练 [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 开始。

VLM 可以把图像和语言放进同一个模型里，例如理解：

```text
图像：桌上有一件皱着的 T-shirt
语言："fold the shirt"
```

但普通 VLM 的输出主要还是离散 language tokens。机器人真正需要的是类似：

\[
[a_t,a_{t+1},\ldots,a_{t+H-1}],
\]

其中每个 $a_t$ 都是连续 motor command / joint-space action。

所以 π0 不是“把 prompt 发给 VLM，然后把一句文字答案变成动作”。它直接把 VLM backbone 改造成 robot policy。

## π0 增加了机器人专用的 Action Expert

π0 的主干可以先画成：

```text
images + language
       │
       ↓
pre-trained VLM backbone
       │
       │  通过 self-attention 与机器人分支交换信息
       │
robot state ───→ Action Expert ←── noisy action chunk
                         │
                         ↓
                 action vector field
```

这里的 [Action Expert](/robot-learning/pi0/action-expert/) 是理解 π0 架构的核心。

图像和语言走较大的 VLM weights；机器人 state 和 action tokens 走另一套更小的 weights。两组表示不是彼此隔离，而是在 Transformer self-attention 中发生信息交互。

所以可以把 π0 理解为：

> **一个继承 VLM 语义能力的大 backbone，旁边接入一套专门处理机器人状态和动作的 expert。**

## π0 不直接回归最终动作

这里出现 π0 与普通 Behavior Cloning 很大的区别。

最直接的 policy 可以写成：

\[
\hat A_t=f_\theta(o_t).
\]

给 observation，一次前向就输出 action chunk。

π0 不是这样。它使用 [Flow Matching](/generative-models/flow-matching/)：先从随机 noise action chunk 开始，再反复预测“现在应该往哪个方向改”，逐步得到最终动作。

```text
random action noise
       │
       ↓  π0 predicts velocity
slightly cleaner action
       │
       ↓  π0 predicts velocity again
       ...
       │
       ↓
final action chunk
```

论文中使用 10 个 flow integration steps。

所以一次 π0 inference 内部实际上包含多次 action-expert forward pass。

## 输入与输出

论文把 observation 写成：

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t],
\]

其中：

- $I_t^i$：第 $i$ 路 RGB camera image；
- $\ell_t$：language instruction；
- $q_t$：proprioceptive robot state。

输出是一整个 action chunk：

\[
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}].
\]

论文实验中使用

\[
H=50.
\]

因此 π0 和 ACT 一样使用 [Action Chunking](/robot-learning/act/action-chunking/) 思想：一次预测多个未来动作，而不是只输出下一步。

但它们生成 action chunk 的方式很不同：ACT 的 decoder 直接输出 chunk；π0 则通过 Flow Matching 从 noise 逐步生成 chunk。

## 一次 inference 的最小完整过程

把内部所有关键部分串起来：

```text
1. Camera images
       │
       ↓
   Vision encoder
       │
       ├─────────────┐
2. Language tokens ──┤
       │             ↓
       │       VLM representations
       │             │
3. Robot state ──────┼────┐
                     │    │
4. Random action noise    │
         │                │
         └──→ Action Expert
                  │
                  ↓
          predicted vector field
                  │
                  ↓
          Euler update action
                  │
            repeat 10 times
                  │
                  ↓
             action chunk
```

这里第 4–6 步就是 [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/)。

## π0 的 attention 不是普通全连接 token mixing

π0 把输入分成三个 block：

```text
[images + language] | [robot state] | [noisy actions]
```

并使用特定的 [Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/)。大方向是：

- VLM prefix 不读取后面新加入的机器人 state / action；
- robot state 不读取 action；
- action tokens 可以读取前面的 observation，也可以彼此双向读取。

这个设计既保留了 VLM pre-training 的输入结构，又方便 inference 时缓存 observation 的 keys / values，因为 flow integration 中反复变化的主要是 action tokens。

## Generalist policy 不只是网络结构

π0 的另一个重点是训练 recipe。

它不是只在一个机器人、一个任务上训练，而是先进行大规模、跨机器人、跨任务的 pre-training，再对困难任务做高质量 post-training：

```text
many robots + many tasks + open robot data
                │
                ↓
            pre-training
                │
                ↓
          general base π0
                │
        high-quality task data
                ↓
           post-training
                │
                ↓
       specialized strong policy
```

这部分在 [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) 中展开。

## π0 与 ACT 最容易混淆的地方

两者都：

- 从 robot observation 预测未来动作；
- 使用 action chunks；
- 可以用于高频 dexterous manipulation。

但核心生成机制不同：

| | ACT | π0 |
|---|---|---|
| 条件 | image + state | image + language + state |
| 大规模语义 backbone | 否 | 预训练 VLM |
| action generation | Transformer decoder 直接输出 | Flow Matching 从 noise 生成 |
| latent / stochastic mechanism | training-time CVAE latent | inference-time noisy action flow |
| action chunk | 是 | 是 |
| Temporal Ensemble | 核心设计之一 | 论文最终未采用 |

因此不能把 π0 简单理解成“ACT 加语言”。两者都属于 robot policy，但内部概率建模和模型骨干都不同。

## 从这里继续深入

建议按下面顺序：

1. [Architecture](/robot-learning/pi0/architecture/)：先看所有输入怎样进入同一个 Transformer 系统；
2. [Action Expert](/robot-learning/pi0/action-expert/)：理解为什么 VLM 之外还需要第二套 weights；
3. [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/)：理解 action 从 noise 到真实动作的生成过程；
4. [Training](/robot-learning/pi0/training/)：训练时怎样构造 noisy action 与 regression target；
5. [Inference](/robot-learning/pi0/inference/)：10 个 integration steps 怎样真正生成动作；
6. [Complete Data Flow](/robot-learning/pi0/complete-data-flow/)：把 tensor/data flow 从输入到执行重新走一遍。

如果只记住一句话：

> **π0 是一个建立在预训练 VLM 之上的 VLA robot policy，它用专门的 Action Expert 和 conditional Flow Matching，把图像、语言和机器人状态转成连续 action chunks。**

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
- Physical Intelligence, **π0: Our First Generalist Policy**. https://www.pi.website/blog/pi0
- Official openpi repository. https://github.com/Physical-Intelligence/openpi
