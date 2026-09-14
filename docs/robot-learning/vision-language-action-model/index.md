---
title: "Vision-Language-Action Model"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/vision-language-action-model/"
prerequisites:
  - "/deep-learning/multimodal/vision-language-model/"
related:
  - "/robot-learning/pi0/"
  - "/robot-learning/cross-embodiment-learning/"
---

# Vision-Language-Action Model

Vision-Language-Action Model（VLA）是一类把视觉、语言与机器人动作放进同一个 policy 中的模型。

最小形式可以写成：

```text
camera images ─────┐
language command ──┼──→ VLA policy ───→ robot actions
robot state ───────┘
```

它和普通 [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 的关键区别在最后一端：VLM 主要学习图像与语言之间的关系，而 VLA 还必须真正输出能够控制机器人的动作。

## 从“理解指令”到“执行指令”

假设用户说：

> fold the shirt

一个 VLM 可以识别 shirt、理解 fold 的语义，也可以根据图像描述衣服当前在哪里。

但机器人 policy 还需要继续回答：

```text
左臂下一步怎么动？
右臂下一步怎么动？
夹爪什么时候闭合？
接下来 0.5 秒的动作轨迹是什么？
```

VLA 的目标就是把前半段的视觉语言理解与后半段的低层机器人控制连接起来。

## VLA 不是一种固定输出形式

“VLA”描述的是模型覆盖的 modality 与任务角色，不规定 action 必须怎样表示。

不同 VLA 可以采用不同 action decoder：

- 把连续动作离散化成 tokens，再像语言一样 autoregressive 生成；
- 直接回归连续动作；
- 使用 diffusion / flow-based 方法生成连续 action chunk。

因此不能把“VLA”与“action tokenization”画等号。

RT-2 代表了一条重要路线：把 robot actions 表示成离散 text-like tokens。OpenVLA 也采用离散 action token 的 autoregressive 方式。π0 则选择另一条路线：**保留连续动作，并用 Flow Matching 生成 action chunk。**

## 为什么 VLA 常从 VLM 开始

机器人数据相比互联网 image-text 数据少得多。预训练 VLM 已经学到大量：

- 物体类别；
- 视觉属性；
- 语言概念；
- 图像与文本之间的对应关系。

VLA 可以把这些已有表示作为起点，再通过 robot demonstrations 学习“这些语义如何对应动作”。

所以一个常见思路是：

```text
Internet image-text data
        │
        ↓
pre-trained VLM
        │
        + robot trajectories
        ↓
       VLA
```

这并不意味着互联网数据直接包含机器人的正确关节控制。它提供的是更通用的视觉语言表示，而动作能力仍需要 embodied robot data 学习。

## 一个 VLA policy 的条件分布

抽象地，可以把 VLA 写成

\[
p(A_t\mid I_t,\ell_t,q_t),
\]

其中：

- $I_t$：当前视觉观测；
- $\ell_t$：语言 instruction；
- $q_t$：机器人 proprioceptive state；
- $A_t$：要执行的 action 或 action chunk。

不同模型的主要区别，就体现在怎样编码条件、怎样表示 $A_t$、怎样训练这个 conditional distribution。

## π0 在 VLA 中的位置

π0 可以先压缩成：

```text
pre-trained VLM
      │
      + robot state
      + Action Expert
      + Flow Matching
      ↓
continuous action chunks
```

它继承 VLM 的图像与语言表示，但没有强迫连续机器人动作变成语言 token，而是专门增加 action expert 处理 robotics-specific continuous inputs/outputs。

## Sources

- Brohan et al., **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, 2023. https://arxiv.org/abs/2307.15818
- Kim et al., **OpenVLA: An Open-Source Vision-Language-Action Model**, 2024. https://arxiv.org/abs/2406.09246
- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
