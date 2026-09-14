---
title: "Vision-Language-Action Model"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/vision-language-action-model/"
prerequisites:
  - "/deep-learning/multimodal/vision-language-model/"
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/cross-embodiment-learning/"
  - "/robot-learning/pi0/"
---

# Vision-Language-Action Model

Vision-Language-Action Model（VLA）是一类把 visual observations、language instructions 与 robot actions 放在同一个 policy learning system 中的模型。

最小 mental model：

```text
camera observation
        +
language instruction
        +
robot state (optional / common)
        ↓
       VLA
        ↓
   robot action(s)
```

它和 [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) 的关键区别是：VLM 的主要输出仍是语义 representation / text / matching result，而 VLA 必须进入**physical control interface**。

## 从理解任务到执行任务

VLM 可以理解：

> “把红色杯子放进盒子。”

它可能识别：

- 哪个是红色杯子；
- 哪个是盒子；
- “放进”表示什么关系。

但 robot policy 还要决定：

- arm 怎样移动；
- gripper 什么时候闭合；
- trajectory 如何避障；
- action magnitude 多大；
- 下一时刻根据新 observation 怎么调整。

所以 semantic understanding 和 motor policy 之间仍有很大 gap。

## VLA 是 Model Family，不是一个固定 Architecture

不同 VLA 可以输出完全不同的 action representation：

- discrete action tokens；
- continuous action vectors；
- action chunks；
- trajectories；
- diffusion / flow generated continuous actions。

因此不能用“把 action token 当文字生成”来定义所有 VLA。

更一般的 conditional policy 可以写成：

\[
p(a_{t:t+k-1}
\mid
I_{\le t},l,s_{\le t}).
\]

其中：

- $I$：vision；
- $l$：language；
- $s$：robot state / proprioception；
- $a$：action 或 action chunk。

## 从 VLM 到 Action Policy

Internet-scale VLM / multimodal model 已经学习大量：

- object concepts；
- attributes；
- spatial / semantic relations；
- language grounding；
- broad visual features。

Robot demonstration data 则通常规模更小、采集更贵。

因此一个重要路线是：

```text
large-scale vision-language pretraining
             ↓
retain semantic representation
             ↓
train / adapt action prediction
             ↓
robot policy
```

RT-2 正是这一思想的重要代表：把 robot actions 表示进 VLM-style output space，让 web-scale knowledge 迁移到 control。

## Action Tokenization 路线

RT-2 将 continuous robot actions discretize / tokenize，使 action 可以被 autoregressive model 当作特殊 tokens 输出。

这种方法的优点是可以最大程度复用 language-model output machinery。

但 quantization 会带来 precision / representation tradeoff，而且 autoregressive token-by-token generation 也影响 control latency。

所以后续 VLA 不一定沿用同一种 action representation。

## Continuous Action 路线

OpenVLA 仍采用 discrete action tokenization strategy，而 π0 则引入 continuous action expert 与 flow matching，直接生成 continuous action chunks。

这说明 VLA architecture 正在围绕一个核心问题分化：

> 如何把高容量 vision-language representation 接到高频、精确、continuous robot control 上？

## Pretraining 与 Robot Data

VLA training 常组合两类数据：

1. vision-language / Internet-scale data；
2. robot demonstration data。

这两类 supervision 的作用不同：

- web data 提供 broad semantic knowledge；
- robot data 把 representations 对齐到 action space 与 physical interaction。

只拥有 VLM 能力并不会自动得到 robot control ability。

## Closed-Loop Control

VLA 最终仍是 policy。

执行 action 后 environment 改变，下一轮 observation 也改变：

```text
observe
 ↓
VLA policy
 ↓
action
 ↓
environment changes
 ↓
observe again
```

因此 robot performance 不只取决于单次 action prediction accuracy，还取决于 latency、control horizon、action chunk strategy、error recovery 和 observation refresh。

## Generalist VLA

VLA 的长期目标通常不是“一条 instruction 一个模型”，而是：

- many tasks；
- many objects；
- many environments；
- potentially many embodiments。

这与 [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) 紧密相关。

Open X-Embodiment、RT-X、OpenVLA、π0 等工作都在探索数据规模和 model capacity 是否能带来 generalist robot policies。

## VLA 的边界

一个模型同时输入 image 和 text，并不自动是 VLA。

必须存在 action prediction / control output，并且该 output 与 robot behavior 建立训练关系。

同样，一个 robot policy 输入 image 但不使用 language，也不是 vision-language-action model。

## Sources

- Brohan et al. / Zitkovich et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. CoRL, 2023.
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. 2024.
- Open X-Embodiment Collaboration et al. *Open X-Embodiment*. 2023/2024.
