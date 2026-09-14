---
title: "Pre-training and Post-training in π0"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/pretraining-and-posttraining/"
prerequisites:
  - "/robot-learning/pi0/"
  - "/robot-learning/cross-embodiment-learning/"
related:
  - "/robot-learning/pi0/training/"
---

# Pre-training and Post-training in π0

π0 的能力不仅来自 architecture，也来自把训练分成两个目的不同的阶段：**pre-training 负责覆盖广泛经验，post-training 负责把行为质量集中到特定任务。**

可以先看成：

```text
大量、广泛、质量不完全一致的数据
              │
              ↓
         pre-training
              │
              ↓
       broad base policy
              │
   少量/中量更高质量任务数据
              │
              ↓
         post-training
              │
              ↓
   更熟练、更稳定的 downstream policy
```

## Pre-training 的目标是覆盖

π0 的 pre-training mixture 包含：

- Physical Intelligence 自己收集的多机器人 dexterous manipulation 数据；
- Open X-Embodiment 中的部分 open-source robot data；
- VLM 已经获得的 Internet-scale image-language pre-training knowledge。

机器人部分在论文报告的 pre-training mixture 中超过 **10,000 小时**，Physical Intelligence 自有数据覆盖 **7 种 robot configurations、68 个 tasks**，并进一步混入 Open X-Embodiment 数据。这里的数字是论文数据规模描述，不是 π0 architecture 的定义。

这一阶段最重要的不是让某一个动作做得极致，而是让模型见到：

- 多种物体；
- 多种场景；
- 多种机器人；
- 成功轨迹；
- 不够完美的执行；
- correction 与 recovery behavior。

这会扩大 policy 训练分布覆盖范围。

## High-Quality Data 的覆盖缺口

高质量 demonstration 往往意味着执行非常顺利。

问题是：真实部署中机器人会偏离完美轨迹。如果训练数据里几乎没有错误、纠正和恢复，模型可能只会“顺风局”。

π0 论文给出的思路是：

```text
广泛 pre-training data
→ 学到多样状态、错误与恢复

高质量 post-training data
→ 学到任务应该怎样高效、稳定地完成
```

两者承担不同作用。

## Post-training 的目标是行为质量

对于高难度任务，例如 laundry folding、box assembly 等，base model 可以再用更集中、更高质量的数据 fine-tune。

这一步不再追求最大 diversity，而更重视：

- 动作是否流畅；
- 策略是否一致；
- 任务是否高效完成；
- 复杂多阶段行为是否稳定。

所以“post-training”不是重新发明另一套 π0 architecture，而是用更窄的数据分布继续更新同一个模型，使其向目标 behavior specialization。

## 与 LLM post-training 类比的边界

论文主动使用了类似 LLM 的 pre-training / post-training language，但类比只应停在训练 recipe：

```text
broad pre-training → task/behavior specialization
```

机器人 post-training 的监督信号仍然来自 embodied trajectories 和 robot actions，并不是简单套用语言模型的 instruction tuning 数据格式。

## Cross-Embodiment 的作用

π0 的 pre-training 同时跨 tasks 与 robot embodiments。

[Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) 让不同机器人数据共享更高层的视觉、语义和 manipulation knowledge，而 action/state 通过统一接口与对应 transforms 适配不同 embodiment。

因此 base model 的“generalist”有两个维度：

```text
任务多样性
   ×
机器人多样性
```

## 与 Architecture 的关系

VLM pre-training 主要提供 Internet-scale visual-semantic knowledge；robot pre-training 再把这些表示连接到 physical actions；post-training 则进一步提高特定 manipulation task 的执行质量。

所以 π0 的完整训练来源不能简化成“用机器人数据 fine-tune PaliGemma”一句话。实际上存在不同数据来源和不同阶段目标。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Sections III and V. https://www.pi.website/download/pi0.pdf
- Physical Intelligence, **π0: Our First Generalist Policy**. https://www.pi.website/blog/pi0
