---
title: "Vision-Language Model"
kind: "canonical"
domain: "Deep Learning / Multimodal Models"
parent: "Multimodal Models"
canonical: "/deep-learning/multimodal/vision-language-model/"
prerequisites:
  - "/deep-learning/transformer/"
  - "/deep-learning/core/embedding/"
related:
  - "/robot-learning/vision-language-action-model/"
  - "/robot-learning/pi0/architecture/"
---

# Vision-Language Model

Vision-Language Model（VLM）是同时处理视觉信息和语言信息的模型。

它最简单的整体形状是：

```text
image ─────┐
           ├──→ Vision-Language Model ───→ multimodal representation / text output
text  ─────┘
```

与只处理文字的语言模型相比，VLM 多了一件关键能力：**让图像中的内容进入与语言可以交互的表示空间。**

## 图像不能直接作为语言 token 使用

一张 RGB 图像本质上是大量像素值：

\[
I\in\mathbb R^{H\times W\times3}.
\]

而 Transformer 通常处理一串 hidden vectors：

\[
X\in\mathbb R^{N\times d}.
\]

因此 VLM 首先需要视觉 encoder，把图像变成一组视觉 tokens：

```text
image
  │
  ↓
Vision Encoder
  │
  ↓
[v1, v2, ..., vm]
```

文字则经过 tokenizer 与 embedding 得到语言 tokens：

```text
"fold the shirt"
       │
       ↓
Tokenizer + Embedding
       │
       ↓
[t1, t2, ..., tn]
```

之后模型需要让两种 token 可以进入同一套 Transformer 信息流。

## 多模态表示

一种常见做法是把视觉 features 投影到语言模型使用的 hidden dimension，再把视觉和语言 tokens 放在同一序列中：

```text
visual tokens      language tokens
v1 v2 ... vm       t1 t2 ... tn
      │                  │
      └────────┬─────────┘
               ↓
         Transformer
               ↓
      multimodal states
```

这时语言 token 可以通过 Attention 读取视觉 token，模型便能把“红色杯子”“桌上的衣服”这类语言概念与图像区域联系起来。

具体 VLM 的融合方式很多，并不是所有模型都采用完全相同的结构。这里最重要的是理解角色分工：

- vision encoder 把像素变成可处理表示；
- language embedding 把文本变成 token representations；
- multimodal Transformer 让视觉与语言发生信息交换。

## VLM 学到的不是机器人动作

VLM 可以有很强的视觉识别、语言理解和语义推理能力，但普通 VLM 的输出通常仍是语言 token 或其他视觉语言任务的结果。

例如它可能理解：

> “把桌上的红色杯子拿起来”

并识别图像中哪一个物体是红色杯子，但这还没有告诉机器人：

- 肩关节应该转多少；
- 手腕应该怎样移动；
- 夹爪何时闭合；
- 接下来几十个控制周期应该输出什么。

从 VLM 到机器人 policy，中间还需要把语义理解连接到动作空间。

## 从 VLM 到 VLA

[Vision-Language-Action Model](/robot-learning/vision-language-action-model/) 在 VLM 基础上进一步把 **action** 纳入模型输入输出体系：

```text
vision + language
       │
       ↓
 semantic understanding
       │
       + robot state
       │
       ↓
     actions
```

π0 就属于这一类模型。它从预训练 VLM 获得视觉和语言知识，再加入机器人 state 与连续 action 的处理结构。

## π0 为什么从 VLM 开始

π0 论文使用 PaliGemma 作为 base VLM。目的不是让机器人“先回答一个问题再执行动作”，而是继承 VLM 已经从大规模 image-text 数据中学到的语义表示能力。

然后 π0 再通过机器人数据训练，把这套表示连接到连续机器人控制。

因此在 π0 中，可以把 VLM backbone 理解成：

> **负责理解“看到了什么、语言要求什么”的大型语义骨干。**

而 action expert 负责把这些信息转成高频连续动作。

## Sources

- Beyer et al., **PaliGemma: A versatile 3B VLM for transfer**, 2024. https://arxiv.org/abs/2407.07726
- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
