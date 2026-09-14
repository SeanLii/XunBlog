---
title: "CLS Token"
kind: "canonical"
domain: "Deep Learning / Transformer"
parent: "Transformer"
canonical: "/deep-learning/transformer/cls-token/"
prerequisites:
  - "/deep-learning/core/embedding/"
  - "/deep-learning/transformer/transformer-encoder/"
related:
  - "/robot-learning/act/cvae-in-act/"
---

# CLS Token

CLS Token 是放在序列中特定位置的可学习特殊 token，其输出 hidden state 可以被用作整个输入序列的聚合表示。这个用法最广为人知地来自 BERT。

## BERT 中的结构

BERT 把特殊 token `[CLS]` 放在输入序列最前面。经过多层 bidirectional Transformer encoder 后，最终 `[CLS]` 位置的 hidden state 被用于 sequence-level classification tasks。

重要的是，`[CLS]` 一开始并不“包含整句话”。它只是一个有独立 embedding 的特殊位置。经过 self-attention layers 后，它能够从其他 tokens 读取信息；训练目标再推动这个位置的最终表示对任务有用。

## 聚合不是平均

CLS representation 不是简单平均所有 token。其更新由多层 learned self-attention、residual paths 和 feed-forward transformations 决定。因此它可以学习非均匀地使用不同位置的信息。

## ACT 中的借用

ACT 论文把 CVAE encoder 描述为 BERT-like Transformer encoder：输入序列由一个 learned `[CLS]` token、当前 joint positions 和 demonstration action chunk 构成。经过 encoder 后，只取 `[CLS]` 对应的 output feature，并用它预测 latent distribution 的参数。

所以 ACT 借用的是“设置一个聚合槽位，让它通过 Transformer 读取整个序列，再从该槽位读出全局表示”的结构思想，而不是使用预训练 BERT 模型。

## Sources

- [BERT — Devlin et al., 2018](https://arxiv.org/abs/1810.04805)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
