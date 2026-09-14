---
title: "BERT"
kind: "canonical"
domain: "Deep Learning / BERT"
parent: "Deep Learning"
canonical: "/deep-learning/bert/"
prerequisites:
  - "/deep-learning/transformer/transformer-encoder/"
  - "/deep-learning/attention/self-attention/"
  - "/deep-learning/sequence-modeling/positional-encoding/"
related:
  - "/deep-learning/bert/cls-token/"
---

# BERT

BERT（Bidirectional Encoder Representations from Transformers）是一个基于 Transformer Encoder 的预训练语言模型。它的核心目标不是直接生成下一段文字，而是先从大规模文本中学习能够同时利用左右上下文的 token representations，再把这些 representations 用于分类、问答、序列标注等下游任务。

最小结构可以写成：

```text
tokens
  ↓
token + position + segment representations
  ↓
stacked Transformer Encoders
  ↓
contextual token representations
```

因此 BERT 的主体是 encoder-only Transformer。

## Bidirectional Context

自回归语言模型在位置 $i$ 预测 token 时通常只能读取左侧上下文。

BERT 的预训练表示则允许一个 token 在 encoder 中同时和左右两侧 token 交换信息：

```text
left context ← token → right context
```

这使输出 $h_i$ 成为双向 contextual representation，而不是只由前缀决定。

## Masked Language Modeling

BERT 的主要预训练目标是 Masked Language Modeling（MLM）。

训练时随机选取部分 token，并要求模型根据周围上下文恢复它们。抽象地：

\[
x_{\setminus i}
\rightarrow
p(x_i\mid x_{\setminus i}).
\]

因为被预测位置可以读取左右两边的信息，MLM 与标准 left-to-right next-token prediction 的条件结构不同。

## Special Tokens

原始 BERT 输入中常见：

- `[CLS]`：放在 sequence 开头，用作 sequence-level representation 的读取位置；
- `[SEP]`：分隔句子或表示 sequence 结束；
- `[MASK]`：用于 MLM 预训练。

其中 [CLS Token](/deep-learning/bert/cls-token/) 后来被很多模型借用，但 `[CLS]` 是 BERT 的具体输入设计，不是 Transformer architecture 自带的标准组件。

## Input Representation

BERT 把多个 embedding 相加：

\[
e_i=
e_i^{token}
+
e_i^{position}
+
e_i^{segment}.
\]

然后送入 Transformer Encoder stack。

Segment embedding 用来区分 sentence A / sentence B；position representation 提供顺序；token embedding 表示词元身份。

## Pre-training 与 Fine-tuning

BERT 的基本范式是：

```text
large-scale pre-training
        ↓
general contextual representations
        ↓
task-specific fine-tuning
```

原论文使用 MLM 和 Next Sentence Prediction（NSP）进行预训练。后续模型经常改变甚至移除 NSP，但这不改变 BERT 原始设计。

## BERT 与 Transformer 的关系

BERT 没有发明 Transformer Encoder。它使用 Transformer 作为基础架构，并改变训练目标与输入组织方式，形成面向双向语言表示的预训练模型。

因此：

- Transformer 属于更早的通用架构；
- BERT 是建立在 Transformer Encoder 上的具体模型；
- `[CLS]` 是 BERT-specific design，不是 Transformer architecture 的通用必备组件。

## 与 ACT 的连接

ACT 的 training-only latent encoder 使用一个 learnable CLS-like token 来汇总 joint state 与 action sequence，再从该位置预测 latent distribution 参数。

这是对 BERT-style summary token 思路的借用，不意味着 ACT 使用完整 BERT。

## Sources

- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018.
