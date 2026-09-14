---
title: "Architecture"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/architecture/"
prerequisites:
  - "/robot-learning/pi0/"
  - "/deep-learning/multimodal/vision-language-model/"
  - "/deep-learning/attention/self-attention/"
related:
  - "/robot-learning/pi0/action-expert/"
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
  - "/robot-learning/pi0/complete-data-flow/"
---

# Architecture

π0 的 architecture 最容易理解成一个 **两套 Transformer weights、共享 attention interaction 的 VLA system**。

先看整张结构：

```text
RGB images ─→ Vision Encoder ─┐
                              │
Language tokens ──────────────┼─→ VLM expert ─────┐
                              │                    │
Robot state q_t ──────────────┼─→ Action Expert ──┼─→ shared self-attention interaction
                              │                    │
Noisy action chunk A_t^τ ─────┘→ Action Expert ───┘
                                                    │
                                                    ↓
                                      action-token hidden states
                                                    │
                                                    ↓
                                         vector field v_θ
```

图像和语言主要沿用预训练 VLM；机器人 state 与 noisy actions 使用额外的 [Action Expert](/robot-learning/pi0/action-expert/) weights。

## 第一组输入：Images

π0 的 observation 可以包含多路 RGB images：

\[
I_t^1,I_t^2,\ldots,I_t^n.
\]

论文实验中不同机器人通常使用 2 或 3 路图像。

每张图像先通过 image encoder 得到视觉 tokens，再进入 VLM Transformer。论文基于 PaliGemma；当前官方 openpi 实现使用 SigLIP `So400m/14` image module 生成视觉 tokens。

关键不是记具体 encoder 名字，而是数据类型发生了变化：

```text
RGB pixels
   │
   ↓
vision encoder
   │
   ↓
sequence of visual embeddings
```

到这里，image 才成为 Transformer 能处理的 token sequence。

## 第二组输入：Language

语言 instruction $\ell_t$ 被 tokenizer 转成离散 token IDs，再通过语言 embedding 变成 vectors。

例如：

```text
"fold the shirt"
       │
       ↓
[token ids]
       │
       ↓
[token embeddings]
```

图像 tokens 与 language tokens 一起组成 VLM prefix。

## 第三组输入：Robot State

π0 还需要 proprioceptive state：

\[
q_t.
\]

论文中主要把它描述为 joint-angle vector。这个输入不是 VLM pre-training 中原本存在的 modality，因此它通过一个额外 projection 映射到 action-expert hidden dimension。

可以先理解成一个 state token：

```text
q_t ∈ R^d
  │
  ↓ Linear
state embedding
```

当前 openpi 的 π0 implementation 也把 continuous state 投影成单个 state token。

## 第四组输入：Noisy Action Chunk

π0 不是直接让 Transformer“凭空”输出 $H$ 个动作，而是在 Flow Matching 中输入当前的 noisy action chunk：

\[
A_t^\tau=
[a_t^\tau,\ldots,a_{t+H-1}^\tau].
\]

每个 noisy action 都对应一个 action token，所以总共有 $H$ 个 action positions。

论文使用

\[
H=50.
\]

这些 action vectors 先做 linear projection，然后与 flow timestep $\tau$ 的 embedding 结合，再送入 action expert。

## Flow timestep 也必须进入网络

同一个 action value 在不同 flow timestep 上含义不同：

- 靠近 noise 端时，它可能仍然很随机；
- 靠近 data 端时，它已经接近可执行动作。

所以网络必须知道当前 $\tau$。

论文把 noisy action embedding 与 sinusoidal timestep embedding 通过 MLP 混合。抽象写成：

\[
e(a^\tau,\tau)
=
\operatorname{MLP}
(\operatorname{concat}(W_a a^\tau,\phi(\tau))).
\]

因此 action expert 接收的并不是“裸 action vector”，而是带 flow-time 信息的 action representation。

## 三个 token blocks

整个 Transformer sequence 可以按功能看成三块：

```text
Block 1              Block 2       Block 3
images + language | robot state | noisy action chunk
```

这些 block 不使用完全对称的 attention 权限，而是通过 [Blockwise Causal Attention Mask](/robot-learning/pi0/blockwise-causal-attention-mask/) 控制信息方向。

特别重要的是 action block：它可以读取完整 observation，并且 action tokens 彼此可以双向 attention。

这意味着第 20 个 future action 并不是独立预测的；整个 action chunk 可以在 Transformer 内部共同建模。

## 输出不是 action，而是 vector field

Action Expert 最终产生 action-token hidden states，然后 linear projection 得到：

\[
v_\theta(A_t^\tau,o_t).
\]

它的 shape 与 action chunk 一致。

但这个值不是最终动作，而是：

> **当前 noisy action chunk 在 flow space 中下一步应该怎样变化。**

然后 [Euler Method](/mathematics/numerical-methods/euler-method/) 用这个 vector field 更新 action chunk。

所以一次 network forward 的真正 I/O 更准确地写成：

\[
(o_t,A_t^\tau,\tau)
\longrightarrow
v_\theta(A_t^\tau,o_t).
\]

## VLM 与 Action Expert 的参数分工

如果直接把 state 与 action 当作普通 VLM tokens，模型必须让原本为了 image/text pre-training 学到的 weights 同时承担一种完全不同的 continuous robotics modality。

π0 选择保留两套 weights：

```text
image / text  → large pretrained VLM weights
state / action → smaller robotics-specific weights
```

但两边仍通过 self-attention 交换信息。

这让模型一方面继承 VLM pre-training，一方面给连续机器人输入输出更专门的参数空间。

## 论文规模

论文使用的 PaliGemma base model约为 3B 参数，并新增约 300M 参数的 action expert，形成约 3.3B parameter model。

这里的 300M action expert不是另一个完整独立模型，而是同一 Transformer 体系中的第二套较小 weights。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Section IV and Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi implementation, `src/openpi/models/pi0.py`. https://github.com/Physical-Intelligence/openpi
