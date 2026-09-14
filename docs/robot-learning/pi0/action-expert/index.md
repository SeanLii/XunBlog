---
title: "Action Expert"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/action-expert/"
prerequisites:
  - "/robot-learning/pi0/architecture/"
  - "/deep-learning/attention/self-attention/"
related:
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
---

# Action Expert

Action Expert 是 π0 为 **robot state 与 continuous actions** 增加的一套 Transformer weights。

它存在的原因可以从 VLM 的训练经历理解：预训练 VLM 原本主要见过 image representations 与 language tokens，却没有见过“机器人第 7 个关节当前是 0.42 rad”或“未来 50 步连续 action vector”这种数据。

π0 因此没有简单地把这些新输入完全交给原来的 VLM weights，而是加入第二套更小的 weights 专门处理 robotics-specific tokens。

## 两个 expert，不是两个完全分开的网络

最容易产生的误解是：

```text
VLM model ─→ 输出一个语义结果
                    ↓
            Action model 再读取
```

π0 不是这种串联结构。

论文描述的是一个 Transformer 内部存在两套 weights：

```text
image + language tokens ─→ VLM expert weights ─┐
                                               │
state + action tokens ────→ Action Expert ─────┤
                                               ↓
                                      self-attention interaction
```

不同 token 根据自己的 modality 使用不同 expert 的 projection / MLP 等参数，但它们在 self-attention 层中仍然交互。

所以 Action Expert 并不是“VLM 后面的动作头”那么简单。

## 哪些 token 进入哪个 expert

论文中的 routing 很明确：

### VLM expert

处理：

\[
[I_t^1,\ldots,I_t^n,\ell_t].
\]

也就是 images 与 language prompt。

### Action Expert

处理：

\[
[q_t,A_t^\tau].
\]

也就是 robot state 与 noisy action chunk。

这种划分对应一个很清楚的知识边界：

```text
VLM expert：世界里是什么、语言要求什么
Action Expert：机器人现在怎样、动作应该怎样变化
```

真实模型里两者通过 attention 融合，所以不是独立推理。

## Action Expert 的较小规模

Flow Matching inference 需要反复更新 action chunk。Observation prefix 可以缓存，但 action tokens 每个 integration step 都变化，因此 action expert 要被重复运行。

为了降低这部分成本，论文把 action expert 缩小到大约：

```text
width = 1024
MLP dim = 4096
parameter count ≈ 300M
```

相比之下，PaliGemma language backbone 基于 Gemma 2B，hidden width 更大。

这个缩小不是随意压参数，而是直接针对 inference pattern：**反复计算的是 action suffix，所以让 action-side computation 更轻。**

## 输入 action 之前还要加入 flow timestep

Action Expert 处理的 action 并不是最终 action：

\[
A_t^\tau.
\]

因此它还必须知道当前 flow timestep $\tau$。

论文把每个 noisy action 先投影，再与 sinusoidal timestep embedding 拼接并通过 MLP：

\[
W_3\operatorname{swish}
\left(
W_2\operatorname{concat}
(W_1a_{t'}^\tau,\phi(\tau))
\right).
\]

这一步的作用可以直接理解为：

> 给 action token 同时标注“我现在是什么值”和“我现在处于从 noise 到 data 的哪个阶段”。

## Action Expert 输出什么

经过 Transformer 后，π0 只取对应 action positions 的 hidden states，再通过 linear projection 得到：

\[
v_\theta(A_t^\tau,o_t).
\]

其 shape 与 action chunk 相同：

\[
H\times d_a.
\]

每一个位置都给出当前 action token 的 flow velocity。

这就是为什么 Action Expert 不能被理解成普通 classification head。它产生的是一个高维 vector field，用于下一次 action update。

## 当前 openpi 中的实现

官方 openpi 的 π0 config 默认使用：

```text
VLM variant:        gemma_2b
Action Expert:      gemma_300m
action horizon:     50
action dimension:   32
```

`action_dim=32` 是开源实现统一接口的默认最大动作维度，不等于论文中的每一种机器人都真的有 32 个物理自由度。具体 embodiment 的 state/action 会经过 data transforms、padding 与 masks 适配到模型接口。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi `pi0_config.py` and `pi0.py`. https://github.com/Physical-Intelligence/openpi
