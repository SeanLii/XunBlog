---
title: "Paper and Released Implementation"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/paper-and-released-implementation/"
prerequisites:
  - "/robot-learning/pi0/"
related:
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/inference/"
  - "/robot-learning/pi0/action-expert/"
---

# Paper and Released Implementation

π0 的论文与当前官方 openpi repository 并不是逐行相同的静态版本。理解模型时应把 **paper description** 与 **released implementation** 分开记录，尤其不要直接混用 flow-time 符号。

## Flow timestep convention

### Paper description

论文定义：

\[
\tau=0 \quad\text{是 noise},
\qquad
\tau=1 \quad\text{是 data}.
\]

训练中：

\[
A^\tau
=
\tau A+(1-\tau)\epsilon,
\]

目标 velocity：

\[
u=A-\epsilon.
\]

Inference 从 $\tau=0$ 向 $1$ 积分。

### Released implementation

当前 `openpi/src/openpi/models/pi0.py` 在 `sample_actions` 中明确写出：采用 diffusion literature 更常见的相反 convention。

```text
t = 1 : noise
t = 0 : target data
```

Training 中：

\[
x_t=t\epsilon+(1-t)A,
\]

\[
u_t=\epsilon-A.
\]

Inference 使用负步长

\[
dt=-\frac{1}{N}
\]

从 1 积分到 0。

两者是同一生成路径的相反参数方向。读代码时不能把 paper 的 $\tau$ 公式直接套进去。

## Action dimension

### Paper description

论文强调多个 embodiment 拥有不同 configuration spaces 与 action representations，并训练同一个 cross-embodiment model。

具体每个机器人实际动作自由度不同。

### Released implementation

当前 `Pi0Config` 默认：

```text
action_dim = 32
action_horizon = 50
```

模型输入 state 与 action tensor 统一到 32 维接口。

这不表示所有机器人都有 32 个真实 actuator。Repository 的 data transform / normalization / padding pipeline 会把具体 robot dataset 适配到统一模型 spec。

## Camera interface

### Paper description

论文写 observation 中包含 2 或 3 路 RGB images，具体取决于 robot。

### Released implementation

当前 base model interface 固定定义三个 image keys：

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

并提供 image masks，因此缺失的 camera view 可以通过 mask 处理。

这属于 released model API 的统一化设计，而不是论文声称每台 robot 实际都有完全相同三台相机。

## VLM 与 image encoder

### Paper description

论文描述 π0 基于 PaliGemma，一个约 3B parameter VLM；action expert 新增约 300M parameters。

### Released implementation

当前 code config 使用：

```text
paligemma_variant = "gemma_2b"
action_expert_variant = "gemma_300m"
```

image module 在 JAX implementation 中使用 SigLIP `So400m/14`，与 PaliGemma family 的视觉编码设计一致。

## Flow Matching integration steps

### Paper description

论文实验使用 10 个 Euler integration steps：

\[
\delta=0.1.
\]

### Released implementation

`sample_actions(..., num_steps=10)` 仍默认 10 steps。

这一点 paper 与 released implementation 保持一致，只是 integration 时间方向相反。

## Temporal Ensemble

### Paper description

论文 Appendix D 明确说：早期尝试过 ACT-style temporal ensembling，但降低 performance，因此最终采用 open-loop action chunk execution，并周期性重新 inference。

### Released implementation

核心 `Pi0.sample_actions` 负责生成一个 action chunk，本身并不实现 ACT 的 temporal ensemble。

真正部署时执行 chunk 的多少步、何时重新调用 policy，属于上层 policy serving / robot control loop 的职责。

## Repository 已经不只包含原始 π0

当前 openpi repository 后续已经加入 π0-FAST 与 π0.5 等模型。因此阅读 `main` branch 时会看到兼容新模型的 configuration 与 conditional branches。

例如当前 `Pi0Config` 中存在 `pi05` 相关选项。这些代码不能反过来被当成 2024 π0 paper 的原始 architecture definition。

这里讨论的是原始 π0。π0-FAST、π0.5 等后续模型具有不同的 action representation 或 training design，不应和原始 π0 的细节混为一谈。

## Sources

- Black et al., **π0: A Vision-Language-Action Flow Model for General Robot Control**, 2024. https://arxiv.org/abs/2410.24164
- Physical Intelligence official openpi repository. https://github.com/Physical-Intelligence/openpi
- `pi0.py`: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- `pi0_config.py`: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py
