---
title: "Inference"
kind: "act-topic"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/inference/"
prerequisites:
  - "/robot-learning/act/architecture/"
  - "/robot-learning/act/action-chunking/"
related:
  - "/robot-learning/act/temporal-ensemble/"
  - "/robot-learning/act/why-z-zero-at-inference/"
---

# Inference

ACT Inference 把训练好的 policy 放进 closed-loop robot rollout。Training-only CVAE encoder 不再存在；每次 policy forward 都使用当前 observation，并固定 latent $z=0$，输出一个 future action chunk。

## Initial State

Released evaluation 读取 training 时保存的 qpos/action normalization statistics，并加载最低 validation loss 对应的 checkpoint。

当前 qpos 按训练统计量标准化：

\[
\tilde q_t
=
\frac{q_t-\mu_q}{\sigma_q}.
\]

各 camera image 转为 tensor、缩放到 $[0,1]$，随后进入与训练相同的 image normalization。

## Latent Input

推理没有 ground-truth future action sequence，因此无法调用

\[
q_\phi(z|A_t,\bar o_t).
\]

官方 model 在 `actions is None` 时直接创建

\[
z=\mathbf0\in\mathbb R^{32}.
\]

然后将它投影到 Transformer hidden dimension。详细原因属于 [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)。

## Policy Output

一次 forward 得到

\[
\hat A_t
=
(
\hat a_{t|t},
\hat a_{t+1|t},
\ldots,
\hat a_{t+k-1|t}
).
\]

默认 ALOHA setting 中

\[
\hat A_t\in\mathbb R^{100\times14}.
\]

这些值仍在 normalized action space 中。选出当前要执行的 action 后，需要使用 training statistics 反标准化：

\[
a=\tilde a\odot\sigma_a+\mu_a.
\]

这里的 $\mu_a,\sigma_a$ 来自 demonstration dataset。

## Without Temporal Ensemble

Released code 默认 `query_frequency = num_queries = k`。因此 policy 每 $k$ 步重新 query 一次。

假设刚在 timestep $s$ 得到 chunk $\hat A_s$，接下来的 timestep 依次使用

\[
\hat A_s[0],\hat A_s[1],\ldots,\hat A_s[k-1].
\]

代码用

\[
t\bmod k
\]

选择 chunk 内的 action slot。

这最接近朴素 action chunking：一次 observation 支撑接下来一整段动作，因此反馈只在下一个 query boundary 更新。

## With Temporal Ensemble

启用 `--temporal_agg` 后，released code 把 query frequency 改为 1。每个 timestep 都重新获得完整 action chunk，并把它写进一个二维 time/action buffer。

对当前 timestep $t$，提取所有历史 query 对 $t$ 的预测：

\[
A_t=
\{\hat a_{t|s}:s\le t<s+k\}.
\]

然后使用

\[
w_i=e^{-0.01i}
\]

做归一化加权平均。这样每一步都使用最新 observation，同时保持来自前几个 chunks 的一致性约束。

## Closed-Loop Step

最终反标准化后的 14 维 target joint positions 被发送给 environment / robot。环境变化后获得新的 image 和 qpos，再进入下一 timestep。

完整闭环是

\[
o_t
\rightarrow
\hat A_t
\rightarrow
\text{select/ensemble }a_t
\rightarrow
\text{robot}
\rightarrow
o_{t+1}.
\]

因此 ACT 虽然一次预测长 action sequence，启用 temporal ensemble 时仍然是高频 closed-loop policy，而不是 open-loop trajectory playback。

## Determinism and Remaining Variability

固定 checkpoint、固定 observation 和 $z=0$ 时，ACT policy 本身按这条路径是 deterministic 的。机器人真实 rollout 仍可能因为 sensor noise、physical interaction、environment variation 等因素产生不同 observations，从而得到不同后续 actions。

“deterministic decoding”只说明 latent sampling 不再主动注入随机性，并不意味着整个现实系统轨迹必然完全重复。

## Sources

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware — Zhao et al., 2023](https://arxiv.org/abs/2304.13705)
- [ACT official implementation — tonyzhaozh/act](https://github.com/tonyzhaozh/act)
