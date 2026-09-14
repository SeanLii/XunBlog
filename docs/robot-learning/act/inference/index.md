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

ACT inference 比 training graph 简单，因为真实 future action 不存在，training-only latent encoder 也随之消失。

部署时每个 timestep 的主流程是：

```text
最新 images + 最新 qpos
          │
          ↓
        z = 0
          │
          ↓
      ACT policy
          │
          ↓
   future action chunk
          │
          ↓
 temporal aggregation
          │
          ↓
  execute current action
          │
          ↓
 receive next observation
```

## 第一步：获得当前 observation

机器人在时刻 $t$ 读取多路 camera images 与 current qpos。

官方 evaluation code 会使用 training dataset statistics 对 qpos 做 normalization；预测出的 action 随后再通过 action mean/std 做反标准化。

Normalization 改变数值尺度，不改变 policy 的语义。

## 第二步：latent 固定为 0

训练时 latent encoder 产生 $\mu,\log\sigma^2$ 并 sample $z$。

推理时 code 直接创建

\[
z=\mathbf 0.
\]

其维度在 released model 中为 32。这个 zero vector 再经过 `latent_out_proj` 映射到 Transformer hidden dimension。

更详细的设计含义见 [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/)。

## 第三步：一次产生整个 action chunk

Policy forward pass 输出

\[
\hat A_t=
(\hat a_t^{(t)},\hat a_{t+1}^{(t)},\ldots,
\hat a_{t+k-1}^{(t)}).
\]

如果只使用 chunk execution，可以顺序执行其中动作；但 ACT 的 temporal aggregation 模式会在每个 timestep 都重新 query policy。

## 第四步：把当前 chunk 写入历史预测表

可以把所有 chunks 组织成二维表：

```text
query time ↓       action time →

0   a0 a1 a2 a3 a4 ...
1      a1 a2 a3 a4 ...
2         a2 a3 a4 ...
3            a3 a4 ...
```

每产生一个新 chunk，就把它写入对应一行。

## 第五步：取当前列做 Temporal Ensemble

执行 $t$ 时刻时，取所有历史 rows 中对 $t$ 的预测：

\[
\{\hat a_t^{(i)}\}_{i\le t}.
\]

再按 exponential weights 加权平均，得到最终 raw action。

官方代码中 temporal aggregation 打开后，`query_frequency` 被设为 1，因此每个 timestep 都获取最新 policy prediction。

## 第六步：反标准化并执行

预测通常处于 normalized action space。反标准化后得到真实 joint target：

\[
a_t^{real}
=
\hat a_t\odot \sigma_a+\mu_a.
\]

然后环境/机器人低层 controller 执行该 target，进入下一个 timestep。

## Inference 与 Training 的关键区别

| Training | Inference |
|---|---|
| 有 ground-truth future actions | 没有 ground-truth future actions |
| 运行 latent encoder | latent encoder 不运行 |
| sample $z$ from approximate posterior | 固定 $z=0$ |
| 计算 L1 + KL loss | 不计算训练 loss |
| 目标是更新参数 | 目标是输出并执行动作 |
| 不需要 temporal rollout aggregation 来定义 loss | deployment 可以用 Temporal Ensemble |

把这两张图混在一起，是理解 ACT 时最常见的错误来源之一。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official evaluation loop: https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py
- Official model: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
