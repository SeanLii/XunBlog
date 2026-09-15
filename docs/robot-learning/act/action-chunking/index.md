---
title: "Action Chunking"
kind: "canonical"
domain: "Robot Learning / ACT"
parent: "ACT"
canonical: "/robot-learning/act/action-chunking/"
prerequisites:
  - "/robot-learning/behavior-cloning/"
related:
  - "/robot-learning/act/temporal-ensemble/"
  - "/robot-learning/act/inference/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Action Chunking

Action Chunking 是把 policy 的基本预测单位从“一个动作”改成“未来一段连续动作”。

单步 policy 的输出是

\[
\hat a_t=\pi_\theta(o_t),
\]

而 action-chunk policy 输出

\[
(\hat a_t,\hat a_{t+1},\ldots,\hat a_{t+k-1})
=\pi_\theta(o_t).
\]

其中 $k$ 是 chunk size。

ACT 借用了更早的 “action chunking” 名称，并把它具体化为机器人 imitation learning 中“根据当前 observation 一次预测未来 $k$ 步动作”的设计；本文只讨论这一 ACT formulation。

## Prediction Unit

假设一个机器人任务总共需要 400 个 control steps。

单步预测时：

```text
observe → predict a1 → execute
observe → predict a2 → execute
observe → predict a3 → execute
...
```

policy 必须把 400 个局部决定正确衔接起来。

如果 chunk size 为 100，一次预测可以直接表示 100 个未来 joint targets：

```text
observe
   │
   ↓
policy
   │
   ↓
[a_t, a_t+1, ..., a_t+99]
```

这不是把 100 个动作简单复制出来，而是让网络把这 100 个位置作为一个联合输出结构来预测。

## Action Chunk 表达局部时间结构

机器人动作在时间上通常高度相关。机械臂正在向杯子移动时，下一毫秒的动作和后面几十毫秒的动作并不是彼此独立的随机决定，而属于同一段运动。

Action chunk 让模型可以直接表示这种局部时间结构：

```text
接近物体 → 合拢夹爪 → 抬起
```

这些动作可以在同一次 forward pass 中被共同预测。

因此 chunking 改变的不只是“输出 tensor 变长了”，也改变了 policy 对时间结构的建模方式。

## Effective horizon

如果 policy 每次只决定一步，那么长度为 $T$ 的任务在 policy 层面大约需要 $T$ 次顺序决策。

如果每个 chunk 覆盖 $k$ 步，一个粗略的 open-loop 直觉是决策段数缩短到大约

\[
\frac{T}{k}.
\]

ACT 论文把这称为降低 task 的 effective horizon。

但这不意味着实际执行时一定每 $k$ 步才观察一次。ACT 还可以每个 timestep 重新预测一个新 chunk，再通过 [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) 融合重叠结果。

所以要区分两件事：

- **预测结构**：一次输出 $k$ 步；
- **policy query frequency**：多久重新根据新 observation 调一次模型。

ACT 使用 temporal aggregation 时可以让第二项保持为每一步。

## 训练样本怎样构造

假设 demonstration 是

\[
(a_1,a_2,\ldots,a_T).
\]

以时刻 $t$ 为训练起点，就取未来的一段：

\[
a_{t:t+k-1}.
\]

输入是当前 observation $o_t$，target 是整个 future chunk：

```text
o_t  ──→ policy ──→ [a_hat_t ... a_hat_t+k-1]
                         │
                         ↓
                  compare with
                  [a_t ... a_t+k-1]
```

接近 episode 末尾时，如果剩余动作不足 $k$，实现需要 padding，并用 mask 避免 padded positions 参与 reconstruction loss。

## Chunk size 的含义

更大的 $k$ 能让单次预测覆盖更长动作结构，但也意味着模型要更远地预测未来；越远的动作通常越依赖之后才会看到的新信息。

更小的 $k$ 则更接近 single-step behavior cloning，局部预测容易一些，但对长时间动作结构的直接建模变弱。

因此 chunk size 是时间尺度选择，不是越大越好。

ACT 原论文使用 $k=100$ 作为其主要配置，但这属于具体实验设置，不是 Action Chunking 的定义。

## 与 Temporal Ensemble 的关系

若只在 $t=0,k,2k,\ldots$ 查询 policy，则形成 strict open-loop chunk execution：一个 chunk 执行完后才使用新的 observation。

ACT 更重要的做法是：每一步都重新预测。

```text
query at t=0:  [0, 1, 2, 3, ...]
query at t=1:     [1, 2, 3, 4, ...]
query at t=2:        [2, 3, 4, 5, ...]
```

这样同一时刻有多个候选动作，[Temporal Ensemble](/robot-learning/act/temporal-ensemble/) 决定怎样把它们合成最终执行动作。

## Sources

- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
- Official ACT implementation. https://github.com/tonyzhaozh/act
