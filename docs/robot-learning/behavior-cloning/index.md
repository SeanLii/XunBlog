---
title: "Behavior Cloning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/behavior-cloning/"
prerequisites:
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/act/"
  - "/robot-learning/act/action-chunking/"
---

# Behavior Cloning

Behavior Cloning（BC）是最直接的 imitation learning 方法：把专家 demonstration 中的“当前观测 → 专家动作”当成监督学习数据，训练一个 policy 去复制专家。

一条 demonstration trajectory 可以写成

\[
(o_1,a_1),(o_2,a_2),\ldots,(o_T,a_T).
\]

Behavior Cloning 把其中每一对当作训练样本：

```text
observation o_t  ──→ policy πθ ──→ predicted action a_hat_t
                                      │
expert action a_t ────────────────────┘
                       supervised loss
```

如果 action 是连续向量，常见目标可以是 L1、L2 或 negative log-likelihood；如果 action 是离散类别，则可以使用 classification loss。损失函数的具体形式不是 BC 的定义，核心在于**直接监督 policy 模仿 demonstration actions**。

## 训练数据来自专家走过的状态

设专家 policy 为 $\pi^*$。专家执行任务时会访问某些状态，并形成一个状态分布。可以把它记成

\[
d_{\pi^*}(s).
\]

Behavior Cloning 的训练数据主要来自这个分布：

\[
(s,a)\sim d_{\pi^*}(s)\pi^*(a\mid s).
\]

所以模型真正学到的是：

> 当我处在**专家通常会到达的状态**时，应该做什么动作。

这和普通图片分类有一个重要区别：部署时输入分布并不是固定的。

## Policy 一旦自己行动，后续输入也被自己改变

部署时，模型执行的动作会改变下一个状态：

```text
s_t
 │
 ↓
πθ
 │
 ↓
a_t
 │
 ↓
environment
 │
 ↓
s_{t+1}
```

因此如果某一步动作产生了小误差，机器人可能进入 demonstration 很少出现的状态。下一步 policy 仍然必须做决定，但此时它面对的输入已经偏离训练分布。

继续执行后，会形成 model policy 自己诱导的状态分布：

\[
d_{\pi_\theta}(s).
\]

而它未必等于训练时的

\[
d_{\pi^*}(s).
\]

这就是 imitation learning 中重要的 distribution shift。

## Compounding Error

单步预测误差很小时，整条 rollout 仍然可能失败，因为错误会改变未来输入，再带来更多错误。

Ross 等人的分析说明，在简单假设下，纯 supervised imitation 的 long-horizon cost 可以随着 horizon 出现比单步误差更严重的累积；这正是后来 DAgger 等方法关注的问题。

直观地看：

```text
expert states:
A → B → C → D → E

learned policy:
A → B' → C'' → D???
```

BC 并不是在 $B'$、$C''$ 上训练得很充分，因为专家 demonstration 可能几乎从不访问这些状态。

## DAgger 的核心改动

DAgger 不再只收集专家自己走过的数据。它让当前 policy 实际 rollout，在 policy 会访问的状态上再次询问 expert：

```text
current policy rollout
        │
        ↓
 states actually visited
        │
        ↓
 ask expert for correct action
        │
        ↓
 add to dataset and retrain
```

这样训练分布会逐渐覆盖 learned policy 自己会到达的状态。

DAgger 是解决 distribution shift 的一种方法，但它通常要求在 learner rollout 状态上还能获得 expert label，这在真实机器人上可能很贵。

## BC 与 ACT 的关系

ACT 仍然属于 demonstration-based supervised policy learning。它并没有从原则上消除 Behavior Cloning 的 distribution-shift 问题。

ACT 主要改变的是**动作预测单位和模型结构**：

\[
o_t\rightarrow a_t
\]

变成

\[
o_t\rightarrow a_{t:t+k-1}.
\]

也就是 [Action Chunking](/robot-learning/act/action-chunking/)。这可以缩短 policy 层面的 effective horizon，让模型一次表达一段相互关联的动作，但如果机器人进入 training demonstrations 没覆盖的状态，仍然可能失败。

所以可以把关系记成：

```text
Imitation Learning
      │
      └── Behavior Cloning
             │
             └── ACT: chunk-level behavior cloning + specific architecture
```

## Sources

- Ross, Gordon & Bagnell, **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**, AISTATS 2011. https://proceedings.mlr.press/v15/ross11a.html
- Zhao et al., **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, 2023. https://arxiv.org/abs/2304.13705
