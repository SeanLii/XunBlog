---
title: "Imitation Learning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/imitation-learning/"
prerequisites: []
related:
  - "/robot-learning/behavior-cloning/"
  - "/robot-learning/dagger/"
  - "/robot-learning/act/"
---

# Imitation Learning

Imitation Learning 是从 expert demonstrations 学习行为策略的一类方法。

与 reinforcement learning 主要依赖 reward signal 不同，imitation learning 的核心 supervision 来自“expert 在什么状态下采取了什么行为”。

一条 demonstration trajectory 可以写成：

\[
\tau=
(o_1,a_1,o_2,a_2,\ldots,o_T,a_T),
\]

其中：

- $o_t$：agent 在时刻 $t$ 能观察到的信息；
- $a_t$：expert 执行的 action。

目标是学习 policy：

\[
\pi(a\mid o)
\]

或 history-conditioned policy，使 learned agent 在环境中产生接近 expert 的行为。

## “模仿”真正意味着什么

表面上看，demonstration dataset 和 supervised learning dataset 很像：

\[
(o_t,a_t).
\]

但 sequential decision-making 有一个关键不同：

> **agent 的 action 会改变未来自己看到的 observations。**

所以 prediction error 不是只影响当前 sample；错误 action 会把系统带到新的 state，改变后续 input distribution。

这也是 imitation learning 不能完全当成普通 i.i.d. supervised learning 的原因。

## Expert Demonstration

Demonstrations 可以来自：

- 人类直接控制机器人；
- teleoperation；
- scripted / planning expert；
- another policy；
- simulation expert；
- offline logged trajectories。

数据质量不仅取决于数量，还取决于：

- expert 是否稳定；
- state coverage 是否足够；
- 是否包含 recovery behavior；
- observation / action synchronization；
- action representation 是否适合学习。

因此 imitation learning 的数据问题本身就是算法表现的一部分。

## Behavior Cloning

最直接的方法是 [Behavior Cloning](/robot-learning/behavior-cloning/)：把 demonstration pairs 当 supervised examples，训练：

\[
\hat a_t=\pi_\theta(o_t)
\]

去匹配 expert action：

\[
a_t^*.
\]

它简单、可扩展，并且是大量现代 robot policies 的训练基础。

但 Behavior Cloning 只在 expert visited states 上直接得到监督。

## Distribution Shift

训练数据来自 expert policy induced distribution：

\[
d_{\pi^*}(o).
\]

部署时 observations 来自 learned policy：

\[
d_{\pi_\theta}(o).
\]

一旦 learned policy 发生小错误：

```text
expert trajectory
───────────────→

learned policy
──────↘
       new state
```

它可能进入 training data 很少出现的状态。

此时 prediction 变差，又产生更大 deviation，形成 compounding error。

这不是 BC 的“代码 bug”，而是 sequential imitation 的基本 distribution mismatch。

## Interactive Imitation Learning

一种解决方向是让 learner 在自己会遇到的 states 上获得 expert labels。

[DAgger](/robot-learning/dagger/) 的典型流程：

```text
current policy rollout
        ↓
visit learner-induced states
        ↓
expert labels those states
        ↓
aggregate into dataset
        ↓
retrain policy
```

这样 training distribution 会逐渐覆盖 learned policy 自己产生的 states。

## Offline 与 Interactive 两种数据条件

现实 robotics 中，expert query 可能昂贵甚至不可用。

因此可以粗分：

- offline imitation：只使用已经收集好的 demonstrations；
- interactive imitation：训练中还能让 expert 对新 states 提供反馈。

算法选择会受到这个数据条件直接限制。

## Multimodality

同一个 observation 可能存在多种都合理的 expert actions。

例如绕障碍物可以从左边，也可以从右边。

如果用简单 unimodal regression，模型可能预测两种动作的平均，而平均动作反而不可行。

因此 modern imitation policies 会使用：

- latent-variable models；
- mixture distributions；
- diffusion / flow generative policies；
- chunked sequence prediction。

这些是在解决 output distribution structure，不只是提高 network size。

## Partial Observability

如果 observation $o_t$ 没有包含决策所需全部 state，单步 mapping：

\[
a_t=\pi(o_t)
\]

可能本身就是 ambiguous。

可以加入：

- observation history；
- recurrent state；
- temporal context；
- multiple camera views；
- proprioception。

所以“更多 demonstration”无法自动修复 information 本身缺失的问题。

## Policy 表示形式

Imitation learning 并不规定 policy 一定输出单个 action。

可以输出：

- one-step action；
- probability distribution over actions；
- action sequence / chunk；
- latent plan；
- tokenized actions；
- continuous generative action trajectory。

ACT 的 action chunking、π0 的 flow-based action generation 都属于 imitation learning policy design 的不同选择。

## 现代 Robot Policy 中的位置

现代 robot imitation policies 会在同一套基本问题上做不同选择：如何表示 action、怎样处理 multimodality、是否预测 action chunk、是否使用 generative objective，以及如何在 closed-loop rollout 中获得恢复能力。

ACT、π0 等模型只是这些设计空间中的具体实例；demonstration supervision、distribution shift、multimodality 与 closed-loop rollout 本身属于 imitation learning 的一般问题。

## Sources

- Pomerleau. *ALVINN: An Autonomous Land Vehicle in a Neural Network*. 1989.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011.
