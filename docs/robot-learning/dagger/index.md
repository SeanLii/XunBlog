---
title: "DAgger"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/dagger/"
prerequisites:
  - "/robot-learning/behavior-cloning/"
related:
  - "/robot-learning/imitation-learning/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# DAgger

DAgger（Dataset Aggregation）是 Ross、Gordon 与 Bagnell 提出的 interactive imitation-learning algorithm。它的核心目标是让 training data 覆盖 learner 自己在部署时会访问的 states。

[Behavior Cloning](/robot-learning/behavior-cloning/) 主要在 expert distribution：

\[
d_{\pi^*}
\]

上训练，而 learned policy 实际运行时产生：

\[
d_{\pi_\theta}.
\]

DAgger 通过反复执行 learner、在 learner-induced states 上查询 expert，并把新 supervision 聚合回 dataset 来缩小这一 distribution mismatch。

## Problem Setting

设 expert policy 为：

\[
\pi^*,
\]

当前 learner 为：

\[
\hat\pi_i.
\]

若只使用初始 expert demonstrations：

\[
\mathcal D_0,
\]

learner 可能在 rollout 中进入 $\mathcal D_0$ 很少覆盖的 state。DAgger 直接把这些 states 变成新的 supervised training examples。

## Dataset Aggregation

第 $i$ 轮的基本流程为：

1. 在当前 aggregated dataset $\mathcal D_{i-1}$ 上训练 learner $\hat\pi_i$；
2. 使用 learner 或 expert/learner mixture policy 在环境中 rollout；
3. 记录 rollout 访问的 states $s$；
4. 对这些 states 查询 expert action $\pi^*(s)$；
5. 将新 pairs 加入 dataset：

\[
\mathcal D_i
=
\mathcal D_{i-1}
\cup
\{(s,\pi^*(s))\}.
\]

经过多轮后，dataset 不再只描述 expert trajectory，而开始覆盖 learner 的实际 state distribution。

## Rollout Policy

原始 DAgger 可以使用 mixture policy：

\[
\pi_i
=
\beta_i\pi^*
+
(1-\beta_i)\hat\pi_i,
\]

其中 $\beta_i$ 随 iteration 下降。

早期 learner 较差时，expert contribution 可以减少危险或完全失控的 rollout；随后逐渐让 learner 决定更多行为。具体 schedule 可以不同，但 collected states 必须逐渐接近 learner deployment distribution。

## Expert Label 与 Executed Action

在某个 visited state $s$ 上，expert label 为：

\[
a^*=\pi^*(s).
\]

这与环境中实际执行的 action 是两个概念。learner 可以负责执行并进入新的 states，同时 expert 只负责提供这些 states 上的监督标签。

因此 DAgger 的核心资源要求是能够对 learner 访问的 state 查询 expert。

## No-Regret Interpretation

DAgger 把 imitation learning 与 online learning / no-regret learning 联系起来。每轮 learner 面对由当前 policy 引出的 state distribution，并在这些 states 上产生 supervised loss。

当 base learner 具有相应 no-regret property 时，aggregated training process 可以得到比固定 expert dataset 上 supervised imitation 更好的 sequential performance guarantee。

经典分析的重要结果之一是：在相应 assumptions 下，DAgger 可以把 vanilla supervised imitation 中不利的 horizon dependence 从 roughly quadratic behavior 改善到 roughly linear behavior。

理论 bound 的意义在于揭示 distribution mismatch，而不是为所有实际任务提供固定数值预测。

## Recovery States

DAgger 的 dataset 往往自然包含 expert trajectory 之外的 states。learner 偏离目标路径后产生的状态可以获得 expert recovery label，这类 supervision 对 closed-loop robustness 很重要，因为普通 expert demonstrations 通常集中在成功轨迹附近。

## Relation to Behavior Cloning

DAgger 并没有规定最终 policy 必须如何表示。每轮重新训练时仍可以使用普通 Behavior Cloning objective：

\[
\mathcal L
=
\mathbb E_{(s,a^*)\sim\mathcal D_i}
[\ell(\pi_\theta(s),a^*)].
\]

它改变的是 training data distribution。

因此 DAgger 可以概括为：

\[
\text{Behavior Cloning}
+
\text{learner-state collection}
+
\text{expert relabeling}
+
\text{dataset aggregation}.
\]

## Practical Constraints

DAgger 的主要成本来自 online interaction：

- expert 必须在训练过程中持续可用；
- human labeling 可能昂贵；
- real robot rollout 有 safety risk；
- early learner 可能访问非常差的 states；
- expert 对极端 off-distribution state 也可能难以给出稳定 action；
- 多轮 deploy–label–retrain pipeline 成本高。

这些限制解释了为什么 offline robotics 常通过更广泛 demonstrations、recovery data、perturbation collection 或 simulation coverage 来改善 coverage，而不是严格执行 DAgger。

## Definition Boundary

一次性增加 recovery demonstrations、random perturbation augmentation、固定 offline dataset 上 oversampling rare states 或 domain randomization 都可能改善 coverage，但它们不自动构成 DAgger。

DAgger 的定义性结构是：

\[
\text{learner-induced states}
\rightarrow
\text{expert query}
\rightarrow
\text{dataset aggregation}.
\]

## Sources

- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011.
