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
---

# DAgger

DAgger（Dataset Aggregation）是 Ross、Gordon 与 Bagnell 在 2011 年提出的 interactive imitation learning algorithm。

它针对 Behavior Cloning 的核心问题：训练数据来自 expert states，但部署时 policy 会访问自己诱导出的 states。

DAgger 的关键改变是：

> **让当前 policy 自己 rollout，然后请 expert 给这些 learner-visited states 标注正确 action，并把新数据不断加入 training set。**

## Behavior Cloning 的 Coverage Problem

普通 BC dataset：

\[
\mathcal D_0
\sim d_{\pi^*}.
\]

learner 训练后 deployment distribution：

\[
d_{\pi_1}.
\]

如果 $d_{\pi_1}$ 包含 expert dataset 很少覆盖的 states，policy 的 supervised accuracy 就没有充分保证。

DAgger 不试图只在固定 expert dataset 上把 model 做得更复杂，而是改变**数据收集 distribution**。

## Algorithm Data Flow

初始化 expert demonstration dataset：

\[
\mathcal D\leftarrow\mathcal D_0.
\]

每一轮 $i$：

```text
1. 在聚合数据集 D 上训练 policy π_i

2. 用 π_i 在环境中 rollout

3. 记录 learner 实际访问的 states o

4. 对这些 states 查询 expert action π*(o)

5. 把 (o, π*(o)) 加入 D

6. 继续下一轮
```

最终 dataset 逐渐包含：

\[
d_{\pi_1},
 d_{\pi_2},
 \ldots
\]

下 learner 真正会访问的 states。

## Expert Query 与 Behavior Execution 是两件事

在某个 state $o_t$，expert 可以提供：

\[
a_t^*=\pi^*(o_t)
\]

作为 label，但实际环境中执行的 action 不一定必须完全来自 expert。

原始 DAgger framework 可以使用 expert / learner mixture policy 进行 rollout，以控制早期 learner 太差带来的风险。

随着训练进行，mixing coefficient 可以逐渐让 learner 承担更多控制。

## Dataset Aggregation

名字中的“Aggregation”很重要。

DAgger 不是每轮只用最新 rollout data，而是持续聚合：

\[
\mathcal D_i
=
\mathcal D_{i-1}
\cup
\{(o,\pi^*(o))
:\ o\sim d_{\pi_i}\}.
\]

这让 training data 同时覆盖早期 expert-like states 和 later learner-induced states。

## 理论动机

Ross 等把 imitation learning 与 online learning / no-regret learning 联系起来。

Vanilla supervised imitation 在 sequential setting 中可能出现 roughly quadratic horizon dependence：

\[
O(T^2\epsilon).
\]

DAgger 通过在 learner-induced state distribution 上训练，可以在相应 assumptions 下把 performance degradation 改善到 roughly linear horizon dependence：

\[
O(T\epsilon).
\]

具体 bound 有前提，但结论背后的 intuition 非常清楚：

> 如果训练时已经见过自己会犯错后到达的 states，就不必在 rollout 中一直靠 extrapolation 生存。

## Recovery Behavior

DAgger 收集的数据天然可能包含“偏离 expert trajectory 后怎么回来”。

例如：

```text
normal expert path
────────────→

learner drifts away
        ↘
         state not in original BC data
         ↓
expert provides recovery action
```

这类 labels 对 closed-loop robustness 很有价值。

## Practical Cost

DAgger 的缺点也来自 interactive nature：

- 需要训练中持续访问 expert；
- human expert labeling 可能昂贵；
- learner early rollout 可能不安全；
- real robot online iteration 成本高；
- expert 对 off-distribution states 也未必容易提供稳定 action。

所以大规模 robotics dataset 时代，纯 DAgger 并不总是最实际方案。

## DAgger 与 Offline Data Expansion

如果不能在线 query expert，可以通过：

- recovery demonstrations；
- perturbation data；
- broader teleoperation coverage；
- simulation augmentation；

尝试获得类似 coverage improvement。

但这些不是严格意义上的 DAgger，因为 DAgger 的核心是 policy-induced states 上的 iterative expert relabeling / aggregation。

## Sources

- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, 2011.
