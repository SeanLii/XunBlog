---
title: "Behavior Cloning"
kind: "canonical"
domain: "Robot Learning"
parent: "Robot Learning"
canonical: "/robot-learning/behavior-cloning/"
prerequisites:
  - "/robot-learning/imitation-learning/"
related:
  - "/robot-learning/dagger/"
  - "/robot-learning/act/"
---

# Behavior Cloning

Behavior Cloning（BC）把 imitation learning 直接转成 supervised learning：给定 expert demonstrations，训练 policy 在 expert observations 上预测 expert actions。

Dataset：

\[
\mathcal D=
\{(o_i,a_i^*)\}_{i=1}^{N}.
\]

Policy：

\[
\pi_\theta(a\mid o).
\]

训练目标通常可以写成 negative log-likelihood：

\[
\theta^*
=
\arg\min_\theta
\mathbb E_{(o,a^*)\sim\mathcal D}
[-\log\pi_\theta(a^*\mid o)].
\]

如果 policy 是 deterministic Gaussian mean / regression head，这可能进一步表现为 MSE 或 L1 等 supervised losses。

## Training Mental Model

```text
expert observation o_t
        ↓
     policy πθ
        ↓
 predicted action a_hat_t
        ↓
compare with expert action a*_t
        ↓
 supervised loss
```

训练阶段不需要 policy 自己真正执行 action 才能计算每个 demonstration sample 的 loss。

这也是 BC 易于扩展的原因：一旦 demonstrations 收集好，可以完全 offline training。

## Closed-Loop Deployment 改变了问题

训练时 sample：

\[
o_t\sim d_{\pi^*}
\]

来自 expert trajectories。

部署时：

\[
o_t\sim d_{\pi_\theta}
\]

来自 learned policy 自己的 rollout。

而 environment transition：

\[
o_{t+1}\sim P(\cdot\mid o_t,a_t)
\]

意味着当前 action 会改变未来 input。

所以即使 supervised validation error 很低，也不保证 closed-loop rollout 一定稳定。

## Covariate Shift

如果 learned policy 发生一个小动作误差，robot state 可能偏离 expert trajectory。

新 observation：

\[
\tilde o_t
\]

可能在 training data 中很少出现。

policy 在这种 out-of-distribution state 上更容易犯错，于是进一步偏离。

这个 mechanism 就是 BC 最经典的 failure mode。

## Compounding Error

设每一步在 expert distribution 上发生 error 的概率约 $\epsilon$。

如果 sequence horizon 为 $T$，naive supervised intuition 可能认为总损失只线性增长。

但在 sequential rollout 中，一个 early error 会改变后续 states，使后面多个 timesteps 都进入 unfamiliar region。

经典分析显示，在 worst-case assumptions 下 vanilla imitation / BC 的 expected cost gap 可出现 $O(T^2\epsilon)$ 级别增长，而 interactive methods 可以改善这种 horizon dependence。

重要的是理解 mechanism，而不是死记一个 bound：

> **错误会改变未来输入，所以错误影响可以沿时间传播。**

## Deterministic Regression 与 Multimodality

假设同一 observation 下 expert dataset 有两种 actions：

\[
a_A,
\qquad a_B.
\]

如果用 MSE 回归单一 mean：

\[
\hat a\approx\frac{a_A+a_B}{2}.
\]

这个平均 action 可能并不是 expert 真正执行过的合理模式。

因此 BC 并不等于“必须用 MSE 预测一个动作”。BC 只规定 supervision 来自 expert behavior；policy distribution 可以很丰富。

## Sequence Prediction 仍然可以是 Behavior Cloning

如果 policy 一次输出 action chunk：

\[
\hat A_t=
(\hat a_t,\ldots,\hat a_{t+k-1}),
\]

然后用 expert future chunk：

\[
A_t^*
\]

做 supervised imitation，它仍然属于 Behavior Cloning。

ACT 就是这个思路的重要例子。

因此 Action Chunking 没有把 ACT 变成“不是 BC”；它改变的是 policy output granularity 和 architecture。

## Dataset Coverage

BC 的能力上限很大程度取决于 dataset：

```text
what states are demonstrated
+
what actions are labeled there
+
how diverse expert behavior is
```

如果 recovery states 从未出现，policy 很难凭 supervised objective 自动知道如何恢复。

这推动了 [DAgger](/robot-learning/dagger/) 等 interactive data aggregation methods。

## BC 的优势

即使有 distribution shift，BC 仍然非常重要，因为它：

- objective 简单；
- 可以纯 offline；
- 不需要 reward engineering；
- 可以利用大规模 teleoperation data；
- 能与大型 neural architectures 和 generative policies 结合。

现代 robot learning 中很多强模型仍以 BC / maximum-likelihood-style imitation 为训练基础，只是在 architecture、action representation 与 data scale 上更复杂。

## Sources

- Pomerleau. *ALVINN*. 1989.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. 2011.（distribution shift / compounding-error 理论背景）
