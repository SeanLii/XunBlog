---
title: "Behavior Cloning：为什么训练集上动作预测很准，机器人 Rollout 还是会崩？"
description: "从 Behavior Cloning 的监督学习目标出发，严格解释 imitation learning 中的 policy-induced distribution shift、covariate shift、compounding error、T²ε 误差累积、DAgger 的 dataset aggregation 思想，并重新审视 ACT Action Chunking 究竟缓解了什么、又没有从根本上解决什么。"
status: reviewed
pageType: concept
canonical: /robot-learning/imitation-learning/behavior-cloning-distribution-shift
updated: "2026-09-15"
---

# Behavior Cloning：为什么训练集上动作预测很准，机器人 Rollout 还是会崩？

假设你收集了很多专家 demonstration：

```text
observation → expert action
observation → expert action
observation → expert action
...
```

然后训练一个神经网络：

\[
\pi_\theta(a_t\mid s_t)
\]

让它在当前状态：

\[
s_t
\]

预测专家动作：

\[
a_t^*
\]

训练结果非常漂亮：

```text
Training Loss 很低
Validation Loss 很低
```

甚至可能：

> 单步动作预测误差非常小。

于是你把 policy 放到真实机器人上执行。

最开始：

```text
第1步：差一点点
第2步：又差一点点
第3步：偏得更多
第4步：机械臂已经进入 demonstration 中几乎没见过的位置
第5步：模型彻底不知道该怎么办
```

最后：

> rollout 失败。

这就是 imitation learning 中最重要的问题之一：

\[
\boxed{
\text{Compounding Error}
}
\]

但如果只把它理解为：

> “很多小误差加起来变成大误差”

还远远不够。

真正的问题不是简单的数值加法：

\[
0.01+0.01+0.01+\cdots
\]

而是：

\[
\boxed{
\text{你的动作会改变下一时刻你看到的数据。}
}
\]

也就是说：

> **Policy 不只是对数据做预测，它还在制造自己未来的数据分布。**

这就是普通 supervised learning 和 sequential control 最关键的区别。

这一篇要彻底讲清：

1. Behavior Cloning 到底是什么？
2. 为什么它看起来只是普通 supervised learning？
3. 为什么训练时数据分布和 rollout 时数据分布不同？
4. \(d_{\pi^*}\) 和 \(d_\pi\) 到底是什么？
5. 什么叫 policy-induced distribution？
6. 为什么这个问题常被叫 covariate shift？
7. 它和普通 supervised-learning covariate shift 有什么区别？
8. 为什么一个单步 error rate \(\epsilon\) 可能变成 \(O(T^2\epsilon)\) 的 sequential cost？
9. 这个 \(T^2\epsilon\) 到底怎么来的？
10. 为什么 recovery behavior 特别重要？
11. 为什么 demonstration 里只有“正确轨迹”反而可能有问题？
12. DAgger 到底改变了什么？
13. 为什么 DAgger 不是“训练更久的 BC”？
14. 为什么 DAgger 要让 learner 自己跑？
15. 为什么还要让 expert 给 learner 访问到的状态打标签？
16. DAgger 和 offline BC 的根本差异是什么？
17. ACT 的 Action Chunking 为什么能缓解 error compounding？
18. “effective horizon 降低 \(k\) 倍”该怎样严谨理解？
19. Action Chunking 是否真正解决 distribution shift？
20. Temporal Ensemble 是否解决 distribution shift？
21. CVAE 是否解决 distribution shift？
22. 2026 年的新研究为什么认为“horizon reduction”不是 Action Chunking 的完整解释？
23. ACT 为什么即使不用 DAgger 也能在很多任务上工作得很好？
24. 什么时候你应该怀疑 policy 的问题不是 network capacity，而是 state-distribution mismatch？

---

# 1. Behavior Cloning 是什么？

Behavior Cloning，简称：

\[
\boxed{
BC
}
\]

是 imitation learning 最直接的方法之一。

假设有 expert policy：

\[
\pi^*
\]

它产生 demonstrations：

\[
\tau
=
(s_1,a_1^*,s_2,a_2^*,\ldots,s_T,a_T^*)
\]

其中：

\[
a_t^*
=
\pi^*(s_t)
\]

简化为 deterministic expert。

---

# 2. 收集成 Supervised Dataset

从 demonstrations 中拆出：

\[
\mathcal D
=
\{
(s_i,a_i^*)
\}_{i=1}^{N}
\]

然后训练：

\[
\pi_\theta
\]

最小化：

\[
\boxed{
\min_\theta
\mathbb E_{(s,a^*)\sim\mathcal D}
[
\ell(
\pi_\theta(s),
a^*
)
]
}
\]

---

# 3. 连续动作可以用 Regression Loss

比如机器人 joint action：

\[
a\in\mathbb R^{14}
\]

可以用：

\[
L_1
\]

或：

\[
MSE
\]

例如：

\[
\ell(
\pi_\theta(s),
a^*
)
=
\|
\pi_\theta(s)-a^*
\|_1
\]

---

# 4. 离散动作可以用 Classification

如果 action：

```text
left
right
forward
stop
```

则 policy 输出 logits，

用：

\[
CrossEntropy
\]

训练。

所以 BC 的模型训练形式：

> 看起来完全就是普通 supervised learning。

---

# 5. 为什么这很诱人？

因为不需要：

- reward design；
- environment gradient；
- reinforcement learning；
- exploration reward；
- value function。

只要：

> expert demonstrations。

然后：

```text
observation → target action
```

直接训练。

---

# 6. ACT 本质上仍属于 Behavior Cloning 范式

ACT 的 target不是单个：

\[
a_t
\]

而是 action chunk：

\[
a_{t:t+k-1}
\]

所以：

\[
\boxed{
\pi_\theta(
a_{t:t+k-1}
\mid
o_t
)
}
\]

依然是在 imitation dataset 上做：

> supervised action prediction。

---

# 7. ACT Training 没有 Environment Reward

训练时不会：

```text
预测动作
↓
真的让机器人执行
↓
看成功没成功
↓
奖励反传
```

而是：

```text
demonstration observation
↓
policy
↓
predicted action chunk
↓
compare with demonstration action chunk
↓
L1 + KL
```

所以 ACT 是：

\[
\boxed{
\text{offline supervised imitation learning}
}
\]

---

# 8. 那问题在哪里？

普通 supervised learning 有一个基本思维：

> Train 和 Test 输入来自相同或相近 distribution。

例如猫狗分类：

训练：

\[
x\sim p_{\text{train}}(x)
\]

测试：

\[
x\sim p_{\text{test}}(x)
\]

如果两者很接近：

> validation error可以很好预测test error。

---

# 9. 但 Robot Policy 有一个额外 Feedback Loop

当前 state：

\[
s_t
\]

输入 policy：

\[
a_t=\pi(s_t)
\]

environment dynamics：

\[
s_{t+1}
=
f(
s_t,
a_t,
\xi_t
)
\]

其中：

\[
\xi_t
\]

代表可能的环境随机性。

---

# 10. 所以 Action 会改变 Future State

\[
a_t
\]

不是一个“预测完就结束”的 label。

它会真正影响：

\[
s_{t+1}
\]

于是：

\[
s_{t+1}
\]

又成为 policy 下一次的 input。

形成：

\[
\boxed{
s_t
\rightarrow
\pi
\rightarrow
a_t
\rightarrow
environment
\rightarrow
s_{t+1}
\rightarrow
\pi
\rightarrow
\cdots
}
\]

---

# 11. 这就是 Sequential Decision Making 的闭环

Policy不只：

> 被动读取数据。

它还通过 action：

> 主动改变未来要读取的数据。

这就是核心。

---

# 12. Expert Demonstration 的 State Distribution

如果一直执行 expert：

\[
\pi^*
\]

在时间：

\[
t
\]

到达状态的分布记作：

\[
\boxed{
d_{\pi^*}^t
}
\]

Ross、Gordon、Bagnell 的 DAgger 论文就是这样定义的。

---

# 13. Learner 的 State Distribution

如果一直执行 learner：

\[
\pi
\]

则时间 \(t\) 的 state distribution：

\[
\boxed{
d_\pi^t
}
\]

---

# 14. 平均 Occupancy / State Distribution

论文定义：

\[
\boxed{
d_\pi
=
\frac1T
\sum_{t=1}^{T}
d_\pi^t
}
\]

可以理解成：

> 在一个长度 \(T\) 的 rollout 中随机挑一个 timestep，执行 policy \(\pi\) 时看到的 state distribution。

---

# 15. Behavior Cloning 真正在优化什么？

传统 supervised imitation：

\[
\boxed{
\hat\pi_{BC}
=
\arg\min_{\pi\in\Pi}
\mathbb E_{s\sim d_{\pi^*}}
[
\ell(s,\pi)
]
}
\]

也就是说：

> **让 policy 在 expert 会访问的 states 上表现好。**

---

# 16. 但 Deployment 真正关心什么？

部署 learner 时，

它访问的不是：

\[
d_{\pi^*}
\]

而是：

\[
\boxed{
d_\pi
}
\]

真正希望小的是：

\[
\boxed{
\mathbb E_{s\sim d_\pi}
[
\ell(s,\pi)
]
}
\]

---

# 17. 这两个 Objective 不是同一个东西

BC：

\[
\mathbb E_{s\sim d_{\pi^*}}
[
\ell
]
\]

Deployment：

\[
\mathbb E_{s\sim d_\pi}
[
\ell
]
\]

核心 mismatch：

\[
\boxed{
d_{\pi^*}
\neq
d_\pi
}
\]

---

# 18. 为什么会不一样？

因为：

\[
\pi
\neq
\pi^*
\]

哪怕只差一点点，

action差异也会通过 dynamics：

\[
s_{t+1}=f(s_t,a_t)
\]

逐步改变 future states。

---

# 19. 一个最直观的 Driving Example

Expert始终开在道路中央：

```text
        road center
──────────●──────────
```

demonstration dataset几乎都是：

> 居中的相机画面。

BC学到：

> 在“居中画面”下怎么打方向盘。

---

# 20. Learner 第一次稍微偏右

```text
────────────●────────
```

这个 state：

> expert demonstrations里出现得比较少。

Policy在这里：

> 误差可能更大。

---

# 21. 下一步更偏

```text
────────────────●────
```

现在 observation更 out-of-distribution。

Policy再犯错的概率：

> 继续增加。

最终：

> 驶出道路。

---

# 22. 关键不是 Error 简单相加

真正 feedback：

```text
小动作误差
↓
产生偏离 expert trajectory 的 state
↓
这个 state 在 training data 中更少
↓
policy 在这里更不准
↓
更大的动作误差
↓
进入更陌生的 state
↓
...
```

所以：

\[
\boxed{
\text{error changes future input distribution}
}
\]

---

# 23. 这就是 Compounding Error 的核心

不是：

\[
e_1+e_2+\cdots
\]

而是：

\[
\boxed{
e_t
\rightarrow
s_{t+1}\text{ shift}
\rightarrow
future\ prediction\ difficulty
\rightarrow
e_{t+1}
}
\]

---

# 24. 为什么叫 Covariate Shift？

在普通 supervised learning 中，

covariate shift通常指：

\[
p_{\text{train}}(x)
\neq
p_{\text{test}}(x)
\]

而：

\[
p(y\mid x)
\]

被假设保持相同。

---

# 25. 在 Imitation Learning 中

可以把：

\[
x
\]

对应：

\[
s
\]

或 observation：

\[
o
\]

训练 inputs主要来自：

\[
d_{\pi^*}
\]

deployment inputs来自：

\[
d_\pi
\]

所以：

\[
\boxed{
d_{\pi^*}(s)
\neq
d_\pi(s)
}
\]

常被称为：

> covariate shift / distribution shift。

---

# 26. 但它比普通 Covariate Shift 更特殊

普通 dataset shift：

> test distribution可能由外部世界决定。

Imitation learning：

\[
d_\pi
\]

本身取决于：

\[
\boxed{
\pi
}
\]

也就是说：

> **模型的预测行为制造了自己的 test distribution。**

---

# 27. 这是 Endogenous Distribution Shift

Policy：

\[
\pi
\]

改变，

state distribution：

\[
d_\pi
\]

也改变。

因此真正 objective：

\[
\mathbb E_{s\sim d_\pi}
[
\ell(s,\pi)
]
\]

里面的 distribution：

> 本身依赖优化变量 \(\pi\)。

---

# 28. 这就是为什么问题比普通 Supervised Learning 难

Ross et al. 直接指出：

> sequential prediction中，future observations取决于previous predictions/actions，因此违反普通 statistical learning常见的 i.i.d. assumptions。

更关键的是：

\[
\boxed{
\text{input distribution depends on the learned policy itself}
}
\]

---

# 29. 不要把“不 i.i.d.”理解成“神经网络不能训练”

当然可以训练。

问题是：

> 普通 supervised generalization guarantee不再直接等价于 rollout performance。

单步 validation loss：

> 不能完整描述closed-loop performance。

---

# 30. 一个非常重要的区别

### Open-Loop Prediction

输入 sequence由外部固定：

```text
x1
x2
x3
...
```

你的 prediction错了：

> 不改变下一个 \(x\)。

---

### Closed-Loop Control

\[
a_t
\]

错了：

> 改变 \(s_{t+1}\)。

所以：

\[
\boxed{
\text{prediction error feeds back into future inputs}
}
\]

---

# 31. 为什么 Offline Validation 可能很好看？

Validation trajectories通常仍来自：

\[
\pi^*
\]

也就是：

> expert demonstration distribution。

所以 validation测试：

\[
\pi
\]

在：

\[
d_{\pi^*}
\]

上的能力。

---

# 32. 但 Rollout 测试的是

\[
\pi
\]

在：

\[
d_\pi
\]

上的能力。

所以：

\[
\boxed{
\text{offline action prediction accuracy}
\neq
\text{closed-loop policy success}
}
\]

---

# 33. 这是 Robot Learning 最值得牢牢记住的一句话之一

一个 policy：

> 可以是非常好的 action predictor，

但不是非常好的 controller。

因为 controller必须：

> 对自己造成的 state deviations 有恢复能力。

---

# 34. “Recovery” 为什么这么关键？

Expert demonstrations通常展示：

> 怎样正确完成任务。

但 learner真正 deployment时还需要：

> **犯错以后怎样回到正确轨迹。**

如果 demonstration中：

> 几乎从来没有错误状态，

模型就没学过：

> recovery action。

---

# 35. 一个 Pick-and-Place Example

Expert每次：

```text
gripper
↓
准确对准杯子
↓
抓起
```

training data几乎都在：

> 杯子正中央附近。

---

# 36. Learner 偏了 2 cm

现在 gripper在：

> 杯子侧边。

这种 observation：

> expert demonstration中可能极少。

真正正确的 action可能应该：

> 横向修正。

但 learner只熟悉：

> “继续往下抓”。

于是：

> 撞到杯子。

---

# 37. 这不是 Model “不知道杯子是什么”

可能视觉 representation很好。

真正缺的是：

\[
\boxed{
\text{state-action supervision for recovery states}
}
\]

---

# 38. Demonstration Coverage 是 Behavior Cloning 的关键限制

Dataset覆盖：

\[
\mathcal S_{\text{demo}}
\]

Deployment可能进入：

\[
\mathcal S_{\pi}
\]

如果：

\[
\mathcal S_{\pi}
\not\subset
\mathcal S_{\text{demo}}
\]

那么：

> policy必须 extrapolate。

Neural network在 OOD states上的行为：

> 不受训练 loss充分约束。

---

# 39. 为什么 Fine Manipulation 特别严重？

例如：

- 插孔；
- 扎线；
- 扣盖；
- battery insertion。

这些任务：

> 对 millimeter-level errors敏感。

一个很小 deviation：

> 就可能让 contact geometry完全改变。

所以 state distribution shift：

> 来得非常快。

ACT 论文也明确强调：

> high-precision imitation learning中特别受到 compounding errors影响。

---

# 40. 现在进入 \(T^2\epsilon\)

Ross et al. 给出了经典 supervised imitation guarantee。

假设 policy在 expert state distribution：

\[
d_{\pi^*}
\]

上的 0-1 error：

\[
\boxed{
\epsilon
}
\]

那么传统 supervised imitation在最坏情况下可能满足：

\[
\boxed{
J(\pi)
\le
J(\pi^*)
+
T^2\epsilon
}
\]

---

# 41. 更重要的是：这个 Quadratic Bound 可以是 Tight 的

也就是说：

> 不只是 proof松了一点。

确实存在问题，

使得 extra cost：

\[
\Theta(T^2\epsilon)
\]

量级发生。

---

# 42. 这个 \(T^2\epsilon\) 为什么出现？

我们先做一个非常直观的 union-bound 推导。

注意：

> 这是理解 scaling 的简化直觉，不代替 Ross 等论文完整 theorem assumptions。

---

# 43. 假设每个 Expert-Like State 犯错概率 ε

如果前：

\[
t
\]

步都还待在 expert-like trajectory附近，

每一步犯错概率大约：

\[
\epsilon
\]

---

# 44. 到时间 t 以前至少犯过一次错的概率

用 Union Bound：

\[
P(
\text{error before }t
)
\le
t\epsilon
\]

当：

\[
t\epsilon
\]

较小时是很自然的近似尺度。

---

# 45. 一次错可能导致后面很多步进入坏状态

如果犯错后：

> 系统进入 expert demonstrations没有覆盖的区域，

那么后续每一步都可能产生额外 cost。

---

# 46. 第 t 步处于 Bad Regime 的概率可能随 t 增大

近似：

\[
O(t\epsilon)
\]

---

# 47. 把所有 T Steps 的 Extra Cost 加起来

\[
\sum_{t=1}^{T}
O(t\epsilon)
\]

\[
=
O\left(
\epsilon
\sum_{t=1}^{T}
t
\right)
\]

---

# 48. 而

\[
\sum_{t=1}^{T}t
=
\frac{T(T+1)}2
\]

因此：

\[
\boxed{
O(T^2\epsilon)
}
\]

---

# 49. 所以 Quadratic 的根源

一个 error有两种“时间放大”：

### 第一层

任务越长：

> 犯至少一次错的机会越多。

### 第二层

越早犯错：

> 可能污染越多 future steps。

两者叠加：

\[
T\times T
\]

得到：

\[
T^2
\]

---

# 50. 一个数字直觉

假设单步 expert-distribution error：

\[
\epsilon=0.01
\]

任务：

\[
T=100
\]

简单独立直觉下：

> 100 步中至少犯一次错误的概率已经很可观。

最坏 sequential cost scaling：

\[
T^2\epsilon
=
10000\times0.01
=
100
\]

当然真实 task cost有自己的bounded normalization，

这里主要是看：

> scaling 为什么会恶化。

---

# 51. “1% Error 很小”在 Sequential Control 里可能完全不小

Image classification：

> 1% error也许意味着99%图片正确。

Robot rollout：

> 只要其中一次 error把系统送进危险 state，

整个 episode可能失败。

---

# 52. Sequence Success 甚至会乘法下降

一个更简单但不同的 toy intuition：

如果每一步都必须独立正确，

单步成功率：

\[
1-\epsilon
\]

T步全部成功：

\[
(1-\epsilon)^T
\]

例如：

\[
0.99^{100}
\approx0.366
\]

也就是说：

> 99%单步正确，不代表100步任务有99%成功率。

---

# 53. 但这个 \((1-\epsilon)^T\) 不是 DAgger 的 \(T^2\epsilon\) Theorem

二者是不同 toy models。

前者：

> 简单独立 step-success直觉。

后者：

> sequential distribution shift下的 worst-case cost guarantee。

不要混淆。

---

# 54. 为什么 Error 会越来越严重，而不是每一步都仍然 ε？

因为：

\[
\epsilon
\]

只是在：

\[
d_{\pi^*}
\]

上测的。

一旦进入：

\[
d_\pi
\]

的新区域，

error rate：

> 完全可能远大于 \(\epsilon\)。

所以 worst-case可以快速恶化。

---

# 55. 这就是 Training Accuracy 最大的陷阱

你测到：

\[
\mathbb E_{s\sim d_{\pi^*}}
[
\ell
]
=
\epsilon
\]

但 deployment真正需要：

\[
\mathbb E_{s\sim d_\pi}
[
\ell
]
\]

没有理由自动相等。

---

# 56. 如果 Learner 永远不犯错呢？

若：

\[
\pi=\pi^*
\]

则：

\[
d_\pi=d_{\pi^*}
\]

distribution shift自然不存在。

但现实中：

- finite data；
- sensory noise；
- model approximation；
- stochastic environment；

使完美复制极难。

---

# 57. 如果 Environment 有很强 Self-Correction 呢？

Compounding可能没那么严重。

例如小误差后 dynamics自然：

> 回到稳定轨迹。

那么 deviation不会无限增长。

Ross et al. 的分析也通过：

> expert recovery / cost-to-go difference

说明，如果专家能快速恢复，单次错误的长期代价可以较小。

---

# 58. 所以 Compounding Error 不是所有系统都同样严重

取决于：

- dynamics stability；
- task precision；
- expert recovery；
- observation robustness；
- horizon；
- policy error。

---

# 59. 一个 Stable System

例如：

> 小车被轨道物理约束住。

即使 steering稍有误差：

> 轨道把它拉回中心。

BC可能很稳。

---

# 60. 一个 Unstable / Precision-Critical System

比如：

> 将插头对准狭窄插孔。

1 mm误差可能导致：

- 卡住；
- 接触改变；
- camera geometry改变。

这时：

> tiny policy error → large future shift。

---

# 61. 为什么 Recovery Demonstrations 能帮助？

如果数据中不仅有：

> expert perfect trajectory，

还包含：

```text
偏左 → 往右修正
偏右 → 往左修正
抓偏 → 重新对准
接触失败 → 撤回再试
```

那么 policy在 learner-like off-nominal states：

> 也有监督。

---

# 62. 这本质上是在扩大 Training State Distribution

从：

\[
d_{\pi^*}
\]

附近，

扩展到：

> learner可能访问的 states。

这已经接近 DAgger 的核心思想。

---

# 63. DAgger 是什么？

全名：

\[
\boxed{
Dataset\ Aggregation
}
\]

Ross、Gordon、Bagnell 提出：

> 不要只在 expert trajectory 上训练。

让：

\[
\boxed{
learner\ 自己执行
}
\]

然后在 learner实际访问的 states 上：

> 询问 expert “这里正确动作是什么？”

再把这些数据加入 dataset。

---

# 64. DAgger 最简单流程

初始化：

\[
\mathcal D=\emptyset
\]

先收一些 expert data。

训练：

\[
\hat\pi_1
\]

---

# 65. 然后让 Current Learner Rollout

执行：

\[
\hat\pi_i
\]

得到它真正会访问的：

\[
s\sim d_{\hat\pi_i}
\]

---

# 66. 但 Label 仍然问 Expert

对于这些 states：

\[
s
\]

获取：

\[
\boxed{
a^*
=
\pi^*(s)
}
\]

所以 learner可能已经：

> 偏离正常轨迹，

expert告诉它：

> 在这个偏离状态应该怎么做。

---

# 67. Aggregate

新数据：

\[
\mathcal D_i
=
\{
(s,\pi^*(s))
\}
\]

然后：

\[
\boxed{
\mathcal D
\leftarrow
\mathcal D
\cup
\mathcal D_i
}
\]

---

# 68. 再重新训练 Policy

\[
\hat\pi_{i+1}
=
Train(
\mathcal D
)
\]

不断迭代。

---

# 69. DAgger 的核心不是“更多数据”

这是非常重要的。

普通 BC也可以：

> 收更多 expert demonstrations。

但如果这些 demonstrations仍然都在：

\[
d_{\pi^*}
\]

上，

distribution mismatch可能仍存在。

---

# 70. DAgger 的关键是 Data Distribution 发生变化

它专门收：

\[
\boxed{
s\sim d_{\pi_i}
}
\]

也就是：

> learner 真正会访问的 state。

因此训练 distribution逐渐接近：

> deployment distribution。

---

# 71. DAgger 真正想优化的东西

我们理想上想要：

\[
\boxed{
\min_\pi
\mathbb E_{s\sim d_\pi}
[
\ell(s,\pi)
]
}
\]

但：

\[
d_\pi
\]

只有真正执行：

\[
\pi
\]

才能采样。

所以 DAgger迭代：

> policy → state distribution → expert labeling → new policy。

---

# 72. 这就是 On-Policy Data Collection 的意味

数据状态来自：

> 当前 learner / learner-expert mixture 的 rollout。

虽然 labels来自：

> expert。

---

# 73. DAgger Algorithm 中的 Expert Mixing

Ross et al. 还允许执行：

\[
\boxed{
\pi_i
=
\beta_i\pi^*
+
(1-\beta_i)\hat\pi_i
}
\]

意思是：

> 早期 learner很差时，可以让 expert部分接管。

---

# 74. 为什么需要 Expert Mixing？

一开始 policy可能：

> 很快进入完全无意义或危险 states。

适当 expert intervention：

> 让数据收集更安全、相关。

随着训练：

\[
\beta_i\rightarrow0
\]

让 learner逐渐主导。

---

# 75. DAgger 的重要 Guarantee

论文核心结论之一：

> 经过合适 dataset aggregation / no-regret learning，可以找到 policy，使其在**自己 induced state distribution**上的 surrogate loss很低。

这就是与传统 BC 的关键区别。

---

# 76. 传统 BC 优化

\[
\boxed{
d_{\pi^*}
}
\]

---

# 77. DAgger 试图适配

\[
\boxed{
d_{\pi}
}
\]

这就是最重要的对比。

---

# 78. 为什么 DAgger 可以把 Horizon Scaling 改善？

在合适 assumptions 下，

如果 policy在自己的 distribution上 error：

\[
\epsilon
\]

那么 cost gap可以具有：

\[
O(T\epsilon)
\]

或近线性 horizon dependence，

而不是传统 supervised approach 的 worst-case：

\[
O(T^2\epsilon)
\]

---

# 79. 直觉是什么？

如果 learner偏了：

> 它曾经在训练中访问过类似偏离 state。

Expert已经告诉它：

> 怎么修回来。

所以第一步 error：

> 不必导致之后所有 steps彻底 OOD。

---

# 80. DAgger 学的是“犯错以后怎么办”

这句话虽然直觉化，

但非常准确地抓住核心：

\[
\boxed{
\text{train on learner-induced states}
}
\]

---

# 81. 为什么 DAgger 在真实机器人很麻烦？

因为它要求：

1. learner真正执行；
2. expert能对 learner访问的 state提供正确 action；
3. 反复收集、重训。

---

# 82. Safety 问题

早期 learner可能：

- 撞桌子；
- 掉物体；
- 夹坏东西；
- 进入危险 joint configuration。

所以让 learner主动制造 mistakes：

> 在真实机器人上可能代价很大。

---

# 83. Human Burden

如果 expert是人类 teleoperator，

每轮都要：

> 在线给 learner states提供 corrections。

非常累。

---

# 84. DART 为什么会出现？

2017 的 DART 提出另一思路：

> 在 expert demonstrations中主动注入合适 noise，

让 expert自然展示：

> 如何从偏差状态恢复。

这样试图在 offline data collection阶段：

> 更接近 learner error distribution。

---

# 85. DART 的高层思想

不是：

> learner犯错 → expert修。

而是：

> demonstration时人为制造可控 error → expert修。

这样可以得到：

> recovery examples。

---

# 86. 所以 Robust Imitation 的一个核心主题

各种方法都在问：

\[
\boxed{
\text{怎么让训练分布覆盖部署时可能访问的状态？}
}
\]

方式可以不同：

- DAgger；
- noise injection；
- recovery demonstrations；
- environment randomization；
- more diverse demonstrations；
- action chunking；
- robust architectures。

---

# 87. 现在回到 ACT

ACT 论文明确把：

> compounding errors

列为 imitation learning 在 high-precision manipulation中的核心挑战。

然后提出：

\[
\boxed{
Action\ Chunking
}
\]

---

# 88. Vanilla One-Step BC

每一步：

\[
\pi_\theta(
a_t
\mid
s_t
)
\]

如果任务长度：

\[
T
\]

那么要进行：

\[
T
\]

次高层 action prediction / decision。

---

# 89. Naive Action Chunking

改成：

\[
\boxed{
\pi_\theta(
a_{t:t+k-1}
\mid
s_t
)
}
\]

每次 observation：

> 直接预测 \(k\) 个 actions。

然后执行整个 chunk。

---

# 90. 如果每 k 步 Query 一次

高层重新决策次数：

\[
\approx
\frac{T}{k}
\]

原始 ACT 的解释是：

\[
\boxed{
\text{effective horizon reduced by factor }k
}
\]

从而：

> error compounding机会减少。

---

# 91. 一个直观例子

任务：

\[
T=500
\]

控制 steps。

One-step BC：

\[
500
\]

次新 prediction。

如果：

\[
k=100
\]

naive chunk：

\[
5
\]

次 chunk-level prediction。

---

# 92. 这为什么可能帮助？

如果每次 policy重新做高层判断：

> 都有机会发生 prediction error。

Chunking让：

> 同一次 observation产生一段 coherent actions。

因此独立重新决策的次数：

> 大幅减少。

---

# 93. 但这里要非常小心

不能简单理解成：

\[
\boxed{
\text{100个动作只会有1个误差}
}
\]

因为整个 chunk里的每一个 action：

> 都仍然可能预测不准。

Action Chunking没有让动作误差消失。

---

# 94. 它改变的是 Temporal Prediction Structure

One-step：

\[
a_t
\leftarrow
s_t
\]

下一步：

\[
a_{t+1}
\leftarrow
s_{t+1}
\]

---

Chunk：

\[
[
a_t,
a_{t+1},
\ldots,
a_{t+k-1}
]
\leftarrow
s_t
\]

后面多个动作：

> 基于同一个 earlier state生成。

---

# 95. 这减少了某类 Feedback-Induced Decision Drift

在 chunk内部，

policy不会：

> 每个微小 deviation后立刻重新从一个可能已经偏移的新 state做新的 high-level prediction。

因此某些误差 feedback链：

> 被改变。

---

# 96. 但也带来 Open-Loop Tradeoff

如果整个 chunk全部执行：

> chunk内部看不到新的真实 observation。

如果环境突然变化：

> policy无法立即响应。

所以：

\[
\boxed{
\text{less frequent replanning}
\leftrightarrow
\text{less reactivity}
}
\]

---

# 97. 这就是为什么 ACT 又引入 Temporal Ensemble

Final ACT不是简单：

```text
observe
↓
predict 100
↓
全部执行100
↓
observe
```

而是：

> 每个 timestep重新 query policy，

得到 overlapping chunks。

---

# 98. 因此 Final ACT 实际仍然 Closed-Loop

每时刻：

\[
o_t
\]

都会产生新 chunk：

\[
\hat A_t
\]

然后只执行当前要用的 action prediction。

所以：

> 新 observation持续影响控制。

---

# 99. 那“effective horizon reduced by k”还严格成立吗？

这里必须分层说。

### 对 Naive Chunk Execution

非常直观：

> 每 \(k\) 步重新决策一次。

---

### 对 Final ACT + Temporal Ensemble

每 timestep仍重新 inference，

所以：

> “实际 controller只做 \(T/k\) 次 inference” 并不成立。

---

# 100. 原论文为什么仍说 Chunking Reduces Effective Horizon？

因为 training/policy output的基本预测单元：

> 从 single action变成 action sequence。

Action chunking改变了：

> temporal abstraction与误差结构。

但如果把它机械解释成：

> “Final ACT只决策 \(T/k\) 次”

是不准确的。

---

# 101. Temporal Ensemble 到底做什么？

同一个 execution time：

\[
t
\]

可能有来自过去多个 query times的预测：

\[
\hat a_t^{(t)}
\]

\[
\hat a_t^{(t-1)}
\]

\[
\hat a_t^{(t-2)}
\]

……

然后：

> exponentially weighted average。

---

# 102. Temporal Ensemble 最直接作用

原 ACT 主要强调：

> 提高 motion smoothness，并缓解 chunk boundaries带来的 jerky behavior。

它融合：

> 对同一 timestep 的多个 predictions。

---

# 103. Temporal Ensemble 是否等于 DAgger？

完全不是。

TE没有：

- 收 learner states；
- 问 expert correction；
- 更新 dataset。

所以：

\[
\boxed{
TE
\neq
distribution\ aggregation
}
\]

---

# 104. Temporal Ensemble 是否从根本上消除 State Distribution Shift？

也没有。

训练数据依然主要来自：

\[
d_{\pi^*}
\]

deployment依然来自：

\[
d_\pi
\]

TE只改变：

> action execution aggregation。

---

# 105. CVAE 又解决什么？

ACT 的 CVAE主要针对：

> human demonstration variability / multimodality / non-stationarity。

例如同一个 observation附近：

> 人可能采用不同细微动作轨迹。

---

# 106. CVAE 是否解决 Covariate Shift？

不是它的主要机制。

它不会自动收集：

> learner偏离后的 recovery states。

所以：

\[
\boxed{
CVAE
\neq
DAgger
}
\]

---

# 107. ACT 的三个核心部件解决不同问题

### Action Chunking

主要改变：

> temporal prediction structure、effective decision horizon / compounding behavior。

### CVAE

建模：

> demonstration variability。

### Temporal Ensemble

改善：

> overlapping chunk execution的平滑与稳定。

---

# 108. 但 ACT 仍是 Offline BC

它没有：

> learner rollout → expert relabel → aggregate dataset

这个 DAgger loop。

所以从经典 imitation-learning理论角度：

\[
\boxed{
\text{ACT does not eliminate policy-induced distribution shift at its source.}
}
\]

---

# 109. 那为什么 ACT 仍然可以非常强？

因为解决 robustness 不只有 DAgger 一条路。

ACT同时通过多个设计：

> 降低 distribution shift造成灾难的可能性。

---

# 110. 第一：Action Chunk Predicts Temporal Structure

不是每一步独立回归：

\[
a_t
\]

而是学习：

\[
a_{t:t+k-1}
\]

让模型捕获：

> local trajectory coherence。

---

# 111. 第二：Transformer Capacity

Observation-to-action mapping可以利用：

- 多视角视觉；
- proprioception；
- long structured output。

高容量模型降低：

> expert-distribution上的 base error。

如果：

\[
\epsilon
\]

本身更小，

compounding自然减轻。

---

# 112. 第三：大量 Demonstration Points

虽然 episode数量可能不大，

但每个 episode有：

> 很多 timesteps。

overlapping chunks产生大量 supervised training windows。

---

# 113. 第四：Temporal Ensemble

多个 temporal predictions：

> 对同一 action进行融合。

可以减少：

- jitter；
- occasional bad prediction；
- abrupt chunk discontinuity。

---

# 114. 第五：Low-Level PID

ACT输出：

> target joint positions。

底层 Dynamixel/PID负责：

> 高频跟踪。

这给系统增加了：

> low-level control stability。

Policy不需要直接学所有 motor dynamics。

---

# 115. 第六：任务本身可能具有一定 Recoverability

真实 manipulation虽然精细，

但有些阶段：

- free-space reaching；
- slow motion；
- quasi-static contact；

小误差未必立即不可恢复。

---

# 116. 第七：Human Demonstrations 本身可能存在一定 Variation

不同 demonstrations并不完全重合。

这会自然扩大：

> training state coverage。

相比单一完美轨迹：

> 更有 robustness。

---

# 117. 所以 Offline BC 并不是必然失败

Distribution shift是：

> 风险与理论限制。

不是：

> “任何 BC 都一定不能工作”。

很多实际任务中：

> 充分数据 + 合适 architecture +稳定 dynamics

可以让 BC表现很好。

---

# 118. 什么时候 BC 特别容易成功？

通常更有利的条件：

- horizon较短；
- dynamics稳定；
- demonstrations覆盖丰富；
- learner base error小；
- task对误差容忍；
- low-level controller强；
- perception可靠。

---

# 119. 什么时候 BC 特别容易崩？

- long horizon；
- unstable dynamics；
- narrow precision tolerance；
- contact-rich；
- rare recovery states；
- multimodal action labels；
- partial observability；
- demonstration coverage窄。

---

# 120. Partial Observability 会让问题更复杂

如果 policy只看到：

\[
o_t
\]

而不是完整 state：

\[
s_t
\]

那么同一个：

\[
o_t
\]

可能对应不同 hidden situations。

这不仅是 distribution shift，

还有：

> state aliasing / non-Markovianity。

---

# 121. ACT 的 Action Chunking 也帮助 Non-Markovian Demonstration Structure

原 ACT强调：

> human behavior可能 non-Markovian。

例如 pause：

当前 observation相似，

但 action取决于：

> 此前人的操作状态/意图。

预测一段 action：

> 可以捕捉 temporally correlated behavior。

---

# 122. 这和 Compounding Error 是两个不同理由

Action Chunking原论文给出至少两类直觉：

1. 减少 effective horizon / error compounding；
2. 捕捉 temporally correlated、non-Markovian human behavior。

不要把它们合并成一个理由。

---

# 123. 到这里需要加入 2026 年的新研究

2026-08-03，

Lazzati、Stachowicz、Chen、Metelli、Wagenmaker、Levine 提出：

**Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?**

专门研究：

> Action Chunking 为什么有效。

---

# 124. 这篇工作的一个重要结论

他们发现：

> 传统常见解释——temporal consistency、horizon reduction、representation learning——**单独都不足以完整解释** action chunking的性能优势。

这不是说：

> horizon reduction完全不存在。

而是：

\[
\boxed{
\text{“减少决策次数”不是全部机制解释。}
}
\]

---

# 125. 新研究强调 Non-Markovian Expressivity

Action Chunking同时学习：

\[
a_t\mid o_t
\]

以及由于 overlapping temporal relationships：

\[
a_t\mid o_{t-1}
\]

\[
a_t\mid o_{t-2}
\]

……

这让 policy可以利用：

> past observations与当前 action之间的关系。

---

# 126. 一个很反直觉的发现

他们发现某些 delayed policies：

\[
\boxed{
a_t
\sim
\pi(
o_{t-n}
)
}
\]

也就是：

> 每一步仍只预测一个 action，

但基于过去的 observation。

在很多 settings中：

> 可以捕获 Action Chunking一部分关键收益。

---

# 127. 为什么 Past Observation 反而可能帮助？

当前：

\[
o_t
\]

是经过 learner之前所有错误后产生的。

它可能已经：

> 偏离 demonstration distribution。

更早的：

\[
o_{t-n}
\]

可能是在：

> 误差还没有累积那么严重的时候。

---

# 128. 所以“基于 Past State 预测 Future Action”会改变 Compounding Structure

这比简单说：

> “decision horizon除以 k”

更细致。

Action Chunk中的后面 action：

\[
a_{t+i}
\]

正是基于更早 observation：

\[
o_t
\]

预测的。

---

# 129. 2026 论文还提出 Implicit Ensembling

Action Chunking学习多种 temporal relationship：

\[
a_t|o_t
\]

\[
a_t|o_{t-1}
\]

\[
a_t|o_{t-2}
\]

……

当部署时组合不同 chunk predictions，

它表现出类似：

> ensemble

的效果。

---

# 130. 这与 ACT Temporal Ensemble 尤其容易混淆

注意两个概念：

### ACT Temporal Ensemble

显式对：

> 同一 timestep 来自不同 chunks 的 actions加权平均。

---

### 2026 Paper 的 Implicit Ensembling Interpretation

更广泛地说：

> action-chunked model同时学习不同 temporal relationships，本身带来 ensemble-like generalization/robustness。

两者相关但：

> 不是完全同一个概念定义。

---

# 131. 所以我们应该怎样更新对 ACT 的理解？

不要删掉原论文说法。

应该分层：

### 原始 ACT 2023 的设计解释

\[
\boxed{
Action\ Chunking
\rightarrow
reduced\ effective\ horizon
\rightarrow
mitigate\ compounding\ errors
}
\]

---

### 2026 后续机制研究

\[
\boxed{
\text{这是真实直觉的一部分，但不足以完整解释性能。}
}
\]

还需要考虑：

- non-Markovian expressivity；
- past-observation conditioning；
- reduced compounding structure；
- implicit ensembling。

---

# 132. 这就是“Paper Fact”和“后来理解”必须分开

不能用2026结果倒过来说：

> ACT作者2023就证明了implicit ensembling。

没有。

也不能因为后续解释更精细就说：

> 原 ACT 的 effective-horizon claim 完全错。

更准确：

> 原始解释是有用的设计直觉，但后续研究表明它不是充分机制解释。

---

# 133. Action Chunking 是否解决 Distribution Shift？

答案：

\[
\boxed{
\text{缓解，但没有从数据分布层面彻底消除。}
}
\]

为什么？

训练依旧：

\[
s\sim d_{\pi^*}
\]

没有显式加入：

\[
s\sim d_\pi
\]

的 expert labels。

---

# 134. 它改变 Error Dynamics，而不是直接修 Training Distribution

这是非常精确的说法。

DAgger：

\[
\boxed{
\text{changes the training state distribution}
}
\]

Action Chunking：

\[
\boxed{
\text{changes temporal policy parameterization/execution and resulting error dynamics}
}
\]

---

# 135. 两者解决问题的层级不同

### DAgger

问：

> Learner会去哪里？在那里 expert会怎么做？

---

### Action Chunking

问：

> 怎样组织 action predictions，让 rollout 更 coherent、减少某些 error-compounding机制？

---

# 136. 所以二者理论上可以结合

完全可以：

> 用 DAgger收 learner-state corrections，

然后训练：

> action-chunking policy。

它们不是互斥思想。

---

# 137. Offline Robot Learning 为什么通常不直接用 DAgger？

实际原因：

- expert在线成本；
- safety；
- reset成本；
- hardware wear；
- human teleoperation burden。

所以现代 robotics大量研究：

> 如何仅凭 fixed offline demonstrations也获得鲁棒 policy。

---

# 138. ACT 正是这个方向的代表

不依赖：

> iterative expert intervention。

只需要：

> demonstration dataset。

这就是其实际吸引力之一。

---

# 139. 一个非常重要的 Diagnostic 思维

如果你的 policy：

```text
offline validation action loss 很低
```

但：

```text
real rollout 很差
```

第一反应不要只问：

> “模型是不是不够大？”

还应该问：

\[
\boxed{
\text{rollout state distribution 是否偏离 demonstration distribution？}
}
\]

---

# 140. 怎么判断是 Distribution Shift？

可以观察：

- rollout失败前 observation是否越来越偏离demo；
- gripper/object relative pose是否进入demo很少见区域；
- policy confidence/variance是否变化；
- nearest-neighbor demo distance是否变大；
- recovery state是否几乎没有训练样本。

---

# 141. Offline Loss 为什么可能看不出问题？

因为它只测：

> demo-distribution generalization。

一个 policy可以：

\[
L_{val}\approx0
\]

但在：

\[
s\notin\mathcal D
\]

上行为任意差。

---

# 142. Rollout Evaluation 才是 Policy 的最终测试

机器人 policy真正 metric：

- success rate；
- completion；
- robustness；
- recovery；
- safety；
- closed-loop performance。

不能只看：

> action prediction loss。

---

# 143. 这也是 Robot Learning 与普通静态 Prediction 最大的认知差异

静态任务：

> prediction不会改变下一 test example。

控制：

> prediction决定下一 test example。

记住：

\[
\boxed{
\text{Your model is part of the data-generating process.}
}
\]

---

# 144. Behavior Cloning 和 Reinforcement Learning 的一个根本区别

BC直接拟合：

\[
\pi^*(a|s)
\]

RL则通过：

> environment interaction + reward

优化：

\[
J(\pi)
\]

因此 RL天然采样：

> 当前 policy induced states，

但又带来：

- exploration；
- reward；
- sample efficiency；

等其他困难。

---

# 145. DAgger 位于一个有趣的中间位置

它让 learner：

> on-policy访问 states，

但 action label仍来自：

> expert。

所以不需要自己从 sparse reward猜正确动作。

---

# 146. BC 的优点

- 简单；
- stable supervised training；
- data-efficient relative to RL；
- easy offline training；
- human demonstrations可直接用。

---

# 147. BC 的缺点

核心就是：

\[
\boxed{
\text{train on }d_{\pi^*}
\quad
\text{deploy on }d_\pi
}
\]

以及：

- multimodality；
- partial observability；
- demonstration noise；
- expert suboptimality。

---

# 148. Compounding Error 和 Multimodality 不是一个问题

### Compounding Error

\[
\boxed{
\text{state distribution mismatch over rollout}
}
\]

---

### Multimodality

同一个 observation：

> 可能有多个合理 actions。

MSE可能平均出：

> 不合理动作。

ACT 的 CVAE主要针对后者。

---

# 149. Compounding Error 和 Non-Markovianity 也不是完全同一个问题

### Non-Markovianity

当前 observation：

> 不足以唯一决定 expert action。

需要：

- history；
- latent intent；
- action chunk；

等。

---

### Compounding Error

learner action：

> 让未来输入偏离训练 distribution。

两者可以同时存在。

---

# 150. 为什么 Action Chunking 一次同时触碰这两个问题？

因为它：

1. 输出 temporally structured action sequence；
2. 后续 action基于过去 observation；
3. 减少某些 decision feedback；
4. 表达 demonstration temporal correlations。

所以它对：

- compounding；
- non-Markovianity；

都有帮助。

---

# 151. 一个简单 One-Step Policy

\[
a_t
=
\pi(o_t)
\]

如果：

\[
o_t
\]

相同，

输出：

> 相同 distribution / deterministic action。

---

# 152. 但 Human Demonstrator 可能有 Hidden Phase

例如：

```text
手停在杯子前面
```

视觉状态几乎一样。

但：

- 刚刚正在靠近；
- 或已经抓取失败准备重试；

正确 action可能不同。

这说明：

\[
o_t
\]

不是完整 Markov state。

---

# 153. Action Chunk 可以隐式暴露 Temporal Plan

\[
[
a_t,a_{t+1},...,a_{t+k-1}
]
\]

作为整体，

能更好表示：

> 当前 demonstration segment属于什么 motion pattern。

再配合：

\[
z
\]

可以进一步编码 style variation。

---

# 154. 但 ACT Policy Inference 本身没有显式 Observation History

Canonical ACT policy输入当前：

- images；
- qpos；
- z=0。

它不是显式：

\[
o_{t-m:t}
\]

history model。

因此 Action Chunking提供的 temporal information：

> 主要来自 output-side temporal structure，而不是 input history。

---

# 155. 2026 Action-Chunking Paper 让这个现象更值得重新理解

它指出：

> predicting actions from past observations 本身就是重要机制。

这说明 chunking的优势可能不只是：

> “输出更平滑”。

而是：

> 它改变了 action 与 observation time之间的 conditioning relation。

---

# 156. 一个统一 Timeline

假设 chunk length：

\[
k=4
\]

在：

\[
t=0
\]

预测：

\[
[
\hat a_0|o_0,
\hat a_1|o_0,
\hat a_2|o_0,
\hat a_3|o_0
]
\]

---

# 157. 到 t=2

当前动作：

\[
a_2
\]

可能同时有：

\[
\hat a_2|o_0
\]

\[
\hat a_2|o_1
\]

\[
\hat a_2|o_2
\]

多个 temporal relationships。

---

# 158. Temporal Ensemble 再融合它们

所以最终：

\[
a_2
\]

不是只依赖：

\[
o_2
\]

还融合：

\[
o_0,o_1
\]

产生的 predictions。

---

# 159. 这实际上构造了一个隐式 Observation History

虽然网络单次 forward只看当前：

\[
o_t
\]

execution通过 overlapping chunks：

> 把过去 observations的预测保留下来。

所以控制 action最终：

> 含有过去 observation信息。

---

# 160. 这是理解 ACT 很高级的一层

Canonical policy network：

> 单次 inference不是 recurrent history model。

但整个 execution system：

\[
\boxed{
\text{overlapping chunk buffer + Temporal Ensemble}
}
\]

让最终 action拥有：

> multi-time observation conditioning。

---

# 161. 这与 Distribution Shift 有什么关系？

更老的：

\[
o_{t-n}
\]

可能发生在：

> learner还没有积累那么多近期偏差时。

融合不同 temporal predictions：

> 可能让控制对当前瞬时误差更robust。

这与2026 paper的 delayed-policy/implicit-ensemble解释有呼应。

---

# 162. 但不要把它说成严格“消除 OOD”

即便过去 observation也可能：

> 已经 OOD。

而且环境发生真正新变化时，

旧 prediction：

> 可能反而过时。

所以仍存在 tradeoff。

---

# 163. Temporal Responsiveness Tradeoff

更多依赖过去：

> 增加稳定/ensemble效果。

更多依赖当前：

> 提高对新变化的响应。

ACT 的 exponential weights正是在某种程度上：

> 平衡 old vs new predictions。

---

# 164. 这也解释为什么 TE Weight Decay 是重要超参数

较快偏重新预测：

> 更 responsive。

较慢衰减：

> 更平滑、更利用过去 chunk predictions。

我们在 [Temporal Ensemble](../act/temporal-ensemble.md) 已详细讲过。

---

# 165. Behavior Cloning 最核心的数学对象：Occupancy Distribution

如果以后进入更高级 imitation/RL，

你会经常看到：

\[
d_\pi(s)
\]

或：

\[
\rho_\pi(s,a)
\]

---

# 166. State-Action Occupancy

可以定义：

\[
\boxed{
\rho_\pi(s,a)
=
d_\pi(s)\pi(a|s)
}
\]

表示：

> 执行 policy \(\pi\) 时，多大概率在 state \(s\) 采取 action \(a\)。

---

# 167. BC 在做什么？

主要通过 expert demonstrations拟合：

\[
\pi(a|s)
\]

但数据采样于：

\[
d_{\pi^*}
\]

---

# 168. 更高级 Imitation Learning 方法会直接考虑 Occupancy Matching

例如 adversarial imitation：

> 尝试让 learner的 state-action occupancy接近 expert occupancy。

这是 GAIL 等方法的方向。

---

# 169. DAgger 则从另一个方向解决

不直接 match occupancy distribution，

而是：

> 在 learner occupancy中不断请 expert提供正确 labels。

---

# 170. 所以 Imitation Learning 有多个层级

### Behavior Cloning

\[
\text{supervised action matching}
\]

### DAgger

\[
\text{learner-state supervised action matching}
\]

### Adversarial / Occupancy Matching

\[
\text{match trajectory/state-action distributions}
\]

---

# 171. ACT 属于哪层？

核心训练：

\[
\boxed{
\text{offline behavior cloning with structured generative action chunks}
}
\]

不是：

- DAgger；
- GAIL；
- RL。

---

# 172. 为什么理解这个定位很重要？

否则容易看到 ACT success就误以为：

> “Distribution shift问题已经被Transformer解决了。”

没有。

Transformer主要解决：

> function approximation / contextual sequence modeling。

Distribution shift：

> 是数据与closed-loop interaction问题。

---

# 173. 再大的 Model 也不能保证 OOD Recovery

如果 training data从来没有：

> “杯子被碰歪以后怎么抓”

那么大模型可能凭 generalization猜对，

但 BC objective本身：

> 没有直接监督它必须猜对。

---

# 174. Foundation Model / VLA 为什么可能改善？

大规模预训练可能让模型有：

> 更广泛视觉与行为 priors。

因此在 demonstration coverage外：

> extrapolation可能更好。

但这仍不意味着：

\[
d_{\pi^*}=d_\pi
\]

distribution shift数学结构消失了。

---

# 175. VLA 也仍是 Closed-Loop Policy

只要：

\[
action_t
\]

改变：

\[
observation_{t+1}
\]

就仍然存在：

> policy-induced distribution。

所以这个知识会一直陪你进入：

- OpenVLA；
- π0；
- GR00T；
- robot foundation models。

---

# 176. Diffusion Policy 是否解决 Compounding Error？

Diffusion Policy主要改善：

> multimodal, high-dimensional action sequence modeling。

通常也预测：

> action horizon / chunk。

所以同样可以因 temporal action prediction获得 robustness。

但它也不是：

> 自动的 DAgger。

---

# 177. 为什么现代 Robot Policies 几乎都爱 Action Horizon？

因为 manipulation中：

> 单步 action prediction往往太短视、易抖、易受distribution shift影响。

预测 action sequence：

> 让模型学习局部 motion structure。

ACT、Diffusion Policy等都体现这类思想。

---

# 178. 但 2026 新结果提醒我们

不要把这个现象过度简化成：

\[
\boxed{
\text{chunking works only because horizon becomes }T/k
}
\]

机制可能更丰富。

---

# 179. 如果只看 Original ACT，应该怎么表述？

最忠实：

> ACT提出 action chunking，将多个动作作为一个预测单元；作者解释这样把 task 的 effective horizon降低约 \(k\) 倍，从而缓解 behavior cloning 中的 compounding errors，同时也能捕捉 human demonstrations中的 temporally correlated non-Markovian behavior。

---

# 180. 如果结合 2026 Knowledge，怎么升级？

> 后续机制研究表明，单纯“减少决策 horizon”不足以完整解释 action chunking 的优势；基于过去 observations 的 action prediction、non-Markovian expressivity，以及多 temporal relationships形成的 implicit ensembling也是重要因素。

---

# 181. 这两个说法可以同时存在

一个是：

> 原方法的设计动机。

一个是：

> 后续更精细的因果/机制分析。

高质量知识库应该：

\[
\boxed{
\text{都保留，并明确时间与证据层级。}
}
\]

---

# 182. Behavior Cloning 的 Offline Evaluation 应该看什么？

除了 action loss，

还应考虑：

- multi-step rollout；
- success rate；
- perturbation recovery；
- unseen initial states；
- object pose variation；
- distractors；
- camera shift。

---

# 183. Recovery Test 特别有价值

故意把机器人放在：

> demonstration轨迹稍微偏离的位置。

看 policy：

> 会回去，还是继续偏。

这比纯 validation L1：

> 更直接测试 distribution robustness。

---

# 184. 为什么 Randomized Initial State 有帮助？

如果 demonstration从不同：

- object poses；
- robot initial joints；
- camera conditions；

开始，

dataset覆盖的：

\[
d_{\pi^*}
\]

本身就更宽。

learner偏离一点后：

> 更可能仍落在训练覆盖附近。

---

# 185. Data Diversity 是 Offline BC 最重要的 Robustness Lever 之一

如果无法 DAgger，

可以通过：

- varied starts；
- intentional perturbations；
- recovery demos；
- multiple human styles；
- environmental randomization；

扩大 state coverage。

---

# 186. “更多 Demonstrations”也要看 Diversity

1000条几乎完全相同 trajectory：

> 不一定比100条高 diversity trajectories更能解决 covariate shift。

关键是：

\[
\boxed{
\text{coverage}
}
\]

而不仅是：

\[
N
\]

---

# 187. 为什么 Human Data 有时反而帮 Coverage？

人类teleoperation：

> 每次轨迹都有小差异。

这自然产生：

> local state/action diversity。

当然也带来：

> non-stationarity / multimodality。

ACT的 CVAE正是为了处理后一个问题。

---

# 188. 所以 Demonstration Variation 是“双刃剑”

### 好处

增加：

> state coverage / robustness。

### 问题

同一 observation附近：

> actions不一致，监督变多模态。

---

# 189. ACT 用 CVAE 尝试吃下这种 Variation

这就是为什么 ACT 的三个设计彼此配合：

- chunking；
- CVAE；
- temporal ensemble。

它们并不是随便堆在一起。

---

# 190. 一个高级统一图

```text
Human demonstrations
        │
        ├── limited learner-state coverage
        │       ↓
        │   compounding error / covariate shift
        │       ↓
        │   Action Chunking helps mitigate dynamics
        │   but does not aggregate learner-state labels
        │
        ├── non-stationary / multimodal actions
        │       ↓
        │      CVAE
        │
        └── overlapping noisy chunk predictions
                ↓
         Temporal Ensemble
```

---

# 191. DAgger 则直接瞄准第一条

```text
learner rollout
↓
states learner actually visits
↓
expert labels those states
↓
aggregate dataset
↓
retrain
```

---

# 192. 一张图看 BC Distribution Mismatch

```text
TRAINING

Expert policy π*
      │
      ▼
 states near expert trajectory
      │
      ▼
      D
      │
      ▼
train π


DEPLOYMENT

learned π
   │
   ▼
small mistake
   │
   ▼
different state
   │
   ▼
π acts on unfamiliar input
   │
   ▼
larger mistake
   │
   ▼
further distribution shift
```

---

# 193. 一张图看 DAgger

```text
Iteration 1:
Expert states
↓
train π₁

Iteration 2:
run π₁
↓
learner states
↓
expert labels them
↓
aggregate
↓
train π₂

Iteration 3:
run π₂
↓
new learner states
↓
expert labels
↓
aggregate
↓
train π₃
...
```

目标：

\[
\boxed{
\text{training distribution follows learner distribution}
}
\]

---

# 194. 一张图看 ACT 的不同策略

```text
One-step BC:

o_t → a_t
      ↓ environment
o_{t+1} → a_{t+1}
          ↓
...

Every step immediately conditions on
the newly induced state.


Action Chunking:

o_t
 ↓
[a_t, a_{t+1}, ..., a_{t+k-1}]

A single earlier observation predicts
multiple future actions.


Final ACT:

o_0 → chunk 0
o_1 → chunk 1
o_2 → chunk 2
...

same execution time receives
multiple temporally offset predictions

↓
Temporal Ensemble
↓
executed action
```

---

# 195. Common Misconception 1：Behavior Cloning 就是普通 Supervised Learning，所以 Validation Loss 足够

**错误。**

Training形式是 supervised，

但 deployment是：

> closed-loop sequential control。

---

# 196. Common Misconception 2：Compounding Error 只是数值误差简单相加

**错误。**

最关键的是：

> error改变future state distribution。

---

# 197. Common Misconception 3：单步准确率 99% 就意味着整段任务成功率 99%

**错误。**

Sequential success可能随着 horizon显著恶化。

---

# 198. Common Misconception 4：训练和部署看到的是同一种 State

**一般不保证。**

训练：

\[
d_{\pi^*}
\]

部署：

\[
d_\pi
\]

---

# 199. Common Misconception 5：只要 Expert Demo 足够“完美”，BC 就更好

不一定。

如果只有完美轨迹：

> recovery states可能完全没有覆盖。

---

# 200. Common Misconception 6：Recovery 是 Model 自己自然学会的

没有数据/先验：

> 不保证。

---

# 201. Common Misconception 7：Covariate Shift 是因为 Camera Distribution 变了

那只是一个可能原因。

Imitation learning特有的关键 shift：

> policy自己的动作造成 state distribution变化。

---

# 202. Common Misconception 8：DAgger 只是收更多 Expert Demonstrations

**错误。**

它专门在：

\[
learner-induced states
\]

收 expert labels。

---

# 203. Common Misconception 9：DAgger 训练时执行的都是 Expert Actions

不一定。

算法允许：

> learner/expert mixture，

最终 learner占比增加。

---

# 204. Common Misconception 10：DAgger 是 Reinforcement Learning

**不是。**

它仍然利用：

> expert action supervision，

而不是主要从 reward探索动作。

---

# 205. Common Misconception 11：DAgger 的关键是更大的 Neural Network

**不是。**

关键是：

> 数据分布的交互式聚合。

---

# 206. Common Misconception 12：Action Chunking 完全解决了 Covariate Shift

**错误。**

ACT training仍然是 offline demonstrations。

---

# 207. Common Misconception 13：Action Chunking 等价于 DAgger

**完全错误。**

一个改 temporal prediction structure，

一个改 training-state distribution。

---

# 208. Common Misconception 14：Chunk Size k=100，就意味着只有1/100的动作会有 Prediction Error

**错误。**

100个 predicted actions都可以有误差。

---

# 209. Common Misconception 15：Final ACT 每100步才看一次 Observation

**错误。**

Temporal Ensemble配置下：

> 每 timestep重新query。

---

# 210. Common Misconception 16：既然每 timestep重新query，Action Chunking 就完全没有 Horizon Reduction意义

也不准确。

Chunk prediction改变：

- output temporal structure；
- action-observation conditioning；
- overlapping predictions。

只是不能简单理解成：

> inference次数严格变成 \(T/k\)。

---

# 211. Common Misconception 17：Temporal Ensemble 就是在训练时增强 Data Distribution

**错误。**

它是 inference mechanism。

---

# 212. Common Misconception 18：CVAE 解决 Compounding Error

不是主要作用。

它主要处理：

> demonstration variability / multimodality。

---

# 213. Common Misconception 19：Transformer Capacity 足够大就可以彻底解决 Distribution Shift

**没有这样的保证。**

OOD state supervision仍是数据问题。

---

# 214. Common Misconception 20：\(T^2\epsilon\) 表示所有 BC 实际 Error 一定精确等于这个值

**错误。**

这是经典 worst-case performance scaling / bound，

不是每个任务的精确预测公式。

---

# 215. Common Misconception 21：\(T^2\epsilon\) 只是 Proof 太松，没有实际意义

Ross 等工作指出：

> 这个 quadratic scaling存在 tight examples。

所以它表达一个真实的最坏情形问题。

---

# 216. Common Misconception 22：如果 Expert Error 为0，Learner就一定没有 Shift

Expert本身可以完美，

但 learner：

> 仍可能因为有限数据/model error偏离。

---

# 217. Common Misconception 23：Distribution Shift 只发生在视觉输入

State shift可以包括：

- robot joints；
- object pose；
- contact state；
- velocities；
- camera pixels；
- task phase。

---

# 218. Common Misconception 24：Action Chunking 的唯一作用是 Smoothness

**错误。**

它还改变：

- temporal expressivity；
- conditioning relationship；
- compounding structure。

---

# 219. Common Misconception 25：Action Chunking 的唯一作用是把 Horizon 除以 k

截至2026年的机制研究表明：

> 这个解释不充分。

---

# 220. Common Misconception 26：2026 Paper 证明 ACT 的原始解释完全错误

**也错误。**

更准确：

> 后续研究表明 horizon reduction并不足以独立解释全部收益，并提出更丰富机制。

---

# 221. Common Misconception 27：Delayed Policy 意味着机器人故意慢 k Steps

不是简单的 actuator delay。

它指：

> 当前 action prediction使用过去 observation作为 conditioning。

具体 deployment设计需看论文定义。

---

# 222. Common Misconception 28：Implicit Ensembling 就等于 ACT 的 Exponential Temporal Ensemble

**不完全等同。**

后者是明确的 execution averaging rule。

前者是：

> 多 temporal relationships 带来的 ensemble-like机制解释。

---

# 223. Common Misconception 29：Offline BC 没有 DAgger 就一定不能做机器人

**错误。**

ACT、Diffusion Policy等大量成功工作证明：

> offline imitation可以很强。

但经典 distribution-shift问题仍然存在。

---

# 224. Common Misconception 30：如果 Rollout 失败，就一定是 Covariate Shift

也不一定。

还可能是：

- bad perception；
- multimodal averaging；
- latency；
- calibration；
- controller mismatch；
- hardware；
- insufficient model capacity；
- bad normalization。

需要诊断。

---

# 225. 一个实用的诊断顺序

如果：

> offline loss低，rollout差，

优先检查：

### 1. State Coverage

失败前是否进入demo少见状态？

### 2. Recovery

policy能否从人为小perturbation恢复？

### 3. Multimodality

同状态附近target actions是否冲突？

### 4. Observation Aliasing

当前 observation是否缺 history信息？

### 5. Execution Mismatch

训练action与真实controller实际执行是否一致？

### 6. Latency / Synchronization

observation与action是否时间对齐？

---

# 226. 如果只记住一个 Behavior Cloning 公式

BC训练：

\[
\boxed{
\hat\pi_{BC}
=
\arg\min_\pi
\mathbb E_{s\sim d_{\pi^*}}
[
\ell(s,\pi)
]
}
\]

---

# 227. 如果只记住一个 Deployment 公式

真正执行时：

\[
\boxed{
s\sim d_\pi
}
\]

而不是：

\[
d_{\pi^*}
\]

---

# 228. 如果只记住一个 Distribution-Shift 公式

\[
\boxed{
d_{\pi^*}
\neq
d_\pi
}
\]

这是 BC sequential failure的核心来源之一。

---

# 229. 如果只记住一个 Compounding-Error 公式

经典 worst-case supervised imitation：

\[
\boxed{
J(\pi)
\le
J(\pi^*)
+
O(T^2\epsilon)
}
\]

其中：

\[
\epsilon
\]

是在 expert state distribution上的 imitation error。

---

# 230. 如果只记住一个 DAgger 公式

理想目标从：

\[
\mathbb E_{s\sim d_{\pi^*}}
[
\ell
]
\]

推进到：

\[
\boxed{
\mathbb E_{s\sim d_\pi}
[
\ell
]
}
\]

通过：

> learner rollout + expert labeling + dataset aggregation。

---

# 231. 如果只记住一句话理解 Behavior Cloning 的根本问题

> **Behavior Cloning 的困难不只是“模型会预测错动作”，而是“预测错动作会改变接下来模型要面对的输入”：训练数据来自 expert policy 诱导的状态分布 \(d_{\pi^*}\)，但部署时 learner 的动作不断改变环境，使自己实际看到 \(d_\pi\)；一旦小误差把系统带到 demonstrations 很少覆盖的状态，policy 的误差可能进一步增大并继续把状态推得更远，因此单步 supervised error 会通过 closed-loop dynamics 发生 compounding。**

---

# 232. 如果只记住一句话理解 DAgger

> **DAgger 并不是把 Behavior Cloning 的网络换得更复杂，而是直接修正训练分布：它让当前 learner 自己 rollout，采集 learner 真正会访问的 states，再让 expert 对这些 states 给出正确 actions，并把这些数据不断聚合进训练集，从而让 policy 不只会模仿“专家从不犯错时怎么做”，也逐渐学会“自己已经偏离以后应该怎么恢复”。**

---

# 233. 如果只记住一句话理解 ACT 与 Compounding Error

> **ACT 没有像 DAgger 那样把 learner-induced states重新交给 expert标注，因此它没有从数据收集层面消除 \(d_{\pi^*}\neq d_\pi\)；它选择通过 action chunking 改变 temporal policy structure——一次预测一段未来动作、联合建模局部轨迹，并在最终系统中融合 overlapping chunk predictions——从而降低某些 error-feedback机制的破坏性，让 offline behavior cloning在高精度机器人任务中更稳定。**

---

# 234. 如果只记住一句话理解 2026 的新修正

> **原始 ACT 用“action chunking 将 effective horizon缩短约 \(k\) 倍”作为缓解 compounding error的重要设计解释；截至2026年的后续机制研究进一步表明，这个解释并不足以单独说明 action chunking 的全部收益，基于过去 observations 的 action prediction、更强的 non-Markovian expressivity，以及同时学习 \(a_t|o_t,a_t|o_{t-1},\ldots\) 所形成的 implicit ensembling，也构成了重要机制。**

---

# 235. 这篇之后，你应该怎样重新理解 ACT？

以前可能是：

```text
ACT = Transformer + CVAE + Action Chunk
```

现在应该进一步变成：

```text
ACT
│
├─ still fundamentally offline Behavior Cloning
│
├─ therefore inherits policy-induced distribution shift
│
├─ Action Chunking changes temporal error dynamics
│
├─ CVAE handles demonstration variability
│
├─ Temporal Ensemble stabilizes overlapping predictions
│
└─ but none of these is literally DAgger-style learner-state relabeling
```

---

# 236. 下一篇：Posterior Collapse & Latent Usage

现在我们已经把：

> ACT 作为 imitation-learning algorithm 的外部理论问题

讲清楚了。

下一块最值得进入的是 CVAE 内部真正高级的问题：

> **模型到底有没有真的使用 \(z\)？**

下一篇建议：

> **[Posterior Collapse：CVAE 明明有 z，Decoder 为什么可能完全不理它？](../../generative-models/posterior-collapse.md)**

会深入解释：

- 什么叫 latent usage；
- 什么叫 posterior collapse；
- 为什么强 decoder可能直接忽略 \(z\)；
- 当：
  \[
  q_\phi(z|x,y)\approx p(z)
  \]
  时发生了什么；
- KL接近0到底是好还是坏；
- \(\beta\) 太大/太小分别怎样；
- mutual information：
  \[
  I(z;y|x)
  \]
  与 latent usage的关系；
- reconstruction与KL的 rate-distortion视角；
- KL annealing；
- free bits；
- decoder weakening；
- 如何实际诊断 ACT 的 \(z\) 是否被使用；
- 既然 ACT inference固定 \(z=0\)，为什么 training-time latent仍然可能重要；
- “CVAE ablation success显著下降”能说明什么、又不能说明什么。

---

## Primary Source：DAgger

Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell.

**A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.**  
AISTATS 2011.

- arXiv: https://arxiv.org/abs/1011.0686
- Full text: https://arxiv.org/html/1011.0686

这是本文关于：

- policy-induced state distribution；
- traditional supervised imitation；
- \(d_\pi\)；
- \(d_{\pi^*}\)；
- \(T^2\epsilon\) compounding-error guarantee；
- DAgger；

的主要理论来源。

论文定义：

\[
d_\pi^t
\]

为：

> 执行 policy \(\pi\) 到时间 \(t\) 时的 state distribution。

平均：

\[
\boxed{
d_\pi
=
\frac1T
\sum_{t=1}^{T}
d_\pi^t
}
\]

传统 supervised imitation求：

\[
\boxed{
\hat\pi_{sup}
=
\arg\min_{\pi\in\Pi}
\mathbb E_{s\sim d_{\pi^*}}
[
\ell(s,\pi)
]
}
\]

但真正想要的 policy应在：

\[
d_\pi
\]

上表现良好。

论文 Theorem 2.1 给出：

若：

\[
\mathbb E_{s\sim d_{\pi^*}}
[
\ell(s,\pi)
]
=
\epsilon
\]

在相应 assumptions 下：

\[
\boxed{
J(\pi)
\le
J(\pi^*)
+
T^2\epsilon
}
\]

并指出该 quadratic horizon dependence存在 tight cases。

---

## DAgger Algorithm

同一论文 Section 3：

1. 初始化 aggregate dataset：
   \[
   \mathcal D
   \]
2. 执行当前 policy / expert mixture：
   \[
   \pi_i
   \]
3. 收集 visited states；
4. 对这些 states询问 expert：
   \[
   \pi^*(s)
   \]
5. 得到：
   \[
   \mathcal D_i
   =
   \{
   (s,\pi^*(s))
   \}
   \]
6. 聚合：
   \[
   \mathcal D
   \leftarrow
   \mathcal D\cup\mathcal D_i
   \]
7. 在 aggregate dataset上训练下一 policy。

核心不是单纯：

> more demonstrations。

而是：

\[
\boxed{
\text{demonstrations on states induced by the learner}
}
\]

---

## Historical Behavior-Cloning Background：ALVINN

Dean A. Pomerleau.

**ALVINN: An Autonomous Land Vehicle In a Neural Network.**  
NeurIPS, 1989.

Carnegie Mellon Robotics Institute:

https://publications.ri.cmu.edu/alvinn-an-autonomous-land-vehicle-in-a-neural-network

ALVINN使用 neural network从：

- camera；
- laser range data；

预测：

> vehicle travel direction。

它是早期 neural behavior cloning / autonomous driving的经典例子之一。

后续 imitation-learning研究经常用 autonomous driving说明：

> 只从正常 expert states训练会缺乏 recovery behavior。

---

## DART：Recovery / Noise Injection

Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, Ken Goldberg.

**DART: Noise Injection for Robust Imitation Learning.**  
CoRL 2017.

- PMLR: https://proceedings.mlr.press/v78/laskey17a.html
- PDF: https://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf

DART明确将 Behavior Cloning描述为 off-policy imitation method，并指出：

> learner drift away from supervisor demonstrations以后会产生 compounding errors。

其方法不执行标准 DAgger式 learner/expert interactive labeling，

而是在 supervisor demonstration中注入经过设计的 noise，

让 supervisor展示：

> recovery from errors。

它是理解：

\[
\boxed{
\text{state-distribution coverage}
}
\]

为什么重要的很好的补充来源。

---

## ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- PDF: https://arxiv.org/pdf/2304.13705
- Project: https://tonyzhaozh.github.io/aloha/
- Code: https://github.com/tonyzhaozh/act

ACT在摘要中明确指出：

> imitation learning在高精度任务中的关键挑战包括 policy error over time compounding，以及 human demonstrations的non-stationarity。

ACT通过：

\[
\boxed{
Action\ Chunking
}
\]

学习 future action sequences，

作者的核心设计解释之一是：

> 减少 effective horizon，从而缓解 compounding error。

---

## ACT Project Description

Official project page:

https://tonyzhaozh.github.io/aloha/

官方项目页面明确描述：

> policy errors can compound over time and drift outside the training distribution。

并把 Action Chunking描述为：

> 通过预测动作 chunks 降低 effective horizon。

这与 ACT paper的设计动机一致。

---

## 2026 Update：Why Does Action Chunking Improve Behavioral Cloning?

Filippo Lazzati, Kyle Stachowicz, William Chen, Alberto Maria Metelli, Andrew Wagenmaker, Sergey Levine.

**Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?**  
arXiv:2608.02547, submitted August 3, 2026.

- arXiv: https://arxiv.org/abs/2608.02547
- Project: https://action-chunking.github.io/

这篇后续工作专门研究：

> Action Chunking究竟为什么有效。

其结论对理解ACT非常重要：

1. temporal consistency；
2. horizon reduction；
3. representation learning；

这些常见解释：

> **不足以单独完整解释 action chunking的收益。**

论文进一步指出：

- action chunking具有更强的non-Markovian expressivity；
- 它可以减少compounding error；
- delayed policies在许多setting中可以捕获这些收益；
- action chunking还学习多种temporal relationships：
  \[
  a_t|o_t,\quad
  a_t|o_{t-1},\quad
  \ldots
  \]
- 这产生了作者称为：
  \[
  \boxed{
  \text{implicit ensembling}
  }
  \]
  的额外收益。

因此本文把：

> ACT 2023 原始“effective horizon”设计解释

与：

> 2026 更深入机制分析

明确分开，而不是互相替代。

---

## 本文知识连接

### Imitation Learning

- [Imitation Learning](../imitation-learning.md)
- [Behavior Cloning](./behavior-cloning-distribution-shift.md)
- DAgger
- [Covariate Shift](./behavior-cloning-distribution-shift.md)
- [Distribution Shift](./behavior-cloning-distribution-shift.md)
- Occupancy Measure
- GAIL
- Recovery Demonstrations

### Sequential Decision Making

- Markov Decision Process
- Policy
- State Distribution
- On-Policy vs Off-Policy
- Closed-Loop Control

### ACT

- [ACT 到底解决了什么问题？](../act/act-what-problem-does-it-solve.md)
- [Action Chunking](../act/action-chunking.md)
- [Temporal Ensemble](../act/temporal-ensemble.md)
- [ACT Training](../act/training.md)
- [ACT Inference](../act/inference.md)
- [ACT Complete Data Flow](../act/complete-data-flow.md)

### Generative Modeling

- [CVAE](../../generative-models/cvae.md)
- [CVAE in ACT](../act/cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](../act/why-z-zero-at-inference.md)

### 下一步

- [Posterior Collapse：CVAE 明明有 z，Decoder 为什么可能完全不理它？](../../generative-models/posterior-collapse.md)
