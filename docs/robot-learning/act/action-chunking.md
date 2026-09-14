---
title: "Action Chunking：为什么一次预测一段动作？"
description: "从单步行为克隆的累积误差出发，严格解释 ACT 原论文中的 Action Chunking、effective horizon、非 Markov 示范，以及 chunk size k 的取舍。"
status: reviewed
pageType: application
canonical: /robot-learning/act/action-chunking
updated: "2026-09-15"
---

# Action Chunking：为什么一次预测一段动作？

如果已经读过 [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)，你应该已经知道 ACT 的第一个核心变化：

普通行为克隆通常预测一个动作：

\[
\pi_\theta(a_t\mid s_t)
\]

ACT 则预测一段未来动作：

\[
\pi_\theta(a_{t:t+k}\mid s_t)
\]

这就是 **Action Chunking**。

但如果只记成：

> “Action Chunking = 一次预测很多步”

其实还没有真正理解它。

我们至少还需要回答四个问题：

1. 为什么一次预测一段动作能够缓解累积误差？
2. 论文所说的 **effective horizon 缩短 \(k\) 倍** 到底是什么意思？
3. Action Chunking 为什么还能帮助处理人类示范中的 non-Markovian behavior？
4. \(k\) 是不是越大越好？

这一篇只解决 Action Chunking 本身。

至于“为什么最终 ACT 每一步仍然重新预测”“重叠的 action chunks 怎么融合”，放到下一篇：

- [Temporal Ensemble](./temporal-ensemble.md)

---

# 1. 从最普通的单步行为克隆开始

假设我们有一条专家示范：

\[
(s_1,a_1),(s_2,a_2),\ldots,(s_T,a_T)
\]

其中：

- \(s_t\)：时间步 \(t\) 的状态或观察；
- \(a_t\)：专家在这个时间步执行的动作；
- \(T\)：一条完整轨迹的长度。

最简单的 [行为克隆（Behavior Cloning）](../imitation-learning/behavior-cloning-distribution-shift.md) 学习：

\[
\pi_\theta(a_t\mid s_t)
\]

也就是说：

> 给我现在的状态，我预测现在应该执行什么动作。

训练时，我们拥有专家真正访问过的状态：

\[
s_1,s_2,\ldots,s_T
\]

所以模型不断学习：

```text
专家状态
   ↓
专家动作
```

如果模型每一步都完全正确，这当然没有问题。

问题在于：

> 推理时，机器人并不会永远待在专家轨迹上。

---

# 2. 单步预测为什么会产生 Compounding Error？

假设专家在某个状态 \(s_t\) 应该执行：

\[
a_t
\]

而机器人预测：

\[
\hat a_t
\]

并且：

\[
\hat a_t \neq a_t
\]

哪怕只是一个很小的误差，也可能导致下一时刻机器人到达：

\[
\hat s_{t+1}
\]

而不是专家示范中的：

\[
s_{t+1}
\]

于是：

\[
\hat s_{t+1}\neq s_{t+1}
\]

接下来模型必须在自己造成的新状态上继续预测：

\[
\pi_\theta(a_{t+1}\mid \hat s_{t+1})
\]

但训练时它主要看到的是专家状态分布。

如果 \(\hat s_{t+1}\) 已经偏离训练数据，模型可能更容易再次出错。

然后：

\[
\hat s_{t+2}
\]

继续偏离。

于是产生：

```text
动作有一点误差
      ↓
下一状态偏一点
      ↓
进入训练中较少见的状态
      ↓
下一次预测更困难
      ↓
状态进一步偏离
      ↓
错误继续积累
```

这就是：

> **Compounding Error（累积误差）**

ACT 原论文明确把它作为高精度模仿学习中的核心问题。

---

# 3. 为什么高频机器人控制尤其容易遇到这个问题？

ALOHA 中的动作数据是高频采集的。

论文中的遥操作控制频率为：

\[
50\text{ Hz}
\]

也就是每秒大约 50 个控制时间步。

假设一个操作持续 10 秒，那么轨迹长度大约就是：

\[
T=50\times10=500
\]

也就是说，一个看起来并不算特别长的动作，可能已经需要经历数百个时间步。

如果策略每一步都独立预测：

\[
a_1,a_2,\ldots,a_{500}
\]

那么机器人就要连续经历很长的预测链。

ACT 作者希望改变的，正是这种：

> **“一个长任务被拆成大量单步决策”**

的表示方式。

---

# 4. Action Chunking：改变“预测的基本单位”

普通单步策略的基本单位是：

\[
a_t
\]

Action Chunking 把基本单位改成：

> **一段动作序列。**

为了避免下标产生歧义，我们在这篇文章里定义：

\[
\mathbf A_t
=
(a_t,a_{t+1},\ldots,a_{t+k-1})
\]

其中：

- \(\mathbf A_t\)：从时间步 \(t\) 开始的一个 action chunk；
- \(k\)：chunk size，也就是一个 chunk 中包含多少个动作。

于是策略从：

\[
\pi_\theta(a_t\mid s_t)
\]

变成：

\[
\pi_\theta(\mathbf A_t\mid s_t)
\]

即：

\[
\pi_\theta(
a_t,a_{t+1},\ldots,a_{t+k-1}
\mid s_t
)
\]

ACT 原论文写作：

\[
\pi_\theta(a_{t:t+k}\mid s_t)
\]

并将其描述为预测接下来的 \(k\) 个动作。

这里最重要的变化不是符号。

而是：

> **模型不再只学习“下一瞬间怎么动”，而是学习“从当前状态出发，接下来这一小段行为如何展开”。**

---

# 5. 一个最简单的例子

假设机器人要把电池插进插槽。

把这个过程极度简化成 8 个动作：

```text
1. 靠近电池
2. 对准电池
3. 夹住电池
4. 抬起电池
5. 移向插槽
6. 调整角度
7. 插入
8. 松开
```

如果：

\[
k=1
\]

模型每次只预测一个动作：

```text
状态
↓
动作 1

新状态
↓
动作 2

新状态
↓
动作 3

...
```

如果：

\[
k=4
\]

则可以把未来动作组织成：

```text
Chunk 1
[靠近, 对准, 夹住, 抬起]

Chunk 2
[移向插槽, 调整角度, 插入, 松开]
```

模型预测的对象不再是完全孤立的瞬时动作，而是一个具有内部时间关系的动作片段。

---

# 6. 什么叫 Effective Horizon？

这是 Action Chunking 中最容易被“听懂了但其实没懂”的词。

先看普通单步预测。

假设任务需要：

\[
T
\]

个物理时间步。

如果每一步是一个决策单位，那么从策略角度看，需要处理大约：

\[
T
\]

个连续决策单位。

我们可以把这种决策链的长度直观地理解成任务的 horizon。

---

如果每个 chunk 包含：

\[
k
\]

个动作，并且先考虑论文描述的最朴素 chunk 执行方式：

```text
观察一次
↓
预测 k 个动作
↓
依次执行
↓
重新观察
```

那么整个任务大约会被划分成：

\[
\left\lceil \frac{T}{k}\right\rceil
\]

个 chunk。

如果 \(T\) 恰好能被 \(k\) 整除：

\[
\frac{T}{k}
\]

于是原来长度约为：

\[
T
\]

的单步决策链，变成长度约为：

\[
\frac{T}{k}
\]

的 chunk-level 决策链。

所以论文称它带来：

\[
k\text{-fold reduction in effective horizon}
\]

即：

> **有效 horizon 缩短约 \(k\) 倍。**

---

# 7. 一个具体数字例子

假设：

\[
T=500
\]

也就是说任务有 500 个控制时间步。

### 单步策略

\[
k=1
\]

有效决策链大约是：

\[
500
\]

个单位。

---

### Action Chunking

假设：

\[
k=100
\]

那么如果使用最朴素的 chunk 执行方式：

\[
\frac{500}{100}=5
\]

任务可以被理解为大约 5 个 chunk-level 单位。

从这个角度：

\[
500\rightarrow5
\]

effective horizon 缩短了约：

\[
100\times
\]

---

# 8. 但必须马上澄清：物理动作没有减少

这句话非常重要：

> **Action Chunking 缩短的是 effective horizon，不是物理轨迹长度。**

如果机器人原本需要执行：

\[
500
\]

个关节目标，

用了 Action Chunking 后，它仍然需要执行：

\[
500
\]

个关节目标。

并不会变成只执行 5 个动作。

变化的是：

> **策略把未来动作按照长度 \(k\) 的序列来预测和建模。**

所以：

```text
物理控制步数
500
```

并没有变成：

```text
5
```

而是从策略建模的角度：

```text
500 个独立单步单位
```

被重新组织成：

```text
若干个动作序列单位
```

这就是“effective”的含义。

---

# 9. 为什么 horizon 变短可能缓解累积误差？

这里需要非常谨慎。

Action Chunking **不是数学上把 compounding error 消灭了**。

它做的是改变策略需要建模的时间结构。

单步策略不断进行：

\[
s_t\rightarrow a_t
\]

然后依赖执行后的新状态继续：

\[
s_{t+1}\rightarrow a_{t+1}
\]

每一个动作都处在一个很长的逐步闭环链条里。

而 chunk policy 一次预测：

\[
s_t
\rightarrow
(a_t,\ldots,a_{t+k-1})
\]

于是模型直接学习：

> 当前状态和接下来一段专家行为之间的映射。

在论文的解释中，这相当于让任务具有更短的 effective horizon，从而减轻单步策略中误差跨大量决策步骤持续累积的问题。

可以直觉地理解为：

```text
单步策略：

现在怎么办？
↓
下一瞬间怎么办？
↓
下一瞬间怎么办？
↓
下一瞬间怎么办？
↓
...


Action Chunking：

从现在开始，
接下来这一小段应该怎么完成？
```

第二种表示方式把一段局部行为作为整体建模。

---

# 10. 这并不意味着“预测越远越准确”

这里很容易产生错误推理：

> 如果预测 \(k\) 步可以缩短 horizon，那把 \(k\) 设成整个 episode 不就最好？

不是。

当：

\[
k=T
\]

时，策略在第一次观察后直接预测完整任务：

\[
(a_1,a_2,\ldots,a_T)
\]

这实际上接近：

> **Fully Open-Loop Control**

即：

```text
一开始看一次环境
↓
一次性生成整个任务
↓
一路执行到底
```

这会失去非常重要的能力：

> 根据执行过程中真正发生的情况进行修正。

而精细操作偏偏非常依赖闭环视觉反馈。

所以 \(k\) 存在一个基本 trade-off：

```text
k 太小
↓
接近单步预测
↓
effective horizon 很长
↓
累积误差问题严重


k 太大
↓
预测序列过长
↓
更难建模
↓
反应能力下降
↓
趋近 open-loop control
```

因此：

> **\(k\) 不是越大越好。**

---

# 11. 原论文的 Ablation 正好验证了这一点

ACT 论文专门改变 chunk size \(k\) 做了 ablation。

作者给出两个极端：

### \(k=1\)

等价于：

> 没有 Action Chunking。

因为每个 chunk 只有一个动作。

---

### \(k=\text{episode length}\)

等价于：

> 完全 open-loop。

机器人根据第一次观察一次性输出整条 episode 的动作序列。

---

作者在 4 个实验设置上比较不同 \(k\)。

在关闭 Temporal Ensemble、单独观察 Action Chunking 作用时，ACT 的平均成功率从：

\[
1\%\quad (k=1)
\]

提升到：

\[
44\%\quad (k=100)
\]

但继续增大到：

\[
k=200,\;400
\]

性能又略微下降。

论文给出的解释是：

1. 太接近 open-loop，缺少 reactive behavior；
2. 很长的 action sequence 本身也更难建模。

这正好说明：

> **Action Chunking 的目标不是让 chunk 无限长，而是在“缩短 effective horizon”和“保留反应能力”之间找到合适尺度。**

---

# 12. 一个重要细节：Action Chunking 不依赖 Transformer 才成立

因为 ACT 的名字叫：

> Action Chunking with Transformers

很容易让人误以为：

> Action Chunking 是 Transformer 才能实现的技巧。

其实不是。

论文的 ablation 还把 Action Chunking 加到了另外两个 baseline 上。

例如对于 BC-ConvMLP，作者只需要把输出维度增加为：

\[
k\times \text{action\_dim}
\]

模型就可以一次输出 \(k\) 个动作。

也就是说：

> **Action Chunking 首先是一种 action representation / policy prediction strategy，而不是 Transformer 独有的结构。**

Transformer 只是 ACT 用来建模复杂视觉输入和动作序列的重要架构。

这两个概念应该分开。

---

# 13. 从 Tensor Shape 看会更直观

假设单个动作维度是：

\[
D_a
\]

batch size 是：

\[
B
\]

---

普通单步策略的输出可以写成：

\[
[B,D_a]
\]

即每个样本预测一个动作。

---

如果 chunk size 为：

\[
k
\]

Action Chunking 的输出则可以写成：

\[
[B,k,D_a]
\]

含义是：

```text
Batch
│
├── Sample 1
│   ├── action t
│   ├── action t+1
│   ├── ...
│   └── action t+k-1
│
├── Sample 2
│   └── ...
│
└── ...
```

因此，从最朴素的监督学习角度看，Action Chunking 的训练目标并不神秘。

输入一个当前观察：

\[
s_t
\]

target 不再是：

\[
a_t
\]

而是：

\[
\mathbf A_t
=
(a_t,\ldots,a_{t+k-1})
\]

---

# 14. Training Data 是怎么构造出来的？

假设一条示范轨迹为：

\[
(a_1,a_2,a_3,a_4,a_5,a_6,\ldots)
\]

并且：

\[
k=4
\]

那么在时间步 \(t=1\)，训练 target 可以是：

\[
(a_1,a_2,a_3,a_4)
\]

在 \(t=2\)：

\[
(a_2,a_3,a_4,a_5)
\]

在 \(t=3\)：

\[
(a_3,a_4,a_5,a_6)
\]

所以：

> **训练中的 action chunks 本身可以高度重叠。**

它们来自同一条 expert trajectory，只是起始 timestep 不同。

这也为后面的 [Temporal Ensemble](./temporal-ensemble.md) 埋下了伏笔：

当推理阶段每个 timestep 都重新预测 chunk 时，同一个未来时间步也会被多个不同 chunk 重复预测。

---

# 15. Action Chunking 还有第二个作用：建模时间相关行为

到这里，我们一直围绕：

> Compounding Error

讲 Action Chunking。

但 ACT 原论文还给出了另一个动机：

> **Action Chunking 可以帮助处理人类示范中的 non-Markovian behavior。**

这是什么意思？

先从 Markov Property 的直觉开始。

一个理想的单步 Markov policy 假设：

> 当前动作只需要由当前状态决定。

也就是：

\[
a_t\sim\pi(a_t\mid s_t)
\]

如果两个时刻的：

\[
s_t
\]

完全一样，那么策略看到的信息也一样。

---

# 16. “停顿”为什么可能让单步策略困惑？

论文举了一个非常好的例子：

> 人类示范中间的 pause（停顿）。

假设机器人看到同样的视觉和关节状态：

\[
s
\]

但人类有时正处在：

```text
停顿的第 1 帧
```

有时正处在：

```text
停顿的第 5 帧
```

如果状态表示本身没有编码“我已经停了多久”，那么从单步策略看：

\[
s=s
\]

输入几乎一样。

可是行为的时间上下文可能不同。

这种 temporally correlated confounder 会让：

\[
\pi(a_t\mid s_t)
\]

很难单独表示行为。

换句话说：

> 当前动作不只与“现在长什么样”有关，还与这一小段行为当前进行到哪里有关。

---

# 17. Chunk 为什么能够帮助？

因为 chunk policy 不再独立预测一个瞬间：

\[
a_t
\]

而是联合预测：

\[
(a_t,a_{t+1},\ldots,a_{t+k-1})
\]

如果某个时间相关因素完整地落在 chunk 内，那么模型可以直接学习：

> **这一整段动作的时间模式。**

例如：

```text
[停, 停, 停, 开始移动]
```

可以作为一个序列模式被预测。

而不是要求模型在每个“看起来几乎一样”的暂停状态上，单独判断：

> 现在到底该继续停，还是开始动？

ACT 论文对此的表述非常谨慎：

> 当 confounder 位于一个 chunk 内时，Action Chunking 可以缓解这个问题。

注意是：

> **mitigate**

而不是：

> 完全解决所有 non-Markovian behavior。

这一点不要夸大。

---

# 18. Action Chunking 和 History-Conditioned Policy 有什么区别？

处理非 Markov 信息，一个自然想法是：

> 那我把过去几帧历史全部输入模型不就好了？

例如：

\[
\pi(
a_t
\mid
s_{t-h:t}
)
\]

这叫 history-conditioned policy。

它确实可以提供过去信息。

但 ACT 论文提到，history-conditioned policies 还可能面对 **causal confusion** 等问题。

Action Chunking 采取的是另一个方向：

> 不一定增加大量过去输入，而是直接让输出包含未来一段时间结构。

因此它不只是：

```text
看更多过去
```

而是：

```text
一次建模更多未来
```

两者不是一回事。

---

# 19. 一个极其重要的边界：Naive Action Chunking ≠ 最终 ACT

到目前为止，我们为了理解 effective horizon，一直使用最朴素的执行方式：

```text
observe
↓
predict k actions
↓
execute k actions
↓
observe again
```

ACT 原论文自己也先这样定义 Action Chunking。

但是作者马上指出：

> **naive implementation can be sub-optimal**

原因是新的环境观察每隔 \(k\) 步才突然加入。

这会带来两个问题：

### 1. Reactive behavior 下降

在 chunk 内发生环境变化时，策略不能立刻根据新观察修正。

### 2. 动作可能不平滑

每次从旧 chunk 切换到新 chunk 时，新 observation 会突然改变预测结果，产生 jerky motion。

所以：

> **最终 ACT 并没有简单地每 \(k\) 步才 query 一次 policy。**

它改成：

> 每一个 timestep 都重新 query policy。

然后得到大量重叠的 action chunks。

这就需要：

- [Temporal Ensemble](./temporal-ensemble.md)

---

# 20. 那最终 ACT 每步都重新预测，为什么还叫“缩短 Effective Horizon”？

这是最值得澄清的问题之一。

如果最终 ACT 在推理时：

> 每个 timestep 都会重新跑 policy，

那么显然不能说：

> “最终 ACT 的模型 forward 次数从 \(T\) 降成了 \(T/k\)。”

这不是事实。

Temporal Ensemble 甚至会增加 inference-time computation。

所以：

> **论文中的 \(k\)-fold effective-horizon reduction，不应该理解成最终 ACT 只做 \(T/k\) 次网络推理。**

更准确地说，它描述的是 **Action Chunking 这种输出表示带来的时间抽象**：

\[
\pi(a_t\mid s_t)
\]

被改为：

\[
\pi(a_{t:t+k}\mid s_t)
\]

策略一次直接对未来一段行为进行建模，而不是只对一个瞬时动作进行建模。

最朴素执行时，这确实对应大约 \(T/k\) 个 chunk-level 决策。

而最终 ACT 为了恢复高频闭环反馈，又在推理阶段对这些 chunks 进行重叠预测和 Temporal Ensemble。

因此可以把两件事分开：

```text
Action representation / learning
        ↓
用 chunk 建模未来 k 步
        ↓
降低有效时间决策尺度


Inference execution
        ↓
每 timestep 重新 query
        ↓
保留高频闭环反馈
```

这两者并不矛盾。

---

# 21. Action Chunking 真正带来的是什么？

现在可以更准确地总结。

它至少带来三个效果。

## 1. Temporal abstraction

把：

\[
\text{single action}
\]

提升为：

\[
\text{short action sequence}
\]

作为策略的预测单位。

---

## 2. Shorter effective horizon

从论文的 chunk-level 视角：

\[
T
\]

变为大约：

\[
T/k
\]

从而缓解长序列单步模仿学习中的累积误差问题。

---

## 3. Sequence-level temporal structure

模型可以直接学习一段行为内部的时间关系：

\[
(a_t,a_{t+1},\ldots,a_{t+k-1})
\]

因此能够更好地表示某些短时间范围内的 non-Markovian behavior。

---

# 22. 它没有解决什么？

同样重要的是：

> 不要把 Action Chunking 神化。

Action Chunking 本身没有自动解决：

### 人类示范的多模态性

同一个 observation 可能有多种合理 action chunks。

ACT 通过 [CVAE](../../generative-models/cvae.md) 处理这个问题。

---

### Chunk 切换和平滑问题

Naive chunking 甚至会产生 jerky motion。

ACT 通过 [Temporal Ensemble](./temporal-ensemble.md) 改善。

---

### 视觉理解

图像中哪个物体重要、机械臂当前在哪里，不是 Action Chunking 本身负责。

ACT 还需要视觉 backbone 和 [Transformer](../../deep-learning/transformer.md)。

---

### 无限长时序建模

\(k\) 太长时，序列本身反而更难预测，并且策略会越来越接近 open-loop。

---

# 23. 常见误解一：Action Chunking = 每 k 步才看一次环境

**不完整。**

这是论文用来介绍 Action Chunking 的 naive implementation。

最终 ACT：

> 每一个 timestep 都重新 query policy。

所以不能把 ACT 的最终执行方式简单写成：

```text
看一次
执行 100 步
再看一次
```

---

# 24. 常见误解二：Action Chunking 减少了机器人动作数量

**错误。**

如果轨迹仍然有：

\[
T
\]

个控制时间步，

机器人仍然需要执行：

\[
T
\]

个动作目标。

Action Chunking 改变的是：

> **策略如何组织和预测这些动作。**

---

# 25. 常见误解三：k 越大越好

**错误。**

论文 ablation 已经显示：

\[
k=1
\]

不好。

但 \(k\) 接近 episode length 也会下降。

因为：

```text
更大 k
→ effective horizon 更短
```

同时也会：

```text
更大 k
→ 序列更难预测
→ reactive behavior 更弱
→ 更接近 open-loop
```

所以存在 trade-off。

---

# 26. 常见误解四：Transformer 才让 Action Chunking 成立

**错误。**

Action Chunking 是一种更一般的策略设计。

只要模型能够输出：

\[
[B,k,D_a]
\]

原则上就可以实现 Action Chunking。

ACT 论文甚至在 BC-ConvMLP 和 VINN 上进行了相应扩展，并观察到类似的收益趋势。

---

# 27. 常见误解五：Action Chunking 本质只是“减少推理次数”

**错误。**

最终 ACT 结合 Temporal Ensemble 后，反而会在每个 timestep query policy。

所以 Action Chunking 更深层的意义在于：

> **改变策略预测的时间单位和输出结构。**

不是单纯为了省算力。

---

# 28. 用一句公式重新理解

普通行为克隆：

\[
\boxed{
\pi_\theta(a_t\mid s_t)
}
\]

问题是：

> 每次只描述下一瞬间。

---

Action Chunking：

\[
\boxed{
\pi_\theta(
a_t,a_{t+1},\ldots,a_{t+k-1}
\mid s_t
)
}
\]

变化是：

> **给定现在，直接描述接下来这一小段行为。**

这就是 Action Chunking 最核心的数学变化。

---

# 29. 从设计思想上重新看 ACT

Action Chunking 体现了一个非常重要的机器人学习思想：

> 有时问题不一定是“模型不够强”，而是“预测对象选错了”。

如果一直预测：

\[
\text{one action}
\]

那么即使换一个更大的网络，仍然在解决：

> 单步动作预测

这个问题。

ACT 首先改变的是：

\[
\text{prediction target}
\]

从：

\[
a_t
\]

变成：

\[
a_{t:t+k}
\]

然后才选择 Transformer 来建模这种序列结构。

因此理解顺序应该是：

```text
为什么单步预测不够？
        ↓
为什么要预测 action sequence？
        ↓
Action Chunking
        ↓
这种序列怎么建模？
        ↓
Transformer
```

而不是反过来：

```text
Transformer 很强
↓
所以输出很多动作
```

---

# 30. 下一步：为什么还需要 Temporal Ensemble？

现在我们已经理解 Action Chunking 的优势。

但它制造了一个新的矛盾：

```text
更长的 chunk
↓
更短的 effective horizon
↓
这是好事
```

可是：

```text
更长的 chunk
↓
如果整段执行完才重新观察
↓
反应能力下降
↓
这是坏事
```

ACT 的解决方式非常巧妙：

> **保留 action chunks，但每一步都重新预测。**

这样同一个真实 timestep 会同时拥有来自多个 chunk 的预测。

例如：

\[
\hat a_t^{(t-3)},
\hat a_t^{(t-2)},
\hat a_t^{(t-1)},
\hat a_t^{(t)}
\]

然后把它们组合起来。

这就是下一篇：

> [Temporal Ensemble：为什么同一个动作会有多个预测？](./temporal-ensemble.md)

---

# 31. 如果只记住一件事

> **Action Chunking 不是“让机器人一次执行很多动作”，而是把策略的预测单位从单个动作改成一段动作序列。**

它让模型直接学习：

\[
\text{当前状态}
\rightarrow
\text{未来短时间行为}
\]

从而缩短论文意义上的 effective horizon，并帮助建模短时间范围内的时间相关行为。

但 chunk 不能无限变长。

因为机器人仍然需要：

> **持续观察、持续修正。**

而这正是 Temporal Ensemble 接下来要解决的问题。

---

## Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html

本文关于 Action Chunking 的主要内容依据原论文：

- Section I — Introduction
- Section II — Related Work
- Section IV-A — Action Chunking and Temporal Ensemble
- Section VI-A — Action Chunking and Temporal Ensembling
- Figure 6
- Figure 9(a)

---

## 本文知识连接

### 前置知识

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [模仿学习（Imitation Learning）](../imitation-learning.md)
- [行为克隆（Behavior Cloning）](../imitation-learning/behavior-cloning-distribution-shift.md)

### 本文概念

- Markov Property

### 下一步

- [Temporal Ensemble](./temporal-ensemble.md)

### ACT 中的其他组件

- [Transformer](../../deep-learning/transformer.md)
- [CVAE](../../generative-models/cvae.md)
