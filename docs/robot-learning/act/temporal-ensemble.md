---
title: "Temporal Ensemble：为什么同一个动作会有多个预测？"
description: "严格解释 ACT 推理阶段的 Temporal Ensemble：重叠 Action Chunks 如何产生多个同一时刻预测、预测数量如何计算、指数权重如何定义，以及它为什么不同于普通时间平滑。"
status: reviewed
pageType: application
canonical: /robot-learning/act/temporal-ensemble
updated: "2026-09-15"
---

# Temporal Ensemble：为什么同一个动作会有多个预测？

在上一篇 [Action Chunking：为什么一次预测一段动作？](./action-chunking.md) 中，我们留下了一个矛盾：

> Action Chunking 希望一次预测更长的一段动作，从而缩短 effective horizon；  
> 但精细机器人操作又要求持续观察环境、持续修正。

如果最朴素地执行一个 Action Chunk：

```text
观察一次
↓
预测 k 个动作
↓
把 k 个动作全部执行完
↓
再观察
```

那么在这 $k$ 个控制步里，新视觉信息都无法进入策略。

这对于毫米级精度的任务并不理想。

ACT 的解决方法是：

> **Action Chunk 仍然保留，但推理时每一个 timestep 都重新 query policy。**

这样做之后，会产生一个看起来很奇怪的现象：

> **同一个真实执行时刻，会同时拥有多个动作预测。**

ACT 不丢掉这些预测，而是把它们组合起来。

这就是：

> **Temporal Ensemble（时间集成）**

---

## 1. 先回到 Action Chunking

为了统一符号，这篇文章规定：

一个长度为 $k$ 的 Action Chunk 为：

$$
\hat{\mathbf A}_t
=
(
\hat a_t^{(t)},
\hat a_{t+1}^{(t)},
\ldots,
\hat a_{t+k-1}^{(t)}
)
$$

这里上标和下标分别表达不同含义。

例如：

$$
\hat a_{t+2}^{(t)}
$$

表示：

> **在时间 $t$ 进行预测时，对真实执行时刻 $t+2$ 所预测的动作。**

为了避免混乱，请先牢牢记住：

- 下标：**这个动作要在哪一个 timestep 执行**
- 上标：**这个预测是在什么时候产生的**

所以：

$$
\hat a_{8}^{(5)}
$$

意思不是“第 5 个动作”。

而是：

> 在 $t=5$ 时，根据当时的观察，模型提前预测了 $t=8$ 应该执行什么。

---

## 2. Naive Action Chunking 为什么会有问题？

假设：

$$
k=4
$$

最朴素的 Action Chunking 会在 $t=0$ 预测：

$$
[
\hat a_0^{(0)},
\hat a_1^{(0)},
\hat a_2^{(0)},
\hat a_3^{(0)}
]
$$

然后：

```text
t = 0：执行 â₀⁽⁰⁾
t = 1：执行 â₁⁽⁰⁾
t = 2：执行 â₂⁽⁰⁾
t = 3：执行 â₃⁽⁰⁾
```

等到 $t=4$ 才重新观察并生成新 chunk。

问题是：

> 从 $t=0$ 到 $t=3$，环境可能已经发生变化，但策略无法利用这些新信息。

ACT 原论文把这个问题描述得很直接：

> 新环境 observation 每隔 $k$ 步才突然被纳入，会导致机器人运动不够平滑，甚至出现 jerky motion。

所以作者没有停在 naive Action Chunking。

---

## 3. ACT 的关键修改：每一步都重新预测

ACT 最终在推理阶段：

> **每一个 timestep 都 query policy。**

仍然假设：

$$
k=4
$$

---

### 在 $t=0$

模型看到观察 $o_0$，输出：

$$
[
\hat a_0^{(0)},
\hat a_1^{(0)},
\hat a_2^{(0)},
\hat a_3^{(0)}
]
$$

---

### 在 $t=1$

机器人已经执行了一步，获得新的观察 $o_1$。

模型再次运行：

$$
[
\hat a_1^{(1)},
\hat a_2^{(1)},
\hat a_3^{(1)},
\hat a_4^{(1)}
]
$$

---

### 在 $t=2$

再次得到新观察 $o_2$，模型重新预测：

$$
[
\hat a_2^{(2)},
\hat a_3^{(2)},
\hat a_4^{(2)},
\hat a_5^{(2)}
]
$$

---

### 在 $t=3$

模型又生成：

$$
[
\hat a_3^{(3)},
\hat a_4^{(3)},
\hat a_5^{(3)},
\hat a_6^{(3)}
]
$$

把它们排在一起：

```text
预测产生时刻       对未来的预测

t = 0        a₀   a₁   a₂   a₃
t = 1             a₁   a₂   a₃   a₄
t = 2                  a₂   a₃   a₄   a₅
t = 3                       a₃   a₄   a₅   a₆
```

你应该已经能看到 Temporal Ensemble 的来源了。

---

## 4. 为什么同一个动作会有多个预测？

现在只看真实执行时刻：

$$
t=3
$$

有多少个 chunk 对它做过预测？

从上面的表里可以看到：

$$
\hat a_3^{(0)}
$$

来自 $t=0$ 的 chunk；

$$
\hat a_3^{(1)}
$$

来自 $t=1$ 的 chunk；

$$
\hat a_3^{(2)}
$$

来自 $t=2$ 的 chunk；

$$
\hat a_3^{(3)}
$$

来自 $t=3$ 的最新 chunk。

因此，在真正要执行 $t=3$ 的动作时，我们手里其实有：

$$
\boxed{
\hat a_3^{(0)},
\hat a_3^{(1)},
\hat a_3^{(2)},
\hat a_3^{(3)}
}
$$

一共四个预测。

它们预测的是：

> **同一个执行时刻。**

只是它们来自不同时间的观察。

---

## 5. 这些预测各自“知道”的信息不同

这一点非常重要。

$$
\hat a_3^{(0)}
$$

是在 $t=0$ 时预测出来的。

它只知道：

$$
o_0
$$

而：

$$
\hat a_3^{(3)}
$$

是在 $t=3$ 才预测出来。

它已经看到了最新的：

$$
o_3
$$

所以不同预测之间最大的区别是：

> **它们依据的环境观察新旧不同。**

可以直觉地画成：

```text
o₀ ───────────────→ 预测 t=3
o₁ ──────────→      预测 t=3
o₂ ─────→            预测 t=3
o₃ →                  预测 t=3
```

越新的预测，使用了越新的 observation。

而较旧的预测，则来自更早规划出的 action chunk。

---

## 6. 到底有多少个预测？

这是学习 Temporal Ensemble 时最常见的问题之一。

答案取决于：

- chunk size $k$
- 当前处于 episode 的哪个 timestep

---

假设 episode 从：

$$
t=0
$$

开始。

每一次 query 都预测未来 $k$ 个动作：

$$
t,t+1,\ldots,t+k-1
$$

对于当前执行时刻 $t$，能够覆盖它的历史 query time 为：

$$
j\in
[
\max(0,t-k+1),\,t
]
$$

因此预测数量为：

$$
N_t
=
t-\max(0,t-k+1)+1
$$

进一步可以写成：

$$
\boxed{
N_t=\min(k,t+1)
}
$$

---

## 7. 用 $k=4$ 看预测数量

假设：

$$
k=4
$$

那么：

#### $t=0$

只有：

$$
\hat a_0^{(0)}
$$

所以：

$$
N_0=1
$$

---

#### $t=1$

有：

$$
\hat a_1^{(0)},\hat a_1^{(1)}
$$

所以：

$$
N_1=2
$$

---

#### $t=2$

有：

$$
\hat a_2^{(0)},
\hat a_2^{(1)},
\hat a_2^{(2)}
$$

所以：

$$
N_2=3
$$

---

#### $t=3$

有四个：

$$
\hat a_3^{(0)},
\hat a_3^{(1)},
\hat a_3^{(2)},
\hat a_3^{(3)}
$$

所以：

$$
N_3=4
$$

---

从这以后，在 episode 中间的大部分时间：

$$
N_t=k
$$

也就是说，当：

$$
k=100
$$

并且已经运行超过前 99 个 timestep 后，

理论上当前执行时刻最多可以拥有：

$$
100
$$

个来自不同 action chunks 的预测。

这正是 Temporal Ensemble 要融合的对象。

---

## 8. ACT 并不是“选一个最好预测”

面对：

$$
\hat a_t^{(j_1)},
\hat a_t^{(j_2)},
\ldots
$$

一种做法可以是：

> 永远使用最新预测。

也就是：

$$
a_t=\hat a_t^{(t)}
$$

这样能够最快使用最新 observation。

另一种做法可以是：

> 永远坚持最早的规划。

但 ACT 都没有这么做。

它采用：

> **加权平均。**

也就是把多个针对同一 timestep 的预测组合成真正执行的动作。

---

## 9. Temporal Ensemble 的正式公式

假设当前时间步 $t$ 有 $N_t$ 个预测。

把它们按照：

> **从最旧到最新**

排列为：

$$
A_t[0],A_t[1],\ldots,A_t[N_t-1]
$$

其中：

$$
A_t[0]
$$

是最旧的预测。

ACT 原论文定义指数权重：

$$
\boxed{
w_i=e^{-mi}
}
$$

其中：

- $i$：预测在“从旧到新”序列中的索引；
- $m>0$：控制权重衰减速度的超参数；
- $w_0$：最旧预测的权重。

真正执行的动作是：

$$
\boxed{
a_t
=
\frac{
\sum_{i=0}^{N_t-1}w_iA_t[i]
}{
\sum_{i=0}^{N_t-1}w_i
}
}
$$

这就是 Temporal Ensemble。

---

## 10. 为什么还要除以 $\sum_i w_i$？

先看没有归一化的情况：

$$
\sum_i w_iA_t[i]
$$

如果预测数量变化，权重总和也会变化。

例如 episode 开头只有两个预测：

$$
w_0+w_1
$$

运行稳定后可能有 $k$ 个预测：

$$
w_0+w_1+\cdots+w_{k-1}
$$

这样动作整体尺度会随着参与预测数量改变。

因此需要除以总权重：

$$
\sum_iw_i
$$

定义归一化权重：

$$
\alpha_i
=
\frac{w_i}
{\sum_jw_j}
$$

那么：

$$
\sum_i\alpha_i=1
$$

最终动作可以写成：

$$
a_t
=
\sum_i\alpha_iA_t[i]
$$

也就是一个标准的：

> **weighted average（加权平均）**

如果还不熟悉加权平均，可以阅读：

- Weighted Average

---

## 11. 一个完整数字例子

假设当前需要执行的是一个一维动作，例如：

$$
a_t=\text{某个关节目标角度}
$$

现在有 4 个预测：

$$
A_t=
[
1.00,\;
1.10,\;
0.95,\;
1.20
]
$$

从左到右：

> 最旧 → 最新。

假设：

$$
m=0.1
$$

则：

$$
w_0=e^0=1
$$

$$
w_1=e^{-0.1}\approx0.9048
$$

$$
w_2=e^{-0.2}\approx0.8187
$$

$$
w_3=e^{-0.3}\approx0.7408
$$

权重总和：

$$
W
=
1+0.9048+0.8187+0.7408
\approx3.4643
$$

归一化以后：

$$
\alpha
\approx
[
0.2887,\;
0.2612,\;
0.2363,\;
0.2138
]
$$

于是最终执行：

$$
a_t
=
0.2887(1.00)
+
0.2612(1.10)
+
0.2363(0.95)
+
0.2138(1.20)
$$

大约得到：

$$
a_t\approx1.057
$$

所以机器人不是执行某一个单独预测。

而是执行：

> **多个重叠 action chunks 对当前 timestep 的加权共识。**

---

## 12. 一个容易让人疑惑的地方：为什么最旧预测权重最高？

根据论文定义：

$$
w_i=e^{-mi}
$$

并且：

$$
w_0
$$

对应 **oldest action prediction**。

所以：

$$
w_0=1
$$

而对于：

$$
i>0
$$

有：

$$
w_i<1
$$

因此论文确实给：

> **较旧预测更高的权重。**

这一点很容易反直觉。

因为我们很自然会想：

> 最新预测看到了最新 observation，难道不应该最可信、权重最大吗？

---

## 13. 这里不要替论文发明不存在的理论解释

必须非常严谨地区分：

#### 论文明确规定的事实

ACT 使用：

$$
w_i=e^{-mi}
$$

并定义：

$$
w_0
$$

为最旧预测的权重。

所以权重随预测变新而指数减小。

---

#### 论文没有做的事情

ACT 论文并没有给出一个严格理论证明说：

> “为什么 oldest prediction 必然比 newest prediction 更正确。”

因此不应该自行编造类似：

> “因为旧预测一定更稳定”

或者：

> “因为旧预测长期规划能力更强”

然后把它写成论文结论。

这是不严谨的。

更准确的理解应该从：

> $m$ 如何控制“旧规划”和“新 observation”之间的平衡

来理解。

---

## 14. 参数 $m$ 到底控制什么？

权重为：

$$
w_i=e^{-mi}
$$

考虑两个极端。

---

### 当 $m$ 很大

例如：

$$
m=2
$$

则：

$$
w_0=1
$$

$$
w_1=e^{-2}\approx0.135
$$

$$
w_2=e^{-4}\approx0.018
$$

更新的预测权重很快衰减。

结果：

> 最旧的规划占据主导。

新 observation 产生的新预测很难迅速改变当前动作。

---

### 当 $m$ 很小

例如：

$$
m=0.01
$$

则：

$$
w_0=1
$$

$$
w_1\approx0.990
$$

$$
w_2\approx0.980
$$

$$
w_3\approx0.970
$$

各个预测的权重非常接近。

这意味着：

> 较新的 prediction 也能够获得接近旧预测的贡献。

因此论文写道：

> **smaller $m$ means faster incorporation of new observations**

也就是说：

$$
m\downarrow
$$

会让新预测相对于旧预测获得更大的相对权重。

---

## 15. 注意：“更快吸收新 observation”不等于“最新预测权重最大”

这是一个很细但很重要的区别。

ACT 的权重排列始终是：

```text
oldest
权重最大

↓

newest
权重较小
```

只要：

$$
m>0
$$

这个顺序都不会改变。

所谓：

> smaller $m$ → faster incorporation

意思是：

> 当 $m$ 变小时，权重下降得没那么快，因此 newer predictions 不再被强烈压低。

如果：

$$
m\to0
$$

那么：

$$
e^{-mi}\to1
$$

所有预测逐渐接近：

$$
w_i=1
$$

也就是趋向普通平均：

$$
a_t
\approx
\frac{1}{N_t}
\sum_iA_t[i]
$$

---

## 16. 原始官方实现中 $m$ 是多少？

ACT 官方开源代码中的 Temporal Ensemble 使用：

```python
k = 0.01
exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
exp_weights = exp_weights / exp_weights.sum()
```

这里代码变量叫：

```python
k
```

但注意：

> **这里的 `k = 0.01` 不是 Action Chunking 的 chunk size。**

它对应论文 Temporal Ensemble 公式中的衰减参数：

$$
m
$$

只是代码里的局部变量命名与论文符号不同。

因此阅读实现时一定不要把：

```python
k = 0.01
```

误认为：

> chunk size = 0.01

真正的 chunk size 在官方代码中使用：

```python
num_queries
```

等相关变量表示。

这是读源码时非常容易混淆的地方。

---

## 17. 官方代码实际上怎样保存这些预测？

ACT 官方实现建立了：

```python
all_time_actions
```

可以把它想象成一个二维时间表。

概念上：

```text
行 = prediction/query time
列 = action execution time
```

例如：

```text
                要执行的 timestep
            0    1    2    3    4    5

query t=0   a0   a1   a2   a3
query t=1        a1   a2   a3   a4
query t=2             a2   a3   a4   a5
query t=3                  a3   a4   a5   a6
```

当真正运行到时间：

$$
t=3
$$

代码取：

```python
actions_for_curr_step = all_time_actions[:, 3]
```

于是得到这一“列”里所有已经产生的预测：

```text
a₃ from query 0
a₃ from query 1
a₃ from query 2
a₃ from query 3
```

然后才做指数加权。

这个二维表是理解 Temporal Ensemble 最直观的方法之一。

---

## 18. 用矩阵写出来

仍然假设：

$$
k=4
$$

可以把所有预测组织为：

$$
\begin{bmatrix}
\hat a_0^{(0)} &
\hat a_1^{(0)} &
\hat a_2^{(0)} &
\hat a_3^{(0)} &
&\\
&
\hat a_1^{(1)} &
\hat a_2^{(1)} &
\hat a_3^{(1)} &
\hat a_4^{(1)}&
\\
&
&
\hat a_2^{(2)} &
\hat a_3^{(2)} &
\hat a_4^{(2)}&
\hat a_5^{(2)}
\\
&
&
&
\hat a_3^{(3)} &
\hat a_4^{(3)}&
\hat a_5^{(3)}
\end{bmatrix}
$$

每一行：

> 一个 Action Chunk。

每一列：

> 对同一个真实 timestep 的多个预测。

Temporal Ensemble 做的，就是：

> **沿着列进行聚合。**

而不是沿着行进行平均。

---

## 19. 为什么它不是普通的 Temporal Smoothing？

这个区别是整篇最重要的知识点之一。

普通时间平滑可能会把：

$$
a_{t-1},a_t,a_{t+1}
$$

这些**不同真实时刻**的动作进行平均。

例如：

$$
\tilde a_t
=
\frac{
a_{t-1}+a_t+a_{t+1}
}{3}
$$

这里被平均的是：

```text
昨天的动作
今天的动作
明天的动作
```

它们本来就可能应该不同。

所以这种 smoothing 会引入 temporal bias。

例如本来机器人需要快速转向：

```text
t-1：向左
t  ：转向
t+1：向右
```

如果直接平均相邻时间动作，就可能把真实需要的快速变化抹平。

---

## 20. Temporal Ensemble 平均的是“同一个时刻”

ACT 做的是：

$$
\hat a_t^{(t-k+1)},
\ldots,
\hat a_t^{(t)}
$$

注意所有下标都是：

$$
t
$$

也就是说：

> 这些预测全部在回答同一个问题：

> **“真实时刻 $t$，机器人应该执行什么动作？”**

因此它们在语义上是可以比较和聚合的。

ACT 原论文特别强调：

> unlike typical smoothing, we aggregate actions predicted for the same timestep.

这就是 Temporal Ensemble 和普通 temporal smoothing 最本质的区别。

---

## 21. 一个例子：为什么普通平滑可能有偏差？

假设机器人正在快速关闭夹爪。

真实理想动作：

$$
a_{t-1}=0.8
$$

$$
a_t=0.4
$$

$$
a_{t+1}=0.0
$$

这些数字可以理解为夹爪开度。

如果普通 smoothing 做：

$$
\frac{0.8+0.4+0}{3}=0.4
$$

碰巧这里等于 $a_t$。

但如果动作变化不是线性的：

$$
a_{t-1}=1.0,\qquad
a_t=0.1,\qquad
a_{t+1}=0
$$

平均变成：

$$
\frac{1+0.1+0}{3}
\approx0.367
$$

机器人真正需要：

$$
0.1
$$

却被平滑成：

$$
0.367
$$

快速动作被抹掉了。

Temporal Ensemble 则不会拿：

$$
a_{t-1}
$$

和：

$$
a_{t+1}
$$

来平均。

它只平均：

$$
\hat a_t^{(1)},
\hat a_t^{(2)},
\hat a_t^{(3)},\ldots
$$

这些针对同一个 $t$ 的不同预测。

---

## 22. Temporal Ensemble 发生在训练阶段吗？

**不发生。**

这是一个非常重要的结论。

ACT Training 的核心是：

```text
observation + demonstration action chunk
↓
CVAE
↓
predicted action chunk
↓
reconstruction loss + KL loss
```

训练时模型直接学习预测：

$$
\hat a_{t:t+k}
$$

并和 demonstration action chunk：

$$
a_{t:t+k}
$$

计算训练目标。

Temporal Ensemble **不参与 training loss**。

ACT 原论文明确说明：

> Temporal Ensemble does not incur additional training cost.

它只增加：

> **inference-time computation**

---

## 23. 为什么训练时不需要 Temporal Ensemble？

因为训练阶段处理的是：

> 从 demonstration dataset 中采样出的监督样本。

例如：

$$
(o_t,a_{t:t+k})
$$

模型的任务只是学习：

$$
o_t
\rightarrow
a_{t:t+k}
$$

而 Temporal Ensemble 解决的是 rollout 时出现的问题：

> 当策略每一步重新 query 后，多个 overlapping chunks 同时对当前 timestep 提供预测，该执行哪一个？

这是：

> **Inference aggregation problem**

不是：

> **Training objective problem**

所以它自然属于推理阶段。

---

## 24. 训练和推理流程对比

### Training

```text
Dataset
   │
   ├── oₜ
   └── ground-truth action chunk
            │
            ▼
           ACT
            │
            ▼
   predicted action chunk
            │
            ▼
    reconstruction loss
          +
       KL loss
```

没有 Temporal Ensemble。

---

### Inference

```text
t = 0
observation
↓
predict chunk

t = 1
new observation
↓
predict overlapping chunk

t = 2
new observation
↓
predict overlapping chunk

...

同一 timestep 出现多个预测
↓
Temporal Ensemble
↓
执行一个最终 action
```

这是两套不同的流程。

---

## 25. Temporal Ensemble 会增加什么代价？

因为最终 ACT：

> 每个 timestep 都 query policy

所以相比“预测一个 chunk 后把整块执行完”的 naive 方法，需要更多网络 forward。

论文明确指出：

> **没有额外训练成本，但会增加 inference-time computation。**

所以 Temporal Ensemble 不是：

> 免费获得平滑。

它是用更多 inference computation 换取：

- 高频 observation incorporation；
- 更平滑的动作；
- 多个模型预测之间的集成。

---

## 26. 它为什么可能让动作更平滑？

这里需要区分：

#### Paper Fact

论文实验发现 Temporal Ensemble 对 ACT 和 BC-ConvMLP 有提升，并认为它有助于产生 precise and smooth motion。

---

#### Intuition

一个单独 neural network prediction 可能存在小幅 modeling error：

```text
1.02
0.98
1.05
1.00
```

如果这些都是对同一真实 timestep 的合理估计，那么加权平均：

$$
\approx1.01
$$

可能减少单次预测中的小波动。

可以把它直觉理解成：

> **多个 overlapping plans 对当前动作进行一次 ensemble。**

但这只是理解方式。

不能由此推导出：

> averaging 在所有策略和所有任务上一定更好。

论文的 ablation 正好说明它并非总是如此。

---

## 27. 原论文的 Ablation 怎么说？

ACT 论文分别测试有无 Temporal Ensemble。

结果显示：

#### ACT

Temporal Ensemble 带来约：

$$
+3.3\%
$$

的平均成功率提升。

---

#### BC-ConvMLP

提升约：

$$
+4\%
$$

---

#### VINN

反而下降约：

$$
-20\%
$$

因此：

> Temporal Ensemble 并不是一个“无条件一定提高性能”的技巧。

---

## 28. 为什么 VINN 反而下降？

论文给出的解释是一个 **hypothesis（假设）**：

> Temporal Ensemble 主要帮助 parametric methods 平滑 modeling errors。

ACT 和 BC-ConvMLP 都是通过参数化模型：

$$
f_\theta(x)
$$

预测动作。

神经网络的连续预测可能带有 modeling error。

而 VINN 是非参数方法，主要从数据集中 retrieve ground-truth actions。

论文猜测：

> 因为 VINN 并不具有相同形式的模型预测误差，所以 Temporal Ensemble 对它不一定有益。

注意这里应写成：

> **论文作者的 hypothesis**

而不是已经被理论证明的结论。

---

## 29. Temporal Ensemble 与“多个模型投票”一样吗？

不完全一样。

传统 ensemble 往往是：

```text
Model 1
Model 2
Model 3
```

三个不同模型预测同一个输入。

而 ACT Temporal Ensemble 通常只有：

> **同一个 policy**

不同的是：

```text
不同 query time
+
不同 observation
```

产生的 overlapping predictions。

因此更准确地说，它是在 ensemble：

> **同一个模型在不同时间、依据不同 observation，对同一未来 timestep 所作的预测。**

---

## 30. Temporal Ensemble 与 Model Predictive Control 有点像吗？

从非常高层的直觉上，可以看到一点相似性：

```text
不断重新观察
↓
不断重新规划未来
↓
只执行当前需要的动作
```

但不要因此把 ACT Temporal Ensemble 直接等同于：

> Model Predictive Control

MPC 通常涉及：

- dynamics model；
- objective / cost；
- online optimization；
- receding horizon optimization。

ACT 的 policy 是：

> 从 demonstrations 学到的神经网络策略。

它直接预测 action chunks，并通过 Temporal Ensemble 聚合重叠预测。

因此：

> **可以类比“receding horizon 的感觉”，但理论机制不同。**

---

## 31. 再回答一次：到底有多少个预测？

现在我们可以给出严格答案。

如果：

- chunk size 为 $k$
- 每 timestep query 一次 policy
- 每个 query 输出从当前时刻开始的 $k$ 个动作

那么当前 timestep $t$ 最多有：

$$
\boxed{k}
$$

个预测。

episode 开始阶段：

$$
\boxed{
N_t=\min(k,t+1)
}
$$

所以：

```text
t = 0 → 1 个
t = 1 → 2 个
t = 2 → 3 个
...
t = k-1 → k 个
t ≥ k-1 → 通常 k 个
```

---

## 32. 再回答一次：权重从哪里来？

权重：

> **不是模型学出来的。**

不是 neural network 输出。

也不是 attention score。

也不是训练过程中优化出来的参数。

ACT 论文直接人工规定：

$$
\boxed{
w_i=e^{-mi}
}
$$

其中：

$$
m
$$

是一个人为选择的超参数。

然后归一化：

$$
\alpha_i
=
\frac{e^{-mi}}
{\sum_j e^{-mj}}
$$

最终：

$$
a_t=\sum_i\alpha_iA_t[i]
$$

所以 Temporal Ensemble 本质上是：

> **一个手工设计的 inference-time aggregation rule。**

---

## 33. 再回答一次：Temporal Ensemble 会参与 Loss 吗？

不会。

训练 loss 仍然是 ACT 的 CVAE objective：

$$
\mathcal L
=
\mathcal L_{\text{reconst}}
+
\beta\mathcal L_{\text{reg}}
$$

Temporal Ensemble：

$$
\boxed{\text{Inference only}}
$$

它不改变模型参数训练过程。

---

## 34. 用伪代码完整表示

假设已经训练好了 ACT policy。

```python
for t in range(T):

    # 1. 获取当前 observation
    observation = get_observation()

    # 2. 每个 timestep 都重新预测一个 action chunk
    chunk = policy(observation)

    # chunk:
    # [a_t, a_{t+1}, ..., a_{t+k-1}]

    # 3. 把每一个预测保存到对应的未来 timestep
    for offset in range(k):
        target_time = t + offset
        predictions[target_time].append(chunk[offset])

    # 4. 找到所有对当前 t 的历史预测
    current_predictions = predictions[t]

    # 5. oldest -> newest
    weights = [
        exp(-m * i)
        for i in range(len(current_predictions))
    ]

    # 6. normalize
    weights = weights / sum(weights)

    # 7. weighted average
    action = weighted_average(
        current_predictions,
        weights
    )

    # 8. 真正执行
    robot.step(action)
```

这段伪代码基本就是 Temporal Ensemble 的完整思想。

---

## 35. 整个过程再画一次

假设：

$$
k=4
$$

```text
t=0:
o₀ → policy
     └─ [a₀⁰, a₁⁰, a₂⁰, a₃⁰]
          │
          └──── execute a₀⁰


t=1:
o₁ → policy
     └─ [a₁¹, a₂¹, a₃¹, a₄¹]

现在 t=1 有：
a₁⁰
a₁¹
  ↓
weighted average
  ↓
execute a₁


t=2:
o₂ → policy
     └─ [a₂², a₃², a₄², a₅²]

现在 t=2 有：
a₂⁰
a₂¹
a₂²
  ↓
weighted average
  ↓
execute a₂


t=3:
o₃ → policy
     └─ [a₃³, a₄³, a₅³, a₆³]

现在 t=3 有：
a₃⁰
a₃¹
a₃²
a₃³
  ↓
weighted average
  ↓
execute a₃
```

从这里开始，只要处在 episode 中间且 chunk 都是完整的，当前 timestep 通常维持：

$$
k
$$

个候选预测。

---

## 36. Temporal Ensemble 真正解决了什么矛盾？

现在可以回到最初的问题。

Action Chunking 希望：

```text
一次预测更长的未来
↓
缩短 effective horizon
```

但精细控制又希望：

```text
每一步都看最新环境
↓
及时修正
```

如果只选其中一个：

#### 只做长 chunk

容易变得接近 open-loop。

#### 只做单步

又失去 Action Chunking 的优势。

ACT 的 Temporal Ensemble 让两者可以同时存在：

```text
Action Chunking
→ 保留长一些的局部行为规划

+

Every-step re-query
→ 每一步使用新 observation

+

Temporal Ensemble
→ 不突然丢弃旧 chunk，而是平滑融合
```

所以它的真正作用不是单纯：

> “让曲线更平滑。”

而是：

> **在保留 Action Chunking 的同时，把高频闭环反馈重新引入执行过程。**

---

## 37. 一个非常重要的认识：Temporal Ensemble 没有改变模型学什么

模型训练时仍然学习：

$$
o_t
\rightarrow
a_{t:t+k}
$$

Temporal Ensemble 并没有改变这个 mapping。

它改变的是：

> **模型预测出来以后，我们如何决定当前真正执行哪一个动作。**

因此可以把 ACT 分成两层：

### Policy Learning

学习：

$$
\pi_\theta(a_{t:t+k}\mid o_t)
$$

---

### Execution Rule

从 overlapping chunks 中计算：

$$
a_t
$$

Temporal Ensemble 属于第二层。

---

## 38. 常见误解一：训练时一个动作会计算多个 loss

**错误。**

Temporal Ensemble 是 inference-only。

训练时，从一个 observation 预测一个 action chunk，然后直接与 demonstration chunk 计算 reconstruction loss。

不会因为某个 action 在多个训练 chunk 中出现，就在同一次 forward 里做 Temporal Ensemble。

数据集中 overlapping chunks 和推理阶段 Temporal Ensemble 是两个不同概念。

---

## 39. 常见误解二：预测数量是提前固定好的一个超参数

**不完全正确。**

最大的预测数量由：

$$
k
$$

决定。

但 episode 开头实际上只有：

$$
1,2,3,\ldots
$$

个。

因此当前真正参与 ensemble 的数量：

$$
N_t
$$

是随 timestep 变化的，直到达到 $k$。

---

## 40. 常见误解三：Temporal Ensemble 平均前后几个 timestep

**错误。**

它不是：

$$
a_{t-1},a_t,a_{t+1}
$$

之间的平均。

它是：

$$
\hat a_t^{(t-k+1)},
\ldots,
\hat a_t^{(t)}
$$

之间的平均。

所有 prediction 都对应：

$$
\boxed{\text{same timestep }t}
$$

---

## 41. 常见误解四：权重是 Attention 学出来的

**错误。**

Temporal Ensemble 权重由手工公式：

$$
e^{-mi}
$$

计算。

与 Transformer 的 attention weight 不是一个东西。

---

## 42. 常见误解五：最新 prediction 权重最大

按照 ACT 原论文：

**不是。**

论文定义：

$$
w_0
$$

为 oldest prediction 的权重。

并且：

$$
w_i=e^{-mi}
$$

所以 $m>0$ 时：

$$
w_0>w_1>w_2>\cdots
$$

也就是旧预测权重更高。

---

## 43. 常见误解六：越大的 $m$ 越快使用新 observation

**恰好相反。**

当：

$$
m\uparrow
$$

权重随 $i$ 衰减更快。

较新的 prediction 被压得更低。

论文明确说明：

$$
m\downarrow
$$

意味着：

> faster incorporation of new observations.

---

## 44. 常见误解七：Temporal Ensemble 没有计算成本

**错误。**

它没有：

> additional training cost

但有：

> extra inference-time computation

因为 policy 从“每隔一个 chunk query”变成：

> 每 timestep 都 query。

---

## 45. 如果只记住一件事

> **Temporal Ensemble 不是把相邻时间的动作做平滑，而是把多个 overlapping action chunks 对“同一个执行时刻”的预测进行指数加权平均。**

ACT 在每个 timestep 都重新观察并预测一个新的 action chunk。

因此：

$$
\text{旧 observation 产生的预测}
$$

和：

$$
\text{新 observation 产生的预测}
$$

会同时存在。

Temporal Ensemble 用：

$$
w_i=e^{-mi}
$$

把它们融合成真正执行的：

$$
a_t
$$

从而在：

> **Action Chunking 的时间抽象**

和：

> **高频闭环反馈**

之间建立连接。

---

## 46. 下一步

现在我们已经理解 ACT 的两个执行层核心设计：

```text
Action Chunking
+
Temporal Ensemble
```

但还有另一个完全不同的问题没有解决：

> **同一个 observation 下，人类可能有多种合理的 action sequence。**

如果直接用一个确定性网络去平均这些 demonstration，会发生什么？

为什么 ACT 要引入一个 latent variable：

$$
z
$$

为什么训练时要有：

$$
\mu,\sigma^2
$$

为什么还要加入：

$$
D_{KL}
$$

这就是下一条知识链：

1. [Latent Variable](../../generative-models/latent-variable.md)
2. [VAE](../../generative-models/vae.md)
3. [CVAE](../../generative-models/cvae.md)
4. [CVAE in ACT](./cvae-in-act.md)

在真正进入 CVAE 之前，建议先把最基础的概率概念补齐：

- [Normal Distribution](../../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean
- Variance

---

### Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html

本文关于 Temporal Ensemble 的核心技术事实依据：

- Section IV-A — Action Chunking and Temporal Ensemble
- Algorithm 2 — ACT Inference
- Figure 5
- Section VI-A — Action Chunking and Temporal Ensembling
- Figure 8(b)

---

### Official Implementation

ACT 官方代码：

- Repository: https://github.com/tonyzhaozh/act
- Inference implementation: `imitate_episodes.py`

官方实现中：

```python
query_frequency = 1
```

在启用 Temporal Ensemble 后，每个 timestep 都 query policy。

核心权重实现为：

```python
exp_weights = np.exp(
    -0.01 * np.arange(len(actions_for_curr_step))
)

exp_weights = exp_weights / exp_weights.sum()
```

代码中的候选预测按照 query time 从旧到新排列，因此与论文中：

$$
w_0=\text{oldest prediction weight}
$$

的定义一致。

---

### 本文知识连接

#### 前置知识

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- Weighted Average

#### ACT 核心组件

- [ACT Architecture](./architecture.md)
- [CVAE in ACT](./cvae-in-act.md)

#### 后续知识

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [CVAE](../../generative-models/cvae.md)

#### 相关概念

- Model Predictive Control
