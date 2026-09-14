---
title: "ACT 到底解决了什么问题？"
description: "从模仿学习的累积误差、人类示范的非平稳性，到 Action Chunking、Temporal Ensemble 与 CVAE：基于 ACT 原论文理解它为什么被提出。"
status: reviewed
pageType: paper
canonical: /robot-learning/act/act-what-problem-does-it-solve
updated: "2026-09-15"
---

# ACT 到底解决了什么问题？

> **Action Chunking with Transformers（ACT）不是“把 Transformer 用到机器人上”这么简单。**
>
> 它真正要解决的是：当机器人通过人类示范学习精细操作时，**单步动作预测会让小误差不断累积，而人类示范本身又不是完全确定、完全平稳的。**
>
> ACT 的几个核心设计——Action Chunking、Temporal Ensemble、CVAE、Transformer——都是围绕这些问题展开的。

---

### 1. 先别急着看网络结构：ACT 为什么会出现？

ACT 来自 2023 年 RSS 论文：

> **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**  
> Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn

这篇论文不仅提出了 ACT，还提出了低成本双臂遥操作平台 ALOHA。

论文关心的不是“让机器人抓起一个方块”这种相对宽松的任务，而是更精细的双臂操作，例如：

- 把电池插进卡槽；
- 打开透明调料杯的盖子；
- 穿扎带；
- 操作薄膜、袋子等柔性物体。

这些任务有一个共同点：

> **毫米级的小误差，都可能让任务失败。**

因此机器人必须不断根据视觉反馈修正动作，而不能只靠一套提前算好的固定轨迹。

这也是为什么 ACT 的问题背景，本质上是：

> **高精度、闭环、长时间序列的模仿学习。**

如果你还不熟悉什么是模仿学习，可以先阅读：

- [模仿学习（Imitation Learning）](../imitation-learning.md)
- [行为克隆（Behavior Cloning）](../imitation-learning/behavior-cloning-distribution-shift.md)

---

## 2. 最简单的机器人模仿学习：一步一步预测动作

先假设我们已经收集了一批人类示范。

在每个时间步 $t$，数据中都有：

- 当前观察 $s_t$
- 人类在这个时刻执行的动作 $a_t$

最简单的 [行为克隆（Behavior Cloning）](../imitation-learning/behavior-cloning-distribution-shift.md) 会学习一个策略：

$$
\pi_\theta(a_t \mid s_t)
$$

这里：

- $s_t$：机器人当前看到和感知到的状态；
- $a_t$：当前时间步应该执行的动作；
- $\pi_\theta$：参数为 $\theta$ 的策略模型。

可以先把它理解成：

> **“看到现在的情况，然后预测下一步应该怎么动。”**

例如机器人要把电池插进卡槽：

```text
看到电池还在卡槽左边
        ↓
预测下一步：向右移动一点
        ↓
执行
        ↓
重新观察
        ↓
再预测下一步
```

看起来非常合理。

甚至直觉上会觉得：

> 每一步都重新看环境、重新决定动作，不应该最稳吗？

问题恰恰出现在这里。

---

## 3. 问题一：一步错，不只是错一步

假设人类示范中的理想轨迹是：

```text
s₀ → s₁ → s₂ → s₃ → s₄ → ...
```

训练时，模型在 $s_0$ 上学习：

$$
s_0 \rightarrow a_0
$$

在 $s_1$ 上学习：

$$
s_1 \rightarrow a_1
$$

在 $s_2$ 上学习：

$$
s_2 \rightarrow a_2
$$

训练数据里的这些状态，都是人类正确执行任务时真正到过的状态。

但推理时，机器人自己控制自己。

假设在第一步，模型没有完全预测正确：

$$
\hat a_0 \neq a_0
$$

那么机器人执行 $\hat a_0$ 后，下一时刻到达的状态可能不是训练数据里的 $s_1$，而是：

$$
\hat s_1
$$

于是下一次模型面对的输入变成：

$$
\pi_\theta(a_1 \mid \hat s_1)
$$

问题在于：

> **模型可能从来没有在训练中见过 $\hat s_1$。**

于是第二步更容易预测错误。

第二步的错误又会把机器人推到更陌生的状态：

$$
\hat s_2
$$

然后：

$$
\hat s_3,\hat s_4,\ldots
$$

误差就可能像这样传播：

```text
一个很小的动作误差
        ↓
状态偏离人类示范
        ↓
模型进入不熟悉的状态
        ↓
预测变得更不可靠
        ↓
状态继续偏离
        ↓
误差越来越大
```

这就是模仿学习中的经典问题：

> **累积误差（Compounding Error）**

ACT 论文明确把它视为精细操作中的核心困难之一。

---

## 4. 为什么精细操作尤其怕累积误差？

如果任务只是：

> “把手伸到桌子另一边。”

几毫米误差可能无所谓。

但如果任务是：

> “让一根手指从杯盖下面插进去，然后轻轻撬开。”

那么几毫米可能意味着：

```text
正确：
手指进入杯盖下方
        ↓
成功撬起

错误：
手指偏了 4 mm
        ↓
碰到杯壁
        ↓
杯子移动
        ↓
下一帧位置更不一样
        ↓
后续动作全部开始偏离
```

ACT 原论文在介绍精细操作时直接强调：

> millimeters of error could lead to task failure.

所以这里的关键不是：

> 模型某一步预测差了多少。

而是：

> **这个误差会不会改变后面的状态，从而影响后面所有预测。**

---

## 5. Horizon：为什么“走得越长”，累积误差越危险？

假设一个任务需要 $T$ 个时间步。

普通单步策略相当于连续做：

$$
T
$$

次预测决策：

$$
\pi_\theta(a_1|s_1),
\pi_\theta(a_2|s_2),
\ldots,
\pi_\theta(a_T|s_T)
$$

这里的 $T$ 可以理解为任务的 **horizon（时域长度）**。

如果机器人控制频率很高，例如几十 Hz，那么一个只有几秒钟的动作，也可能包含几百个时间步。

时间步越多：

> **前面产生的小误差，就有越多机会继续影响后面。**

于是 ACT 的第一个核心问题变成：

> **能不能减少这条长轨迹中“需要独立决定”的次数？**

这就引出了 Action Chunking。

---

## 6. ACT 的第一个核心思想：不要只预测下一步

普通行为克隆学习：

$$
\pi_\theta(a_t \mid s_t)
$$

ACT 改成预测一段未来动作：

$$
\pi_\theta(a_{t:t+k} \mid s_t)
$$

这里的：

$$
a_{t:t+k}
$$

表示从当前时刻开始的一段动作序列。

为了直观理解，假设：

$$
k = 4
$$

普通单步策略：

```text
当前观察 sₜ
   ↓
预测 aₜ
```

ACT：

```text
当前观察 sₜ
   ↓
预测
[aₜ, aₜ₊₁, aₜ₊₂, aₜ₊₃]
```

这一串动作就叫：

> **Action Chunk（动作块 / 动作分块）**

详细原理参见：

- [Action Chunking](./action-chunking.md)

---

## 7. 为什么预测一串动作能够缓解累积误差？

先考虑最朴素的 Action Chunking。

假设一个任务原来需要：

$$
T
$$

个时间步。

如果每次预测 $k$ 个动作，并把这一整块视为一次高层决策，那么有效决策长度大约从：

$$
T
$$

缩短为：

$$
\frac{T}{k}
$$

这就是论文所说的：

> **effective horizon 被缩短 $k$ 倍。**

注意这里说的是：

> **effective horizon**

不是说机器人真的少执行了 $k$ 倍的物理动作。

机器人仍然要执行每一个关节目标。

变化的是：

> 原来模型每次只预测一步，现在一次预测一段具有时间结构的动作。

例如原来：

```text
观察
→ 预测第 1 步

观察
→ 预测第 2 步

观察
→ 预测第 3 步

观察
→ 预测第 4 步
```

现在可以先理解成：

```text
观察
→ 一次预测 [第1步, 第2步, 第3步, 第4步]
```

这样模型学习的就不再只是：

> “下一瞬间应该怎么动？”

而是：

> **“从当前情况出发，接下来这一小段动作应该怎样展开？”**

---

## 8. Action Chunk 不是随便把几个动作绑在一起

这一点很重要。

Action Chunking 的价值并不只是：

> 减少调用模型的次数。

它还改变了模型学习的对象。

例如一个“抓住包装袋角落”的动作，可能包含：

```text
手靠近
↓
调整方向
↓
夹爪闭合
↓
稍微向上提
```

如果只做单步预测，模型分别学习这些瞬间动作。

而预测 Action Chunk 时，模型可以一次看到并生成：

> **具有连续时间结构的一整小段行为。**

论文还指出，这有助于处理人类示范中的某些 **non-Markovian behavior**，例如示范过程中出现具有时间相关性的停顿。

如果你还不熟悉 Markov / non-Markov 的区别，后续可以阅读：

- Markov Property

---

## 9. 但这里出现了一个新的问题

如果我们真的按照最朴素的方法：

```text
观察一次
↓
预测 k 步
↓
连续执行 k 步
↓
再观察
```

会发生什么？

假设：

$$
k=100
$$

那么机器人可能在这一段执行期间都不重新利用新的视觉观察。

但 ACT 原本面对的恰恰是：

> **需要闭环视觉反馈的精细操作。**

这就产生矛盾：

```text
Action Chunking
想一次规划更长的动作
        ↕

精细操作
又需要不断吸收新观察
```

因此：

> **“每 $k$ 步才看一次环境”只是最朴素的 Action Chunking，并不是 ACT 最终的执行方式。**

这是理解 ACT 时非常容易搞错的一点。

---

## 10. ACT 最终不是“预测 k 步，然后闭眼执行 k 步”

ACT 论文明确指出，朴素 Action Chunking 会有一个问题：

> 每隔 $k$ 步才突然加入一次新的环境观察，动作容易发生不连续和抖动。

因此 ACT 最终在推理时：

> **每一个 timestep 都重新 query policy。**

假设：

$$
k=4
$$

在 $t=0$：

```text
预测：

a₀⁽⁰⁾
a₁⁽⁰⁾
a₂⁽⁰⁾
a₃⁽⁰⁾
```

到了 $t=1$，机器人重新观察环境，再预测：

```text
a₁⁽¹⁾
a₂⁽¹⁾
a₃⁽¹⁾
a₄⁽¹⁾
```

到了 $t=2$：

```text
a₂⁽²⁾
a₃⁽²⁾
a₄⁽²⁾
a₅⁽²⁾
```

注意现在发生了一件很有意思的事情。

对于时间步 $t=2$，我们已经拥有多个预测：

$$
a_2^{(0)},\quad
a_2^{(1)},\quad
a_2^{(2)}
$$

它们都是：

> **对同一个真实执行时刻 $t=2$ 的预测。**

区别只是这些预测来自不同时间产生的 Action Chunk。

于是 ACT 不需要强行只选其中一个。

它可以把这些预测结合起来。

这就是：

> **Temporal Ensemble（时间集成）**

详细内容参见：

- [Temporal Ensemble](./temporal-ensemble.md)

---

## 11. Temporal Ensemble 解决的是什么？

ACT 对同一个 timestep 的多个动作预测进行加权平均。

形式上，可以写成：

$$
a_t =
\frac{
\sum_i w_i \hat a_t^{(i)}
}{
\sum_i w_i
}
$$

其中论文采用指数权重：

$$
w_i = e^{-mi}
$$

这里：

- $\hat a_t^{(i)}$：对同一时间步 $t$ 的第 $i$ 个预测；
- $w_i$：该预测的权重；
- $m$：控制权重衰减速度的超参数。

需要特别注意：

> **这不是把相邻时间步 $a_{t-1},a_t,a_{t+1}$ 做平滑。**

ACT 聚合的是：

> **多个不同 Action Chunk 对“同一个 timestep”作出的预测。**

这是非常重要的区别。

因此 ACT 可以同时获得两件事：

#### Action Chunking

让模型学习短时间动作序列，而不是完全独立的单步动作。

#### Temporal Ensemble

又让新的观察能够在每个 timestep 被重新纳入决策。

于是：

```text
长一点的动作结构
+
持续闭环修正
```

可以同时存在。

---

## 12. 问题二：同一个状态，人类不一定只有一种“正确动作”

到这里还没有解决另一个问题。

假设两个人都在完成同一个任务。

当机器人看到几乎相同的场景时，人类 A 可能：

```text
先略微向左
↓
再向前
```

人类 B 可能：

```text
先向前
↓
再略微调整
```

两条轨迹都能成功。

甚至同一个人重复做两次，动作也不完全一样。

这意味着：

$$
s_t
$$

并不一定只对应唯一的：

$$
a_{t:t+k}
$$

更准确地说：

> **人类 demonstration 本身具有 variability。**

ACT 原论文还强调，人类在“不需要高精度”的区域会表现得更随机。

所以如果我们强迫一个普通确定性网络：

```text
同一个 observation
↓
只能输出唯一 action chunk
```

它可能会把不同合理轨迹“平均”到一起。

而轨迹的平均：

> 不一定仍然是一条合理轨迹。

---

## 13. ACT 为什么引入 CVAE？

为了解决人类示范中的这种变化，ACT 没有把策略单纯训练成一个确定性的回归网络。

论文把 Action Chunking policy 训练成：

> **Conditional Variational Autoencoder（CVAE）**

也就是一个条件生成模型。

可以先把它粗略理解为：

```text
当前 observation
+
一个 latent style variable z
        ↓
生成未来 Action Chunk
```

这里的 $z$ 用于承载 demonstration 中某些变化因素。

例如可以直觉地理解成：

> **“这一段动作准备以哪一种合理方式完成？”**

但必须注意：

> 这只是帮助理解的直觉，不是 $z$ 的严格数学定义。

要真正理解它，需要继续学习：

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [CVAE](../../generative-models/cvae.md)
- [Normal Distribution](../../mathematics/normal-distribution.md)
- [KL Divergence](../../mathematics/kl-divergence.md)

ACT 原论文中的 CVAE 细节，我们会在：

- [CVAE in ACT](./cvae-in-act.md)

单独讨论。

---

## 14. 那 Transformer 又在 ACT 里做什么？

到这里，其实 ACT 的核心问题已经大致出现了：

```text
输入：
当前视觉
当前关节状态

输出：
未来一段动作序列
```

这里天然涉及：

> **序列建模。**

因此 ACT 使用 [Transformer](../../deep-learning/transformer.md) 来实现主要序列建模模块。

论文中：

- CVAE encoder 使用 BERT-like Transformer encoder；
- policy / CVAE decoder 中使用图像编码器、Transformer encoder 和 Transformer decoder；
- 最终预测未来 $k$ 个动作。

但请注意：

> Transformer 不是 ACT 最核心的“问题来源”。

ACT 并不是因为：

> “Transformer 很强，所以把 Transformer 用到机器人。”

才产生的。

正确的逻辑顺序是：

```text
精细模仿学习
↓
累积误差严重
↓
需要 Action Chunking
↓
需要处理序列输入 / 序列输出
↓
Transformer 是合适的实现架构
```

同时：

```text
人类示范具有变化性
↓
需要生成式建模
↓
引入 CVAE
```

这两个方向最后组合成：

> **Action Chunking with Transformers**

---

## 15. 所以 ACT 到底解决了几个问题？

可以把论文动机压缩成三个主要问题。

### 问题一：Compounding Error

单步模仿学习中的误差会改变未来状态，使机器人逐渐离开训练分布。

ACT 的主要应对：

> **Action Chunking**

通过预测动作序列缩短有效 horizon。

---

### 问题二：Action Chunk 与闭环控制之间的矛盾

如果每次预测 $k$ 步后全部执行完再重新观察：

> 新观察进入得太慢，而且动作可能不够平滑。

ACT 的应对：

> **Temporal Ensemble**

每个 timestep 都重新预测 Action Chunk，再融合对同一 timestep 的重叠预测。

---

### 问题三：Human Demonstration Variability

同一个 observation 可能对应多种合理的人类动作轨迹。

ACT 的应对：

> **CVAE**

将 Action Chunk policy 建模为条件生成模型。

---

最后 Transformer 负责：

> **实现这些序列信息的建模和动作序列生成。**

可以总结成：

```text
                ACT
                 │
        ┌────────┼────────┐
        │        │        │
        ▼        ▼        ▼
 Compounding   Human    Closed-loop
    Error    Variability  Smoothness
        │        │        │
        ▼        ▼        ▼
   Action      CVAE    Temporal
   Chunking            Ensemble
        \        |        /
         \       |       /
          └─ Transformer ┘
```

注意这张图是帮助理解的概念图，不是论文原始架构图。

---

## 16. ACT 的完整输入和输出是什么？

暂时忽略 CVAE encoder 的训练细节。

在真正控制机器人时，可以先把 ACT policy 看成：

$$
\pi_\theta(
\hat a_{t:t+k}
\mid
o_t,z
)
$$

其中：

$$
o_t =
(\text{images}_t,\text{joint positions}_t)
$$

输入包括：

#### 视觉观察

论文中的 ALOHA 系统使用多个相机视角。

#### 当前机器人关节位置

告诉模型机器人当前姿态。

#### latent style variable $z$

用于 CVAE 的生成建模。

输出：

$$
\hat a_{t:t+k}
$$

也就是：

> **未来一段 target joint positions。**

论文中的一个 action 对应：

> 下一时间步两个机械臂的 target joint positions。

这些 target positions 最后交给机器人底层高频 PID controller 去跟踪。

因此 ACT 本身并不是：

> 直接输出电机电流。

这一点也要区分清楚。

---

## 17. 训练和推理并不完全一样

ACT 的一个重要特点是：

> CVAE encoder 只在训练阶段存在。

训练时：

```text
当前 observation
+
示范中的未来 action sequence
        ↓
CVAE Encoder
        ↓
z
        ↓
Policy / Decoder
        ↓
重建未来 action sequence
```

而推理时：

```text
CVAE Encoder
    ✕
被丢弃
```

论文直接令：

$$
z=0
$$

也就是 prior distribution 的均值，然后确定性地进行解码。

为什么这样仍然能够工作，是理解 ACT 时非常关键的问题。

我们会在：

- [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)

单独完整推导。

这里先不要把 $z=0$ 理解成：

> “latent information 没用了。”

真正的原因与：

- [VAE](../../generative-models/vae.md)
- Standard Normal Distribution
- [KL Divergence](../../mathematics/kl-divergence.md)

密切相关。

---

## 18. 一个常见误解：ACT = 一次预测很多动作

现在可以看出，这个说法只说对了一小部分。

如果只有：

> 一次预测 $k$ 个动作

那只是 Action Chunking。

完整 ACT 还包括至少三层关键设计：

```text
Action Chunking
      +
Temporal Ensemble
      +
CVAE
      +
Transformer implementation
```

其中每一层都在解决不同的问题。

因此理解 ACT 最好不要记：

> ACT = Transformer 输出 action chunk。

而应该记：

> **ACT 是针对高精度模仿学习中的累积误差和人类示范变化性设计的一套 action-sequence policy；Action Chunking 缩短有效 horizon，Temporal Ensemble 保持闭环和平滑，CVAE 建模示范中的变化，而 Transformer 实现序列信息的融合与生成。**

---

## 19. 另一个常见误解：Action Chunking 就是不频繁观察

不准确。

论文首先用“每 $k$ 步观察一次、预测并执行 $k$ 步”解释最朴素的 Action Chunking。

但随后作者明确指出：

> 这种 naive implementation 会导致新的 observation 每 $k$ 步才突然进入系统，从而产生 jerky motion。

因此 ACT 最终推理时：

> **每一步都 query policy。**

Action Chunk 之间产生重叠。

再通过 Temporal Ensemble 聚合。

所以不要把最终 ACT 推理理解成：

```text
看一眼
↓
闭眼执行 100 步
↓
再看一眼
```

它实际上仍然是：

> **持续吸收新 observation 的闭环策略。**

---

## 20. ACT 最值得学习的地方其实不是 Transformer

如果第一次学习 ACT，很容易把注意力放在：

- Transformer encoder；
- Transformer decoder；
- [CLS]；
- positional embedding；
- ResNet；
- CVAE encoder。

这些当然重要。

但从算法思想上看，更值得先理解的是：

> **作者是怎么从问题一步一步走到这些设计的。**

逻辑是：

```text
Fine-grained imitation learning
        ↓
单步 BC 累积误差明显
        ↓
缩短 effective horizon
        ↓
Action Chunking
        ↓
但 naive chunking 降低反馈频率
        ↓
每 timestep 重新预测
        ↓
Action chunks overlap
        ↓
Temporal Ensemble
```

同时另一条线：

```text
Human demonstrations are variable
        ↓
确定性回归不够理想
        ↓
Generative modeling
        ↓
CVAE
```

然后再问：

> 什么架构适合融合观察并输出动作序列？

于是：

> Transformer

才真正进入故事。

这也是后续学习 ACT 网络结构时最重要的主线。

---

## 21. 下一步：真正拆开 ACT

现在我们已经知道：

> **ACT 为什么存在。**

接下来可以分别学习它的几个关键组成。

建议顺序：

1. [行为克隆（Behavior Cloning）](../imitation-learning/behavior-cloning-distribution-shift.md)
2. [Action Chunking](./action-chunking.md)
3. [Temporal Ensemble](./temporal-ensemble.md)
4. [Transformer](../../deep-learning/transformer.md)
5. [VAE](../../generative-models/vae.md)
6. [CVAE](../../generative-models/cvae.md)
7. [CVAE in ACT](./cvae-in-act.md)
8. [ACT Architecture](./architecture.md)
9. [ACT Training](./training.md)
10. [ACT Inference](./inference.md)

如果你现在还没有完全理解 CVAE，不需要停在这里。

这篇文章最重要的是先建立 ACT 的问题地图：

```text
为什么预测 action chunk？
为什么需要 temporal ensemble？
为什么要用 CVAE？
Transformer 在这里承担什么角色？
```

有了这张地图，再进入网络内部时就不会变成：

> “这里突然出现一个 Transformer，这里又突然出现一个 $z$，这里又突然多了一个 KL loss。”

---

## 22. 如果只记住一件事

> **ACT 的核心不是“用 Transformer 预测多步动作”，而是重新设计模仿学习中的动作预测单位：从独立的单步动作变成动作序列，再用 Temporal Ensemble 保留持续闭环反馈，并用 CVAE 处理人类示范的变化性。**

Transformer 是实现这一策略的重要架构。

但：

> **问题驱动设计，才是理解 ACT 的真正入口。**

---

### Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html
- Project: https://tonyzhaozh.github.io/aloha/

本文关于 ACT 的主要技术事实均以该论文的 **Section I、Section II、Section IV 以及 Algorithm 1 / Algorithm 2** 为基础。

---

### 本文涉及的知识节点

#### 前置知识

- [模仿学习（Imitation Learning）](../imitation-learning.md)
- [行为克隆（Behavior Cloning）](../imitation-learning/behavior-cloning-distribution-shift.md)

#### ACT 核心组件

- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)
- [ACT Architecture](./architecture.md)
- [CVAE in ACT](./cvae-in-act.md)

#### 深度学习

- [Transformer](../../deep-learning/transformer.md)
- [Attention](../../deep-learning/attention.md)

#### 生成模型

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [CVAE](../../generative-models/cvae.md)

#### 数学

- [Normal Distribution](../../mathematics/normal-distribution.md)
- Standard Normal Distribution
- [KL Divergence](../../mathematics/kl-divergence.md)
- Markov Property
