---
title: "ACT Inference：机器人每一个 timestep 到底发生什么？"
description: "基于 ACT 原论文 Algorithm 2 与官方实现，完整拆解部署时 observation normalization、z=0、action-chunk prediction、every-step re-query、Temporal Ensemble、反归一化与机器人执行的闭环流程。"
status: reviewed
pageType: application
canonical: /robot-learning/act/inference
updated: "2026-09-15"
---

# ACT Inference：机器人每一个 timestep 到底发生什么？

训练完成后，ACT 真正部署到机器人上时，每一个控制 timestep 都会经历一条完整闭环：

```text
读取当前 observation
↓
normalize qpos
↓
读取当前多路 camera images
↓
z = 0
↓
ACT policy forward
↓
得到未来 k 个 actions
↓
保存这个 action chunk
↓
收集所有“针对当前 timestep”的历史预测
↓
Temporal Ensemble
↓
得到当前真正执行的 normalized action
↓
de-normalize
↓
发送 target joint positions 给机器人
↓
环境发生变化
↓
进入下一 timestep
```

这一篇只回答一个问题：

> **训练好的 ACT 在 rollout 时，到底每一步做了什么？**

---

## 1. Inference 和 Training 最大的区别

训练时：

$$
(o_t,a_{t:t+k})
$$

两者都已知。

所以可以用 ground-truth future action chunk 运行 CVAE encoder：

$$
q_\phi(z\mid a_{t:t+k},q_t)
$$

得到：

$$
\mu,\log\sigma^2
$$

再 sample：

$$
z=\mu+\sigma\odot\epsilon
$$

---

推理时完全不同。

我们只有：

$$
o_t
$$

而：

$$
a_{t:t+k}
$$

正是未知答案。

所以 CVAE encoder 无法使用。

ACT 原论文明确规定：

$$
\boxed{z=0}
$$

即使用 unit Gaussian prior 的均值做 deterministic decoding。

所以 inference policy 可以写成：

$$
\boxed{
\hat a_{t:t+k}
=
\pi_\theta(o_t,z=0)
}
$$

---

## 2. 先加载哪个 Checkpoint？

ACT 原论文写得很明确：

> test time 使用 validation loss 最低的 policy checkpoint。

released code 也会加载：

```text
policy_best.ckpt
```

所以部署开始前：

```text
训练得到很多 checkpoints
↓
根据 validation loss 选择 best
↓
load policy_best.ckpt
↓
policy.eval()
```

同时还要加载训练时保存的：

```text
dataset_stats.pkl
```

因为推理时必须使用：

> **和训练阶段完全一致的 normalization statistics。**

---

## 3. 为什么 Inference 还需要 Training Dataset 的 Statistics？

训练时 qpos 和 actions 做过：

$$
q_{\text{norm}}
=
\frac{q-\mu_q}{\sigma_q}
$$

$$
a_{\text{norm}}
=
\frac{a-\mu_a}{\sigma_a}
$$

因此 policy 学到的是：

> normalized coordinate system 中的 mapping。

如果 test time 直接把真实 joint angles 原始值喂进去：

```text
训练：
normalized qpos

测试：
raw qpos
```

输入分布会完全不一致。

所以官方 eval code 加载：

```python
stats['qpos_mean']
stats['qpos_std']
stats['action_mean']
stats['action_std']
```

并定义：

```python
pre_process = lambda q:
    (q - qpos_mean) / qpos_std

post_process = lambda a:
    a * action_std + action_mean
```

---

## 4. 第 t 个 Timestep：先读取 Observation

每一步 rollout：

```python
obs = ts.observation
```

然后取：

$$
q_t
$$

以及 camera images。

qpos：

$$
q_t\in\mathbb R^{14}
$$

先转成 normalized coordinates：

$$
\boxed{
\tilde q_t
=
\frac{
q_t-\mu_q
}{
\sigma_q
}
}
$$

然后变成 batch tensor：

$$
[1,14]
$$

这里 batch size：

$$
B=1
$$

因为机器人当前只需要处理一个实时 observation。

---

## 5. Camera Images 怎样进入 Policy？

官方：

```python
curr_image =
    get_image(ts, camera_names)
```

会读取当前 timestep 的多路 camera frames。

ACT policy 内部再执行：

```text
image / 255
↓
ImageNet normalization
↓
ResNet18
↓
visual tokens
```

所以 inference 时视觉 pipeline 与 training 必须一致。

如果训练时：

- camera 顺序；
- resize；
- normalization；
- camera naming；

和 inference 不一致，

即使模型参数正确，性能也会明显下降。

---

## 6. CVAE Encoder 在这里已经不存在

这是 inference 最重要的分叉。

训练时：

```text
qpos + target actions
↓
CVAE Encoder
↓
μ, logσ²
↓
z
```

推理时：

```text
CVAE Encoder
    ✕
```

官方 `DETRVAE.forward()` 在：

```python
actions is None
```

时直接走 inference branch：

```python
mu = logvar = None
latent_sample = torch.zeros(...)
```

因此：

$$
\boxed{
z_t=\mathbf 0
}
$$

并不是：

> encoder 根据当前 observation 预测出来的。

它是：

> **ACT 人工规定的 test-time latent choice。**

---

## 7. 一次 Policy Forward 输出什么？

输入：

$$
\tilde q_t
$$

当前 images：

$$
I_t
$$

以及：

$$
z=0
$$

ACT policy 输出：

$$
\boxed{
\hat A_t
=
[
\hat a_t^{(t)},
\hat a_{t+1}^{(t)},
\ldots,
\hat a_{t+k-1}^{(t)}
]
}
$$

shape：

$$
[1,k,14]
$$

这里上标：

$$
(t)
$$

表示：

> 这条预测是在 query time $t$ 产生的。

下标：

> 表示这个 action 对应哪个实际执行 timestep。

例如：

$$
\hat a_{t+17}^{(t)}
$$

就是：

> 在当前 $t$ 时，根据当前 observation，预测未来第 17 步应该执行的 14-D joint target。

---

## 8. 如果 k=100，一次就预测 100 个未来动作

原论文默认：

$$
k=100
$$

因此一次 forward：

$$
o_t
\rightarrow
100\times14
$$

也就是：

$$
1400
$$

个连续 action values。

但它们不是 1400 个互相独立的输出。

它们来自 Transformer decoder 的：

$$
100
$$

个 action-query slots，

彼此可以通过 decoder self-attention 交换信息。

详细架构：

- [ACT Architecture](./architecture.md)

---

## 9. 最朴素的 Action Chunk Inference 会怎么做？

如果没有 Temporal Ensemble，可以采用：

```text
t = 0
↓
query policy
↓
得到 [a0 ... a99]
↓
连续执行这 100 个动作
↓
t = 100
↓
重新 query
```

这就是论文最初用来解释 Action Chunking 的 naive implementation。

如果：

$$
k=100
$$

那么每：

$$
100
$$

步才重新根据新 observation 规划一次。

优点：

> effective horizon 显著缩短。

缺点：

> chunk 内的新视觉信息无法及时改变动作。

这就是为什么 ACT 最终引入 Temporal Ensemble。

---

## 10. 官方代码里“不开 Temporal Ensemble”就是接近这种执行方式

released eval code 初始：

```python
query_frequency =
    policy_config['num_queries']
```

而：

$$
num\_queries=k
$$

所以没有：

```text
--temporal_agg
```

时：

$$
\boxed{
query\_frequency=k
}
$$

policy 只在：

$$
t\bmod k=0
$$

时重新 forward：

```python
if t % query_frequency == 0:
    all_actions =
        policy(qpos, curr_image)
```

然后当前 action：

```python
raw_action =
    all_actions[:, t % query_frequency]
```

因此：

```text
query 一次
↓
把这个 chunk 按顺序执行
↓
chunk 用完再 query
```

---

## 11. 开启 Temporal Ensemble 后发生关键变化

如果：

```python
temporal_agg = True
```

官方直接：

```python
query_frequency = 1
```

于是：

$$
\boxed{
\text{每一个 timestep 都重新 query policy}
}
$$

这正是原论文 Algorithm 2。

所以：

```text
t=0
→ predict chunk 0

t=1
→ new observation
→ predict chunk 1

t=2
→ new observation
→ predict chunk 2

...
```

action chunks 开始大量 overlap。

---

## 12. 重新写出 k=4 的例子

假设：

$$
k=4
$$

#### t = 0

$$
\hat A_0
=
[
\hat a_0^{(0)},
\hat a_1^{(0)},
\hat a_2^{(0)},
\hat a_3^{(0)}
]
$$

#### t = 1

$$
\hat A_1
=
[
\hat a_1^{(1)},
\hat a_2^{(1)},
\hat a_3^{(1)},
\hat a_4^{(1)}
]
$$

#### t = 2

$$
\hat A_2
=
[
\hat a_2^{(2)},
\hat a_3^{(2)},
\hat a_4^{(2)},
\hat a_5^{(2)}
]
$$

排列起来：

```text
query t=0:  a₀⁰  a₁⁰  a₂⁰  a₃⁰
query t=1:       a₁¹  a₂¹  a₃¹  a₄¹
query t=2:            a₂²  a₃²  a₄²  a₅²
query t=3:                 a₃³  a₄³  a₅³  a₆³
```

因此当前真实执行：

$$
t=3
$$

时，

已经有四个 predictions：

$$
\hat a_3^{(0)},
\hat a_3^{(1)},
\hat a_3^{(2)},
\hat a_3^{(3)}
$$

---

## 13. 官方代码怎样保存这些 Overlapping Chunks？

开启 temporal aggregation 后，代码创建：

```python
all_time_actions =
    torch.zeros(
        [
            max_timesteps,
            max_timesteps + num_queries,
            state_dim
        ]
    )
```

概念上是一个三维表：

$$
[
\text{query time},
\text{execution time},
\text{action dim}
]
$$

其中：

$$
state\_dim=14
$$

可以画成：

```text
                  execution time
             0     1     2     3     4
query 0     a0    a1    a2    a3
query 1           a1    a2    a3    a4
query 2                 a2    a3    a4
query 3                       a3    a4
```

---

## 14. 新 Chunk 怎样写进这个 Buffer？

第：

$$
t
$$

步得到：

$$
all\_actions
\in
[1,k,14]
$$

代码：

```python
all_time_actions[
    [t],
    t:t+num_queries
] = all_actions
```

含义：

> 把 query time $t$ 预测的未来 $k$ 个 actions，放到 execution-time 轴的 $t$ 到 $t+k-1$。

因此：

```text
row = 什么时候预测的
column = 什么时候执行
```

这就是理解 Temporal Ensemble 最好的矩阵。

---

## 15. 当前 Timestep 怎样取出所有预测？

假设当前：

$$
t
$$

官方：

```python
actions_for_curr_step =
    all_time_actions[:, t]
```

也就是：

> 取整个二维时间矩阵的第 $t$ 列。

这会得到所有 query times 对：

$$
\text{execution time}=t
$$

产生过的 prediction。

然后过滤没有填过的 zero rows：

```python
actions_populated =
    torch.all(
        actions_for_curr_step != 0,
        axis=1
    )
```

再：

```python
actions_for_curr_step =
    actions_for_curr_step[
        actions_populated
    ]
```

最终得到：

$$
A_t
=
[
\hat a_t^{(j_1)},
\ldots,
\hat a_t^{(j_n)}
]
$$

---

## 16. 这里有一个 Released-Code 的小陷阱：用 0 判断“有没有预测”

官方代码用：

```python
torch.all(
    actions_for_curr_step != 0,
    axis=1
)
```

判断某一行是否已填充。

这隐含假设：

> 一个真实 normalized action vector 不会包含任何精确为 0 的 dimension。

如果一个合法 action 某个维度恰好精确为：

$$
0
$$

这行可能会被错误判断为未填充。

实践中连续网络输出精确等于 0 的概率通常不高，

但从软件工程角度：

> 显式 boolean validity mask 会更稳健。

这是 released implementation detail，

不是论文算法定义。

---

## 17. Temporal Ensemble 权重怎样计算？

原论文定义：

$$
\boxed{
w_i
=
\exp(-mi)
}
$$

其中：

$$
w_0
$$

对应：

> oldest action prediction。

官方 code：

```python
k = 0.01

exp_weights =
    np.exp(
        -k *
        np.arange(
            len(actions_for_curr_step)
        )
    )
```

这里代码变量：

```python
k = 0.01
```

不要和：

> chunk size $k$

混淆。

它其实对应论文里的：

$$
m
$$

即 exponential decay coefficient。

---

## 18. 权重归一化

代码：

```python
exp_weights =
    exp_weights /
    exp_weights.sum()
```

所以：

$$
\alpha_i
=
\frac{
e^{-mi}
}{
\sum_j e^{-mj}
}
$$

满足：

$$
\sum_i\alpha_i=1
$$

然后：

$$
\boxed{
\tilde a_t
=
\sum_i
\alpha_i
A_t[i]
}
$$

这一步仍然发生在：

> **normalized action space**

中。

---

## 19. 为什么 Older Prediction 权重反而更大？

ACT 原论文明确规定：

$$
w_0
$$

对应 oldest prediction，

而：

$$
w_i=e^{-mi}
$$

因此：

$$
w_0>w_1>w_2>\cdots
$$

只要：

$$
m>0
$$

这一点确实有些反直觉。

因为 newest prediction 看到了最新 observation。

但论文没有给一个严格理论证明说：

> oldest prediction 一定更准确。

论文明确说的是：

> $m$ 控制吸收新 observation 的速度，较小 $m$ 意味着更快吸收新 observation。

因此不要替论文发明：

> “旧预测因为长期规划更可靠，所以必须权重大。”

这种解释没有被论文证明。

详细讨论：

- [Temporal Ensemble](./temporal-ensemble.md)

---

## 20. 重要：Temporal Ensemble 平均的是 Normalized Actions

官方 sequence：

```text
policy output
↓
raw_action
↓
temporal weighted average
↓
post_process
```

policy output 仍然在：

> action normalization space。

只有 Temporal Ensemble 得到当前：

$$
raw\_action
$$

之后，

才：

```python
action =
    post_process(raw_action)
```

也就是：

$$
\boxed{
a_t
=
\tilde a_t
\odot\sigma_a
+
\mu_a
}
$$

恢复真实 joint target units。

---

## 21. 在这种线性 Normalization 下，先平均再反归一化其实等价于先反归一化再平均

因为：

$$
post(a)
=
a\odot\sigma+\mu
$$

是 affine transformation。

假设：

$$
\sum_i\alpha_i=1
$$

那么：

$$
post\left(
\sum_i\alpha_i a_i
\right)
$$

等于：

$$
\left(
\sum_i\alpha_i a_i
\right)\sigma+\mu
$$

而：

$$
\sum_i
\alpha_i
(a_i\sigma+\mu)
$$

等于：

$$
\sigma
\sum_i\alpha_i a_i
+
\mu
\sum_i\alpha_i
$$

由于：

$$
\sum_i\alpha_i=1
$$

得到同样结果。

所以这里两种顺序数学上等价。

官方实现选择：

> 先 aggregation，再 post-process。

---

## 22. 得到 Current Action 后，才真正发送给环境

Temporal Ensemble 最终产生：

$$
raw\_action
\in
\mathbb R^{14}
$$

反 normalization：

$$
target\_qpos
=
raw\_action
\odot
\sigma_a
+
\mu_a
$$

然后：

```python
ts =
    env.step(target_qpos)
```

这一步才真正让：

> 仿真或物理机器人执行当前 action。

注意：

> 当前只执行一个 14-D action。

虽然 policy 一次预测了：

$$
k
$$

个 actions，

Temporal Ensemble 模式下此时只执行：

$$
\boxed{
a_t
}
$$

下一 timestep 会重新观察，再重新预测整个 chunk。

---

## 23. 这就是 Receding-Horizon 风格的 Closed Loop

在 $t$：

```text
看当前 observation
↓
预测未来 k 步
↓
只决定当前真正执行什么
```

执行：

$$
a_t
$$

后：

> 环境发生变化。

于是到了：

$$
t+1
$$

重新：

```text
看新的 observation
↓
再预测未来 k 步
↓
重新融合 current-time predictions
```

所以最终 ACT 不是：

> “预测 100 步，然后机械执行 100 步。”

而是：

> **不断用最新 observation 重算未来动作计划。**

---

## 24. 一个完整 k=4 Rollout

假设：

$$
k=4
$$

---

### t = 0

观察：

$$
o_0
$$

normalize qpos。

令：

$$
z=0
$$

policy：

$$
[
\hat a_0^{(0)},
\hat a_1^{(0)},
\hat a_2^{(0)},
\hat a_3^{(0)}
]
$$

当前只有：

$$
\hat a_0^{(0)}
$$

所以执行：

$$
a_0
=
\hat a_0^{(0)}
$$

---

### t = 1

得到新 observation：

$$
o_1
$$

重新 forward：

$$
[
\hat a_1^{(1)},
\hat a_2^{(1)},
\hat a_3^{(1)},
\hat a_4^{(1)}
]
$$

现在针对 $t=1$ 有：

$$
\hat a_1^{(0)}
$$

和：

$$
\hat a_1^{(1)}
$$

Temporal Ensemble：

$$
a_1
=
\alpha_0
\hat a_1^{(0)}
+
\alpha_1
\hat a_1^{(1)}
$$

执行。

---

### t = 2

重新观察：

$$
o_2
$$

新 chunk：

$$
[
\hat a_2^{(2)},
\hat a_3^{(2)},
\hat a_4^{(2)},
\hat a_5^{(2)}
]
$$

当前：

$$
a_2
=
\alpha_0\hat a_2^{(0)}
+
\alpha_1\hat a_2^{(1)}
+
\alpha_2\hat a_2^{(2)}
$$

执行。

---

### t = 3

又重新 query。

现在有 4 个 prediction：

$$
\hat a_3^{(0)},
\hat a_3^{(1)},
\hat a_3^{(2)},
\hat a_3^{(3)}
$$

weighted average 后执行。

从这里开始，在 episode 中间：

> 当前 timestep 通常最多有 $k$ 个 action predictions。

---

## 25. 当前 Timestep 最多有多少个 Candidate Actions？

episode 从：

$$
t=0
$$

开始。

chunk size：

$$
k
$$

则当前 timestep：

$$
t
$$

被以下 query 覆盖：

$$
j
\in
[
\max(0,t-k+1),
t
]
$$

所以 candidate 数量：

$$
\boxed{
N_t
=
\min(k,t+1)
}
$$

因此：

```text
t=0     → 1 个
t=1     → 2 个
...
t=k-1   → k 个
之后     → 通常维持 k 个
```

直到 episode 结束附近也仍由已产生的 chunk 覆盖当前时间。

---

## 26. 为什么不是使用 Future Chunk 中最新那个 Action？

一个简单方法是始终执行：

$$
\hat a_t^{(t)}
$$

也就是最新 observation 产生的 prediction。

但 ACT 选择把：

> 旧计划与新计划

进行 ensemble。

这样可以避免每 timestep 完全把过去规划扔掉，

从而降低 prediction jitter。

论文实验中 Temporal Ensemble 对 ACT 平均带来约：

$$
3.3\%
$$

的性能提升。

但它不是无条件有效的通用技巧。

---

## 27. 为什么 Temporal Ensemble 不是普通 Action Smoothing？

普通 smoothing 可能平均：

$$
a_{t-1},a_t,a_{t+1}
$$

它们本来对应不同实际时刻。

ACT 平均：

$$
\hat a_t^{(t-k+1)},
\ldots,
\hat a_t^{(t)}
$$

全部都在回答：

> **同一个实际 timestep $t$，应该执行什么？**

所以它不会直接把：

> 前一个动作和后一个动作

混在一起。

论文明确强调这一点，因为普通 temporal smoothing 可能引入 bias。

---

## 28. 一个关键区别：Policy Prediction Time 和 Execution Time

学习 ACT inference 最容易混的就是两个“时间”。

必须区分：

### Query / Prediction Time

$$
j
$$

表示：

> 这条 prediction 是什么时候生成的。

---

### Execution Time

$$
t
$$

表示：

> 这个 action 是什么时候真正执行的。

所以：

$$
\hat a_t^{(j)}
$$

表示：

> 在时间 $j$ 预测的、要在时间 $t$ 执行的 action。

只要把这两个 index 分清，

Temporal Ensemble 就不再混乱。

---

## 29. 为什么每一步都重新 Prediction 还保留 Action Chunking 的意义？

有人会问：

> 如果每一步都重新预测，最终又只执行当前动作，那是不是又退回 single-step policy 了？

不是。

因为模型每次 forward 学习和输出的对象仍然是：

$$
\boxed{
a_{t:t+k}
}
$$

而不是：

$$
a_t
$$

也就是说：

> 当前 action prediction 是 action sequence joint modeling 的一部分。

Transformer decoder 中不同未来 action slots 还会互相 self-attend。

因此当前 action：

$$
\hat a_t^{(t)}
$$

并不是一个 single-step model 独立预测的结果。

它来自：

> 一个 action-chunk policy。

---

## 30. Action Chunking 的“有效 Horizon”与 Every-Step Query 并不矛盾

Action Chunking 改变的是：

> policy 的预测单位。

从：

$$
a_t
$$

变成：

$$
a_{t:t+k}
$$

Temporal Ensemble 则改变：

> rollout 时怎样使用这些 chunks。

所以：

```text
Training / representation:
预测 k-step action chunk

Inference:
每 timestep 都重新产生一个 chunk
```

可以同时成立。

不能因为：

> forward frequency = 每一步

就说：

> “Action Chunking 不存在了。”

---

## 31. Inference 时 z=0 会不会每个 Timestep 都重新创建？

概念上：

$$
z_t=0
$$

对每个 timestep 都一样。

官方 model forward 中每次 inference branch 都会创建：

```python
torch.zeros(
    [bs, latent_dim]
)
```

所以每次 query policy：

> latent condition 都固定为 zero。

真正随时间变化的是：

- images；
- qpos；

而不是：

$$
z
$$

---

## 32. 所以 Policy 为什么仍然会每一步输出不同动作？

因为：

$$
\pi_\theta(a\mid o_t,z=0)
$$

虽然：

$$
z
$$

固定，

但 observation：

$$
o_t
$$

不断变化。

机器人动了一步后：

- joint state 变化；
- object position 可能变化；
- camera pixels 变化；
- contact situation 变化。

于是：

$$
o_t\neq o_{t+1}
$$

因此：

$$
\pi(o_t,0)
\neq
\pi(o_{t+1},0)
$$

完全正常。

---

## 33. Inference 时为什么还需要 Image Normalization？

因为 ResNet18 training 时看到的是：

> normalized image distribution。

如果 rollout 时跳过：

```text
/255
ImageNet normalization
```

那么输入 feature statistics 会发生巨大变化。

这属于最典型的：

> train–test preprocessing mismatch。

所以部署模型时：

> preprocessing 是模型的一部分。

不能只保存 neural network weights 而忘记：

- mean；
- std；
- camera ordering；
- image scaling。

---

## 34. Action 为什么必须 De-normalize？

policy 输出：

$$
raw\_action
$$

不是直接的 physical target joint positions。

它位于 training standardization space。

因此必须：

$$
\boxed{
a_{\mathrm{physical}}
=
a_{\mathrm{norm}}
\odot
\sigma_a
+
\mu_a
}
$$

如果忘记 post-process，

机器人收到的是：

> “标准差单位”

而不是实际 joint values。

这会完全错误。

---

## 35. Target Joint Position 后发生什么？

论文定义 action：

> target joint positions for both arms at the next timestep.

这些 target positions 不由 ACT 自己直接转成 motor torque。

而是由 Dynamixel motors 中：

> low-level high-frequency PID controller

跟踪。

所以控制栈：

```text
ACT
↓
14-D target qpos
↓
low-level PID
↓
motors
↓
physical robot
```

ACT 位于：

> learned high-level joint-target policy

这一层。

---

## 36. ACT 的 Control Loop 是多少 Hz？

论文中的 ALOHA teleoperation / control 数据以：

$$
50\text{ Hz}
$$

高频运行。

时间间隔：

$$
\Delta t
=
\frac1{50}
=
0.02\text{ s}
$$

所以每：

$$
20\text{ ms}
$$

大约一个 timestep。

论文强调：

> 高频 closed-loop visual feedback 对 fine manipulation 很重要。

它甚至专门做了：

$$
50\text{ Hz}
$$

对：

$$
5\text{ Hz}
$$

的用户实验。

---

## 37. 但论文 Policy Inference 本身报告约 0.01 秒

ACT 论文报告：

> 在单张 RTX 2080 Ti 上，policy inference time 约 0.01 s。

也就是约：

$$
10\text{ ms}
$$

这比：

$$
20\text{ ms}
$$

的 50 Hz control interval 更短。

所以至少在论文硬件条件下：

> 每 timestep query policy 在计算时间上是可行的。

但真实部署还要考虑：

- camera capture；
- data transfer；
- robot communication；
- preprocessing；
- Python overhead；

等额外 latency。

---

## 38. 官方 Evaluation 为什么使用 torch.inference_mode()？

代码：

```python
with torch.inference_mode():
```

因为 test time 不需要 gradient。

这样 PyTorch 可以：

- 不构建 autograd graph；
- 减少 memory usage；
- 降低部分计算 overhead。

同时：

```python
policy.eval()
```

会让：

- dropout；
- batchnorm 等；

进入 evaluation behavior。

所以部署时通常同时需要：

```python
model.eval()
torch.inference_mode()
```

---

## 39. 为什么 Load Best Validation Checkpoint 后还要 set_seed？

官方 evaluation 会：

```python
set_seed(1000)
```

这主要用于：

> 提高 evaluation pipeline 的可复现性。

特别是 simulation 环境 initialization 等过程可能包含随机性。

但注意：

ACT policy 自己因为：

$$
z=0
$$

而不再引入 latent sampling stochasticity。

所以给定完全相同：

- observation；
- weights；
- deterministic computation；

policy mapping 是 deterministic 的。

---

## 40. 真实环境当然仍然不是完全 Deterministic

论文说：

> policy output deterministic

并不意味着物理世界：

> 每次 rollout 必然一模一样。

真实机器人仍有：

- camera noise；
- actuator error；
- contact uncertainty；
- friction differences；
- object initialization changes；
- communication latency。

这些都会使未来 observation 不同。

ACT 的 deterministic 只表示：

$$
o
$$

固定时：

> 不会因为随机采 $z$ 而额外得到不同 action chunk。

---

## 41. 没有 Temporal Ensemble 时为什么 Query Frequency = k？

因为一条 chunk 本身就有：

$$
k
$$

个 actions。

query 在：

$$
t=0
$$

得到：

$$
a_0,\ldots,a_{k-1}
$$

接下来可以依次执行。

直到：

$$
t=k
$$

才需要新 chunk。

所以：

$$
query\_frequency=k
$$

就是最直接 naive chunk execution。

---

## 42. Temporal Ensemble 时为什么 Query Frequency 必须 = 1？

要让同一个 timestep 有多个 overlapping predictions，

必须让 chunks 发生 overlap。

如果每：

$$
k
$$

步才 query：

```text
chunk 1:
0 ... k-1

chunk 2:
k ... 2k-1
```

它们完全不 overlap。

就没有：

> “多个 prediction 预测同一个 timestep”

这件事。

所以 Temporal Ensemble 的前提就是：

$$
\boxed{
query\ every\ timestep
}
$$

---

## 43. Temporal Ensemble 会让 Policy Forward 数量增加多少？

naive chunk execution：

$$
T/k
$$

次左右 policy queries。

Temporal Ensemble：

$$
T
$$

次 policy queries。

因此 forward 次数大约增加：

$$
k
$$

倍。

如果：

$$
k=100
$$

这是非常显著的 inference compute 增加。

论文明确说：

> Temporal Ensemble 没有额外 training cost，但有额外 inference-time computation。

---

## 44. 但不能把这个 k 倍简单等同于实际 wall-clock 慢 k 倍

因为真实系统还存在：

- fixed control interval；
- GPU parallel compute；
- camera / environment latency；
- implementation overhead；

等。

所以只能说：

> policy forward 的调用次数从约 $T/k$ 变成 $T$。

不能在没有 benchmark 的情况下直接说：

> 整个机器人一定慢了 100 倍。

---

## 45. Buffer 为什么开成 max_timesteps × (max_timesteps + k)？

因为第：

$$
t
$$

个 query 会预测到：

$$
t+k-1
$$

所以 execution-time 轴必须比：

$$
T
$$

再长一些，

才能容纳 episode 尾部 query 的未来预测。

官方：

```python
[max_timesteps,
 max_timesteps + num_queries,
 state_dim]
```

只是一个方便的 dense storage。

理论上也可以用：

- FIFO queues；
- dictionaries；
- ring buffers；

更节省 memory。

论文 Algorithm 2 把它抽象成：

> FIFO buffers $\mathcal B[0:T]$。

---

## 46. 为什么论文说 FIFO Buffer，而代码是大 Tensor？

它们表达的是同一逻辑。

论文层面：

$$
\mathcal B[t]
$$

存：

> 所有预测 execution timestep $t$ 的 actions。

代码层面：

```python
all_time_actions[:, t]
```

就是：

> 当前 $t$ 的所有 predictions。

所以：

```text
Algorithm 2 Buffer
```

和：

```text
all_time_actions matrix
```

只是数据结构实现不同。

---

## 47. Inference 时 Padding 不再出现吗？

training target padding 不再需要。

因为 inference 时没有：

> ground-truth future action chunk。

policy 永远直接输出固定：

$$
k
$$

个 actions。

episode 最后几步时：

> chunk 中超出 rollout horizon 的未来 predictions 只是不会被执行。

它们可以存在于 buffer 中，但 episode 结束后自然被丢弃。

---

## 48. Inference 时还有 KL Loss 吗？

没有。

test time：

- 没有 training posterior；
- 不计算 $\mu,\sigma$；
- 不计算 reconstruction；
- 不计算 KL；
- 不 backward；
- 不 optimizer step。

只做：

$$
\boxed{
\text{forward + action selection}
}
$$

因此不要把：

> CVAE training objective

和：

> inference computation

混起来。

---

## 49. Inference 时 ResNet 和 Transformer 参数还更新吗？

不更新。

所有参数：

$$
\theta
$$

固定。

模型只执行：

> deterministic numerical computation。

所以：

```text
images
↓
ResNet
↓
Transformer
↓
actions
```

每一步都在 forward，

但没有 learning。

---

## 50. 一个完整官方式 Pseudocode

```python
load_best_checkpoint()
load_dataset_stats()

policy.eval()

for t in range(T):

    # 1. 当前 observation
    obs = env.get_observation()

    qpos = obs.qpos
    images = obs.images

    # 2. 与训练一致的 preprocessing
    qpos_norm = (
        qpos - qpos_mean
    ) / qpos_std

    image_tensor = preprocess_images(
        images
    )

    # 3. ACT inference:
    # CVAE encoder 不运行
    # z = 0
    action_chunk = policy(
        qpos_norm,
        image_tensor
    )

    # 4. 保存这个 timestep 预测的整个 chunk
    buffers.add(
        query_time=t,
        predicted_actions=action_chunk
    )

    # 5. 找到所有针对当前 timestep t 的预测
    candidates = buffers[t]

    # 6. Temporal Ensemble
    weights = exp_decay_weights(
        len(candidates)
    )

    action_norm = weighted_average(
        candidates,
        weights
    )

    # 7. 恢复真实 joint units
    action = (
        action_norm * action_std
        + action_mean
    )

    # 8. 当前只执行一个 action
    env.step(action)
```

这就是最终 ACT closed-loop inference 的核心。

---

## 51. 如果关闭 Temporal Ensemble，Pseudocode 会怎样？

```python
for t in range(T):

    if t % k == 0:

        obs = env.get_observation()

        action_chunk = policy(
            normalize(obs.qpos),
            obs.images
        )

    action_norm = action_chunk[t % k]

    action = denormalize(action_norm)

    env.step(action)
```

这两个模式一定要区分。

---

## 52. 为什么论文 Algorithm 2 只写 Temporal Ensemble 版本？

因为 Algorithm 2 描述的是：

> ACT 最终 inference procedure。

论文 Section IV-A 先讨论 naive Action Chunking，

然后说明它可能 jerky，

最终提出：

> every-step query + Temporal Ensemble。

所以 Algorithm 2 直接总结最终方法。

released code 保留：

> `temporal_agg=False`

模式，

方便 ablation 和对比。

---

## 53. ACT Inference 的真正 Closed-Loop 在哪里？

“Closed-loop”意味着：

$$
\text{action}
\rightarrow
\text{environment changes}
\rightarrow
\text{new observation}
\rightarrow
\text{new action}
$$

ACT Temporal Ensemble 模式：

```text
oₜ
↓
predict future chunk
↓
execute aₜ
↓
environment changes
↓
oₜ₊₁
↓
predict a new future chunk
↓
execute aₜ₊₁
```

因此每一步都重新利用实际执行结果。

这就是 fine manipulation 所需的 closed-loop visual feedback。

---

## 54. 为什么仅预测 Chunk 还不够？

因为机器人实际执行永远存在误差：

$$
q_{t+1}^{\text{actual}}
\neq
q_{t+1}^{\text{predicted ideal}}
$$

物体也可能因为 contact 产生偏移。

所以如果完全 open-loop 地执行：

$$
a_t,\ldots,a_{t+k-1}
$$

后半段 prediction 的前提可能已经不成立。

every-step re-query 允许：

> 根据真实的新 observation 重新修正后续计划。

---

## 55. 为什么不干脆变回 Single-Step Prediction？

因为 action chunk 仍提供：

- shorter effective horizon；
- sequence-level modeling；
- temporally coordinated action structure；
- 对某些 non-Markovian human behavior 的缓解。

所以 ACT 想同时保留：

```text
Action Chunking
的序列建模优势

+

Closed-loop
的新 observation 修正能力
```

Temporal Ensemble 就是连接两者的 execution rule。

---

## 56. Inference 中最核心的三个时间尺度

可以把 ACT 看成有三层时间尺度。

### 1. Physical Control Timestep

每一步：

$$
t\rightarrow t+1
$$

机器人执行一个 14-D target。

---

### 2. Action Chunk Horizon

一次 policy forward 预测：

$$
k
$$

步未来。

---

### 3. Episode Horizon

整个任务：

$$
T
$$

个 timesteps。

ACT 的核心就是：

> 在长度 $T$ 的任务里，不断滚动长度 $k$ 的局部 future plan。

---

## 57. ACT 的 Inference 很像 MPC 吗？

在高层直觉上有一点像：

```text
观察当前状态
↓
规划未来一段
↓
执行当前动作
↓
重新观察
↓
重新规划
```

这具有：

> receding horizon

的味道。

但 ACT 不是经典 Model Predictive Control。

ACT：

- 不显式使用 dynamics model；
- 不在线求解 optimization problem；
- future actions 来自 learned imitation policy。

经典 MPC 通常：

- 有 dynamics；
- 有 cost；
- 在线 optimize action sequence。

所以可以类比行为形态，

不能把理论机制等同。

---

## 58. 真实机器人和 Simulation 的 Rollout 有什么共同点？

无论 sim 还是真机：

核心都是：

```text
observation
↓
policy
↓
target qpos
↓
environment step
```

区别主要在：

- environment backend；
- sensor / actuator communication；
- reward availability；
- latency；
- reset mechanism。

ACT policy 本身不需要知道：

> 这是 MuJoCo 还是真实 ViperX。

---

## 59. Simulation 中 Success Rate 怎样计算？

official eval：

每个 rollout 保存 reward sequence。

然后：

```python
episode_highest_reward =
    np.max(rewards)
```

如果：

$$
episode\_highest\_reward
=
env\_max\_reward
$$

就算 success。

最后：

$$
\boxed{
success\ rate
=
\frac{
\text{successful rollouts}
}{
\text{num rollouts}
}
}
$$

默认 simulation evaluation：

$$
50
$$

个 rollouts。

这属于 official benchmark evaluation 逻辑，

不是 ACT policy architecture 的一部分。

---

## 60. 真机上为什么没有同样的 Reward？

真实世界环境通常没有一个完美自动 reward function。

official real-robot branch：

$$
env\_max\_reward=0
$$

真实任务 success 往往需要：

- 人工判断；
- task-specific evaluator；
- additional sensing；

等。

所以算法中的 action inference 不依赖 reward。

再次说明：

> ACT 不是 reinforcement learning。

---

## 61. Inference 时为什么只输出 qpos，不输出 qvel？

虽然 dataset 文件可以包含：

```text
qpos
qvel
```

original ACT policy observation 主要使用：

> qpos + images。

官方 eval 取：

```python
obs['qpos']
```

作为 proprioceptive policy input。

所以不要因为 HDF5 里保存了 qvel 就说：

> original ACT policy 输入 qpos + qvel。

具体模型 input 仍应以论文与 policy code 为准。

---

## 62. Temporal Ensemble 的 m 在官方实现是多少？

official eval：

```python
k = 0.01
```

再次强调，这个变量名只是代码局部命名。

对应论文：

$$
\boxed{
m=0.01
}
$$

权重：

$$
w_i=e^{-0.01i}
$$

例如：

$$
w_0=1
$$

$$
w_1\approx0.9900
$$

$$
w_{10}\approx0.9048
$$

$$
w_{99}\approx0.3716
$$

所以 decay 实际上相当缓慢。

newer predictions 仍然保留显著贡献。

---

## 63. m=0.01 时 k=100 的 Oldest/Newest 权重差多少？

如果已有完整：

$$
100
$$

个 candidate predictions，

oldest：

$$
w_0=1
$$

newest 对应：

$$
i=99
$$

所以：

$$
w_{99}
=
e^{-0.99}
\approx0.3716
$$

也就是说：

> newest prediction 并没有被压到接近 0。

只是 oldest 的 raw weight 大约是它的：

$$
\frac1{0.3716}
\approx2.69
$$

倍。

归一化后所有 100 个 predictions 都共同参与。

---

## 64. Temporal Ensemble 有没有学参数？

没有。

$$
m
$$

不是神经网络学出来的。

权重：

$$
e^{-mi}
$$

是人为设计的 inference rule。

所以：

```text
Transformer Attention Weight
```

和：

```text
Temporal Ensemble Weight
```

是两个完全不同的东西。

前者：

> 由模型根据 QK similarity 动态计算。

后者：

> 根据 prediction age 用固定指数公式计算。

---

## 65. 一个非常容易混淆的问题：哪个 action 最终被执行？

policy output：

$$
[\hat a_t,\hat a_{t+1},...]
$$

不是全部立刻执行。

Temporal Ensemble 模式下：

> 当前 timestep 只执行一个最终 action：

$$
\boxed{
a_t
}
$$

这个：

$$
a_t
$$

由所有针对 timestep $t$ 的 chunk predictions 融合得到。

chunk 里的其他 predictions：

> 被存起来，可能在未来 timestep 参与 ensemble。

---

## 66. 旧 Prediction 会保存多久？

一个 query time：

$$
j
$$

产生：

$$
j,j+1,\ldots,j+k-1
$$

这些 execution-time predictions。

所以它最多会对未来：

$$
k
$$

个 timesteps 有贡献。

当真实时间超过：

$$
j+k-1
$$

这条 chunk 已经无法覆盖当前 timestep，

自然不会再参与 ensemble。

所以 influence window 长度就是：

$$
k
$$

---

## 67. 为什么 Buffer 不需要显式删除旧 Chunk？

因为代码每次只取：

```python
all_time_actions[:, t]
```

也就是 execution-time column $t$。

已经过期的 chunk 在这一列没有值，

所以自然不会被选中。

这相当于：

> 用稀疏矩阵索引实现了 FIFO 的效果。

---

## 68. Episode 开头为什么 Ensemble 很弱？

在：

$$
t=0
$$

只有 1 个 prediction。

在：

$$
t=1
$$

只有 2 个。

直到：

$$
t=k-1
$$

才积累到：

$$
k
$$

个。

所以 Temporal Ensemble 的稳定 aggregation effect：

> 会在 rollout 开始后的前 $k$ 步逐渐建立起来。

这也是公式：

$$
N_t=\min(k,t+1)
$$

的物理含义。

---

## 69. Episode 结束时会执行 Chunk 中超出的部分吗？

不会。

rollout loop 只运行：

$$
t=0,\ldots,T-1
$$

所以最后一次 query 可能预测：

$$
T-1,\ldots,T+k-2
$$

但：

$$
T
$$

以后 episode 已结束。

这些预测不会真正发送给机器人。

它们只是模型固定 chunk length 的自然结果。

---

## 70. 为什么 Inference Output 仍然叫 Action Chunk，而不是 Plan？

叫 plan 作为直觉没问题。

但论文使用：

> action sequence / action chunk。

因为模型并没有显式：

- dynamics rollout；
- cost optimization；
- symbolic planning。

它是一个 learned policy 直接回归：

$$
k
$$

个 future target joint positions。

所以“局部动作计划”是直觉，

正式术语仍然是：

> action chunk。

---

## 71. 常见误解一：推理时还要运行 CVAE Encoder

**错误。**

encoder 依赖 ground-truth future actions，

test time 被完全 discard。

---

## 72. 常见误解二：推理时 z 从 N(0,I) 随机采样

**原始 ACT 不这样做。**

它直接：

$$
\boxed{z=0}
$$

进行 deterministic decode。

---

## 73. 常见误解三：一次 Policy Forward 只输出当前 Action

**错误。**

一次：

$$
o_t
$$

输入产生：

$$
k
$$

个 future actions。

---

## 74. 常见误解四：预测 k 个动作就会全部执行完

**Temporal Ensemble 模式下错误。**

当前只执行：

$$
a_t
$$

下一 timestep 会重新 observation + forward。

---

## 75. 常见误解五：每 timestep forward 等于没有 Action Chunking

**错误。**

模型预测对象仍然是：

$$
a_{t:t+k}
$$

而不是 single-step action。

---

## 76. 常见误解六：Temporal Ensemble 平均的是相邻实际动作

**错误。**

它平均：

> 多个 query times 对同一个 execution timestep 的预测。

---

## 77. 常见误解七：Temporal Ensemble 在 Physical Joint Units 里做

released implementation 中：

> 先在 normalized action space ensemble，

再 de-normalize。

---

## 78. 常见误解八：模型输出可以直接发给 Robot

**错误。**

必须先用 training action statistics：

$$
a=
a_{\text{norm}}\sigma_a+\mu_a
$$

恢复 physical joint coordinate scale。

---

## 79. 常见误解九：ACT 直接输出 Motor Torque

**错误。**

输出是：

> 14-D absolute target joint positions。

底层 PID 再跟踪。

---

## 80. 常见误解十：Temporal Ensemble 是 Transformer 的一部分

**错误。**

它发生在：

> 多次 policy forwards 之后。

是外部 inference aggregation rule。

---

## 81. 常见误解十一：Policy Output Deterministic = Robot World Deterministic

**错误。**

论文只表示：

> 给定 observation，不会因 latent sampling 产生随机输出。

现实环境仍然可能随机。

---

## 82. 常见误解十二：不开 Temporal Ensemble 时仍每一步 Query Policy

released original code：

**不是。**

关闭 temporal aggregation：

$$
query\_frequency=k
$$

开启：

$$
query\_frequency=1
$$

这也是 ablation 能比较两种 execution style 的关键。

---

## 83. 常见误解十三：Temporal Ensemble 权重是训练出来的

**错误。**

$$
w_i=e^{-mi}
$$

是 hand-designed。

---

## 84. 常见误解十四：Newest Prediction 权重最大

按照原论文和官方 code：

**不是。**

oldest：

$$
i=0
$$

权重最大。

---

## 85. 常见误解十五：Inference 还需要 KL

**错误。**

KL 只属于 training objective。

test time 没有 loss。

---

## 86. 用五步记住 ACT Inference

如果所有细节最后只留下五步：

### 1. Observe

$$
o_t=(images_t,q_t)
$$

---

### 2. Predict Chunk

$$
\boxed{
\hat A_t
=
\pi_\theta(
o_t,z=0
)
}
$$

---

### 3. Store Overlapping Predictions

把：

$$
\hat A_t
$$

写入未来各 execution timestep 的 buffers。

---

### 4. Aggregate Current Timestep

$$
\boxed{
a_t^{norm}
=
\frac{
\sum_i e^{-mi}A_t[i]
}{
\sum_i e^{-mi}
}
}
$$

---

### 5. De-normalize and Execute

$$
\boxed{
a_t
=
a_t^{norm}
\odot\sigma_a
+
\mu_a
}
$$

发送给 low-level controller。

然后：

$$
t\leftarrow t+1
$$

重新开始。

---

## 87. 一张完整 Inference Flow

```text
                         TIMESTEP t
                             │
                             ▼
                    Read observation
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
             camera images             qpos
                 │                       │
                 │                       ▼
                 │                normalize qpos
                 │                       │
                 └───────────┬───────────┘
                             │
                     z = 0 (fixed)
                             │
                             ▼
                        ACT Policy
                             │
                             ▼
                 predicted chunk [k,14]
                             │
                             ▼
            store predictions for t ... t+k-1
                             │
                             ▼
           collect all predictions for current t
                             │
                             ▼
                  exponential weighting
                             │
                             ▼
                normalized current action
                             │
                             ▼
                     de-normalize
                             │
                             ▼
                 target qpos [14]
                             │
                             ▼
                    low-level PID
                             │
                             ▼
                       robot moves
                             │
                             ▼
                  new environment state
                             │
                             ▼
                        timestep t+1
                             │
                        repeat loop
```

---

## 88. 训练与推理最后做一次对照

| | Training | Inference |
|---|---|---|
| 当前 images | 有 | 有 |
| 当前 qpos | 有 | 有 |
| Ground-truth future actions | 有 | 没有 |
| CVAE encoder | 使用 | 丢弃 |
| $\mu,\log\sigma^2$ | 计算 | 不计算 |
| $z$ | posterior sample | $0$ |
| Policy 输出 | $k$ actions | $k$ actions |
| L1 loss | 有 | 无 |
| KL loss | 有 | 无 |
| Backward | 有 | 无 |
| Optimizer | 有 | 无 |
| Temporal Ensemble | 无 | 可启用 / final ACT 使用 |
| 实际执行机器人 | 无 | 有 |

这张表就是：

> **ACT train / test asymmetry 的完整总结。**

---

## 89. 一句话重新理解 ACT Inference

> **ACT 推理时在每个 timestep 读取最新多视角图像和 joint state，用训练时保存的 statistics 做相同 preprocessing，固定 latent $z=0$，一次预测未来 $k$ 个 normalized target-joint actions；最终 ACT 每一步都会重新 query policy，使 chunks 相互重叠，再用指数权重融合所有针对当前 execution timestep 的 predictions，随后把结果反归一化成真实 14-D target joint positions，交给底层 PID 执行，并根据执行后的新 observation 进入下一轮闭环。**

这就是从：

$$
\text{camera pixels}
$$

到：

$$
\text{robot motion}
$$

的完整 test-time loop。

---

## 90. 下一步：把 ACT 从头到尾连成一个完整 Data Flow

现在我们已经分别理解：

```text
为什么需要 ACT
↓
Action Chunking
↓
Temporal Ensemble
↓
CVAE
↓
z
↓
Architecture
↓
Training
↓
Inference
```

下一篇最自然的是做一次：

> **完整 ACT Data Flow**

不再引入新的理论。

而是拿一个具体例子：

$$
k=4
$$

从：

```text
一条 training demonstration
```

开始，

走到：

```text
训练完成
```

再走到：

```text
真实 rollout t=0,1,2,3
```

把：

- tensor shape；
- token；
- posterior；
- decoder；
- action chunk；
- overlapping predictions；
- Temporal Ensemble；

全部放到同一张时间线上。

下一篇：

- [ACT Complete Data Flow](./complete-data-flow.md)

---

### Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html
- Project: https://tonyzhaozh.github.io/aloha/

本文主要依据：

- Section IV-A — Action Chunking and Temporal Ensemble
- Section IV-B — Modeling Human Data
- Algorithm 2 — ACT Inference
- Appendix C — Detailed Architecture
- Section VI-A — Action Chunking and Temporal Ensembling

Algorithm 2 的核心过程是：

$$
\hat a_{t:t+k}
\sim
\pi_\theta(
\hat a_{t:t+k}
\mid
o_t,z=0
)
$$

将这些预测加入各 execution timestep buffer，

再：

$$
a_t
=
\frac{
\sum_iw_iA_t[i]
}{
\sum_iw_i
}
$$

其中：

$$
w_i=e^{-mi}
$$

且：

$$
w_0
$$

对应 oldest prediction。

---

### Official Implementation

ACT official repository:

https://github.com/tonyzhaozh/act

本文主要核对：

```text
imitate_episodes.py
policy.py
detr/models/detr_vae.py
```

released evaluation implementation 可确认：

- load `policy_best.ckpt`
- load `dataset_stats.pkl`
- qpos 使用 training mean/std normalize
- actions 使用 training mean/std de-normalize
- temporal aggregation disabled 时 `query_frequency = num_queries`
- temporal aggregation enabled 时 `query_frequency = 1`
- inference 时 CVAE latent 为 zero vector
- 每次 query 输出完整 action chunk
- `all_time_actions` 保存 overlapping chunks
- 当前 timestep 沿 execution-time column 取 candidate actions
- exponential temporal weight coefficient 使用 0.01
- weighted action 先在 normalized space 得到
- 再反 normalization 为 target qpos
- 最终 `env.step(target_qpos)` 执行当前 action

---

### 本文知识连接

#### ACT 主线

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)
- [CVAE in ACT](./cvae-in-act.md)
- [为什么 ACT 推理时令 z = 0？](./why-z-zero-at-inference.md)
- [ACT Architecture](./architecture.md)
- [ACT Training](./training.md)

#### Robot Control

- Joint Position
- PID Controller
- Closed-Loop Control
- Model Predictive Control

#### Deep Learning

- Normalization
- [Transformer](../../deep-learning/transformer.md)

#### 下一步

- [ACT Complete Data Flow](./complete-data-flow.md)
