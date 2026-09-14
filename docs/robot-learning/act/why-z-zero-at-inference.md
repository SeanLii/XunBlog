---
title: "为什么 ACT 推理时令 z = 0？"
description: "从 posterior、standard normal prior、KL regularization 与 deterministic decoding 出发，严格解释 ACT 为什么训练时采样 latent z，推理时却直接使用 z=0。"
status: reviewed
pageType: application
canonical: /robot-learning/act/why-z-zero-at-inference
updated: "2026-09-15"
---

# 为什么 ACT 推理时令 z = 0？

ACT 中最反直觉的一行代码可能就是：

```python
latent_sample = torch.zeros(...)
```

也就是推理阶段直接：

\[
\boxed{z=0}
\]

但训练时明明不是这样。

训练时，ACT 的 CVAE encoder 会根据：

- 当前 joint positions；
- ground-truth future action chunk；

预测：

\[
\mu
\]

和：

\[
\log\sigma^2
\]

再通过：

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
\]

随机采出一个 latent \(z\)。

于是自然会产生一个非常合理的问题：

> **训练时辛辛苦苦学了一个随机 latent variable，为什么测试时直接把它设成 0，模型却还能正常工作？**

甚至更尖锐一点：

> **如果推理时永远不用训练时推断出来的 \(z\)，那训练 CVAE 到底有什么意义？**

要真正回答这个问题，必须把下面几件事放在同一张图里：

1. training posterior；
2. prior；
3. KL regularization；
4. decoder / policy；
5. test-time latent choice。

这一篇只解决这个问题。

---

# 1. 先把训练和推理的 z 来源摆在一起

训练时：

\[
q_\phi(
z
\mid
a_{t:t+k}, q_t
)
\]

会根据 ground-truth action sequence 和当前 proprioception 产生一个 Gaussian posterior：

\[
q_\phi(z|\cdot)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma^2)
)
\]

然后：

\[
z
=
\mu+\sigma\odot\epsilon
\]

其中：

\[
\epsilon\sim\mathcal N(0,I)
\]

---

推理时则完全不同。

ACT 原论文直接规定：

\[
\boxed{
z=0
}
\]

论文原话对应的含义是：

> 在 test time，把 \(z\) 设为 prior distribution 的均值，也就是 0，从而 deterministic decode。

所以第一件必须确认的事实是：

> **ACT 并不是在推理时从 encoder 得到一个 \(\mu\)，然后令 \(z=\mu\)。**

推理阶段：

- CVAE encoder 被丢弃；
- \(\mu\) 不计算；
- \(\sigma^2\) 不计算；
- ground-truth future action 不存在；
- 直接使用 zero vector。

---

# 2. 为什么推理时根本不能继续用 Training Encoder？

训练 encoder 是：

\[
q_\phi(
z
\mid
a_{t:t+k},q_t
)
\]

注意它需要：

\[
a_{t:t+k}
\]

也就是：

> ground-truth future action chunk。

训练时我们当然有这个答案。

因为 demonstration dataset 已经告诉我们：

```text
当前 observation
+
人类接下来实际做了哪些动作
```

所以 encoder 可以看着真实 future actions 去推断：

> 这条 demonstration 对应什么 latent style distribution？

---

但推理时，我们正在问：

> 接下来应该执行哪些 actions？

因此：

\[
a_{t:t+k}
\]

正是未知答案。

如果还想运行：

\[
q_\phi(z\mid a_{t:t+k},q_t)
\]

就变成：

```text
为了预测未来 action
↓
先把未来 action 给 encoder
```

显然不可能。

所以 test time：

> **training recognition encoder 必须消失。**

这不是一个奇怪 hack。

这是 CVAE 的正常结构。

详细背景参见：

- [CVAE](../../generative-models/cvae.md)
- [CVAE in ACT](./cvae-in-act.md)
- [Posterior Collapse](../../generative-models/posterior-collapse.md)

---

# 3. 那一般 CVAE 推理时怎么办？

一般 fixed-prior CVAE 中，训练时可能有：

\[
q_\phi(z\mid c,y)
\]

而测试时 ground-truth \(y\) 不存在。

于是通常从 prior：

\[
p(z)
\]

中得到 latent。

例如：

\[
p(z)
=
\mathcal N(0,I)
\]

那么最自然的生成方式是：

\[
z\sim\mathcal N(0,I)
\]

然后：

\[
y\sim p_\theta(y\mid c,z)
\]

如果每次采不同：

\[
z_1,z_2,z_3
\]

就可能产生不同输出：

\[
y_1,y_2,y_3
\]

这正是 generative model 的典型用法。

---

# 4. 但 ACT 不想在机器人控制时随机抽“风格”

ACT 论文没有选择：

\[
z\sim\mathcal N(0,I)
\]

而是：

\[
\boxed{
z=0
}
\]

论文给出的直接目的非常清楚：

> **deterministically decode**

也就是：

> 给定同一个 observation，希望 policy output 是确定的。

Appendix 进一步明确指出：

> 给定 observation 时，policy output 始终 deterministic，这有利于 policy evaluation。

这很好理解。

假设机器人每次执行同一个任务时都随机：

```text
第一次：
z₁
→ 一种轨迹

第二次：
z₂
→ 另一种轨迹

第三次：
z₃
→ 又一种轨迹
```

那么 policy evaluation 会额外混入：

> latent sampling noise。

当一次任务失败时，你甚至更难判断：

- observation policy 本身有问题；
- 还是刚好采到了不好的 latent style。

所以 ACT 选择一个固定 latent：

\[
z=0
\]

让这一部分随机性消失。

---

# 5. 为什么偏偏是 0？

因为 ACT 使用的 prior 是：

\[
\boxed{
p(z)=\mathcal N(0,I)
}
\]

对于一维 standard normal：

\[
z\sim\mathcal N(0,1)
\]

均值：

\[
\mathbb E[z]=0
\]

对于 \(d\) 维标准高斯：

\[
z\sim\mathcal N(0,I)
\]

均值向量：

\[
\mathbb E[z]
=
\mathbf 0
\]

所以：

\[
\boxed{
z=0
}
\]

就是：

> **prior mean**

它不是随便拍脑袋选的一个数字。

如果不熟悉为什么 standard normal 的均值是 0，可以阅读：

- [Normal Distribution](../../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean

---

# 6. 现在真正困难的问题来了

你可能会说：

> 好，我知道 0 是 prior mean 了。

但是：

> **训练时 encoder 为不同 demonstration 学到的 \(z\) 明明可能不是 0。为什么 decoder 在测试时突然只看到 \(z=0\)，还能够工作？**

这个问题的答案就是：

> **KL regularization。**

---

# 7. 训练时 Posterior 不是可以随便跑

训练阶段，encoder 给出：

\[
q_\phi(z\mid a,q)
\]

如果完全没有限制，

encoder 可以把不同 demonstrations 编成：

```text
sample 1 → z = [100, -80, ...]
sample 2 → z = [-300, 52, ...]
sample 3 → z = [900, 700, ...]
```

只要 decoder 能根据这些 latent codes 重建 action chunk，

reconstruction loss 可能就很满意。

但这样会出现一个严重问题：

> test time 从哪里得到这些奇怪 latent codes？

因为 test time 没有 ground-truth action sequence。

---

# 8. 所以 ACT 给 Encoder 加了一个 Prior Constraint

ACT 的 loss 包含：

\[
D_{KL}
\left(
q_\phi(z\mid a,q)
\parallel
\mathcal N(0,I)
\right)
\]

也就是说：

> training posterior 不能无限自由地分布在 latent space 的任意位置。

它被持续鼓励：

\[
q_\phi(z\mid a,q)
\]

不要离：

\[
\mathcal N(0,I)
\]

太远。

这意味着 decoder 训练期间接触到的 latent samples 被鼓励位于：

> **standard normal prior 所覆盖的区域。**

而：

\[
z=0
\]

恰好位于这个 prior 的中心。

所以 test time 使用：

\[
z=0
\]

不是突然把 decoder 扔到一个完全陌生的位置。

---

# 9. 这就是 KL 和 z=0 之间真正的关系

可以画成：

```text
TRAINING

demonstration
↓
qφ(z | action, qpos)
↓
posterior sample z
↓
ACT decoder

同时：

qφ(z | action, qpos)
        │
        │ KL
        ▼
    N(0, I)
```

因此训练过程中：

> decoder 一边学习根据 \(z\) 和 observation 重建 action chunk，

同时 encoder 的 posterior 又被拉向：

\[
\mathcal N(0,I)
\]

---

推理时：

```text
prior = N(0,I)
↓
choose its mean
↓
z = 0
↓
ACT decoder
```

所以训练和推理并不是两个完全断开的世界。

中间的桥梁就是：

\[
\boxed{
D_{KL}
(
q_\phi(z|\cdot)
\parallel
p(z)
)
}
\]

---

# 10. 但要非常严谨：KL 不保证 z=0 是“最优风格”

这是一个必须划清的边界。

KL 确实鼓励：

\[
q_\phi(z|\cdot)
\]

靠近：

\[
\mathcal N(0,I)
\]

但不能因此推出：

\[
\boxed{
z=0
=
\text{数学意义上的最佳 action style}
}
\]

论文没有证明这种 theorem。

论文做的是一个具体设计选择：

1. prior 是 unit Gaussian；
2. prior mean 是 zero；
3. test time 使用 prior mean；
4. 因此 inference deterministic；
5. 实验上这个设计有效。

所以最准确的表述是：

> **ACT 选择 prior mean 作为一个 canonical deterministic latent condition。**

不是：

> “0 是理论上最正确的 style。”

---

# 11. 一个直觉例子：训练时学习“允许的变化范围”

假设为了直觉，我们把 \(z\) 简化成一维。

不同 human demonstrations 可能形成 posterior：

\[
q_1(z)
=
\mathcal N(-0.4,0.3^2)
\]

\[
q_2(z)
=
\mathcal N(0.2,0.4^2)
\]

\[
q_3(z)
=
\mathcal N(0.5,0.3^2)
\]

因为 KL 要求这些 distribution 不要离：

\[
\mathcal N(0,1)
\]

太远，

它们整体会被约束在 prior 周围。

可以想成：

```text
              prior N(0,1)

       -----------0-----------
          q1     q2     q3
```

推理时：

\[
z=0
\]

位于训练 latent space 的中心区域。

因此 decoder 对这一位置并不陌生。

---

# 12. 但不要把这张图理解得太机械

真实 ACT 中：

\[
z\in\mathbb R^{32}
\]

是高维 latent vector。

而且 posterior 是 diagonal Gaussian：

\[
q_\phi(z|\cdot)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma^2)
)
\]

所以真实情况不是简单的一条数轴。

我们的一维图只是帮助理解：

> KL 把 training posterior 限制在 prior 附近，

而不是说每条 demonstration 真能画成漂亮分离的小高斯峰。

---

# 13. “z=0 就是没有 Style”吗？

这是一个很自然的误解。

因为：

\[
0
\]

听起来像：

> “什么都没有。”

但在 latent space 里：

\[
0
\]

不是“空”。

它只是：

> **一个具体 latent vector。**

例如：

\[
z=
[0,0,\ldots,0]
\]

经过：

```python
latent_out_proj(z)
```

仍然会进入 ACT policy。

而 `latent_out_proj` 是一个带 bias 的 Linear layer。

即使输入是全零：

\[
z=0
\]

输出也不一定是全零：

\[
Wz+b
=
b
\]

所以从网络计算角度：

> `z = 0` 不等于“删除 latent token”。

policy 仍然得到一个确定的 latent conditioning representation。

---

# 14. 这个代码细节非常有意思

官方实现：

```python
latent_sample = torch.zeros(
    [bs, self.latent_dim]
)

latent_input =
    self.latent_out_proj(latent_sample)
```

Linear：

\[
e_z
=
Wz+b
\]

当：

\[
z=0
\]

时：

\[
\boxed{
e_z=b
}
\]

因此 policy Transformer 实际接收到的是：

> latent projection layer 在 zero latent 下产生的一个 learned fixed embedding。

这进一步说明：

> **z=0 绝不是“把 z 这条输入线拔掉”。**

它是把 latent variable 固定到一个特定 canonical point。

---

# 15. 那 z 训练时到底学了什么？

这里要非常谨慎。

ACT 论文把 \(z\) 称为：

> style variable

它的作用是帮助建模 human demonstrations 中的 variability。

但我们不能说：

> \(z\) 明确学到了“昨晚睡得好不好”“力气大小”“动作速度”等具体因素。

因为没有这种 supervision。

更准确地说：

\[
z
\]

可以承载：

> **当前 observation 无法完全解释、但有助于 reconstruction 当前 demonstration action chunk 的 latent variation。**

这可能与：

- trajectory choice；
- timing；
- subtle motion differences；
- human stochasticity；

有关。

但具体每个 latent dimension 代表什么：

> 论文没有给出可解释性保证。

---

# 16. 一个你可能会有的直觉：z 用来编码“额外小因素”

这个直觉有一部分是有价值的。

例如可以想象：

> 当前 observation 已经解释了任务的大部分信息。

但 human action sequence 中还有一些：

> 没有被 observation 唯一决定的变化。

CVAE 允许：

\[
z
\]

帮助表示这些“剩余 variation”。

这一点是合理的。

但是下一步如果说：

> “推理时 \(z=0\)，就是把这些额外因素消除掉”

就需要修正。

---

# 17. 为什么“z=0 = 消除额外因素”不够准确？

因为 latent space 的坐标：

\[
z_1,z_2,\ldots,z_{32}
\]

没有被监督成：

```text
z₁ = 睡眠状态
z₂ = 手臂力量
z₃ = 操作者紧张程度
...
```

因此：

\[
z=0
\]

不能解释成：

> 每一个现实因素都被设为“没有”。

更准确的是：

> **把模型的 latent conditioning 固定在 prior 的中心点。**

它是模型内部 latent coordinate system 中的：

\[
\mathbf 0
\]

不是现实世界所有 nuisance factors 的“零值”。

---

# 18. 用“搬水弯腰”的例子重新修正这个直觉

假设为了教学，我们有一个模型预测：

> 搬水时接下来应该怎样弯腰和发力。

训练 demonstrations 中存在 variation：

```text
动作 A
略微蹲得深一点

动作 B
腰弯得多一点

动作 C
先调整脚的位置
```

这些差异可能来自：

- 个体习惯；
- 状态；
-微小姿态变化；
- 演示随机性。

模型可能利用 \(z\) 帮助 reconstruction 这些不同轨迹。

但：

\[
z=0
\]

不能严格解释为：

> “假设这个人昨晚既没睡好也没睡差，力量等所有额外因素都处于客观标准状态。”

因为 latent dimensions 未必对应这些物理变量。

更准确的说法是：

> **训练时模型允许一个受 Gaussian prior 约束的 latent variable 来吸收 demonstration-specific variation；推理时 ACT 不再选择某条具体 demonstration 的 latent posterior，而固定使用 prior center 这一统一 latent condition。**

这个表达更严谨。

---

# 19. z=0 会不会意味着“平均风格”？

也要小心。

因为：

\[
0
\]

是 Gaussian prior 的 mean。

所以我们可以用直觉说：

> 它是 latent prior 的中心 / canonical style。

但不能直接声称：

> 它等于所有 human action styles 的算术平均轨迹。

这两个概念完全不同。

\[
\mathbb E[z]=0
\]

并不意味着：

\[
\pi(o,\mathbb E[z])
=
\mathbb E_z[\pi(o,z)]
\]

一般非线性神经网络中：

\[
f(\mathbb E[z])
\neq
\mathbb E[f(z)]
\]

所以：

> **用 mean latent 解码，不等于把所有可能 action trajectories 做平均。**

这是非常重要的数学区别。

---

# 20. 为什么 f(E[z]) 不等于 E[f(z)]？

假设一个非常简单的 nonlinear decoder：

\[
f(z)=z^2
\]

如果：

\[
z\sim\mathcal N(0,1)
\]

那么：

\[
\mathbb E[z]=0
\]

所以：

\[
f(\mathbb E[z])
=
f(0)
=
0
\]

但是：

\[
\mathbb E[f(z)]
=
\mathbb E[z^2]
=
1
\]

所以：

\[
\boxed{
f(\mathbb E[z])
\neq
\mathbb E[f(z)]
}
\]

因此 ACT 的：

\[
z=0
\]

不要理解成：

> “输出所有动作风格的平均动作。”

它只是：

> **在 prior mean 这个 latent point 上运行 decoder。**

---

# 21. 那为什么不从 Prior 随机采样，然后多跑几次选最好的？

理论上可以设计这样的系统。

例如：

\[
z_1,z_2,\ldots,z_M
\sim
\mathcal N(0,I)
\]

得到多个 action chunks：

\[
A_1,A_2,\ldots,A_M
\]

然后再用一个 critic / cost function / verifier 选择。

但原始 ACT 没有这样做。

它选择：

> 简单、确定、容易评估的 \(z=0\)。

所以如果以后看到某个 ACT 变体使用：

- latent sampling；
- best-of-N；
- diffusion；
- latent planner；

那是后续模型设计。

不能反过来写进原始 ACT。

---

# 22. β 在这里扮演什么角色？

ACT 使用：

\[
L
=
L_{\text{recon}}
+
\beta
L_{\text{KL}}
\]

其中：

\[
L_{\text{KL}}
=
D_{KL}
(
q_\phi(z|\cdot)
\parallel
\mathcal N(0,I)
)
\]

\(\beta\) 控制：

> KL regularization 有多强。

---

如果：

\[
\beta
\]

很小，

encoder 可以更自由地利用 \(z\)：

```text
不同 demonstrations
→ posterior 可以相差更大
→ z 可以携带更多 action-specific 信息
```

这有利于 reconstruction。

但问题是：

> training posterior 可能离 prior 更远。

那么 test time 直接使用：

\[
z=0
\]

可能更容易进入 training 较少使用的 latent region。

---

如果：

\[
\beta
\]

较大：

```text
posterior
→ 更强地被拉向 N(0,I)
→ latent information capacity 受到更强限制
```

这使 training latent distribution 与 prior 更一致。

但如果过强：

> \(z\) 可能几乎不携带 demonstration information。

所以：

\[
\beta
\]

控制的是一个非常重要的 trade-off。

---

# 23. 为什么 β 不是越大越好？

如果只优化 KL：

\[
q_\phi(z|\cdot)
=
\mathcal N(0,I)
\]

对所有 samples 都一样，

KL 可以做到非常小。

但这时：

\[
z
\]

无法告诉 decoder：

> 当前 demonstration 有什么特殊 variation。

CVAE encoder 几乎失去作用。

---

反过来，如果完全没有 KL：

> encoder 可以把 action sequence 几乎当作秘密密码塞进 \(z\)。

reconstruction 可能很好，

但 test time：

\[
z=0
\]

完全接不上。

所以：

```text
Reconstruction
要求 z 有信息

          ↕

KL
要求 z 接近 prior
```

这就是 VAE/CVAE 最核心的张力。

---

# 24. 为什么 ACT 的 CVAE 对 Human Data 特别重要？

论文指出 human demonstrations 本身具有 stochasticity。

例如同一个 tape handover：

> 每个 episode 的 handover 位置都会不同。

人类没有精确 reference 去保证每次都在完全相同位置交接。

所以 policy 不应该简单 memorise：

> “交接一定发生在这个坐标。”

而应该学习：

> 当前 observation 下怎样合理完成行为。

CVAE 提供了一个机制：

> 训练时允许 demonstration-specific variation 被 latent \(z\) 吸收一部分，

而 decoder 同时学习 observation-conditioned action generation。

---

# 25. 一个很重要的观点：z 的作用主要发生在训练过程中

这可能是理解 ACT 最关键的一句话之一。

很多人会认为：

> 一个变量只有推理时还在动态变化，它才算“有用”。

并不是。

CVAE 的 \(z\) 在 ACT 中一个非常重要的作用是：

> **改变训练问题的组织方式。**

训练时模型面对的不是简单：

\[
o_t
\rightarrow
a_{t:t+k}
\]

而是：

\[
(o_t,z)
\rightarrow
a_{t:t+k}
\]

其中 \(z\) 来自：

\[
q_\phi(z\mid a_{t:t+k},q_t)
\]

于是不同 demonstrations 中的一部分 variation 可以通过 latent channel 被解释。

即使推理时最后固定：

\[
z=0
\]

decoder 参数：

\[
\theta
\]

也已经是在这种 CVAE objective 下学出来的。

所以：

> **test-time 固定 latent，不等于 training-time latent 没有价值。**

---

# 26. 一个类比：训练辅助变量不一定要在推理时保持同样形式

这里只做帮助理解的类比。

例如某些训练方法中：

- data augmentation 只发生在训练；
- teacher forcing 的 ground-truth context 只发生在训练；
- dropout 训练和推理行为不同。

这些机制在 test time 并不原样存在，

但它们仍然改变了：

> 模型学到的参数。

ACT 的 CVAE 当然不是这些技术中的任何一个。

但共同的高层思想是：

> **“训练阶段存在”与“推理阶段必须同样存在”不是一回事。**

---

# 27. 为什么 z=0 仍然让 Observation 起主要作用？

ACT policy 是：

\[
\pi_\theta(
a_{t:t+k}
\mid
o_t,z
)
\]

即使：

\[
z=0
\]

仍然有：

\[
o_t
\]

而 \(o_t\) 包括：

- 4 camera images；
- current joint positions。

所以：

\[
z=0
\]

并没有让模型失去环境信息。

真正决定：

> 当前物体在哪里、机械臂在哪里、下一步任务应该怎么做

的主要信息仍然来自 observation。

\(z\) 是：

> **额外 conditioning variable**

而不是整个 policy information 的唯一来源。

---

# 28. 如果 z=0，模型为什么还能产生复杂动作？

因为复杂动作知识存在于：

\[
\theta
\]

也就是训练好的网络参数中。

训练大量 demonstrations 后，policy weights 已经学会：

```text
image patterns
+
joint state
+
latent condition
↓
action sequence
```

推理时：

\[
z=0
\]

只是固定最后这一项。

真正的 learned behavior 并没有存储在：

> 某个 test-time \(z\) 里面。

\(z\) 不是一个“动作数据库”。

它只是 policy input space 的一个 latent axis / latent condition。

---

# 29. 一个极端思想实验

假设 decoder 完全忽略：

\[
z
\]

也就是说：

\[
\pi_\theta(a\mid o,z)
\approx
\pi_\theta(a\mid o)
\]

那么：

\[
z=0
\]

当然完全没问题。

---

另一个极端：

decoder 极度依赖 \(z\)，

并且 training posteriors 都远离：

\[
0
\]

那么 test time 突然：

\[
z=0
\]

就会很危险。

ACT 的训练目标通过：

\[
KL(q||N(0,I))
\]

正是在避免第二种极端：

> 让 training latent space 与 test-time prior reference 完全脱节。

真实模型处在两个极端之间。

---

# 30. 这也解释了为什么“Posterior Collapse”并不一定简单

如果：

\[
q_\phi(z|\cdot)
\approx
\mathcal N(0,I)
\]

对所有 demonstration 都几乎一样，

decoder 可能学会：

> 完全不使用 \(z\)。

这就是 posterior collapse 风格的退化。

此时 test time：

\[
z=0
\]

当然也能工作，

但 CVAE 没有真正利用 latent variation。

---

而如果 \(z\) 携带太多信息，

posterior 与 prior 差异过大，

test time 又可能出现 mismatch。

所以理想状态是：

> \(z\) 有用，但受到 prior 约束。

这也是 KL-weighted CVAE 的核心 trade-off。

---

# 31. “为什么不直接训练时也一直 z=0？”

这是一个非常好的问题。

如果训练时永远：

\[
z=0
\]

那么模型退化为：

\[
\pi_\theta(
a_{t:t+k}
\mid
o_t
)
\]

CVAE encoder 完全没有作用。

对于同一个或相似 observation 下存在的 demonstration variability，

模型只能全部塞进：

> deterministic observation-to-action mapping

中。

这正是 ACT 引入 CVAE 想改善的情况。

所以：

```text
训练时允许 z 变化
```

让模型有机会解释 human demonstration variability；

而：

```text
推理时固定 z=0
```

让最终机器人 policy deterministic。

这两件事并不矛盾。

---

# 32. 为什么不训练时 z 随机从 N(0,I) 采，而不用 Encoder？

如果训练一开始就：

\[
z\sim\mathcal N(0,I)
\]

且这个 \(z\) 与当前 action chunk 没有任何关系，

那么 decoder 不知道：

> 这次 random \(z\) 应该对应 demonstration 中哪一种 variation。

例如同一个 sample：

```text
ground-truth trajectory A
```

今天随机：

\[
z=1.2
\]

明天又随机：

\[
z=-0.7
\]

完全没有 consistent relationship。

模型最简单的做法可能就是：

> 忽略 \(z\)。

---

而 training encoder：

\[
q_\phi(z\mid a,q)
\]

让 latent 与当前 demonstration 建立学习到的关系。

所以：

> encoder 不是为了制造随机性，

而是为了：

> **做 posterior inference。**

---

# 33. 那为什么 Training Posterior 还要有 σ，不直接用 μ 表示 Style？

因为 ACT 使用的是 VAE/CVAE objective。

posterior 是：

\[
q_\phi(z|\cdot)
=
\mathcal N(
\mu,
\operatorname{diag}(\sigma^2)
)
\]

不是一个 deterministic point。

这样：

- 可以定义 KL；
- 可以进行 variational training；
- decoder 在 posterior 邻域 sample 上训练；
- latent representation 与 prior distribution 建立概率上的匹配。

如果直接：

\[
z=\mu
\]

就变成 deterministic latent encoder，

不再是同一个 CVAE training objective。

---

# 34. z=0 和 Reparameterization 没有直接因果关系

必须再次强调。

训练时：

\[
z
=
\mu+\sigma\epsilon
\]

是：

> **Reparameterization Trick**

解决：

> stochastic posterior sample 如何高效反向传播。

推理时：

\[
z=0
\]

是：

> **ACT-specific inference policy**

解决：

> 没有 ground-truth action 后使用什么 latent，并保持 deterministic decoding。

不能说：

> “因为 reparameterization，所以 inference z=0。”

这两个设计解决完全不同的问题。

---

# 35. z=0 和 KL 也不是“直接推导”的关系

同样不能写：

\[
KL
\Rightarrow
z=0
\]

KL 的数学作用是：

\[
q_\phi(z|\cdot)
\]

接近：

\[
p(z)=\mathcal N(0,I)
\]

而从：

\[
p(z)
\]

中，ACT 仍然有很多可能选择：

### 随机采样

\[
z\sim p(z)
\]

### 使用均值

\[
z=\mathbb E[z]=0
\]

### 多样本生成

\[
z_1,\ldots,z_M\sim p(z)
\]

等等。

ACT **人为选择**：

\[
z=0
\]

以获得 deterministic decoding。

所以更准确的因果链：

```text
KL training
↓
posterior 被 regularize toward N(0,I)
↓
N(0,I) 成为有意义的 test-time reference prior
↓
ACT 选择 prior mean 0
↓
deterministic decoding
```

---

# 36. 这条逻辑链非常重要

可以把整个故事浓缩成：

```text
训练时有 ground-truth future action
        ↓
CVAE encoder 能推断 posterior
        ↓
qφ(z | actions, qpos)
        ↓
sample z
        ↓
decoder 学习 observation + z → action chunk

同时
        ↓
KL 把 posterior 拉向 N(0,I)
        ↓
training latent 与 known prior 建立连接


推理时没有 future action
        ↓
encoder 无法使用，直接丢弃
        ↓
需要从 prior 侧选择 z
        ↓
ACT 不随机 sample
        ↓
选择 prior mean
        ↓
z = 0
        ↓
deterministic policy output
```

如果这条链理解了，

ACT 的：

\[
z=0
\]

就不再反直觉了。

---

# 37. 一个更形式化的视角

训练时，decoder 学的是：

\[
\pi_\theta(a\mid o,z)
\]

其中 \(z\) 的 training distribution 来自：

\[
q_\phi(z\mid a,q)
\]

同时优化：

\[
D_{KL}
(
q_\phi(z\mid a,q)
\parallel
p(z)
)
\]

使：

\[
q_\phi
\]

不要离：

\[
p(z)
\]

太远。

---

test time 则选择：

\[
z^\star
=
\mathbb E_{p(z)}[z]
\]

由于：

\[
p(z)=\mathcal N(0,I)
\]

所以：

\[
\boxed{
z^\star=0
}
\]

最终 policy：

\[
\boxed{
\pi_\theta(a\mid o,z^\star)
=
\pi_\theta(a\mid o,0)
}
\]

这就是 ACT test-time decoder。

---

# 38. 注意：这不是在计算条件期望动作

ACT 并没有计算：

\[
\mathbb E_{z\sim p(z)}
[
\pi_\theta(a\mid o,z)
]
\]

它计算的是：

\[
\pi_\theta(
a\mid o,\mathbb E[z]
)
\]

也就是：

\[
\pi_\theta(a\mid o,0)
\]

由于 decoder 是 nonlinear neural network，

一般：

\[
\boxed{
\pi_\theta(a\mid o,\mathbb E[z])
\neq
\mathbb E_z[
\pi_\theta(a\mid o,z)
]
}
\]

所以：

> `z = 0` 不是“对所有 z 的动作取平均”。

这一点一定要分清。

---

# 39. 那 0 是不是最常出现的 z？

对连续 Gaussian 来说，这个问题也要说得精确。

对于一维：

\[
\mathcal N(0,1)
\]

0 同时是：

- mean；
- median；
- mode。

多维 standard Gaussian 的 density 最大点也是：

\[
\mathbf 0
\]

所以可以说：

> zero 位于 prior 的最高密度中心。

但连续分布中：

> “精确采到 \(z=0\) 的概率”

仍然是：

\[
0
\]

因为单个点的概率质量为 0。

所以不要说：

> “训练时经常随机采到 z=0。”

不对。

训练时几乎不会精确采到全零向量。

但会大量采到：

> prior 中心附近的区域。

而 KL 又鼓励 posterior 与该 prior 匹配。

---

# 40. 这一点进一步解释了为什么 decoder 能处理 zero

虽然 training 时精确：

\[
z=0
\]

这个 32 维点几乎不会被随机命中，

neural network 是连续函数，

训练 samples 会覆盖：

> zero 周围的 latent region。

而：

\[
0
\]

正处于 Gaussian 的中心高密度区域。

因此使用 prior center 通常比选择一个远离 prior mass 的任意 latent vector合理得多。

但再次强调：

> 这是基于连续性和 prior geometry 的解释，不是论文给出的性能保证 theorem。

---

# 41. 一个重要问题：如果 KL 很弱，z=0 会不会变差？

从原理上：

> 有这种风险。

如果 KL weight 太弱，

training posterior 可以远离：

\[
\mathcal N(0,I)
\]

那么 decoder 主要在一些偏离 zero 的 latent regions 上训练。

test time 突然：

\[
z=0
\]

可能产生更大的 train-test mismatch。

这就是为什么：

> prior regularization 对 fixed-prior CVAE 非常重要。

但具体多弱会失败、最优权重是多少：

> 必须通过实验确定。

不能只凭理论给一个固定答案。

---

# 42. 如果 KL 很强，为什么 z=0 可能更自然？

如果 posterior distributions 都被较强地约束到：

\[
\mathcal N(0,I)
\]

附近，

那么 decoder 训练时见到的 \(z\) 会更多覆盖 prior 的高密度区域。

test time 选择：

\[
0
\]

与 training latent geometry 更一致。

但代价是：

> latent channel 能携带的 demonstration-specific information 减少。

所以这仍然不是：

> KL 越强越好。

而是 trade-off。

---

# 43. 可以把 z 理解成“Residual Information”吗？

作为直觉，可以谨慎使用。

observation：

\[
o_t
\]

已经包含：

- scene；
- object positions；
- robot state。

如果这些信息已经决定了大部分动作，

那么 \(z\) 可以帮助解释：

> demonstration 中没有被 observation 唯一确定的剩余变化。

从这个角度可以把它直觉称为：

> residual / extra latent variation。

但必须保留两个限定：

1. 这是直觉，不是 \(z\) 的数学定义；
2. \(z\) 具体编码了什么没有可解释性保证。

---

# 44. 为什么说“推理时消除 variation”只能作为粗略直觉？

ACT 确实：

- 训练时允许 stochastic \(z\)；
- 推理时固定 \(z=0\)。

所以高层上可以说：

> test time 不再随机选择不同 latent variations。

这个说法是合理的。

但如果进一步说：

> “因此模型把人类示范中的噪声全部消除了”

就过头了。

因为 demonstration variability 已经影响了：

- learned weights；
- training targets；
- decoder mapping。

而且 observation 本身仍可能产生复杂行为。

所以更准确的是：

> **ACT 在推理时消除了 latent sampling 这一来源的随机性，而不是数学上删除所有 human variability 的影响。**

---

# 45. 这也是“Deterministic Decode”真正的意思

论文说：

> set \(z\) to prior mean to deterministically decode.

这里的 deterministic 指：

> 给定固定 observation，latent input 不再随机。

因此模型 forward：

\[
o_t
\rightarrow
a_{t:t+k}
\]

在相同参数和确定性算子下会给相同输出。

它不是说：

> 真实机器人执行一定完全 deterministic。

现实系统仍有：

- sensor noise；
- control error；
- environment perturbation；
- floating-point / hardware details；

等因素。

论文说的是：

> **policy mapping 本身不再因为 latent sampling 而随机。**

---

# 46. 为什么这对 Temporal Ensemble 也很方便？

ACT 在 inference 时每个 timestep 都重新 query policy：

\[
o_t
\rightarrow
\hat a_{t:t+k}
\]

如果每次 query 还额外随机：

\[
z_t\sim\mathcal N(0,I)
\]

那么 overlapping chunks 之间的差异会同时来自：

1. observation 更新；
2. latent random sampling。

Temporal Ensemble 聚合时会多一个随机 variation source。

原论文没有把这一点作为 \(z=0\) 的正式动机来展开，

所以不能说这是论文明确证明的理由。

但从系统设计直觉上：

> 固定 \(z\) 会让 overlapping predictions 的变化主要来自 observation，而不是额外 latent sampling。

这与 deterministic control 的目标是相容的。

---

# 47. 一个常见误解：z=0 是 Normalization

不是。

Normalization 通常指：

> 把数据转换到均值 0、方差 1 的尺度。

ACT 的：

\[
z=0
\]

不是在 normalize test input。

而是：

> **选择 latent prior 的 mean 作为固定 latent sample。**

这是完全不同的概念。

---

# 48. 常见误解一：训练时模型努力让所有 z 都等于 0

**错误。**

KL 鼓励的是：

\[
q_\phi(z|\cdot)
\]

接近：

\[
\mathcal N(0,I)
\]

而：

\[
\mathcal N(0,I)
\]

不是：

\[
z=0
\]

这个单点。

它是一个具有方差的 distribution。

---

# 49. 常见误解二：N(0,I) 就是所有 z 都在 0

**错误。**

一维：

\[
z\sim\mathcal N(0,1)
\]

意味着：

- mean = 0；
- variance = 1。

样本可能是：

\[
-0.4,\;0.7,\;1.2,\;-1.8,\ldots
\]

多维同理。

所以：

\[
p(z)=\mathcal N(0,I)
\]

与：

\[
z=0
\]

是：

> distribution

和：

> distribution 中的一个特定点

的区别。

---

# 50. 常见误解三：推理时 z=0，因为 Encoder 学会输出 μ=0

**错误。**

推理时 encoder 根本不运行。

所以不存在：

\[
\mu_{\text{test}}
\]

官方代码直接创建：

```python
torch.zeros(...)
```

---

# 51. 常见误解四：z=0 等于没有 z

**错误。**

zero vector 仍然进入：

\[
latent\_out\_proj
\]

而：

\[
W0+b=b
\]

所以 Transformer 仍然接收一个固定 latent embedding。

更重要的是：

> 网络参数本身是在 latent-conditioned CVAE objective 下训练出来的。

---

# 52. 常见误解五：z=0 等于所有 demonstration 的平均动作

**错误。**

一般：

\[
f(\mathbb E[z])
\neq
\mathbb E[f(z)]
\]

所以：

\[
\pi(o,0)
\]

不是：

> 所有 latent action outputs 的平均。

---

# 53. 常见误解六：z=0 把 human noise 全部删除了

**过度解释。**

它只删除：

> test-time latent sampling stochasticity。

human demonstration variability 已经参与训练并塑造 model weights。

---

# 54. 常见误解七：一般 CVAE 都应该推理 z=0

**错误。**

一般 CVAE 常常：

\[
z\sim p(z)
\]

以产生 diverse outputs。

ACT 选择：

\[
z=0
\]

是它自己的 deterministic control design。

---

# 55. 常见误解八：既然测试不用随机 z，CVAE 完全可以删掉

论文 ablation 表明：

> 对 human demonstrations，CVAE training 本身是重要的。

因此：

> test time encoder 不存在

不能推出：

> training CVAE 没作用。

training objective 本身就会改变最终 policy。

---

# 56. 用一个完整训练例子再走一遍

假设某时刻 observation 大致相似。

human demonstration A：

```text
action chunk A
```

encoder 得到：

\[
q_A(z)
=
\mathcal N(\mu_A,\sigma_A^2)
\]

sample：

\[
z_A
\]

decoder 学：

\[
(o,z_A)
\rightarrow
A
\]

---

demonstration B：

```text
action chunk B
```

encoder 得到：

\[
q_B(z)
\]

sample：

\[
z_B
\]

decoder 学：

\[
(o,z_B)
\rightarrow
B
\]

同时：

\[
q_A(z),q_B(z)
\]

都被 KL 拉向：

\[
\mathcal N(0,I)
\]

---

测试时：

没有 A 或 B 的 ground-truth。

所以不问：

> “这次应该用 \(q_A\) 还是 \(q_B\)？”

ACT 直接：

\[
z=0
\]

然后：

\[
(o,0)
\rightarrow
\text{一个 deterministic action chunk}
\]

这就是完整逻辑。

---

# 57. 为什么 Decoder 能学会在 z=0 附近工作？

因为训练不是：

```text
每个 demonstration
→ 一个完全孤立的 deterministic code
```

而是：

```text
每个 demonstration
→ 一个 Gaussian posterior
→ sample 一片局部 latent region
```

同时这些 Gaussian 又被 KL regularize toward：

\[
N(0,I)
\]

所以训练本身鼓励：

> latent distributions 有共同的 prior geometry。

这就是 VAE/CVAE 相比 deterministic autoencoder 的关键优势之一。

---

# 58. 一个更严格的“睡得好”例子

假设 human demonstration 的 variation 确实部分与操作者当天状态相关。

但 dataset 没有标签：

```text
sleep_quality = 0.8
```

因此 CVAE 不知道“睡眠质量”这个概念。

它只知道：

```text
observation
+
实际 action sequence
```

然后通过 reconstruction + KL 学到某个 latent representation。

所以我们最多可以说：

> \(z\) 有能力帮助表示 demonstration 中 observation 未解释的 variation。

不能说：

> \(z\) 的某一维就是睡眠质量。

因此 test time：

\[
z=0
\]

也不是：

> “假设睡眠质量为 0。”

而是：

> **使用 learned latent coordinate system 的 prior center。**

这是两个完全不同的层次。

---

# 59. 那你之前的理解怎样修正成严谨版本？

一个比较好的版本可以是：

> **训练时，ACT 使用 latent \(z\) 来帮助解释同一类 observation 下 human action sequence 中额外的、没有被 observation 唯一决定的变化。KL 又限制这些 latent posteriors 不要脱离标准高斯 prior。推理时，因为 ground-truth future actions 不存在，训练 encoder 无法使用；ACT 因此不再推断某条具体 demonstration 的 style，而固定使用 prior 的中心 \(z=0\)，得到一个 deterministic 的 canonical latent condition。**

这个理解：

- 保留了“\(z\) 处理额外 variation”的直觉；
- 但没有把 \(z\) 强行解释成具体现实因素；
- 也没有说 \(z=0\) 是“把所有因素消除”。

这是更准确的 mental model。

---

# 60. 用四层关系记住全部内容

## 第一层：Posterior

训练时：

\[
q_\phi(z\mid a,q)
\]

回答：

> 这条 demonstration 对应什么 latent distribution？

---

## 第二层：Prior

规定：

\[
p(z)=\mathcal N(0,I)
\]

回答：

> latent space 的共同 reference distribution 是什么？

---

## 第三层：KL

\[
D_{KL}(q_\phi\parallel p)
\]

回答：

> 如何避免 training posterior 和 test-time prior 完全脱节？

---

## 第四层：Inference Choice

\[
z=\mathbb E_{p(z)}[z]=0
\]

回答：

> test time 没有 target action 后，ACT 具体选哪个 latent？

并且目标是：

> deterministic decoding。

---

# 61. 用三条公式记住

如果最后只记住三条公式：

---

## Training Posterior

\[
\boxed{
q_\phi(
z\mid
a_{t:t+k},q_t
)
}
\]

---

## Prior Regularization

\[
\boxed{
D_{KL}
\left(
q_\phi(z|\cdot)
\parallel
\mathcal N(0,I)
\right)
}
\]

---

## Test-Time Latent

\[
\boxed{
z
=
\mathbb E_{
\mathcal N(0,I)
}[z]
=
0
}
\]

它们连起来就是：

```text
training posterior
↓
regularized toward prior
↓
test time uses prior mean
```

---

# 62. 一句话重新理解

> **ACT 推理时令 \(z=0\)，不是因为 latent variable 没有用，也不是因为模型把所有“额外因素”都清除了，而是因为 training encoder 依赖 ground-truth future actions、测试时无法使用；CVAE 训练通过 KL 把 posterior 约束到标准高斯 prior 附近，因此 ACT 在 test time 选择这个 prior 的均值 \(0\) 作为一个固定的 canonical latent condition，从而得到 deterministic policy output。**

训练时：

\[
z
\]

帮助模型解释 demonstration variability。

推理时：

\[
z=0
\]

则取消 latent sampling 这一额外随机来源。

而最终动作仍然主要由：

- camera observations；
- current joint positions；
- learned policy parameters；

决定。

---

# 63. 下一步

现在 ACT 的 CVAE 主线已经完整：

```text
Latent Variable
↓
VAE
↓
Reparameterization
↓
CVAE
↓
CVAE in ACT
↓
Why z = 0?
```

接下来最自然的是进入完整网络结构：

> [ACT Architecture](./architecture.md)

下一篇会把论文 Figure / Appendix architecture 真正按 tensor flow 拆开：

```text
4 × RGB Images
↓
ResNet18
↓
4 × 300 Visual Tokens
↓
concatenate
↓
1200 × 512

Joint State
↓
Linear
↓
1 × 512

z
↓
Linear
↓
1 × 512

全部进入 Transformer Encoder
↓
1202 × 512 memory

k Action Queries
↓
Transformer Decoder
↓
k × 512

Action Head
↓
k × 14
```

并明确解释：

- Transformer encoder 为什么已经有 image features 还需要 joint / z tokens；
- decoder query 到底是什么；
- Query、Key、Value 在 ACT cross-attention 中分别来自哪里；
- 为什么固定 query embeddings 能输出未来不同 timestep 的动作；
- ACT 中 Transformer encoder / decoder 与 CVAE encoder 到底是什么关系。

---

## Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.  
**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
Robotics: Science and Systems (RSS), 2023.

- Paper: https://arxiv.org/abs/2304.13705
- RSS Proceedings: https://roboticsproceedings.org/rss19/p016.html
- Project: https://tonyzhaozh.github.io/aloha/

本文主要依据原论文：

- Section IV-B — Modeling Human Data
- Figure 2 — ACT CVAE architecture
- Algorithm 1 — ACT Training
- Algorithm 2 — ACT Inference
- Appendix C — Detailed Architecture Diagram

论文明确规定：

\[
z
\]

在 test time 被设置为 prior distribution 的 mean，即：

\[
z=0
\]

以 deterministic decode。

Appendix 进一步说明：

> 给定 observation，policy output 始终 deterministic，从而有利于 policy evaluation。

---

## Official Implementation

ACT official repository:

https://github.com/tonyzhaozh/act

主要对应：

```text
detr/models/detr_vae.py
policy.py
```

官方 inference branch：

```python
mu = logvar = None

latent_sample = torch.zeros(
    [bs, self.latent_dim],
    dtype=torch.float32
).to(qpos.device)

latent_input =
    self.latent_out_proj(latent_sample)
```

这确认：

- test time 不运行 CVAE encoder；
- 不计算 \(\mu\)；
- 不计算 `logvar`；
- 不从 prior 随机采样；
- 直接使用 zero latent vector；
- 再通过 learned `latent_out_proj` 输入 policy。

---

## 本文知识连接

### 前置知识

- [Latent Variable](../../generative-models/latent-variable.md)
- [VAE](../../generative-models/vae.md)
- [Reparameterization Trick](../../generative-models/reparameterization-trick.md)
- [CVAE](../../generative-models/cvae.md)
- [CVAE in ACT](./cvae-in-act.md)

### 数学基础

- Probability Distribution
- [Normal Distribution](../../mathematics/normal-distribution.md)
- Standard Normal Distribution
- Mean
- Variance
- Expectation
- [KL Divergence](../../mathematics/kl-divergence.md)

### ACT

- [ACT 到底解决了什么问题？](./act-what-problem-does-it-solve.md)
- [Action Chunking](./action-chunking.md)
- [Temporal Ensemble](./temporal-ensemble.md)

### 下一步

- [ACT Architecture](./architecture.md)
- [ACT Training](./training.md)
- [ACT Inference](./inference.md)
