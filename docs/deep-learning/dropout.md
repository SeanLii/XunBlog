---
title: "Dropout：训练时随机删神经元，为什么推理时反而全部使用？"
description: "从 Bernoulli mask 出发，严格推导 inverted dropout 为什么训练时要除以 1-p、为什么推理时变成 identity、它怎样形成 stochastic regularization，以及 Transformer / ACT 中 attention-weight dropout、FFN dropout 与 residual dropout 分别位于哪里。"
status: reviewed
pageType: concept
canonical: /deep-learning/dropout
updated: "2026-09-15"
---

# Dropout：训练时随机删神经元，为什么推理时反而全部使用？

Dropout 是一个非常容易“会用，但没真正理解”的模块。

你可能已经见过：

```python
nn.Dropout(0.1)
```

也知道一句标准解释：

> 训练时随机丢掉一些神经元，防止过拟合。

但如果认真追问，很快会出现一连串问题：

1. “随机丢掉”到底是什么意思？
2. 真的是把 neuron 删除了吗？
3. 为什么训练时随机删，推理时反而一个都不删？
4. 如果训练时只剩 90% activation，推理时突然全部恢复，不会让数值整体变大吗？
5. 为什么 PyTorch 训练时还要把剩下的值乘：
   $$
   \frac{1}{1-p}
   $$
   ？
6. 这个缩放为什么能让 train / eval 对齐？
7. Dropout 到底是在模拟 ensemble，还是只是在“制造噪声”？
8. 原始 Dropout 论文中的 $p$ 和 PyTorch `Dropout(p)` 为什么含义居然相反？
9. Transformer 的 dropout 到底加在哪里？
10. Attention dropout 是删掉 neuron，还是删掉 Attention Weight？
11. ACT 论文写 dropout=0.1，那么代码里究竟有哪些地方真的使用了 0.1？
12. `model.eval()` 到底关闭了什么？
13. Dropout 关闭后，LayerNorm 会不会也一起关闭？

这一篇把这些问题全部拆开。

---

## 1. Dropout 最早想解决什么问题？

深度神经网络参数很多。

如果训练数据有限，

网络很容易发展出一些：

> 对训练集特别有效、但对新数据很脆弱的内部依赖。

Dropout 论文把其中一种现象描述为：

> **co-adaptation**

也就是：

> 某些 feature detectors 过度依赖其他特定 units 总会同时存在。

于是作者提出一个很直接的想法：

> 如果每次训练都随机让一些 units 暂时消失，那么剩下的 units 就不能过度依赖某一个固定伙伴。

这就是 Dropout 的原始动机之一。

---

## 2. “Dropout”这个名字是什么意思？

原始论文的定义非常字面：

> 在训练时随机把一些 units 暂时从网络中移除，同时移除与它们相连的 connections。

关键字是：

$$
\boxed{
\text{temporarily}
}
$$

不是：

> 永久删除参数。

下一次 forward：

> 会重新随机采样另一组 units。

---

## 3. PyTorch 里更准确的说法是“随机把 Activation 元素置零”

例如：

```python
dropout = nn.Dropout(p=0.1)
```

输入：

$$
x=
[x_1,x_2,x_3,x_4]
$$

某一次 training forward 可能随机得到：

```text
keep
drop
keep
keep
```

于是某些 activation：

$$
x_i
$$

被设成：

$$
0
$$

但：

- 对应参数没有删除；
- network architecture没有永久改变；
- 下一次 forward可能又保留它。

所以：

$$
\boxed{
\text{Dropout does not prune the network permanently}
}
$$

---

## 4. Dropout 和 Pruning 不是一回事

#### Dropout

训练时：

> 随机、临时关闭 activation。

推理时通常：

> 全部恢复。

#### Pruning

通常：

> 永久移除某些 weights / channels / units，

以达到：

- 压缩；
- 加速；
- 稀疏化。

所以：

$$
\boxed{
\text{Dropout}
\neq
\text{Pruning}
}
$$

---

## 5. 用随机变量正式表示 Dropout

先定义：

$$
q
=
\text{keep probability}
$$

也就是保留概率。

对每个 activation：

$$
x_i
$$

采样：

$$
m_i
\sim
\operatorname{Bernoulli}(q)
$$

其中：

$$
m_i=
\begin{cases}
1,&\text{以概率 }q\\
0,&\text{以概率 }1-q
\end{cases}
$$

---

## 6. 最朴素的 Dropout

最简单：

$$
\tilde x_i
=
m_i x_i
$$

如果：

$$
m_i=0
$$

则：

$$
\tilde x_i=0
$$

如果：

$$
m_i=1
$$

则：

$$
\tilde x_i=x_i
$$

所以：

$$
\boxed{
\tilde x
=
m\odot x
}
$$

其中：

$$
\odot
$$

表示 element-wise multiplication。

---

## 7. 一个具体例子

假设：

$$
x=
[2,-4,6,8]
$$

keep probability：

$$
q=0.75
$$

某次采样：

$$
m=
[1,0,1,1]
$$

朴素 Dropout：

$$
\tilde x
=
[2,0,6,8]
$$

看起来很简单。

但这里马上出现一个问题。

---

## 8. 训练时 Expected Activation 变小了

因为：

$$
E[m_i]=q
$$

所以：

$$
E[\tilde x_i]
=
E[m_i x_i]
$$

把：

$$
x_i
$$

看作当前 forward 中固定值：

$$
=
x_iE[m_i]
$$

所以：

$$
\boxed{
E[\tilde x_i]
=
qx_i
}
$$

如果：

$$
q=0.5
$$

平均 activation只有原来一半。

---

## 9. 这会造成 Train / Test Scale Mismatch

训练时：

$$
E[\tilde x]
=
qx
$$

但如果测试时突然：

> 不 Dropout，全部 units 都在，

那么 test activation：

$$
x
$$

会比 training average：

$$
qx
$$

大：

$$
1/q
$$

倍。

所以必须解决：

> train 与 test 数值尺度不一致。

---

## 10. 原始 Dropout 论文的解决方式

原始 2014 JMLR 论文主要采用：

#### Training

保留 unit：

$$
m_i\sim Bernoulli(q)
$$

直接：

$$
\tilde x_i=m_ix_i
$$

---

#### Test

不再随机 drop，

但把 outgoing weights乘：

$$
q
$$

这样 test contribution就与 training expected contribution对齐。

论文写的是：

$$
\boxed{
W_{\text{test}}
=
qW
}
$$

---

## 11. 为什么 Test Weight × q 有效？

假设某个 unit activation：

$$
h
$$

连到下一层权重：

$$
w
$$

训练：

该 unit以概率：

$$
q
$$

存在。

所以 expected contribution：

$$
E[mhw]
=
qhw
$$

测试时 unit总存在，

但用：

$$
qw
$$

于是：

$$
h(qw)=qhw
$$

和训练期 expectation一致。

---

## 12. 但现代框架通常反过来做

PyTorch不采用：

> train原值，test乘 $q$

这个实现方式。

而使用：

> **inverted dropout**

也就是训练时先把保留下来的 activation放大。

---

## 13. PyTorch 中 p 表示 Drop Probability

这是一个必须牢记的记号冲突。

### 原始 Dropout 论文

论文中的：

$$
p
$$

通常表示：

$$
\boxed{
\text{retention probability}
}
$$

也就是 keep probability。

---

### PyTorch

```python
nn.Dropout(p=0.1)
```

这里：

$$
p
$$

表示：

$$
\boxed{
\text{drop probability}
}
$$

即被置零的概率。

所以为了避免混乱，

本文统一定义：

$$
d
=
\text{drop probability}
$$

$$
q
=
1-d
=
\text{keep probability}
$$

---

## 14. PyTorch 的 Inverted Dropout 公式

训练时：

$$
m_i\sim Bernoulli(q)
$$

输出：

$$
\boxed{
y_i
=
\frac{
m_i x_i
}{
q
}
}
$$

因为：

$$
q=1-d
$$

也可以写：

$$
\boxed{
y_i
=
\frac{
m_i x_i
}{
1-d
}
}
$$

这就是为什么 PyTorch文档说：

> training时 output会乘 $1/(1-p)$。

这里 PyTorch 的：

$$
p=d
$$

是 drop probability。

---

## 15. 为什么要除 q？

直接算 expectation：

$$
E[y_i]
=
E
\left[
\frac{
m_ix_i
}{
q
}
\right]
$$

$$
=
\frac{x_i}{q}
E[m_i]
$$

而：

$$
E[m_i]=q
$$

所以：

$$
\boxed{
E[y_i]=x_i
}
$$

这就是 inverted dropout 的核心。

---

## 16. 于是 Evaluation 什么都不用做

training：

$$
y_i
=
\frac{m_ix_i}{q}
$$

已经保证：

$$
E[y_i]=x_i
$$

所以 evaluation只需：

$$
\boxed{
y_i=x_i
}
$$

即：

> Dropout layer变成 identity function。

PyTorch文档明确说明：

> training时保留的输出按 $1/(1-p)$ 缩放，因此 evaluation时模块直接计算 identity。

---

## 17. 这就是为什么叫 Inverted Dropout

原始方式：

```text
Train:
随机 mask，不放大

Test:
全部使用，但乘 keep probability
```

inverted方式：

```text
Train:
随机 mask，并除以 keep probability

Test:
什么都不改
```

两者只是把 scale correction：

> 从 test搬到了 train。

---

## 18. 一个 p_drop = 0.1 的例子

ACT：

$$
d=0.1
$$

所以：

$$
q=0.9
$$

保留下来的 activation会乘：

$$
\frac1{0.9}
=
1.111\ldots
$$

例如：

$$
x=9
$$

training时：

#### 90%概率保留

输出：

$$
9/0.9=10
$$

#### 10%概率 drop

输出：

$$
0
$$

Expectation：

$$
0.9(10)+0.1(0)=9
$$

所以：

$$
\boxed{
E[y]=x
}
$$

---

## 19. 一个向量例子

输入：

$$
x=
[2,-4,6]
$$

drop probability：

$$
d=0.25
$$

所以：

$$
q=0.75
$$

某次 mask：

$$
m=
[1,0,1]
$$

training output：

$$
y
=
\frac{
m\odot x
}{
0.75
}
$$

所以：

$$
y=
[
2.667,
0,
8
]
$$

注意：

> 保留下来的值反而比原值更大。

这不是 bug。

这是为了补偿那 25% 被随机删除的 probability mass。

---

## 20. 为什么“删一些再放大剩下的”不会互相抵消？

它只在：

$$
\boxed{
\text{expectation}
}
$$

上保持平均尺度。

单次 forward仍然是不同的。

例如：

$$
x=[2,-4,6]
$$

可能得到：

$$
[2.667,0,8]
$$

下一次可能：

$$
[0,-5.333,8]
$$

再下一次：

$$
[2.667,-5.333,0]
$$

所以：

> 每一次训练都加入随机扰动。

只是这些扰动的平均值：

$$
E[y]=x
$$

---

## 21. Dropout 增加的是随机方差

对固定：

$$
x_i
$$

inverted dropout：

$$
y_i=
\frac{m_ix_i}{q}
$$

因为：

$$
Var(m_i)=q(1-q)
$$

所以：

$$
Var(y_i)
=
\frac{x_i^2}{q^2}
q(1-q)
$$

得到：

$$
\boxed{
Var(y_i)
=
x_i^2
\frac{
1-q
}{
q
}
}
$$

因为：

$$
1-q=d
$$

所以：

$$
\boxed{
Var(y_i)
=
x_i^2
\frac{
d
}{
1-d
}
}
$$

---

## 22. p_drop 越大，Training Noise 越强

如果：

$$
d=0.1
$$

variance multiplier：

$$
\frac{0.1}{0.9}
\approx
0.111
$$

如果：

$$
d=0.5
$$

则：

$$
\frac{0.5}{0.5}=1
$$

如果：

$$
d=0.9
$$

则：

$$
\frac{0.9}{0.1}=9
$$

所以越大的 dropout：

> 不只是删得更多，

保留下来的 units还会被放大得更强，

训练噪声急剧增加。

---

## 23. 因此 Dropout Rate 不是越大越好

过小：

> regularization可能不足。

过大：

- 信息被大量删除；
- gradient noise更强；
- optimization更困难；
- 有效容量下降。

所以：

$$
\boxed{
p_{\text{drop}}
}
$$

是需要调节的 hyperparameter。

---

## 24. p_drop = 0 会怎样？

$$
d=0
$$

$$
q=1
$$

mask永远：

$$
m_i=1
$$

scale：

$$
1/q=1
$$

所以：

$$
\boxed{
Dropout(x)=x
}
$$

训练和推理都一样。

即：

> no dropout。

---

## 25. p_drop → 1 会怎样？

当：

$$
d\rightarrow1
$$

keep probability：

$$
q\rightarrow0
$$

scale：

$$
1/q
$$

趋于非常大。

而几乎所有 activations都被 drop。

这会让训练极其不稳定。

所以合法实现通常要求：

$$
0\le d<1
$$

而不会用：

$$
d=1
$$

作为普通 Dropout。

---

## 26. Dropout Mask 每次都一样吗？

不是。

PyTorch明确：

> zeroed elements are chosen independently for each forward call。

也就是说：

同一个 training sample，

如果 forward两次：

> 通常 mask不同。

---

## 27. 同一个 Batch 中所有样本共享 Mask 吗？

标准 `nn.Dropout` 的概念是：

> 对输入 tensor中的元素独立采样。

所以不同 batch sample：

> 通常也不会被强制共享同一 mask。

注意这和：

- SpatialDropout；
- Dropout2d；
- channel dropout；

等 structured dropout不同。

---

## 28. “Dropout 随机删神经元”只是历史直觉

现代 tensor实现更准确是：

> 对 activation tensor随机施加 Bernoulli mask。

例如一个 hidden unit在：

- sample A；
- position 3；

可能被 drop。

但同一个 parameterized feature在：

- sample B；
- position 8；

可能被保留。

所以不应把它想成：

> “这一整个 neuron在这一批训练中永久被拆掉。”

---

## 29. Dropout 为什么是 Regularization？

核心效果是：

> 每个 training forward网络都被随机扰动。

模型不能只在一个固定 deterministic computation graph上：

> 精确记住训练样本的脆弱组合。

它需要学到：

> 在不同随机子网络 / activation masks下仍然有用的 representations。

这会增加训练难度，

但往往改善：

> generalization。

---

## 30. 什么叫 Co-Adaptation？

假设 neuron A 的工作方式是：

> “只要 B 一定在，我就只负责一半。”

B 又想：

> “A 一定会补另一半。”

二者形成强依赖。

训练数据上可能很好，

但新数据稍微变化：

> 这个 fragile collaboration就失效。

Dropout让 B 有时突然消失。

A 被迫：

> 自己也学更独立、有鲁棒性的 feature。

这就是原论文的 co-adaptation intuition。

---

## 31. 但不能说 Co-Adaptation 是 Dropout 唯一被证明的机制

这是需要严谨处理的一点。

原始论文把：

> 减少 co-adaptation

作为重要解释。

同时它还给出：

> approximate model averaging

的理解。

后续关于 Dropout为何有效还有很多解释：

- stochastic regularization；
- noise injection；
- implicit ensemble；
- Bayesian approximations等。

所以 canonical page最安全的写法是：

$$
\boxed{
\text{co-adaptation reduction}
=
\text{original motivation / interpretation}
}
$$

而不是：

> Dropout有效性的唯一完整理论。

---

## 32. 为什么 Dropout 可以看成很多 Thinned Networks？

假设一层有：

$$
n
$$

个 units。

每个 unit：

> 保留或删除。

理论上有：

$$
2^n
$$

种 possible masks。

整个大网络因此对应：

> 大量不同的 thinned subnetworks。

训练时每次随机采样一套 mask，

等价于：

> 每次训练一个共享参数的随机子网络。

---

## 33. 这些 Subnetworks 是完全独立模型吗？

不是。

它们：

$$
\boxed{
\text{share weights}
}
$$

所以不是训练：

$$
2^n
$$

套独立参数。

而是：

> 同一套 weights在很多随机 masks下被优化。

---

## 34. Test Time 为什么不开 Dropout？

原始 Dropout想达到：

> 类似 ensemble averaging 的效果。

如果测试时还随机 mask一次，

你得到的只是：

> 一个随机 thinned model的预测。

标准做法则使用：

> 完整网络 + 合适 scaling

来近似 many subnetworks的平均预测。

---

## 35. 原始论文的 Model Averaging 是“精确”等价吗？

一般深网络：

> 不是。

论文明确把 test-time scaling描述成：

> approximate averaging。

在简单结构下可以有精确关系，

但经过 nonlinearities和多层组合后：

$$
E[f_{\text{dropout}}(x)]
\neq
f(E[\text{dropout input}])
$$

一般成立。

所以不能说：

$$
\boxed{
\text{full network prediction}
=
\text{exact average of all dropout subnetworks}
}
$$

---

## 36. 这和 Jensen / Nonlinearity 有关

例如：

$$
f
$$

非线性时，

通常：

$$
\boxed{
f(E[X])
\neq
E[f(X)]
}
$$

所以即使每层局部 activation expectation被对齐，

整个 nonlinear network的最终 output expectation：

> 不会自动严格相等。

---

## 37. 那为什么 Inverted Dropout 还要保持局部 Expectation？

因为这样至少能避免：

> train与eval activation scale发生明显系统性偏移。

它是一个非常实用的局部 calibration。

但：

$$
E[\text{每层输入}]
$$

对齐，

不代表：

$$
E[\text{最终 network output}]
$$

严格完全一致。

---

## 38. Inverted Dropout 的一个最重要性质

训练：

$$
y=
\frac{m\odot x}{q}
$$

eval：

$$
y=x
$$

因此模型代码不需要在 evaluation时：

> 修改所有 weight matrices。

这是现代实现非常方便的一点。

---

## 39. `model.train()` 与 `model.eval()` 为什么重要？

PyTorch Module有：

```python
model.train()
```

和：

```python
model.eval()
```

它们会改变某些 modules的行为。

Dropout就是最典型之一。

---

## 40. Train Mode

调用：

```python
model.train()
```

后：

```python
nn.Dropout(p=d)
```

执行：

$$
\boxed{
y=
\frac{
m\odot x
}{
1-d
}
}
$$

其中：

$$
m_i
\sim Bernoulli(1-d)
$$

所以每次 forward随机。

---

## 41. Eval Mode

调用：

```python
model.eval()
```

后：

$$
\boxed{
Dropout(x)=x
}
$$

不再随机 mask，

也不需要额外 scale。

PyTorch官方文档明确称：

> evaluation时 Dropout module computes an identity function。

---

## 42. `torch.no_grad()` 会自动关闭 Dropout 吗？

**不会。**

这是非常重要的工程误区。

```python
torch.no_grad()
```

主要作用：

> 不记录 autograd graph / gradient。

但模型仍可能处于：

```python
training=True
```

于是 Dropout仍然随机。

所以 inference通常需要：

```python
model.eval()

with torch.no_grad():
    ...
```

两者职责不同。

---

## 43. `model.eval()` 会自动关闭 Gradient 吗？

也不会。

`model.eval()`只切换：

> module training/eval behavior。

它不会自动：

> 禁止 autograd。

所以标准 inference常同时使用：

```python
model.eval()
torch.no_grad()
```

---

## 44. LayerNorm 会被 eval() 关闭吗？

不会。

上一篇已经讲过：

> LayerNorm训练和推理都根据当前 input计算 statistics。

所以：

```text
model.eval()
```

会让：

> Dropout停止随机。

但 LayerNorm：

> 继续正常计算当前 token mean/variance。

---

## 45. BatchNorm 则不同

BatchNorm在：

- train；
- eval；

通常使用不同 statistics。

所以：

```python
model.eval()
```

不仅影响 Dropout，

还会影响 BatchNorm行为。

ACT Transformer内部主要使用：

> LayerNorm，

所以这里尤其容易区分。

---

## 46. Dropout 训练时会修改 Parameters 吗？

Dropout本身：

> 没有 learned weights。

它只生成随机 mask并改变 forward activations。

真正的：

$$
W
$$

仍通过 optimizer根据 loss gradient更新。

---

## 47. Dropout 自己有 Trainable Parameters 吗？

标准：

```python
nn.Dropout(p=0.1)
```

没有。

它只有 hyperparameter：

$$
p
$$

不是 optimizer学习的 parameter。

---

## 48. Dropout Rate 会被 Gradient 更新吗？

不会。

除非你专门设计：

> learnable dropout / concrete dropout等变体。

标准 Dropout：

$$
p
$$

由人设定。

---

## 49. Dropout 会把 Gradient 一起 Mask 吗？

如果 forward中：

$$
y_i=0
$$

因为：

$$
m_i=0
$$

那么在这一条局部路径上：

$$
\frac{
\partial y_i
}{
\partial x_i
}
=
0
$$

所以这个 activation对应的 gradient不会通过该路径回传。

如果：

$$
m_i=1
$$

则：

$$
\frac{
\partial y_i
}{
\partial x_i
}
=
\frac1q
$$

因此训练时 backward同样受到 mask影响。

---

## 50. Dropout 不只是 Forward Noise

因为每次随机 mask也会改变：

> 哪些参数在这次 example中收到 gradient，以及 gradient magnitude。

原始论文明确指出：

> 每个 training case在不同 thinned network上 forward / backprop。

所以 stochasticity会进入：

> optimization本身。

---

## 51. 为什么训练可能更慢？

Dropout提高了 gradient noise。

原始论文也指出：

> Dropout network常需要更长训练时间。

原因包括：

- 每次看到不同 random architecture；
- gradients更 noisy；
- 单次 update的信息更不稳定。

所以 regularization收益往往伴随：

> optimization cost。

---

## 52. 为什么不在 Test 也随机 Dropout 然后平均很多次？

可以。

如果测试时保持 Dropout打开，

多次 stochastic forward：

$$
f_1(x),f_2(x),\ldots,f_M(x)
$$

再平均，

这可以被用于：

> Monte Carlo Dropout等方法。

但这不是标准 deterministic inference。

标准 PyTorch / ACT evaluation：

> `model.eval()`，Dropout关闭。

---

## 53. Monte Carlo Dropout 是另一种用法

MC Dropout后来常用于：

- uncertainty estimation；
- approximate Bayesian inference。

但 canonical Dropout page应先明确：

$$
\boxed{
\text{standard inference}
=
\text{dropout off}
}
$$

MC Dropout属于：

> 后续特殊 inference strategy。

---

## 54. Transformer 为什么特别需要 Regularization？

Transformer有很多参数和强大的：

- Attention；
- FFN；

尤其在训练数据规模有限时，

容易过拟合。

原始 Transformer因此在 architecture中显式使用 Dropout。

---

## 55. 原始 Transformer 的 Residual Dropout

《Attention Is All You Need》Section 5.4明确：

> Dropout applied to the output of each sub-layer, before it is added to the sub-layer input and normalized.

所以原始 Post-LN：

$$
\boxed{
LN(
x+
Dropout(
Sublayer(x)
)
)
}
$$

不是：

$$
Dropout(
LN(x+Sublayer(x))
)
$$

---

## 56. 这意味着 Identity Shortcut 不经过这次 Dropout

结构：

```text
x ──────────────────────────┐
                            │
Sublayer(x)                 │
↓                           │
Dropout                     │
↓                           │
──────────────────────────→ Add
```

Dropout作用：

> residual branch。

shortcut：

$$
x
$$

保持直接。

---

## 57. 为什么这个 Placement 很自然？

因为 Dropout随机扰动：

> 当前 sub-layer写入的 update。

如果这次某些 update dimensions被 drop，

identity path仍在。

所以 block局部更接近：

$$
y\approx x
$$

这与 residual architecture很契合。

---

## 58. 原始 Transformer 还在哪里 Dropout？

论文还明确说：

> 对 Encoder 和 Decoder stacks底部的 embedding + positional encoding 之和应用 Dropout。

即概念上：

$$
\boxed{
Dropout(
Embedding+
PositionalEncoding
)
}
$$

原始 Base Transformer：

$$
\boxed{
P_{drop}=0.1
}
$$

---

## 59. 原始论文明确写 Attention-Weight Dropout 吗？

在 Section 5.4 的明确文字中，

作者直接说明的是：

1. sub-layer output dropout；
2. embedding + positional encoding sum dropout。

不能仅凭现代实现习惯就把：

> attention-weight dropout

自动写成原论文明确描述的同一件事。

所以应该区分：

$$
\boxed{
\text{Original-paper documented dropout placements}
}
$$

和：

$$
\boxed{
\text{current library / implementation dropout placements}
}
$$

---

## 60. PyTorch MultiheadAttention 的 dropout 参数是什么？

PyTorch当前文档对：

```python
nn.MultiheadAttention(
    ...,
    dropout=p
)
```

明确写：

> `dropout` is the dropout probability on `attn_output_weights`.

也就是说：

$$
\boxed{
\text{它作用于 Attention Weights}
}
$$

不是普通：

> MHA最终 512-D output上的同一个 Dropout。

---

## 61. Attention Weight Dropout 在哪里？

标准 Attention：

$$
A=
softmax(S)
$$

然后：

$$
O=AV
$$

Attention dropout概念上对：

$$
A
$$

施加 Dropout：

$$
\tilde A=
Dropout(A)
$$

再：

$$
O=\tilde AV
$$

所以随机削弱的是：

> 某些 query-to-key routing connections。

---

## 62. 这和 Causal Mask 完全不同

#### Causal Mask

某些 connections：

> 永久/确定性禁止。

例如：

$$
j>i
$$

weight必须：

$$
0
$$

---

#### Attention Dropout

合法 connection中：

> training时随机 drop。

下一次 forward可能又恢复。

所以：

$$
\boxed{
\text{Mask}
\neq
\text{Dropout}
}
$$

---

## 63. Attention Dropout 后 Weight 还会 Sum to 1 吗？

这是一个非常好的细节。

Softmax前：

$$
\sum_jA_{ij}=1
$$

但如果对：

$$
A
$$

使用 inverted dropout：

$$
\tilde A_{ij}
=
\frac{
m_{ij}A_{ij}
}{
q
}
$$

那么某一次 forward：

$$
\sum_j\tilde A_{ij}
$$

一般：

> 不再严格等于 1。

它只在 expectation上：

$$
E[\tilde A_{ij}]=A_{ij}
$$

所以：

$$
E
\left[
\sum_j\tilde A_{ij}
\right]
=
1
$$

---

## 64. 因此 Attention Dropout 后不能再叫 Probability Distribution 吗？

Softmax产生的：

$$
A
$$

可以解释成：

> normalized attention weights。

Dropout后的：

$$
\tilde A
$$

单次 training forward不再严格是：

> sum-to-one probability vector。

它更准确是：

> stochastic reweighted routing coefficients。

这是非常重要的数学细节。

---

## 65. 那为什么仍然能乘 V？

矩阵乘法只要求：

$$
\tilde A
$$

和：

$$
V
$$

shape匹配。

不要求 weights必须 sum=1。

所以：

$$
\tilde AV
$$

仍然是合法 weighted combination。

只是它不再严格是 convex combination。

---

## 66. Dropout 会破坏 Attention 的 Convex Combination 性质

没有 Dropout时：

$$
A_{ij}\ge0
$$

且：

$$
\sum_jA_{ij}=1
$$

于是：

$$
o_i
=
\sum_jA_{ij}v_j
$$

位于 Values的 convex hull。

Training attention dropout后：

$$
\sum_j\tilde A_{ij}
\neq1
$$

一般成立。

所以单次 training output：

> 不再严格是 Values的 convex combination。

Eval时 Dropout关闭，

恢复普通 Attention权重。

---

## 67. ACT 的 Transformer Dropout 具体有几类？

当前官方 ACT `transformer.py` 中至少可以清楚区分：

$$
\boxed{
1.\ Attention\ weight\ dropout
}
$$

$$
\boxed{
2.\ FFN\ hidden\ dropout
}
$$

$$
\boxed{
3.\ Residual\ branch\ output\ dropout
}
$$

这三者用的是同一个 config：

$$
dropout
$$

但位置不同、作用对象不同。

---

## 68. ACT 第一类：Attention Weight Dropout

Encoder：

```python
self.self_attn =
    nn.MultiheadAttention(
        d_model,
        nhead,
        dropout=dropout
    )
```

Decoder：

```python
self.self_attn =
    nn.MultiheadAttention(...)

self.multihead_attn =
    nn.MultiheadAttention(...)
```

PyTorch定义：

> 这个 `dropout` parameter作用于 `attn_output_weights`。

所以 ACT 的：

- Encoder Self-Attention；
- Decoder Self-Attention；
- Decoder Cross-Attention；

训练时都包含：

> attention-weight dropout。

---

## 69. ACT 第二类：FFN Hidden Dropout

官方：

```python
self.dropout =
    nn.Dropout(dropout)
```

FFN：

```python
linear2(
    dropout(
        activation(
            linear1(x)
        )
    )
)
```

所以：

$$
512
\rightarrow
3200
$$

后，

经过 ReLU，

在 3200-D hidden activation上做 Dropout，

再：

$$
3200\rightarrow512
$$

---

## 70. FFN Dropout 的 Shape

ACT Encoder某个 layer：

$$
[B,1202,3200]
$$

的 ReLU hidden activations。

`nn.Dropout(0.1)`：

> 每个 element training时有 10% probability设 0。

shape仍：

$$
[B,1202,3200]
$$

不会删除 token或改变 tensor shape。

---

## 71. ACT Decoder FFN 同理

$$
[B,k,3200]
$$

例如：

$$
[B,100,3200]
$$

training时随机 element-wise dropout。

它不会让：

> 某个 action slot消失。

只是在这个 action slot的 hidden features中：

> 随机屏蔽部分 intermediate activations。

---

## 72. ACT 第三类：Residual Branch Output Dropout

Encoder：

```python
src =
    src +
    self.dropout1(src2)
```

和：

```python
src =
    src +
    self.dropout2(src2)
```

所以：

- Attention output；
- FFN output；

在 residual addition前分别经过 Dropout。

---

## 73. Decoder 有三个 Residual Dropout

```python
tgt =
    tgt +
    self.dropout1(
        self_attn_output
    )
```

```python
tgt =
    tgt +
    self.dropout2(
        cross_attn_output
    )
```

```python
tgt =
    tgt +
    self.dropout3(
        ffn_output
    )
```

所以：

$$
\boxed{
\text{Decoder每个 sub-layer的 update branch都有 dropout}
}
$$

---

## 74. 一个 ACT Encoder Layer 的 Dropout Map

可以画成：

```text
src
 │
 ├──────────────────────────────┐
 │                              │
 ▼                              │
Self-Attention                  │
  ↑                             │
  └─ attention-weight dropout   │
 │                              │
 ▼                              │
dropout1 on attn output         │
 │                              │
 └──────────────→ Add ←─────────┘
                  │
                 LN
                  │
                  ▼
               Linear1
                  │
                 ReLU
                  │
             dropout
                  │
               Linear2
                  │
             dropout2
                  │
 ┌────────────────┴─────────────┐
 │                              │
 └──────────────→ Add ←─────────┘
                  │
                 LN
```

注意：

> 同一个 `dropout=0.1` 出现在不同层级。

---

## 75. 一个 ACT Decoder Layer 的 Dropout Map

```text
tgt
 │
 ▼
Self-Attention
 │    └─ attn-weight dropout
 ▼
dropout1
 ▼
Residual Add + Norm

 │
 ▼
Cross-Attention
 │    └─ attn-weight dropout
 ▼
dropout2
 ▼
Residual Add + Norm

 │
 ▼
Linear1 → ReLU
 │
 ▼
internal FFN dropout
 │
 ▼
Linear2
 │
 ▼
dropout3
 ▼
Residual Add + Norm
```

---

## 76. 一个 Dropout Rate 同时用于这么多位置，会不会“总共只 Drop 10%”？

不是。

每个 Dropout operation：

> 都单独采样自己的随机 mask。

所以：

$$
0.1
$$

不是说：

> 整个模型每次 forward总共只有 10%东西被丢。

而是：

> 每个配置为 $p=0.1$ 的 dropout operation，都以自己的方式对它的输入施加 10% drop probability。

---

## 77. Attention Weight Dropout 和 Residual Dropout 会重复发生

例如 Encoder Self-Attention：

1. 内部 attention weights可能被 dropout；
2. 得到 MHA output；
3. 整个 MHA output又经过：
   $$
   dropout1
   $$
4. 再加 residual。

所以随机 regularization：

> 不止一层。

---

## 78. 为什么需要两个不同 Dropout？

它们扰动不同计算对象。

#### Attention Weight Dropout

扰动：

> “我从哪些 key/value位置读取信息？”

#### Residual Output Dropout

扰动：

> “这个 Attention sub-layer算出的 512-D update有多少 feature写回 residual stream？”

作用层级不同。

---

## 79. FFN Internal Dropout vs FFN Residual Dropout

也有两层：

#### Internal FFN Dropout

$$
ReLU(W_1x)
\rightarrow
Dropout
\rightarrow
W_2
$$

扰动：

> 3200-D intermediate hidden features。

#### Residual Output Dropout

$$
FFN(x)
\rightarrow
Dropout
\rightarrow
+x
$$

扰动：

> 已经投回 512-D 的 final FFN update。

---

## 80. 所以一个 FFN Branch里有两次 Dropout

ACT代码确实：

```python
src2 =
    linear2(
        dropout(
            activation(
                linear1(src)
            )
        )
    )

src =
    src +
    dropout2(src2)
```

因此：

$$
\boxed{
\text{hidden dropout}
+
\text{residual-output dropout}
}
$$

两次随机操作。

---

## 81. ACT 论文的 Dropout Rate 是多少？

ACT Table III：

$$
\boxed{
dropout=0.1
}
$$

也就是：

> 10% drop probability。

这是论文明确列出的 ACT hyperparameter。

---

## 82. ACT Code 如何把这个 0.1 传进去？

`build_transformer(args)`：

```python
dropout=args.dropout
```

传给：

```python
Transformer(...)
```

然后：

> EncoderLayer和DecoderLayer中的多个 Dropout modules都使用这个值。

CVAE Encoder `build_encoder(args)` 也：

```python
dropout = args.dropout
```

再传入：

```python
TransformerEncoderLayer(...)
```

所以 training-only CVAE Encoder同样使用：

> 该 Transformer dropout配置。

---

## 83. 因此 ACT 的三个 Transformer 都有 Dropout

我们之前已经区分 ACT 中三个 Transformer-related stacks：

1. CVAE Transformer Encoder；
2. Policy Transformer Encoder；
3. Policy Transformer Decoder。

这些 layer构造都接收：

$$
args.dropout
$$

所以训练时都存在 dropout regularization。

---

## 84. ResNet18 Backbone 的 Dropout 也是同一个吗？

不能自动这么说。

`args.dropout=0.1` 在 ACT当前代码中明确传给：

> Transformer modules。

ResNet18 backbone是否有 Dropout：

> 要看它自己的 backbone implementation。

标准 ResNet18本身通常并没有在基本 residual blocks中使用 classic Dropout。

所以不要把 ACT Table III 的：

$$
dropout=0.1
$$

理解成：

> 整个模型所有层统一随机删 10%。

---

## 85. ACT 当前 Transformer 有没有原始论文那种 “Embedding + Positional Encoding Sum Dropout”？

当前 DETR-derived `transformer.py` 主要通过：

```python
with_pos_embed(tensor, pos)
```

在 Q/K中加入 position。

在这份代码里没有看到一个与原始语言 Transformer完全对应的：

```python
dropout(
    embedding + positional_encoding
)
```

输入层 Dropout。

所以需要区分：

#### Original Transformer paper

明确有：

> embeddings + positional encodings sum dropout。

#### Current ACT/DETR-derived implementation

显式可见的是：

- attention-weight dropout；
- FFN hidden dropout；
- residual-output dropout。

不要自动把原始 Transformer每个 dropout placement都假设为 ACT代码存在。

---

## 86. Dropout 和 Position Encoding 没有直接数学关系

Position Encoding：

> 提供位置结构。

Dropout：

> 随机 regularization。

原始 Transformer把 embedding+PE的和做 dropout，

只是：

> 在输入 representation上做 regularization。

并不是说：

> Dropout本身编码 position。

---

## 87. Dropout 和 LayerNorm 的顺序很重要

原始 Post-LN：

$$
\boxed{
LN(
x+
Dropout(F(x))
)
}
$$

不是：

$$
Dropout(
LN(x+F(x))
)
$$

顺序不同会产生不同 network function。

---

## 88. 为什么 Residual Dropout 在 Add 前？

因为想随机扰动：

> sub-layer写回 residual stream的 update。

而保留：

$$
x
$$

identity path。

如果 Dropout放到：

$$
x+F(x)
$$

之后，

就可能连 shortcut信息一起随机抹掉。

那是不同 architecture。

---

## 89. Pre-LN 中 Dropout 又在哪里？

ACT Pre-LN：

$$
\tilde x=LN(x)
$$

$$
u=F(\tilde x)
$$

然后：

$$
\boxed{
y=x+Dropout(u)
}
$$

所以虽然 Norm位置改变，

Dropout仍然主要位于：

> residual branch output写回之前。

---

## 90. Dropout 会不会改变 Tensor Shape？

不会。

标准 `nn.Dropout`：

输入：

$$
(*)
$$

输出：

$$
(*)
$$

shape完全一致。

它只是把一些元素变：

$$
0
$$

并缩放剩余元素。

---

## 91. 为什么叫 “Drop Units” 但 Shape 不变？

逻辑上：

> unit在这次 computation中失效。

实现上为了高效 tensor computation：

> 保留原 tensor shape，只把值置 0。

所以“drop”是：

> functional removal，

不是：

> physically remove tensor dimension。

---

## 92. Dropout 会减少实际 FLOPs 吗？

标准 dense Dropout通常：

> 不会自动让后续 dense matrix multiplication少算对应神经元。

因为 tensor shape仍然一样，

GPU通常仍做 dense compute。

所以：

$$
\boxed{
\text{Dropout is not primarily an inference/computation-saving technique}
}
$$

它是 regularization。

---

## 93. 为什么训练还可能更慢？

除了随机和更长收敛，

Dropout本身还增加：

- mask sampling；
- elementwise operations。

但主要训练成本仍来自大 matrix operations。

---

## 94. Dropout 会让模型参数变少吗？

不会。

参数 count：

> 完全不变。

它只是 training-time stochastic computation。

---

## 95. Dropout 会减小 Effective Capacity 吗？

某一次 training forward：

> 只有部分 activations参与，

因此这个 stochastic subnetwork的有效 capacity较小。

但完整 parameterized model：

> 参数仍都存在。

多次 forward中不同 subsets被使用。

---

## 96. Expected Number of Surviving Units

假设：

$$
n
$$

个 independent elements，

keep probability：

$$
q
$$

则存活数：

$$
K
\sim
Binomial(n,q)
$$

所以：

$$
\boxed{
E[K]=nq
}
$$

例如：

$$
n=3200
$$

$$
q=0.9
$$

期望保留：

$$
2880
$$

个 FFN hidden activation elements。

---

## 97. 但实际每次不一定恰好保留 2880

因为：

$$
K
$$

是随机变量。

variance：

$$
Var(K)=nq(1-q)
$$

所以每次 forward survivors会波动。

Dropout不是：

> 精确删除固定 10%数量。

而是：

> 每个 element有 10%独立 drop概率。

---

## 98. 大 Tensor 时比例会接近 p

当：

$$
n
$$

很大，

根据大数规律，

实际 dropped fraction通常接近：

$$
d
$$

例如 3200-D hidden：

> 往往接近 10%。

但不是数学上每次恰好 320 个。

---

## 99. Dropout Mask 是不是可微？

Bernoulli采样本身：

> 不是对概率参数 $p$ 做普通 pathwise differentiable training。

但标准 Dropout中：

$$
p
$$

不是需要训练的 parameter。

对于已采样 mask：

$$
m
$$

forward：

$$
y=
m\odot x/q
$$

对：

$$
x
$$

是简单线性函数，

所以正常 backprop即可。

---

## 100. 为什么这不像 VAE Reparameterization Problem？

VAE中我们想训练：

$$
\mu,\sigma
$$

它们控制 sampling distribution。

因此采样路径必须让 gradient回到：

$$
\mu,\sigma
$$

Dropout中：

$$
p
$$

通常固定，

无需对 Bernoulli probability求 gradient。

所以不需要：

> reparameterization trick

来训练 Dropout probability。

---

## 101. 如果想学习 Dropout Probability 呢？

那就进入：

- Concrete Dropout；
- Relaxed Bernoulli；
- learned stochastic gates；

等更复杂方法。

它们需要处理：

> discrete sampling gradient。

这超出 canonical Dropout主线。

---

## 102. Dropout 和 Data Augmentation 有什么共同点？

两者都在 training时引入随机变化。

#### Data Augmentation

扰动：

> input data。

#### Dropout

扰动：

> internal network activations / connections。

它们都可以迫使模型：

> 不过度依赖某个固定训练配置。

但作用层级不同。

---

## 103. Dropout 和 Weight Decay 有什么区别？

#### Dropout

training-time stochastic activation masking。

#### Weight Decay

直接对参数大小施加优化偏好 / decay。

两者都是 regularization，

但数学机制完全不同。

ACT训练也可能同时使用：

- weight decay；
- dropout。

不能把它们当成替代概念。

---

## 104. Dropout 和 Noise Injection 有关系吗？

有。

Inverted Dropout可写成：

$$
y_i=r_ix_i
$$

其中随机 multiplier：

$$
r_i=
\begin{cases}
1/q,&\text{概率 }q\\
0,&\text{概率 }1-q
\end{cases}
$$

并且：

$$
E[r_i]=1
$$

$$
Var(r_i)=\frac{1-q}{q}
$$

所以 Dropout就是一种：

> multiplicative Bernoulli noise。

原始论文也明确给出了这个视角。

---

## 105. 这和 Gaussian Noise 有什么联系？

原始 Dropout论文还讨论：

> 用 Gaussian multiplicative noise替代 Bernoulli dropout。

如果：

$$
r_g
\sim
\mathcal N
\left(
1,
\frac{1-q}{q}
\right)
$$

它可以和 inverted Bernoulli noise具有相同：

- mean；
- variance。

这说明 Dropout也可以从：

> noise regularization

视角理解。

---

## 106. 但 Bernoulli Dropout 最特殊的地方是什么？

它真的会产生：

$$
0
$$

也就是某些 activation在该 forward：

> 完全消失。

这种 hard zeroing让网络无法依赖：

> 某个 feature永远存在。

---

## 107. 为什么 Dropout 对小数据集可能特别有帮助？

当数据少，

大模型更容易：

> 记忆训练集偶然模式。

Dropout增加 stochastic regularization，

可以减少：

> 过度拟合某一固定 computation configuration。

ACT每个 task训练 demonstration数量并不巨大，

使用：

$$
dropout=0.1
$$

也是其 regularization recipe的一部分。

但不要说：

> ACT成功主要就是因为 Dropout。

方法还依赖：

- action chunking；
- CVAE；
- architecture；
- data；
- temporal ensemble等。

---

## 108. 为什么 ACT Dropout 只有 0.1，而经典 MLP 常听到 0.5？

Dropout最早很多 fully-connected network experiments常使用较大 dropout。

Transformer领域常见更小：

$$
0.1
$$

因为 architecture、data、normalization和optimization不同。

没有普遍规则：

$$
\boxed{
p=0.5\text{ 才叫标准 Dropout}
}
$$

rate是 architecture-specific hyperparameter。

---

## 109. ACT 0.1 意味着 Keep Probability 是多少？

PyTorch：

$$
d=0.1
$$

所以：

$$
\boxed{
q=0.9
}
$$

训练时保留下来的 element：

$$
\times
\frac1{0.9}
\approx1.1111
$$

---

## 110. 不要把 ACT 论文的 dropout=0.1 当成“Retention Probability 0.1”

这是记号最危险的地方。

ACT代码基于 PyTorch：

```python
nn.Dropout(dropout)
```

所以：

$$
0.1
$$

是：

> drop probability。

不是：

> keep probability。

因此 ACT不是每次只保留 10%。

而是平均保留：

$$
90\%
$$

---

## 111. 为什么原始 Dropout 论文 p 和 PyTorch p 相反？

这是历史 API notation差异。

原论文常定义：

$$
p=\text{probability of retaining a unit}
$$

PyTorch API定义：

$$
p=\text{probability of zeroing an element}
$$

所以读公式必须先问：

> 这里的 $p$ 到底是 keep 还是 drop？

---

## 112. 最安全的写法

本文建议：

$$
d=p_{\text{drop}}
$$

$$
q=p_{\text{keep}}=1-d
$$

于是永远写：

$$
\boxed{
y=
\frac{
m\odot x
}{
q
},
\qquad
m_i\sim Bernoulli(q)
}
$$

这样不容易混。

---

## 113. Dropout 和 Softmax Probability 的 p 完全无关

注意：

Dropout的：

$$
p
$$

只是一个超参数 probability。

Attention Softmax输出：

$$
\alpha_{ij}
$$

也是 weights/probabilities-like values。

二者不是同一个：

$$
p
$$

概念。

尤其 Attention dropout是：

> 用 Dropout随机作用于 Softmax得到的 weights。

---

## 114. 一个 Attention Dropout 数值例子

Softmax后：

$$
A=
[0.2,0.3,0.5]
$$

drop probability：

$$
d=0.1
$$

keep：

$$
q=0.9
$$

假设 mask：

$$
m=
[1,0,1]
$$

则：

$$
\tilde A
=
\frac{
[0.2,0,0.5]
}{
0.9
}
$$

$$
\approx
[
0.222,
0,
0.556
]
$$

sum：

$$
0.778
$$

不是：

$$
1
$$

---

## 115. 另一种 Mask

如果：

$$
m=[1,1,1]
$$

则：

$$
\tilde A=
[
0.222,
0.333,
0.556
]
$$

sum：

$$
1.111
$$

所以单次：

> 可能小于1，也可能大于1。

但 expectation：

$$
E[\tilde A]=A
$$

---

## 116. 为什么 Attention Dropout 不重新 Softmax？

标准 implementation一般就是：

> Softmax之后 Dropout。

不再重新 normalization。

这样保留 inverted dropout expectation property。

如果 drop后重新 Softmax：

> 那是另一种 stochastic attention mechanism。

---

## 117. Dropout 是否一定独立 Element-Wise？

标准 `nn.Dropout`：

> 是 element-wise style随机零化。

但 Dropout family还有：

- Dropout1d；
- Dropout2d；
- Dropout3d；

可能整 channel一起 drop。

还有：

- DropPath / Stochastic Depth；

会 drop整条 residual branch。

所以“Dropout”是一个 family。

---

## 118. Transformer 中常说 Stochastic Depth 又是什么？

它不是标准 element-wise Dropout。

Stochastic Depth / DropPath：

> 训练时随机跳过整个 residual block / branch。

和：

$$
nn.Dropout
$$

逐 element zeroing不同。

现代 Vision Transformer常用。

ACT canonical Transformer代码这里用的是：

> standard `nn.Dropout`，

不是 DropPath。

---

## 119. 为什么这个区分重要？

如果看到：

```python
DropPath(0.1)
```

不能解释成：

> 每个 activation有10%概率变0。

它可能是：

> 整个 sample的 residual branch有一定概率被屏蔽。

所以永远看具体 layer semantics。

---

## 120. Dropout 和 Attention Mask 是否可以同时存在？

当然可以。

先通过：

- causal mask；
- padding mask；

决定合法 connections。

再通过 Dropout：

> 对合法 attention weights做 training-time random regularization。

例如：

```text
future:
永远 forbidden

past合法位置:
training时仍可能被 attention dropout随机抑制
```

---

## 121. Mask 的 0 和 Dropout 的 0 含义不同

#### Causal/Padding Mask产生的 0

> 结构上非法。

#### Dropout产生的 0

> 本来合法，但这次 training随机屏蔽。

下一次 forward：

> 可能恢复。

---

## 122. ACT Decoder 没有 Causal Mask，但仍有 Attention Dropout

这是一个很好例子。

ACT action-slot Self-Attention：

> non-causal，所有 slots都允许互看。

但 `nn.MultiheadAttention(dropout=0.1)`：

> 训练时仍会随机 drop一部分 attention weights。

所以：

$$
\boxed{
\text{non-causal}
\neq
\text{no dropout}
}
$$

---

## 123. ACT Inference 时 Attention Dropout 会怎样？

调用：

```python
policy.eval()
```

后，

MultiheadAttention内部 dropout关闭。

所以：

$$
A=
softmax(S)
$$

正常完整使用所有合法 routing weights。

这使同一 observation与参数下：

> attention不再因为 dropout产生随机变化。

---

## 124. ACT Inference 是不是因此完全 Deterministic？

在 canonical ACT inference中：

- Dropout关闭；
- $z=0$；
- model.eval()；
- 没有随机 action sampling。

因此 policy neural forward基本是 deterministic的，

假设：

- input固定；
- hardware/kernel deterministic问题忽略；
- 没有额外 stochastic preprocessing。

所以 Dropout不会成为 inference随机来源。

---

## 125. 训练时同一个 Sample为什么 Loss会略有不同？

即使：

- sample一样；
- weights一样；
- z sampling暂时固定；

不同 Dropout masks也可能使：

$$
\hat A
$$

不同，

从而：

$$
L
$$

不同。

ACT训练还有 CVAE：

$$
z=\mu+\sigma\epsilon
$$

本身也有 sampling随机性。

所以 training forward有多种 stochastic source：

1. latent reparameterization；
2. Dropout；
3. minibatch / start timestep sampling等。

---

## 126. Dropout 和 CVAE Latent Noise作用完全不同

#### CVAE $z$

承担：

> latent variation / conditional generative modeling。

#### Dropout

承担：

> regularization noise。

推理时：

- ACT $z\rightarrow0$ 是模型设计；
- Dropout关闭是标准 evaluation behavior。

不要把二者都叫：

> “训练噪声，所以一样”。

---

## 127. 为什么 Dropout 不是 Latent Variable Model？

Dropout mask：

$$
m
$$

当然数学上也是随机变量。

但标准 neural-network modeling中它不是：

> 希望显式表达数据生成因素的 latent variable。

它主要是：

> training regularizer。

CVAE的：

$$
z
$$

则是 probabilistic model中明确的 latent variable。

---

## 128. Dropout Mask 会进入 Loss 吗？

通常没有显式：

$$
L_{\text{dropout}}
$$

loss项。

Dropout通过：

> 改变 stochastic forward，

间接改变 task loss和 gradient。

所以它是一种：

$$
\boxed{
\text{implicit stochastic regularization mechanism}
}
$$

而不是像：

$$
L_2
$$

那样额外加一个显式 penalty term。

---

## 129. Weight Decay 才更像显式/优化级参数约束

例如：

$$
L+\lambda\|W\|^2
$$

经典 L2 regularization具有显式 penalty形式。

Dropout没有简单：

$$
+\lambda L_{\text{drop}}
$$

这样的通用训练目标。

---

## 130. 为什么 Dropout 有时会降低 Training Performance？

因为训练任务被人为变难：

> 每次只能使用随机部分网络。

所以通常：

- training loss可能更高；
- 收敛更慢。

但 test generalization可能更好。

这正是 regularization常见 tradeoff。

---

## 131. Dropout 会不会总是提升 Test Performance？

不会。

如果：

- 模型本来就欠拟合；
- 数据非常多；
- dropout rate过高；
- architecture已经有其他强 regularization；

Dropout可能：

> 没帮助甚至伤害性能。

所以它不是万能模块。

---

## 132. 为什么现代某些大模型 Dropout 很低甚至 0？

大数据、大规模训练和其他 regularization条件下，

classic dropout不一定必要。

很多现代 LLM确实会使用：

> very small or zero dropout

在某些大规模 pretraining设置中。

这并不否定 Dropout原理。

它只是说明：

> regularization需要匹配 data/model regime。

---

## 133. ACT 为什么仍然使用 0.1？

ACT论文的任务级数据规模远小于大型语言模型。

作者在 Table III明确采用：

$$
\boxed{
dropout=0.1
}
$$

作为 architecture/training hyperparameter。

我们可以说：

> 它是 ACT 的 regularization配置之一。

但不能从论文中编造：

> 作者证明 0.1是这些机器人任务最优值。

---

## 134. Dropout 在 Validation 时应该开还是关？

标准 validation / evaluation：

> 关。

因为我们想评估：

> deterministic full model。

所以通常：

```python
model.eval()
```

然后计算 validation loss。

---

## 135. ACT Training Code 中的 `actions is not None` 和 Dropout Mode不是一回事

在 `detr_vae.py`：

```python
is_training =
    actions is not None
```

这个变量主要决定：

> CVAE encoder是否使用 actions。

但 Dropout是否激活：

> 由 PyTorch module 的 `self.training` 状态决定。

通常外层训练循环通过：

```python
model.train()
```

evaluation通过：

```python
model.eval()
```

来控制。

这两个“training”概念不要混。

---

## 136. 为什么这个区别很重要？

你可以理论上：

> 给 `actions`，

但 module处于 eval mode。

这时：

- CVAE路径可能仍根据代码逻辑执行；
- Dropout却是关闭的。

或者反过来：

> 没有 actions但 module仍处于 train mode，

Dropout可能仍打开。

所以：

$$
\boxed{
\text{algorithmic branch flag}
\neq
\text{PyTorch module training mode}
}
$$

---

## 137. 一个最小 PyTorch Dropout 实验

```python
import torch
import torch.nn as nn

drop = nn.Dropout(p=0.5)

x = torch.tensor(
    [1., 2., 3., 4.]
)

drop.train()

print(drop(x))
print(drop(x))
print(drop(x))
```

可能得到：

```text
[2, 0, 6, 0]
[0, 4, 6, 8]
[2, 4, 0, 8]
```

因为：

$$
1/(1-0.5)=2
$$

所以 retained elements乘 2。

---

## 138. 然后 Eval

```python
drop.eval()

print(drop(x))
```

得到：

```text
[1, 2, 3, 4]
```

Dropout变：

$$
\boxed{
Identity
}
$$

---

## 139. 一个从零实现 Inverted Dropout

```python
def dropout(x, p_drop, training):
    if not training:
        return x

    q_keep = 1.0 - p_drop

    mask = (
        torch.rand_like(x)
        < q_keep
    ).float()

    return (
        mask
        * x
        / q_keep
    )
```

这就是核心数学。

---

## 140. 为什么这个实现与 PyTorch思想一致？

因为：

$$
mask_i
\sim Bernoulli(q)
$$

输出：

$$
\frac{
mask_i x_i
}{
q
}
$$

training expectation：

$$
x_i
$$

evaluation：

$$
x_i
$$

所以 scale对齐。

实际 PyTorch kernel实现会更高效，

但数学主线就是这样。

---

## 141. 为什么不能写成 mask*x 然后 eval 原样 x？

因为训练 expectation：

$$
qx
$$

测试：

$$
x
$$

发生 scale mismatch。

除非你改成：

> test阶段乘 q。

---

## 142. 为什么不能训练和测试都 Dropout？

可以作为特殊方法，

但标准 Dropout不是这样设计。

如果 eval仍随机：

- prediction随机；
- 不再是标准 full-network approximation；
- validation难稳定。

MC Dropout是有意这样做的扩展。

---

## 143. 为什么不能训练和测试都不 Scale？

那会造成：

$$
E[y_{\text{train}}]=qx
$$

但：

$$
y_{\text{test}}=x
$$

后续 layers在测试突然接收更大 activations。

这就是原始论文专门解决的问题。

---

## 144. Expectation 对齐后 Variance 还不同

训练：

$$
Var(y_i)
=
x_i^2
\frac{d}{1-d}
$$

evaluation：

$$
Var_{\text{dropout}}=0
$$

因为不再采样 mask。

所以 train / eval不是：

> distribution完全一致。

只是：

$$
\boxed{
\text{mean scale aligned}
}
$$

training仍有额外 stochastic variance。

---

## 145. 这正是 Regularization Noise 的来源

如果 train/eval完全一样，

就没有 Dropout regularization。

Dropout故意让 training：

> 更 noisy。

Expectation scaling只是避免：

> noise之外再额外引入系统性 scale bias。

---

## 146. Dropout 和 Noise 的一个非常精确表述

训练时：

$$
y_i=r_ix_i
$$

其中：

$$
r_i
=
\begin{cases}
0,&\text{概率 }d\\
1/(1-d),&\text{概率 }1-d
\end{cases}
$$

满足：

$$
\boxed{
E[r_i]=1
}
$$

以及：

$$
\boxed{
Var(r_i)
=
\frac{
d
}{
1-d
}
}
$$

所以它是：

> mean-one multiplicative noise。

---

## 147. 为什么这个表达非常有用？

它把 Dropout从“删神经元”的故事，

提升成严格数学：

$$
\boxed{
\text{activation}
\times
\text{random multiplicative variable}
}
$$

这样很容易分析：

- expectation；
- variance；
- gradient；
- Gaussian dropout relation。

---

## 148. Dropout 对 Negative Activation 一样适用

例如：

$$
x=-4
$$

保留后：

$$
-4/q
$$

仍为负。

Dropout没有：

> ReLU那种只针对负值的语义。

它完全不关心 activation正负。

---

## 149. Dropout 和 ReLU 的 0 完全不同

#### ReLU 输出 0

因为：

$$
x\le0
$$

deterministically。

#### Dropout 输出 0

因为：

> Bernoulli random mask。

即使 activation：

$$
100
$$

也可能随机变 0。

---

## 150. Dropout 后为 0，不代表这个 Feature“不重要”

可能只是：

> 这一次训练随机被屏蔽。

所以不能从单次 Dropout mask解释：

> feature importance。

---

## 151. 为什么 Dropout 不是 Feature Selection？

Feature selection通常想识别：

> 哪些 feature长期有用。

Dropout则：

> 不管 feature是否有用，都随机屏蔽。

它的目标是鲁棒性，

不是找最优 feature subset。

---

## 152. 为什么 Dropout 可能减少依赖单一 Camera Feature？

在 ACT里，

visual memory token经过 Transformer时，

attention和FFN内部有 dropout。

因此训练中某些 routing / hidden feature contributions会随机失效。

从高层直觉上：

> 模型被迫不能依赖某一条内部 feature path每次都完美存在。

但不能直接说：

> “Dropout会随机删除整张 camera。”

当前 standard elementwise dropout并不是 camera-level masking。

---

## 153. 如果想 Randomly Drop Whole Camera 呢？

那应该设计：

> modality dropout / camera dropout。

例如 training时：

> 整个 camera stream置零。

这与 ACT当前 Transformer `nn.Dropout(0.1)`：

> 完全不同。

不要把 elementwise hidden dropout夸大成 sensor augmentation。

---

## 154. 为什么 Attention Dropout 也不是 Randomly Delete Entire Token？

对 attention matrix：

$$
A_{ij}
$$

随机drop的是：

> 某些 query-key routing weight entries。

一个 memory token $j$：

> 可能对 query 1被drop，

但对 query 2仍保留。

所以不等于：

> 整个 token从 sequence删除。

---

## 155. Standard Dropout 和 Token Dropout 不同

Token Dropout：

> 可能把完整 token vector移除/屏蔽。

Standard Transformer Dropout：

> 常对 feature elements或 attention weights随机 zero。

所以看论文里的：

> token dropout

要单独理解。

---

## 156. 为什么 Dropout 可以和 Residual 很好地组合？

因为：

$$
y=x+Dropout(F(x))
$$

即使这次：

$$
F(x)
$$

有部分 update被随机屏蔽，

input：

$$
x
$$

仍直接存在。

所以随机 regularization不会完全切断该 block的 signal path。

---

## 157. 这是不是等于 Stochastic Depth？

不是。

Standard Dropout：

> branch中部分 elements随机为0。

Stochastic Depth：

> 整个 residual branch可能一次整体为0。

二者粒度不同。

---

## 158. 为什么 Residual Dropout 不会永久改变 Residual Stream Dimension？

因为：

$$
Dropout(F(x))
$$

shape仍和：

$$
F(x)
$$

相同。

所以仍可：

$$
x+
Dropout(F(x))
$$

element-wise add。

---

## 159. ACT 的 Dropout 会不会影响 z=0 的含义？

训练时 policy memory受：

- latent $z$；
- dropout；

共同影响。

推理：

- $z=0$；
- dropout off。

所以：

> zero latent表示 ACT选择 prior mean作为 inference condition，

Dropout只是另一条 training regularization机制。

二者没有直接数学绑定。

---

## 160. Dropout 是否进入 KL Loss？

不直接。

KL：

$$
D_{KL}
(
q_\phi(z|x)
\|
p(z)
)
$$

来自 CVAE latent distributions。

Dropout会改变 network hidden activations，

因此间接可能影响：

$$
\mu,\log\sigma^2
$$

和最终 loss。

但 Dropout没有自己的 KL项。

---

## 161. Dropout 会影响 CVAE Encoder 的 μ 和 logvar 吗？

训练时：

> 会间接影响。

因为 CVAE Transformer Encoder内部有 dropout。

同一 `[CLS]+qpos+actions` 输入：

> 不同 dropout masks可能产生略不同 $h_{CLS}$。

于是：

$$
\mu,\log\sigma^2
$$

也可能略变。

这是 standard training stochasticity的一部分。

---

## 162. Inference 时 CVAE Encoder 本来就不用

ACT inference：

> training-only CVAE Encoder被跳过。

所以其 Dropout自然也不参与推理。

Policy Encoder / Decoder则仍存在，

但因为 `.eval()`：

> Dropout关闭。

---

## 163. 为什么机器人推理通常不希望 Dropout随机开着？

实际控制需要：

> 同样 observation尽量得到稳定 action。

如果每 20ms forward都随机 mask不同 internal features，

action可能产生不必要 stochastic jitter。

所以 canonical ACT inference关闭 Dropout是合理且标准的。

---

## 164. 当然机器人策略也可以有 Intentional Stochasticity

例如：

- stochastic policy；
- exploration；
- uncertainty sampling。

但那应该由：

> 明确的 policy distribution / exploration mechanism

控制。

不应把 training Dropout噪声误当成：

> 正式控制策略随机性。

---

## 165. Dropout 是否会让 Policy 更鲁棒？

作为 regularization目标：

> 希望提升 generalization和减少过拟合。

但具体鲁棒性：

- 对光照；
- 对 camera noise；
- 对 distribution shift；

不能只凭“用了 Dropout”就保证。

需要实验验证。

---

## 166. 常见误解一：Dropout 永久删除神经元

**错误。**

只是 training forward中临时 zero activation。

---

## 167. 常见误解二：Dropout 会减少模型参数量

**错误。**

参数量不变。

---

## 168. 常见误解三：Dropout 会让推理更快

标准 Dropout推理时直接关闭，

并不是模型压缩方法。

---

## 169. 常见误解四：PyTorch p=0.1 表示保留 10%

**错误。**

PyTorch：

$$
p=\text{drop probability}
$$

所以保留：

$$
90\%
$$

---

## 170. 常见误解五：原始 Dropout 论文的 p 和 PyTorch p 一样

**不一定。**

经典论文常用：

$$
p=\text{retention probability}
$$

---

## 171. 常见误解六：训练时 Drop 10%，剩下值保持原大小

PyTorch inverted dropout：

**不是。**

保留下来的值乘：

$$
1/0.9
$$

---

## 172. 常见误解七：乘 1/(1-p) 是为了让每次输出完全等于原输出

**错误。**

只保证：

$$
\boxed{
E[y]=x
}
$$

单次 forward仍随机不同。

---

## 173. 常见误解八：Expectation 一样说明 Train/Test Network完全一样

**错误。**

training还有非零 dropout variance。

---

## 174. 常见误解九：局部 expectation 对齐，所以整个 nonlinear network预测 expectation严格对齐

**错误。**

一般：

$$
f(E[X])
\neq
E[f(X)]
$$

---

## 175. 常见误解十：Dropout Test-Time Model是所有子网络预测的精确平均

深 nonlinear network中：

**一般不是。**

原始论文描述为：

> approximate model averaging。

---

## 176. 常见误解十一：Dropout 只有“防 co-adaptation”一个解释

**不完整。**

它还可从：

- multiplicative noise；
- stochastic regularization；
- approximate ensemble；

等角度理解。

---

## 177. 常见误解十二：Dropout Mask 是模型学出来的

**错误。**

标准 mask随机采样。

---

## 178. 常见误解十三：Dropout Probability 会被 Adam 优化

**错误。**

标准：

$$
p
$$

是 hyperparameter。

---

## 179. 常见误解十四：`torch.no_grad()` 会关闭 Dropout

**错误。**

必须切换：

```python
model.eval()
```

---

## 180. 常见误解十五：`model.eval()` 会关闭 Autograd

**错误。**

需要另行：

```python
torch.no_grad()
```

或 inference mode。

---

## 181. 常见误解十六：`model.eval()` 会关闭 LayerNorm

**错误。**

LayerNorm继续正常计算当前输入 statistics。

---

## 182. 常见误解十七：Attention Dropout 就是 Causal Mask

**错误。**

一个是随机 regularization，

一个是结构可见性约束。

---

## 183. 常见误解十八：Attention Dropout 后权重仍严格 Sum=1

**错误。**

inverted dropout后单次 row sum一般不等于1。

---

## 184. 常见误解十九：Attention Dropout 会删除完整 Token

**不一定。**

PyTorch MHA的 dropout作用于：

> attention output weights entries。

---

## 185. 常见误解二十：ACT dropout=0.1只在 FFN里使用一次

**错误。**

当前官方 Transformer代码中它至少用于：

- MHA attention weights；
- FFN hidden activations；
- residual branch outputs。

---

## 186. 常见误解二十一：ACT Transformer所有 Dropout共享同一 Random Mask

**错误。**

不同 Dropout operation独立采样。

---

## 187. 常见误解二十二：ACT dropout=0.1就是每次精确删除10%元素

**错误。**

每个 element独立以0.1概率drop。

实际比例只是统计上接近10%。

---

## 188. 常见误解二十三：Dropout 会改变 Tensor Shape

**错误。**

只改数值，

shape不变。

---

## 189. 常见误解二十四：Dropout 是一种 Attention Mechanism

**错误。**

Dropout是通用 regularization method。

它可以被应用在 Attention中，

但不是 Attention本身。

---

## 190. 常见误解二十五：Dropout 和 CVAE z 都是随机，所以本质一样

**错误。**

$z$属于 generative latent modeling。

Dropout mask属于 regularization noise。

---

## 191. 用一个公式记住 PyTorch Dropout

设：

$$
d
=
p_{\text{drop}}
$$

$$
q=1-d
$$

training：

$$
m_i\sim Bernoulli(q)
$$

$$
\boxed{
y_i
=
\frac{
m_ix_i
}{
q
}
}
$$

evaluation：

$$
\boxed{
y_i=x_i
}
$$

---

## 192. 用两个公式理解它为什么这样 Scale

Expectation：

$$
\boxed{
E[y_i]=x_i
}
$$

Variance：

$$
\boxed{
Var(y_i)
=
x_i^2
\frac{
d
}{
1-d
}
}
$$

所以：

> mean scale不变，

但 training加入随机 variance。

这几乎就是 inverted dropout最核心的数学。

---

## 193. 用一句话理解 Dropout

> **Dropout 是一种 training-time stochastic regularization：它用 Bernoulli mask随机屏蔽 activation 或某些内部连接贡献，并把保留下来的 activation按 keep probability 的倒数进行缩放，使训练期随机输出在局部 expectation 上与完整网络保持同一尺度；模型因此必须在许多随机扰动的 computation paths 下都完成任务，而标准 inference 时关闭 Dropout，直接使用完整 deterministic network。**

---

## 194. 用一句话理解为什么训练随机、推理不随机

> **训练时随机性本身就是 regularization：它故意让模型不能过度依赖某条固定内部路径；而推理阶段我们的目标已经不是继续 regularize，而是使用训练出的全部能力进行稳定预测，所以 inverted dropout把 scale compensation提前放在 training，从而让 evaluation 可以直接使用 identity mapping。**

---

## 195. 一句话连接 Transformer

> **Transformer 不只在一个地方使用 Dropout：原始论文明确在每个 sub-layer output写回 residual之前以及 embedding+position sum上使用 dropout；现代 `MultiheadAttention`实现还可以对 attention weights做 dropout，因此必须区分“Attention routing dropout”“FFN hidden dropout”和“Residual update dropout”，而不能把所有 `dropout=0.1` 想成同一个随机操作。**

---

## 196. 一句话连接 ACT

> **ACT Table III 的 `dropout=0.1` 在当前官方 DETR-style Transformer代码中会进入 Encoder/Decoder `nn.MultiheadAttention` 的 attention-weight dropout、FFN中间 activation dropout，以及每个 Attention/FFN sub-layer输出写入 residual stream前的 dropout；因此训练时 ACT 的 observation memory和action-slot computation都带有随机 regularization，而 `model.eval()` 后这些 Dropout关闭，和固定 $z=0$ 一起形成稳定的 deterministic policy inference。**

---

## 197. 到这里，一个 Transformer Layer 的主要组件已经全部拆完

现在我们已经单独理解：

- Q/K/V；
- Dot Product；
- Softmax；
- Self-Attention；
- Cross-Attention；
- Multi-Head Attention；
- Positional Encoding；
- Causal Mask；
- Encoder；
- Decoder；
- FFN；
- Residual Connection；
- LayerNorm；
- Dropout。

接下来最自然的是：

> 把这些零件重新放到一个真正的 **Multi-Head Attention** 内部，从单头完整推到多头 concat + $W_O$。

虽然前面的文章已经引用过 Multi-Head Attention，

但它还值得一个专门 canonical page。

下一篇：

> **[Multi-Head Attention：为什么一个 Attention Head 不够？](./multi-head-attention.md)**

会严格讲：

- 为什么：
  $$
  d_{\text{model}}=512,\ h=8
  $$
  时每头是 64-D；
- $W_Q^h,W_K^h,W_V^h$ 是怎样分别投影的；
- 为什么多个 heads并不是“同一个 Attention复制8遍”；
- Concatenate后为什么还需要：
  $$
  W_O
  $$
- 为什么原论文让总计算量接近单个 full-dimensional head；
- 每个 head是否一定对应可解释的语义；
- head redundancy和head pruning应该怎样谨慎理解；
- ACT 的 8 heads在 Encoder Self-Attention、Decoder Self-Attention、Cross-Attention分别对应什么 shape；
- attention dropout在每个 head的 weight matrix上怎样工作。

---

### Primary Source：Dropout

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov.

**Dropout: A Simple Way to Prevent Neural Networks from Overfitting.**  
Journal of Machine Learning Research, 2014.

- JMLR: https://www.jmlr.org/papers/v15/srivastava14a.html
- PDF: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

原论文核心事实：

- training时随机 temporary remove units；
- 每个 unit以某个 retention probability独立保留；
- 每次 training case会采样一个不同 thinned network；
- 所有这些 subnetworks共享 parameters；
- 作者把减少 co-adaptation作为核心动机之一；
- test时不实际枚举指数数量 subnetworks，而使用一个 unthinned network做 approximate model averaging。

原论文主要 notation中：

$$
p
=
\text{retention probability}
$$

因此要特别区别现代 PyTorch：

$$
p
=
\text{drop probability}
$$

---

### Original vs Inverted Dropout

原论文主要叙述：

#### Train

$$
m_i\sim Bernoulli(q)
$$

$$
y_i=m_ix_i
$$

#### Test

outgoing weights：

$$
W_{\text{test}}=qW
$$

---

论文 Section 10 同时明确指出一种等价 scaling convention：

> training时把 retained activations乘 $1/q$，test时不修改 weights。

这正是现代常用 inverted dropout：

$$
\boxed{
y_i=
\frac{
m_ix_i
}{
q
}
}
$$

training，

而 evaluation：

$$
\boxed{
y_i=x_i
}
$$

---

### Transformer Primary Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- PDF: https://arxiv.org/pdf/1706.03762

Section 5.4 — Regularization：

#### Residual Dropout

原论文明确：

> 对每个 sub-layer的 output应用 Dropout，然后再与 sub-layer input做 residual addition和 normalization。

即：

$$
\boxed{
LayerNorm(
x+
Dropout(
Sublayer(x)
)
)
}
$$

#### Input Dropout

原论文还对：

> embeddings + positional encodings

的和应用 Dropout。

Base model：

$$
\boxed{
P_{drop}=0.1
}
$$

---

### PyTorch Dropout Reference

PyTorch `nn.Dropout`:

https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html

当前文档明确：

- training时以 probability `p` 随机把 input elements置零；
- 每次 forward独立采样 Bernoulli mask；
- retained outputs在 training时乘：
  $$
  \boxed{
  \frac1{1-p}
  }
  $$
- evaluation时 Dropout module为：
  $$
  \boxed{
  \text{identity function}
  }
  $$

因此 PyTorch中的：

$$
p
$$

是：

> zero/drop probability。

---

### PyTorch Multi-Head Attention Reference

PyTorch `nn.MultiheadAttention`:

https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

当前文档明确：

```python
dropout
```

参数表示：

> **dropout probability on `attn_output_weights`**

所以：

```python
nn.MultiheadAttention(
    d_model,
    nhead,
    dropout=0.1
)
```

不仅仅是在整个 MHA output之后随机 zero features。

它内部对：

> Attention Weight Matrix

施加 dropout regularization。

---

### ACT Primary Source

Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.**  
RSS 2023.

- arXiv: https://arxiv.org/abs/2304.13705
- HTML: https://arxiv.org/html/2304.13705

ACT Table III：

$$
\boxed{
dropout=0.1
}
$$

同时：

$$
\#encoder\ layers=4
$$

$$
\#decoder\ layers=7
$$

$$
d_{\text{model}}=512
$$

$$
d_{\text{ff}}=3200
$$

$$
heads=8
$$

---

### ACT Official Implementation

Official repository:

https://github.com/tonyzhaozh/act

Transformer source:

https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py

#### Attention-Weight Dropout

Encoder：

```python
self.self_attn =
    nn.MultiheadAttention(
        d_model,
        nhead,
        dropout=dropout
    )
```

Decoder：

```python
self.self_attn =
    nn.MultiheadAttention(
        d_model,
        nhead,
        dropout=dropout
    )

self.multihead_attn =
    nn.MultiheadAttention(
        d_model,
        nhead,
        dropout=dropout
    )
```

根据 PyTorch `MultiheadAttention` API，

这里的 `dropout`作用于：

$$
\boxed{
attention\ output\ weights
}
$$

---

#### FFN Hidden Dropout

```python
self.dropout =
    nn.Dropout(dropout)
```

并用于：

```python
linear2(
    dropout(
        activation(
            linear1(x)
        )
    )
)
```

即：

$$
\boxed{
D
\rightarrow
D_{ff}
\rightarrow
Dropout
\rightarrow
D
}
$$

更准确地：

$$
Linear_1
\rightarrow
Activation
\rightarrow
Dropout
\rightarrow
Linear_2
$$

---

#### Residual Branch Dropout

Encoder：

```python
src =
    src +
    self.dropout1(
        attn_output
    )
```

```python
src =
    src +
    self.dropout2(
        ffn_output
    )
```

Decoder：

```python
tgt =
    tgt +
    self.dropout1(
        self_attn_output
    )
```

```python
tgt =
    tgt +
    self.dropout2(
        cross_attn_output
    )
```

```python
tgt =
    tgt +
    self.dropout3(
        ffn_output
    )
```

因此 ACT training中：

> Attention routing、FFN hidden activations、Residual updates

都受到 Dropout regularization。

---

### ACT CVAE Encoder

Official:

https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

`build_encoder(args)`：

```python
dropout =
    args.dropout
```

然后传入：

```python
TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    ...
)
```

所以 training-only CVAE Encoder：

> 也使用 ACT 的 dropout配置。

---

### 本文知识连接

#### 数学

- Bernoulli Distribution
- Binomial Distribution
- Expectation
- Variance
- Random Variable

#### Deep Learning

- Overfitting
- Regularization
- Weight Decay
- Data Augmentation
- Stochastic Depth

#### Transformer

- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Feed-Forward Network](./feed-forward-network.md)
- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Attention](./attention.md)
- [Multi-Head Attention](./multi-head-attention.md)
- [Causal Mask](./causal-mask.md)

#### Generative Modeling

- [Latent Variable](../generative-models/latent-variable.md)
- [Reparameterization Trick](../generative-models/reparameterization-trick.md)
- [CVAE](../generative-models/cvae.md)

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [CVAE in ACT](../robot-learning/act/cvae-in-act.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Inference](../robot-learning/act/inference.md)

#### 下一步

- [Multi-Head Attention](./multi-head-attention.md)
