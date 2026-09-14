---
title: "Feed-Forward Network：Attention 已经交换信息了，为什么还需要 MLP？"
description: "从 Transformer 原论文的 Position-wise Feed-Forward Network 出发，严格理解 512→2048→512 / ACT 512→3200→512 的作用、为什么需要非线性、为什么逐 token 独立却参数共享，以及 FFN 与 Attention 在计算和表示上的互补关系。"
status: reviewed
pageType: concept
canonical: /deep-learning/feed-forward-network
updated: "2026-09-15"
---

# Feed-Forward Network：Attention 已经交换信息了，为什么还需要 MLP？

如果你第一次认真看 Transformer，很容易产生一个疑问：

> Attention 已经让每个 token 从其他 token 读取信息了，为什么后面还要再接一个普通 MLP？

甚至更容易产生一种错觉：

> Transformer 最重要的是 Attention，FFN 只是一个顺手加上的小模块。

事实上完全不是。

原始 Transformer 的每一层，不论 Encoder 还是 Decoder，都明确包含一个：

> **Position-wise Feed-Forward Network**

原论文公式：

$$
\boxed{
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
$$

在 Transformer Base 中：

$$
d_{\text{model}}=512
$$

而 FFN 中间维度：

$$
d_{\text{ff}}=2048
$$

所以每个 token 都经历：

$$
\boxed{
512
\rightarrow
2048
\rightarrow
512
}
$$

ACT 更宽：

$$
\boxed{
512
\rightarrow
3200
\rightarrow
512
}
$$

这已经提示我们：

> FFN 绝对不是一个可以忽略的小尾巴。

这一篇专门回答：

1. Attention 已经有 Softmax 非线性了，为什么还需要 FFN？
2. 为什么先扩维再缩维？
3. 为什么没有 activation 时两层 Linear 等于一层？
4. ReLU 到底让网络多了什么能力？
5. 什么叫 **Position-wise**？
6. 为什么所有 token 共用同一个 FFN 参数？
7. FFN 会不会让不同 token 互相交流？
8. 为什么论文说它也可以看成两个 kernel size 1 的 convolution？
9. FFN 的参数量为什么常常比 Attention 还大？
10. ACT 的 $512\rightarrow3200\rightarrow512$ 到底意味着什么？
11. GELU、SwiGLU 又是在改 FFN 的哪一部分？

---

## 1. 先回到 Transformer Layer

一个原始 Transformer Encoder Layer：

```text
Input Tokens
    │
    ▼
Multi-Head Self-Attention
    │
    ▼
Residual + LayerNorm
    │
    ▼
Feed-Forward Network
    │
    ▼
Residual + LayerNorm
```

Decoder Layer 也是一样有 FFN：

```text
Self-Attention
↓
Cross-Attention
↓
Feed-Forward Network
```

所以 FFN 不是只存在于：

> Encoder。

而是存在于：

$$
\boxed{
\text{每一个 Encoder Layer 和每一个 Decoder Layer}
}
$$

原始 Transformer Section 3.3 明确写道：

> 除 Attention sub-layers 外，Encoder 和 Decoder 的每一层都包含一个 fully connected feed-forward network。

---

## 2. 原论文正式定义

对单个 position 的 hidden vector：

$$
x\in\mathbb R^{d_{\text{model}}}
$$

FFN：

$$
\boxed{
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
}
$$

拆开：

---

### 第一层 Linear

$$
h=xW_1+b_1
$$

---

### ReLU

$$
r=\operatorname{ReLU}(h)
$$

其中：

$$
\operatorname{ReLU}(z)
=
\max(0,z)
$$

---

### 第二层 Linear

$$
y=rW_2+b_2
$$

所以：

$$
\boxed{
x
\rightarrow
Linear_1
\rightarrow
ReLU
\rightarrow
Linear_2
\rightarrow
y
}
$$

---

## 3. 原始 Transformer 的 Shape

Base Transformer：

$$
d_{\text{model}}=512
$$

$$
d_{\text{ff}}=2048
$$

所以：

$$
x:
[512]
$$

第一层：

$$
W_1:
[512,2048]
$$

得到：

$$
h:
[2048]
$$

ReLU：

$$
r:
[2048]
$$

第二层：

$$
W_2:
[2048,512]
$$

得到：

$$
y:
[512]
$$

即：

$$
\boxed{
512
\rightarrow
2048
\rightarrow
512
}
$$

---

## 4. 为什么输入和输出都必须回到 512？

因为 FFN 外面还有 Residual Connection。

原始 Post-LN：

$$
\boxed{
y_{\text{block}}
=
LN(
x+
Dropout(
FFN(x)
)
)
}
$$

为了做：

$$
x+FFN(x)
$$

二者 shape 必须一样。

所以：

$$
\boxed{
FFN:
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
}
$$

中间可以变宽，

最终必须重新回：

$$
d_{\text{model}}
$$

---

## 5. 为什么中间要扩到 2048？

一个最简单但不完整的回答是：

> 为了增加模型容量。

我们需要进一步说明“增加什么容量”。

第一层：

$$
512
\rightarrow
2048
$$

相当于从当前 512-D token representation 中构造：

$$
2048
$$

个 intermediate features。

每一个 intermediate unit：

$$
h_j
=
x^\top w_j+b_j
$$

都可以学习：

> 一种不同的 feature detector / feature combination。

然后 ReLU 决定：

$$
h_j
$$

是否保留。

最后第二层把这些 activated intermediate features：

> 重新组合回 512-D hidden space。

---

## 6. 一个小例子：2 → 4 → 2

假设：

$$
x=[x_1,x_2]
$$

FFN：

$$
2\rightarrow4\rightarrow2
$$

第一层可以生成：

$$
h_1=w_{11}x_1+w_{21}x_2+b_1
$$

$$
h_2=w_{12}x_1+w_{22}x_2+b_2
$$

$$
h_3=w_{13}x_1+w_{23}x_2+b_3
$$

$$
h_4=w_{14}x_1+w_{24}x_2+b_4
$$

也就是说：

> 从原来 2 个 coordinates 构造 4 个不同的 learned feature combinations。

然后：

$$
r_j=\max(0,h_j)
$$

再把：

$$
[r_1,r_2,r_3,r_4]
$$

重新组合成两个输出 dimensions。

---

## 7. 为什么不能直接 512 → 512？

当然可以设计。

但：

$$
512\rightarrow2048
$$

给中间 nonlinear stage 更多 feature channels。

这类似于给网络一个更宽的：

> hidden workspace。

它可以在 2048 个 intermediate directions 上：

- 检测；
- 门控；
- 重组；

特征，再压回 512。

这不意味着：

> “2048 是理论上唯一正确的宽度。”

它只是原始 Transformer 的 architecture choice：

$$
d_{\text{ff}}=4d_{\text{model}}
$$

ACT 则选了：

$$
3200/512=6.25
$$

倍。

---

## 8. 为什么“扩维”本身还不够？

这是理解 FFN 最关键的数学点之一。

假设没有 ReLU：

$$
FFN(x)
=
(xW_1+b_1)W_2+b_2
$$

先忽略 bias：

$$
FFN(x)
=
xW_1W_2
$$

令：

$$
W=W_1W_2
$$

则：

$$
\boxed{
FFN(x)=xW
}
$$

仍然只是：

> 一个 Linear Transformation。

所以即使：

$$
512
\rightarrow
100000
\rightarrow
512
$$

只要中间没有非线性，

仍可以合并成：

$$
512\rightarrow512
$$

的一层 Linear。

---

## 9. 带 Bias 也仍然可以合并

完整：

$$
(xW_1+b_1)W_2+b_2
$$

展开：

$$
xW_1W_2+b_1W_2+b_2
$$

定义：

$$
W'=W_1W_2
$$

$$
b'=b_1W_2+b_2
$$

则：

$$
\boxed{
FFN(x)=xW'+b'
}
$$

仍然只是一个 affine transformation。

因此：

$$
\boxed{
\text{两层 Linear 不自动等于更深的函数}
}
$$

真正让两层无法合并的是：

$$
\boxed{
\text{Nonlinearity}
}
$$

---

## 10. ReLU 打破了这种可合并性

有 ReLU：

$$
FFN(x)
=
ReLU(xW_1+b_1)W_2+b_2
$$

无法把：

$$
ReLU(\cdot)
$$

吸收到一个固定矩阵里。

因为对于不同 input：

$$
x
$$

不同 intermediate neurons 会：

- 打开；
- 关闭。

所以 FFN 对不同 input regions：

> 实际使用不同的有效线性组合。

这就是 piecewise-linear nonlinearity。

---

## 11. ReLU 可以看成一种 Gate

定义：

$$
h_j=x^\top w_j+b_j
$$

如果：

$$
h_j\le0
$$

则：

$$
ReLU(h_j)=0
$$

这个 intermediate feature：

> 被关掉。

如果：

$$
h_j>0
$$

则：

$$
ReLU(h_j)=h_j
$$

这个 feature：

> 被保留。

所以第一层 2048 个 hidden units 可以被直觉化为：

> 2048 个 learned feature channels + input-dependent on/off gates。

这只是直觉。

它们并没有人类预先规定的语义标签。

---

## 12. 为什么这比单个 Linear 更强？

单个 Linear：

$$
y=xW+b
$$

无论输入是什么，

都使用：

> 同一组线性规则。

FFN：

$$
y=W_2ReLU(W_1x+b_1)+b_2
$$

不同 input 会激活不同 hidden subsets。

因此：

> 网络可以对不同输入区域使用不同 feature-processing regimes。

这大幅提高函数表达能力。

---

## 13. 但不能说“FFN 是 Transformer 唯一的 Nonlinearity”

这是一个常见但错误的说法。

Attention 中已经有：

$$
Softmax
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$

Softmax 本身就是非线性。

LayerNorm 也不是一个固定线性映射。

所以：

$$
\boxed{
\text{Transformer 没有 FFN 并不等于纯线性网络}
}
$$

更准确是：

> FFN 提供一个额外、强大的、逐 token 的 nonlinear feature-transformation stage。

---

## 14. Attention 和 FFN 到底怎样分工？

一个非常实用的第一层理解：

$$
\boxed{
\text{Attention}
=
\text{跨 token 取信息}
}
$$

$$
\boxed{
\text{FFN}
=
\text{token 内加工信息}
}
$$

也可以说：

$$
\boxed{
\text{Attention}
:
\text{communication}
}
$$

$$
\boxed{
\text{FFN}
:
\text{computation}
}
$$

---

## 15. 为什么说 Attention 是 Communication？

对 token $i$：

$$
o_i
=
\sum_j
\alpha_{ij}v_j
$$

它显式从：

$$
j=1,\ldots,n
$$

其他 positions 读取信息。

所以 token $i$ 的新 representation：

> 可以依赖整个 sequence。

这是：

$$
\boxed{
\text{cross-position interaction}
}
$$

---

## 16. 为什么说 FFN 是 Per-Token Computation？

FFN：

$$
FFN(x_i)
$$

只读取：

$$
x_i
$$

本身。

它不会直接访问：

$$
x_j,\quad j\neq i
$$

所以：

$$
\boxed{
FFN
\text{ does not itself mix sequence positions}
}
$$

---

## 17. 那 FFN 怎么可能利用“别的 Token 的信息”？

因为 FFN 的输入：

$$
x_i
$$

已经经过了 Attention。

例如：

$$
x_i'
=
LN(
x_i+
Attention_i(X)
)
$$

这个：

$$
x_i'
$$

已经包含从其他 token 聚合来的 context。

FFN 虽然只处理：

$$
x_i'
$$

一个 vector，

但这个 vector 已经是：

> contextualized representation。

所以 FFN 做的是：

> 对“已经收集好的上下文信息”继续加工。

---

## 18. 一个更准确的信息流

```text
原 token i
    │
    ▼
Attention
    │
从其他 tokens 读取信息
    │
    ▼
contextualized token i
    │
    ▼
FFN
    │
对当前综合信息做 nonlinear feature transform
    │
    ▼
更成熟的 token i representation
```

所以二者不是竞争关系。

而是：

$$
\boxed{
\text{先交流，再加工}
}
$$

---

## 19. 为什么 FFN 放在 Attention 后面很自然？

假设 token $i$ 刚刚通过 Attention 得到：

- 自己原来的信息；
- token 3 的信息；
- token 10 的信息；
- token 50 的信息。

现在它需要根据这些组合：

> 形成新的 task-relevant features。

FFN 就像：

> 每个 token 自己的“本地处理器”。

Attention 类似网络通信。

FFN 类似拿到数据后的本地 CPU computation。

这是一个类比，不是定义。

---

## 20. “Position-Wise” 到底是什么意思？

假设：

$$
X
\in
\mathbb R^{N\times D}
$$

例如：

$$
N=4
$$

有四个 tokens：

$$
x_1,x_2,x_3,x_4
$$

FFN 计算：

$$
y_1=FFN(x_1)
$$

$$
y_2=FFN(x_2)
$$

$$
y_3=FFN(x_3)
$$

$$
y_4=FFN(x_4)
$$

每个 position：

> 单独处理。

这就是：

$$
\boxed{
\text{Position-Wise}
}
$$

---

## 21. 但所有 Position 使用同一个 FFN

非常重要。

不是：

$$
FFN_1(x_1)
$$

$$
FFN_2(x_2)
$$

而是：

$$
\boxed{
FFN(x_i)
\quad
\forall i
}
$$

同一层的：

$$
W_1,b_1,W_2,b_2
$$

对所有 sequence positions 共享。

原论文明确写：

> applied to each position separately and identically。

---

## 22. 为什么要共享参数？

如果每一个 position 都有自己的 FFN：

$$
FFN_i
$$

那么：

- 参数量随最大 sequence length 增长；
- position 37 和 position 38 会使用完全不同 feature transform；
- 对变长 sequence 泛化不自然。

共享同一个 FFN 表示：

> “无论这个 token 位于哪里，我对 hidden features 使用同一种局部计算规则。”

位置差异已经编码在：

> token representation 本身。

---

## 23. Position 共享不意味着输出一样

即使：

$$
FFN
$$

参数相同，

只要：

$$
x_i\neq x_j
$$

通常：

$$
FFN(x_i)\neq FFN(x_j)
$$

所以共享函数不等于共享结果。

就像同一个分类器对不同样本输出不同。

---

## 24. 不同 Transformer Layers 的 FFN 是否共享？

原论文明确：

> 同一层不同 positions 使用相同 Linear transformations；

但：

> 不同 layers 使用不同 parameters。

所以：

$$
FFN^{(1)}
\neq
FFN^{(2)}
$$

一般成立。

每层都可以学习：

> 自己这一深度所需要的 feature transformation。

---

## 25. 为什么论文说 FFN 等价于两个 Kernel Size 1 的 Convolution？

考虑 sequence：

$$
X
\in
\mathbb R^{N\times D}
$$

如果把：

$$
D
$$

理解成 channels，

kernel size：

$$
1
$$

的 convolution：

> 不会读取相邻 position。

它只在当前位置的 feature channels 上做线性组合。

这恰好和：

$$
Linear(D,D_{ff})
$$

逐 position 应用一样。

所以原论文写：

> Another way of describing this is as two convolutions with kernel size 1。

---

## 26. 为什么 1×1 Conv 不做 Spatial Mixing？

在一维 sequence 中 kernel width 1：

> receptive field 只覆盖当前位置。

在二维 image 中 1×1 Conv：

> 只看当前 pixel / feature location 的 channels。

所以它主要做：

> channel mixing。

Transformer FFN 也是类似：

> 每个 position 内 feature/channel mixing。

---

## 27. 一个矩阵实现为什么看起来不像“逐 Token for-loop”？

数学上：

$$
y_i=FFN(x_i)
$$

实现上不会真的：

```python
for token in tokens:
    token = ffn(token)
```

GPU 会一次矩阵运算：

$$
XW_1
$$

如果：

$$
X:
[B,N,D]
$$

$$
W_1:
[D,D_{ff}]
$$

则直接得到：

$$
[B,N,D_{ff}]
$$

这只是：

> 把所有独立 position 的相同 Linear 并行算完。

所以：

$$
\boxed{
\text{position-wise}
\neq
\text{sequential computation}
}
$$

---

## 28. Batch 维度也不会互相混合

输入：

$$
X:
[B,N,D]
$$

FFN 不会让：

> sample 1 token 3

和：

> sample 2 token 3

相互作用。

它只是对所有：

$$
B\times N
$$

个 vectors 使用同一个小 MLP。

---

## 29. FFN 的 Shape Flow

一般：

$$
X:
[B,N,D]
$$

第一层：

$$
XW_1+b_1
$$

得到：

$$
\boxed{
[B,N,D_{ff}]
}
$$

activation：

$$
[B,N,D_{ff}]
$$

第二层：

$$
\boxed{
[B,N,D]
}
$$

所以：

$$
\boxed{
[B,N,D]
\rightarrow
[B,N,D_{ff}]
\rightarrow
[B,N,D]
}
$$

Batch 和 token count 都不变。

---

## 30. 原始 Transformer 的完整 Shape

$$
X:
[B,N,512]
$$

第一层：

$$
[B,N,512]
\rightarrow
[B,N,2048]
$$

ReLU：

$$
[B,N,2048]
$$

第二层：

$$
[B,N,2048]
\rightarrow
[B,N,512]
$$

所以最终仍：

$$
\boxed{
[B,N,512]
}
$$

---

## 31. ACT 的完整 Shape

ACT Table III：

$$
d_{\text{model}}=512
$$

$$
d_{\text{ff}}=3200
$$

所以：

$$
\boxed{
[B,N,512]
\rightarrow
[B,N,3200]
\rightarrow
[B,N,512]
}
$$

activation：

> ReLU。

官方 repository 的默认 Transformer activation 也是：

```python
activation="relu"
```

---

## 32. ACT Policy Encoder 中 N 是多少？

$$
N=1202
$$

所以一层 FFN：

$$
[B,1202,512]
$$

先变成：

$$
\boxed{
[B,1202,3200]
}
$$

再回：

$$
\boxed{
[B,1202,512]
}
$$

1202 个 tokens：

> 各自独立经过同一套 512→3200→512 MLP。

---

## 33. 这意味着 1202 个 Tokens 会互相通过 FFN 通信吗？

**不会。**

即使实现 tensor 是：

$$
[B,1202,3200]
$$

也不意味着 position dimension 被混合。

Linear 作用在：

> 最后一个 feature dimension。

所以：

$$
token_i
$$

的 FFN output只由：

$$
token_i
$$

的输入 hidden vector决定。

---

## 34. ACT Decoder 中也一样

如果 chunk：

$$
k=100
$$

Decoder FFN：

$$
[B,100,512]
$$

变：

$$
[B,100,3200]
$$

再回：

$$
[B,100,512]
$$

每一个 action slot：

> 独立经过同一个 FFN。

action-slot 之间真正交流：

> 发生在 Decoder Self-Attention。

---

## 35. CVAE Encoder 中也一样

若：

$$
k=100
$$

CVAE sequence：

$$
[CLS],qpos,a_0,\ldots,a_{99}
$$

共：

$$
102
$$

tokens。

FFN：

$$
[B,102,512]
\rightarrow
[B,102,3200]
\rightarrow
[B,102,512]
$$

所以：

- `[CLS]`；
- qpos；
- action tokens；

都使用同一套 FFN parameters。

---

## 36. 为什么 `[CLS]` 和 Action Token 可以使用同一 FFN？

因为进入 Transformer 后，

它们都被表示成：

$$
512
$$

维 hidden vectors。

FFN 只看到：

> hidden features。

它不知道输入最初来自：

- `[CLS]`；
- qpos；
- action；
- image。

不同 token 类型的差异已经被编码在：

> representation 中。

共享 FFN 提供统一的 feature-processing rule。

---

## 37. 共享 FFN 是 Transformer 的一个强假设

它假设：

> 不同 positions 可以使用同一类局部 computation。

这带来：

- 参数效率；
- 长度泛化；
- sequence symmetry。

Position / modality differences 通过 hidden state表达，

而不是给每个位置一个独立网络。

这是 Transformer 的重要 inductive bias 之一。

---

## 38. FFN 参数量到底有多大？

先算原始 Transformer。

---

### W₁

$$
512\times2048
=
1,048,576
$$

---

### W₂

$$
2048\times512
=
1,048,576
$$

只算 weight matrices：

$$
\boxed{
2,097,152
}
$$

约：

$$
2.10M
$$

---

## 39. 加上 Bias

第一层 bias：

$$
2048
$$

第二层：

$$
512
$$

总：

$$
2560
$$

所以完整：

$$
\boxed{
2,099,712
}
$$

个参数。

约：

> 2.10 million parameters per FFN block。

---

## 40. Attention Projection 有多少参数？

标准：

$$
d=512
$$

MHA 主要 projection weights：

$$
W_Q,W_K,W_V,W_O
$$

每个：

$$
512\times512
$$

所以 weight-only：

$$
4\times512^2
=
1,048,576
$$

约：

$$
1.05M
$$

---

## 41. 一个很重要的结论

原始 Transformer 中，

一个 FFN 的 Linear weight 参数量：

$$
\approx2.10M
$$

而一个 MHA 的 Q/K/V/O projection weights：

$$
\approx1.05M
$$

所以：

$$
\boxed{
\text{FFN weight parameters}
\approx
2\times
\text{MHA projection weights}
}
$$

忽略 bias 等小项。

这就是为什么：

> “Transformer 基本都是 Attention 参数”

是错误印象。

---

## 42. ACT FFN 参数更多

ACT：

$$
512\rightarrow3200\rightarrow512
$$

---

第一层：

$$
512\times3200
=
1,638,400
$$

第二层：

$$
3200\times512
=
1,638,400
$$

weight-only：

$$
\boxed{
3,276,800
}
$$

约：

$$
3.28M
$$

---

## 43. 加 ACT FFN Bias

bias：

$$
3200+512=3712
$$

所以：

$$
\boxed{
3,280,512
}
$$

parameters。

而 MHA projection weight仍大约：

$$
1.05M
$$

所以在 ACT 的这一配置里：

> 一个 FFN block 的 weight 参数甚至超过一组标准 MHA projection weights三倍。

---

## 44. 这不意味着 FFN 一定比 Attention “更重要”

参数多：

$$
\neq
$$

功能更重要。

Attention 的独特能力是：

> 动态跨 token routing。

FFN 的独特能力是：

> 宽维 nonlinear local computation。

两者缺一都会改变 Transformer 的计算性质。

所以不要把比较参数量变成：

> “FFN 才是真正核心，Attention 不重要。”

正确理解是：

$$
\boxed{
\text{它们承担互补角色}
}
$$

---

## 45. FFN 的计算复杂度

对：

$$
N
$$

个 tokens，

第一层矩阵乘：

$$
[N,D]
\times
[D,D_{ff}]
$$

主要成本与：

$$
NDD_{ff}
$$

成正比。

第二层同量级。

所以：

$$
\boxed{
FFN\ complexity
=
O(
NDD_{ff}
)
}
$$

省略常数。

---

## 46. Self-Attention 核心复杂度

Attention token-token interaction：

$$
QK^\top
$$

以及：

$$
AV
$$

主要约：

$$
O(N^2D)
$$

外加：

$$
Q/K/V/O
$$

projections：

$$
O(ND^2)
$$

所以：

$$
\boxed{
Attention
\text{ 和 FFN 的主导成本取决于 }N,D,D_{ff}
}
$$

---

## 47. 为什么短 Sequence 时 FFN 可能很重？

Attention 的 quadratic 项：

$$
N^2D
$$

随着：

$$
N
$$

快速增长。

FFN：

$$
NDD_{ff}
$$

对 N 线性。

如果 sequence 不太长、但：

$$
D_{ff}\gg D
$$

FFN 的矩阵乘同样非常可观。

所以实际 Transformer profiling：

> 不能只盯 Attention。

---

## 48. 为什么长 Context 时 Attention 更容易成为瓶颈？

因为：

$$
N^2
$$

最终增长比：

$$
N
$$

快。

当 sequence length 非常大，

attention score matrix 和相关计算/显存会越来越显著。

这就是：

- sparse attention；
- linear attention；
- FlashAttention；

等研究方向的重要背景之一。

但 FFN 仍然占大量参数和计算。

---

## 49. 为什么 FFN 参数不随 Sequence Length 变化？

FFN 参数：

$$
W_1,W_2
$$

只依赖：

$$
D,D_{ff}
$$

和：

$$
N
$$

无关。

sequence 变长：

> 同一套 FFN被多使用几次。

所以参数量不增加，

只是计算次数随 N 线性增长。

---

## 50. Attention 的参数量也不直接随 N 增长

MHA projections：

$$
W_Q,W_K,W_V,W_O
$$

也和 sequence length无关。

但 attention intermediate matrix：

$$
[N,N]
$$

随 N quadratic 增大。

这说明：

> 参数复杂度和运行时中间激活复杂度是两个不同问题。

---

## 51. 为什么 FFN 可以看成“对 Context 做重新解释”？

假设 Attention 后 token representation：

$$
h_i
$$

已经包含：

- 自己的信息；
- 其他 token 的 context。

但这些信息只是：

> 被聚合进同一个 hidden vector。

FFN 可以学习：

> 哪些 feature combinations 在当前任务下重要。

例如抽象地：

```text
如果 feature 12 和 feature 74 同时高
→ 激活某个 hidden unit

如果 feature 20 为负
→ 某些 ReLU units关闭
```

这让 contextual representation进一步变成：

> task-useful nonlinear features。

---

## 52. 为什么“Attention 是检索，FFN 是计算”很有用？

因为 Attention output：

$$
o_i
=
\sum_j
\alpha_{ij}v_j
$$

很像：

> 从 memory 中取回一批 weighted information。

FFN：

$$
FFN(o_i)
$$

则像：

> 对取回信息进行本地处理。

这个视角能帮助理解 Transformer：

```text
Retrieve
↓
Compute
↓
Retrieve again
↓
Compute again
```

层层交替。

---

## 53. 但这个类比的边界是什么？

Attention 自己也有：

- learned projections；
- Softmax；
- output projection。

所以它不只是一个“纯数据库检索”。

FFN 也不是 CPU 指令集。

这只是：

> 按独特功能做的高层分工。

严格数学仍然是各自公式。

---

## 54. FFN 会不会创建“新信息”？

从信息论上说，网络不会凭空获得外部新观测。

但 representation 层面，

它可以构造：

> 输入 features 的新 nonlinear combinations。

例如原来：

$$
x=[a,b]
$$

FFN 可以形成某种：

$$
ReLU(
w_1a+w_2b+c
)
$$

再和其他 hidden features组合。

所以：

> 它可以生成新的 internal features，

但不是从环境凭空获得新事实。

---

## 55. 为什么 Attention 的 Weighted Sum 之后需要这种 New Feature Construction？

Attention 的核心输出是：

$$
\sum_j\alpha_jv_j
$$

它把多个 Value features混合。

但很多任务需要的不只是：

> “把相关信息相加”。

而是：

> 对组合后的信息应用条件性的 nonlinear transformation。

FFN提供这一步。

---

## 56. 一个极简逻辑例子

假设 token representation中有两个 dimensions：

$$
x_1=
\text{“看到红色物体” feature}
$$

$$
x_2=
\text{“gripper 已接近” feature}
$$

一个 task可能关心：

> 两者同时成立时激活某个更高层 feature。

单个 Linear 只能做：

$$
w_1x_1+w_2x_2
$$

而多 hidden units + nonlinear activation可以构造更复杂的分段条件关系。

实际 Transformer features当然不会如此干净可解释。

这只是说明：

> 非线性 hidden layer可以形成 richer feature interactions。

---

## 57. ReLU 为什么是原始 Transformer 的选择？

2017 Transformer 使用：

$$
ReLU
$$

它：

- 简单；
- 计算便宜；
- 梯度在正区间稳定；
- 当时已经是深度学习常见激活。

但：

$$
\boxed{
\text{Transformer}
\neq
\text{必须 ReLU}
}
$$

后来的模型广泛替换 activation。

---

## 58. GELU 是什么？

GELU：

$$
\boxed{
GELU(x)
=
x\Phi(x)
}
$$

其中：

$$
\Phi(x)
$$

是标准正态分布 CDF。

和 ReLU 的硬：

$$
x\le0\rightarrow0
$$

相比，

GELU 是一种更平滑的 input-dependent gating。

BERT 等后续 Transformer 架构广泛使用 GELU。

但：

> 这不是 2017 原始 Transformer 的 FFN。

---

## 59. ReLU 和 GELU 的直觉差异

ReLU：

$$
x=
-0.01
\rightarrow0
$$

$$
x=
0.01
\rightarrow0.01
$$

在 0 处硬切。

GELU：

> 对输入做平滑衰减/保留。

所以接近 0 的负值：

> 不一定立刻精确归零。

两者都是 activation choice。

FFN 的基本结构：

$$
D
\rightarrow
D_{ff}
\rightarrow
D
$$

没有因此消失。

---

## 60. ACT Official Code 支持哪些 Activation？

当前 `transformer.py`：

```python
if activation == "relu":
    return F.relu

if activation == "gelu":
    return F.gelu

if activation == "glu":
    return F.glu
```

所以通用代码支持：

- ReLU；
- GELU；
- GLU。

但默认：

```python
activation="relu"
```

ACT canonical 配置应理解为：

> ReLU FFN。

---

## 61. GLU 又是什么思路？

Gated Linear Unit 类方法不只是：

$$
activation(Wx)
$$

而是让一条 feature branch：

> 门控另一条 branch。

典型高层形式：

$$
\boxed{
\phi(xW_g)
\odot
(xW_v)
}
$$

其中：

$$
\odot
$$

是 element-wise multiplication。

也就是说：

> 一组 features 决定另一组 features 放行多少。

---

## 62. SwiGLU 是现代 Transformer 中常见的 FFN 变体

一种常见抽象形式：

$$
\boxed{
SwiGLU(x)
=
Swish(xW_g)
\odot
(xW_v)
}
$$

然后再通过 output projection：

$$
W_o
$$

回到：

$$
d_{\text{model}}
$$

它属于：

> gated FFN family。

但这是 2017 Transformer 之后的发展。

不要把：

$$
SwiGLU
$$

写成原始 Transformer FFN。

---

## 63. 为什么现代模型愿意修改 FFN？

因为 FFN：

- 参数量大；
- 计算量大；
- 直接决定 per-token nonlinear processing capacity。

所以改变：

- activation；
- gating；
- hidden width；

会显著影响模型。

这也再次说明：

> FFN 不是一个不重要的附属模块。

---

## 64. FFN Hidden Width 是 Architecture Hyperparameter

原始：

$$
d_{ff}=2048
$$

ACT：

$$
d_{ff}=3200
$$

其他模型可能：

- 更大；
- 更小；
- gated structure需要不同等效宽度。

所以不能把：

$$
4d_{\text{model}}
$$

当成 Transformer 数学定义。

它只是经典 Base Transformer 的设置。

---

## 65. 为什么 ACT 用 3200，不是 2048？

ACT Table III 给出的实验配置就是：

$$
3200
$$

但论文没有提供一个理论推导说：

$$
3200
$$

是机器人动作预测的数学最优值。

所以最严谨的说法：

> 3200 是 ACT 作者采用的 architecture hyperparameter。

不要编造：

> “因为机器人任务比语言复杂，所以必须扩 6.25 倍。”

---

## 66. Code Default 和实验配置为什么又不同？

ACT `Transformer` class：

```python
dim_feedforward=2048
```

只是通用模块默认值。

但 model builder：

```python
dim_feedforward=args.dim_feedforward
```

训练命令官方 README 使用：

```text
--dim_feedforward 3200
```

因此真正 canonical ACT 配置：

$$
\boxed{
3200
}
$$

这再次提醒：

> 读代码不能只看 function signature 的 default。

还要追：

> config / call site。

---

## 67. ACT Encoder FFN 和 Decoder FFN 用相同维度吗？

当前 `build_transformer(args)` 把同一个：

$$
args.dim\_feedforward
$$

传给：

- `TransformerEncoderLayer`
- `TransformerDecoderLayer`

所以 Policy Transformer：

> Encoder 和 Decoder FFN 都使用 3200 hidden width。

---

## 68. CVAE Encoder 呢？

官方 `build_encoder(args)` 同样会基于 Transformer Encoder Layer构造 CVAE encoder。

在 ACT configuration下也使用对应：

$$
d_{\text{model}},
d_{\text{ff}},
nheads
$$

配置。

因此训练时 CVAE Transformer Encoder 也具有宽 FFN stage。

---

## 69. 一个 ACT Policy Encoder Token 的完整局部路径

假设某个 visual token：

$$
x_i
$$

先经过 Self-Attention：

$$
a_i
=
\sum_j
\alpha_{ij}v_j
$$

Residual + Norm：

$$
h_i
=
LN(x_i+a_i)
$$

然后 FFN：

$$
u_i
=
ReLU(
h_iW_1+b_1
)
$$

其中：

$$
u_i\in\mathbb R^{3200}
$$

再：

$$
f_i=u_iW_2+b_2
$$

回：

$$
512
$$

最终：

$$
y_i=
LN(
h_i+f_i
)
$$

这就是一个 token 在一层中的：

> 通信 → 本地计算。

---

## 70. 一个 ACT Action Slot 的路径也一样

Decoder 某个 future slot：

先：

> Self-Attention with other action slots。

再：

> Cross-Attention read observation memory。

此时得到：

$$
h_i
$$

随后：

$$
512
\rightarrow
3200
\rightarrow
512
$$

FFN。

所以 FFN 在这里加工的是：

> 已经融合“其他未来 slots + 当前 observation”的 action representation。

---

## 71. 为什么 Encoder 和 Decoder 可以使用同一个 FFN 形式？

因为 FFN 并不关心：

> hidden vector 是来自 image token 还是 action slot。

它只操作：

$$
d_{\text{model}}
$$

维 representation。

所以同样：

$$
D\rightarrow D_{ff}\rightarrow D
$$

的抽象可以用在：

- Encoder；
- Decoder；
- text；
- vision；
- robotics。

---

## 72. FFN 能不能跨 Camera 融合信息？

直接：

> 不能。

某个 camera token和另一个 camera token之间的 interaction：

> 发生在 Self-Attention。

但 Attention 已经融合后的 token representation再进入 FFN，

所以 FFN可以：

> 加工已经包含多-camera context 的 feature。

因此说：

> “FFN 完全与多模态无关”

也不准确。

它不做直接 token mixing，

但可以加工 multimodal-contextualized representation。

---

## 73. FFN 会不会看到 Position Encoding？

取决于 position information 怎样进入 hidden representation。

原始 Transformer：

$$
x=e+p
$$

position已经进入 residual representation，

FFN自然间接接收到。

ACT/DETR-style 显式 `pos` 主要加在 Q/K，

但经过 Attention output和 residual 后，

当前 hidden representation可以包含 position-dependent interaction结果。

所以 FFN不需要一个独立：

> positional argument。

---

## 74. 为什么 FFN 不需要 Q/K/V？

因为它没有 memory retrieval问题。

Q/K/V 是为了：

> 判断不同 positions之间如何匹配和传递信息。

FFN只处理一个当前 vector：

$$
x_i
$$

不需要：

> “我要去哪个 token 查什么？”

所以普通 Linear/activation 足够。

---

## 75. FFN 和 Attention 的权重是否 Input-Dependent？

Attention aggregation weights：

$$
\alpha_{ij}(X)
$$

会随 input动态变化。

FFN参数：

$$
W_1,W_2
$$

是固定 learned parameters。

但 ReLU activation pattern：

$$
1[h_j>0]
$$

会随 input改变。

所以 FFN也具有：

> input-dependent effective computation path，

只是机制不同。

---

## 76. Attention 是动态 Routing，FFN 是固定参数 + 动态 Activation

可以粗略写：

#### Attention

$$
\text{input}
\rightarrow
\text{动态生成 token-token weights}
$$

#### FFN

$$
\text{input}
\rightarrow
\text{固定 learned matrices}
+
\text{input-dependent nonlinear gates}
$$

这说明二者都不是简单 static linear layer。

---

## 77. FFN 可以理解成“特征词典”吗？

作为直觉，可以有限度这样理解：

第一层：

$$
W_1
$$

的 columns 定义很多 learned directions / detectors。

输入：

$$
x
$$

与这些方向组合得到 hidden activations。

ReLU 决定哪些 features被激活。

第二层：

$$
W_2
$$

再把 activated features写回 residual representation。

这个视角有助于理解：

> 宽 FFN 为什么能存很多 feature transformations。

但“每一个 neuron 就是一个明确概念”：

> 并不保证。

---

## 78. 后续研究为什么会把 FFN 联系到 Key-Value Memory？

后来的研究提出一种解释：

> Transformer FFN 的第一层可能像大量 learned keys / pattern detectors，第二层像对应 values / output directions。

这是很有启发性的分析视角。

但它是：

> **Transformer 之后的 interpretability hypothesis / empirical analysis。**

不是 2017 原论文对 FFN 的定义。

所以 canonical 理论必须先建立在：

$$
Linear
\rightarrow
Activation
\rightarrow
Linear
$$

上。

---

## 79. 为什么不能把 FFN 当成另一个 Attention？

因为它没有：

$$
QK^\top
$$

没有：

$$
softmax
$$

也没有：

> 根据 sequence 中其他 token 动态生成读取权重。

FFN的 hidden units来自：

$$
xW_1+b_1
$$

它是 feature-space transformation，

不是 token-memory retrieval。

---

## 80. 为什么不能用一个更大的 Attention 替代 FFN？

这是一种 architecture research问题，

不是数学上“不可能”。

但标准 Transformer选择：

> token mixing 和 feature transformation 分成两个不同模块。

这带来清晰结构：

```text
跨 token:
Attention

token 内:
FFN
```

实践证明这一组合非常有效。

所以 canonical Transformer 不应把 FFN省略。

---

## 81. 为什么不能只堆 FFN、不用 Attention？

如果只有 Position-wise FFN：

$$
y_i=FFN(x_i)
$$

不同 token 永远不会直接交流。

无论堆多少层，

如果没有其他 token-mixing机制，

position $i$ 仍然只依赖：

$$
x_i
$$

自己。

所以 sequence dependency无法建立。

这说明 Attention不可替代。

---

## 82. 为什么不能只堆 Attention、不用 FFN？

从数学上当然可以构造纯 Attention 网络。

但标准 Transformer加入 FFN为每个 contextualized token提供额外的宽 nonlinear computation capacity。

经验上这成为 Transformer block的核心组成。

所以：

> Attention解决“交流”，FFN增加“局部计算深度和 feature capacity”。

---

## 83. 一个极简对照

假设两个 tokens：

$$
x_1,x_2
$$

只有 FFN：

$$
x_1'
=
FFN(x_1)
$$

$$
x_2'
=
FFN(x_2)
$$

它们永远不交流。

---

只有 Attention：

$$
x_i'
=
\sum_j\alpha_{ij}Vx_j
$$

可以交流，

但缺少标准 Transformer中额外的宽 per-token MLP transform。

---

两者：

```text
Attention
→ context exchange

FFN
→ nonlinear feature refinement
```

组合才形成标准 block。

---

## 84. 为什么 FFN 的宽度通常比 d_model 大？

这是一个很常见的 neural-network design：

> bottleneck/residual stream 保持适中的 model width，

局部计算模块暂时扩到更大的 hidden space。

优点包括：

- 更多 intermediate feature channels；
- 更强 nonlinear transformation capacity；
- 最后仍回到固定 residual width。

这类似很多网络里的：

> expansion → projection。

---

## 85. 为什么不让整个 Residual Stream 都保持 2048？

那会让：

- Attention Q/K/V；
- residual；
- memory；
- 所有层 hidden states；

全部变得更宽，

显著增加整体计算和显存。

FFN expansion让模型：

> 只在局部 MLP 内暂时使用宽表示，

再压回：

$$
d_{\text{model}}
$$

这是更经济的结构折中。

---

## 86. 为什么 ACT 可以把 FFN 扩到 3200，但 Residual Stream 仍 512？

因为中间 3200 只存在于：

> FFN内部。

Attention、Cross-Attention、Residual Memory等主要 representation仍：

$$
512
$$

维。

所以 ACT可以增加：

> per-token nonlinear computation width，

而不必把所有 attention heads 和 memory全部扩大到 3200。

---

## 87. FFN 输出会不会改变 Token 的“位置身份”？

不会改变 token count或索引。

position $i$ 的 FFN output：

> 仍属于 position $i$。

它只是修改：

$$
x_i
$$

的 feature vector。

所以：

$$
\boxed{
\text{FFN changes representation, not sequence topology}
}
$$

---

## 88. FFN 会不会改变 Action Slot 对应哪个未来时间？

不会。

ACT slot $i$ 经 FFN后仍是：

> slot $i$。

它的 temporal identity主要来自 query positional structure。

FFN只加工当前 slot hidden features。

---

## 89. Dropout 在 FFN 哪里？

原始 Transformer和 ACT-style code会在 FFN内部以及 residual branch使用 dropout。

ACT current code：

```python
src2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(src)
            )
        )
    )
```

所以概念上：

$$
Linear_1
\rightarrow
Activation
\rightarrow
Dropout
\rightarrow
Linear_2
$$

然后外面：

$$
Dropout
$$

再进入 residual。

---

## 90. 为什么训练时有 Dropout，推理没有？

Dropout是 regularization。

训练时：

> 随机关闭部分 intermediate activations / sublayer outputs。

推理：

```python
model.eval()
```

后关闭随机 dropout。

所以 FFN在 inference 中是 deterministic mapping。

---

## 91. FFN 有 BatchNorm 吗？

标准 Transformer FFN：

> 没有。

Layer normalization放在 Transformer block结构中。

ACT同样使用：

> LayerNorm。

FFN内部本身就是：

```text
Linear
Activation
Dropout
Linear
```

---

## 92. FFN 自己有 Residual 吗？

如果严格区分模块：

> `FFN(x)` 本身通常不包含 residual。

Transformer Layer在 FFN 外面做：

$$
x+
Dropout(
FFN(x)
)
$$

所以：

$$
\boxed{
FFN module
\neq
FFN sublayer with residual wrapper
}
$$

读代码时要看层级。

---

## 93. FFN 和 MLP 是不是同一个东西？

FFN在这里本质上就是：

> 一个两层 MLP。

但“Feed-Forward Network”是更广泛概念。

在 Transformer语境中：

> FFN 通常特指 position-wise MLP sub-layer。

所以：

$$
\boxed{
\text{Transformer FFN}
\approx
\text{per-token MLP}
}
$$

---

## 94. 为什么叫 Feed-Forward？

因为内部没有：

- recurrent loop；
- feedback state。

信息：

$$
x
\rightarrow
hidden
\rightarrow
output
$$

单向前传。

但整个 Transformer也基本是 feed-forward computation graph，

所以这个名字在 Transformer中更多是历史性的模块称呼：

> fully-connected MLP sub-layer。

---

## 95. FFN 是不是每次 Forward 都学习新参数？

不是。

参数：

$$
W_1,W_2,b_1,b_2
$$

训练过程中通过 gradient descent更新。

一次 inference forward中：

> 参数固定。

不同 input只改变：

- pre-activations；
- ReLU gate pattern；
- output values。

---

## 96. Gradient 怎样训练 FFN？

Loss：

$$
L
$$

通过后续网络回传到 FFN output：

$$
\frac{\partial L}{\partial y}
$$

再通过：

$$
W_2
$$

回到 activated hidden：

$$
r
$$

通过 ReLU derivative：

$$
\frac{dReLU(z)}{dz}
=
\begin{cases}
1,&z>0\\
0,&z<0
\end{cases}
$$

再回到：

$$
W_1
$$

和 input。

所以 training会学习：

> 哪些 hidden feature detectors和 output combinations能降低最终任务 loss。

---

## 97. ReLU 的“Dead Unit”问题

如果某个 unit长期：

$$
h_j<0
$$

那么 ReLU输出：

$$
0
$$

并且局部 gradient：

$$
0
$$

它可能变得难以重新激活。

这就是常说的：

> dying ReLU。

GELU等平滑 activation在某些现代架构中有不同 gradient behavior。

但原始 Transformer成功使用 ReLU，

所以不能说 ReLU“不适合 Transformer”。

---

## 98. 为什么 ACT 仍使用 ReLU？

ACT继承 DETR-style Transformer architecture，

论文/代码配置保留 ReLU FFN。

这说明：

> 对 ACT 的规模和任务，经典 ReLU Transformer FFN 已经足以工作。

是否换成 GELU/SwiGLU更好：

> 是一个需要实验验证的模型修改问题。

不能自动假设现代 LLM 的 activation替换在机器人 imitation learning中一定更优。

---

## 99. 如果想改 ACT FFN，可以改哪些东西？

研究/工程上可以改变：

- $d_{ff}$；
- ReLU → GELU；
- gated FFN；
- dropout；
- pre/post norm；
- layer count。

但每个改动都会改变：

- 参数量；
- FLOPs；
- optimization；
- possibly performance。

所以 canonical ACT page应先保持论文配置。

---

## 100. 为什么 3200 会直接影响模型大小？

每层 FFN weight约：

$$
2\times512\times3200
$$

如果 Encoder 4 层 + Decoder 7 层，

Policy Transformer中有：

$$
11
$$

个 FFN blocks。

仅这些 FFN的大 weight matrices总量粗略：

$$
11\times3,276,800
$$

约：

$$
36.0M
$$

weights。

这里还没算：

- CVAE Encoder；
- Attention；
- ResNet；
- embeddings；
- heads。

所以 FFN width会显著影响整个模型参数规模。

---

## 101. 这个 36M 是精确 ACT 总参数吗？

**不是。**

这里只是在做：

> Policy Transformer 11 个 FFN block 的 weight-only粗略加总。

它不包括：

- bias；
- CVAE Encoder FFNs；
- MHA；
- backbone；
- projections；
- norms；
- query embeddings。

所以不能把它称为：

> ACT total parameters。

它只是帮助建立量级直觉。

---

## 102. 为什么 Canonical Article 要讲参数量？

因为只有看到：

$$
FFN\approx3.28M
$$

per ACT block，

你才会真正意识到：

> FFN不是“Attention后面那个小 MLP”。

它是 Transformer capacity 的主要来源之一。

---

## 103. FFN 会不会“存知识”？

现代 LLM interpretability研究中有很多工作研究：

> FFN neurons / parameters是否存储 factual associations。

这是一个有价值的研究方向。

但 canonical FFN定义不能直接简化成：

> “FFN 就是知识库。”

更严谨：

> FFN是大规模 per-token nonlinear transform；后续研究发现其中某些参数/activations可能表现出类似 key-value memory和知识关联的性质。

理论定义和 interpretability观察要分开。

---

## 104. 对 ACT 也不能说“FFN 存机器人动作知识”

ACT FFN训练后当然会编码对任务有用的 parameters。

但不能把某个：

$$
W_1
$$

neuron直接因果解释成：

> “抓杯子神经元”。

除非做：

- activation analysis；
- intervention；
- ablation；

等实验。

所以文章只把：

> feature transformation capacity

作为可靠主线。

---

## 105. 为什么 Layer 1 FFN 和 Layer 4 FFN 可以不同？

因为它们处理的 representations处于不同深度。

Layer 1：

> token刚完成第一次 context exchange。

Layer 4：

> token已经经过多轮 contextualization。

所以每层拥有独立：

$$
W_1^{(l)},W_2^{(l)}
$$

允许不同深度学习不同 feature-processing rules。

---

## 106. 为什么所有 Layers 不共享同一个 FFN 能增加 Capacity？

如果共享：

$$
FFN^{(1)}
=
FFN^{(2)}
=\cdots
$$

每层都使用同一个 local transformation。

独立参数则允许：

$$
f_1,f_2,\ldots,f_L
$$

逐层形成不同 computation stages。

当然也有参数共享型 Transformer研究，

但不是原始 Transformer / ACT canonical structure。

---

## 107. FFN 的 Hidden Units 有位置编码吗？

FFN hidden units：

$$
2048
$$

或：

$$
3200
$$

不是 sequence positions。

它们是：

> feature dimensions。

不要把：

```text
3200 hidden units
```

误解成：

> 3200 个 tokens。

这是完全不同的 axis。

---

## 108. Token Axis 和 Feature Axis 再区分一次

ACT Policy Encoder：

$$
[B,1202,512]
$$

#### 1202

是：

> token / sequence positions。

#### 512

是：

> hidden feature dimensions。

FFN：

$$
[B,1202,512]
\rightarrow
[B,1202,3200]
$$

变化的是：

$$
\boxed{
\text{feature axis}
}
$$

不是：

$$
\boxed{
\text{token axis}
}
$$

---

## 109. Attention 恰好相反吗？

Attention 的 unique operation主要建立：

$$
1202\times1202
$$

token-token relation。

所以它显式作用于：

> sequence axis。

但内部也有 feature projections。

因此“恰好相反”只是一种简化。

更准确：

> FFN不做 token mixing；Attention会做 token mixing。

---

## 110. 这也是为什么 MLP-Mixer 叫 Mixer

后来一些 architecture显式把：

- token mixing；
- channel mixing；

拆成两个 MLP。

Transformer则使用：

- Attention 做动态 token mixing；
- FFN 做 channel/feature processing。

这是架构思想上的一个有趣联系。

但不是理解原始 Transformer必须依赖的背景。

---

## 111. 为什么 FFN 可以高度并行？

每个 token独立使用同一 MLP。

所以所有：

$$
B\times N
$$

vectors可以打包成大矩阵乘法。

没有：

$$
token_i
$$

等待：

$$
token_{i-1}
$$

的 sequential dependency。

因此 FFN非常适合 GPU/TPU dense matrix computation。

---

## 112. 为什么 FFN 常是硬件友好的？

核心运算：

$$
XW_1
$$

和：

$$
HW_2
$$

都是 dense GEMM。

现代 accelerator对这种大矩阵乘高度优化。

所以虽然 FFN计算量大，

它通常具有良好的硬件利用率。

---

## 113. Attention 和 FFN 哪个更难优化硬件？

这取决于：

- sequence length；
- batch；
- model width；
- kernel implementation。

Attention涉及：

- QK matrix；
- Softmax；
- memory traffic；
- quadratic intermediates。

FFN主要是：

> 大 dense GEMMs。

所以它们的硬件瓶颈模式不同。

不要简单用 FLOP 数就推断 wall-clock latency。

---

## 114. ACT 里的 FFN 对 1202 Visual/State Tokens 是不是同一个 Matrix Multiplication？

实现上：

> 是 batch化的大矩阵运算。

概念上：

> 1202 个 token分别应用同一 MLP。

这两个描述并不矛盾。

---

## 115. 为什么这和卷积权重共享思想很像？

CNN：

> 同一个 kernel在不同 spatial locations共享。

Transformer FFN：

> 同一个 MLP在不同 sequence positions共享。

共同思想：

$$
\boxed{
\text{shared local computation across positions}
}
$$

区别是：

- CNN kernel可读取邻域；
- Transformer position-wise FFN只读取当前位置。

---

## 116. 原论文为什么专门强调“same across positions, different across layers”？

因为它准确界定了参数共享边界：

```text
同一层:
token 1, token 2, token 3
→ 同一 FFN

不同层:
layer 1 FFN
layer 2 FFN
→ 不同参数
```

这是阅读 Transformer代码时非常重要的结构知识。

---

## 117. PyTorch 代码怎样体现 Position-Wise？

ACT Encoder：

```python
self.linear1 =
    nn.Linear(
        d_model,
        dim_feedforward
    )

self.linear2 =
    nn.Linear(
        dim_feedforward,
        d_model
    )
```

输入可能是：

$$
[N,B,D]
$$

`nn.Linear` 默认作用于：

> 最后一个 dimension。

所以：

$$
D
\rightarrow
D_{ff}
$$

自动对前面的：

$$
N,B
$$

所有位置广播/批处理。

这就是 position-wise computation。

---

## 118. ACT Encoder Post-LN FFN 代码

当前官方：

```python
src2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(src)
            )
        )
    )

src =
    src +
    self.dropout2(src2)

src =
    self.norm2(src)
```

翻成数学：

$$
u=W_1x+b_1
$$

$$
r=ReLU(u)
$$

$$
r'=Dropout(r)
$$

$$
f=W_2r'+b_2
$$

$$
y=
LN(
x+Dropout(f)
)
$$

---

## 119. ACT Decoder FFN 完全同类

当前代码：

```python
tgt2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(tgt)
            )
        )
    )

tgt =
    tgt +
    self.dropout3(tgt2)

tgt =
    self.norm3(tgt)
```

所以 Encoder/Decoder 的 FFN mathematical form：

> 一致。

只是输入 representation语义不同。

---

## 120. Pre-LN 时 FFN 公式怎样变化？

Pre-LN：

先：

$$
\tilde x=LN(x)
$$

再：

$$
f=FFN(\tilde x)
$$

最后：

$$
\boxed{
y=x+Dropout(f)
}
$$

所以 FFN本身公式没变。

改变的是：

> LayerNorm 相对于 sub-layer/residual 的位置。

---

## 121. Post-LN ACT 默认

ACT current default：

```python
normalize_before=False
```

所以 FFN处于：

$$
\boxed{
x
\rightarrow
FFN
\rightarrow
Residual Add
\rightarrow
LayerNorm
}
$$

与 2017 Transformer Post-LN结构一致。

---

## 122. FFN 为什么不负责 Normalization？

因为 FFN 和 LayerNorm是不同模块。

FFN：

> learned nonlinear feature transformation。

LayerNorm：

> input-dependent normalization + learned scale/shift。

把它们分开有助于分析每个模块功能。

---

## 123. FFN 为什么不改变 Batch/Sequence Length？

因为：

$$
Linear
$$

只变最后 feature dimension。

ReLU：

> element-wise。

第二个 Linear：

> 再改 feature dimension。

所以：

$$
B,N
$$

全程保留。

这也是 residual compatibility的必要条件。

---

## 124. 一个 Token 的 FFN 可以怎样逐行理解？

ACT：

```python
u = linear1(x)
```

把：

$$
512
$$

个当前 features重新组合成：

$$
3200
$$

个 candidate features。

---

```python
u = relu(u)
```

根据当前 token content：

> 保留正 activation，抑制负 activation。

---

```python
u = dropout(u)
```

训练时随机 regularize。

---

```python
f = linear2(u)
```

把 activated 3200-D hidden pattern重新写回：

$$
512
$$

维 residual space。

---

```python
x = norm(x + dropout(f))
```

将这个：

> feature update

写回 token representation。

---

## 125. 为什么第二层 Linear 很重要？

如果只：

$$
512\rightarrow3200
$$

那 Transformer hidden width就永久改变。

第二层：

$$
3200\rightarrow512
$$

不仅恢复 shape，

更重要：

> 学习把不同 hidden features如何组合成 residual update directions。

所以：

$$
W_1
$$

和：

$$
W_2
$$

扮演不同角色。

---

## 126. W₁ 可以怎样直觉理解？

有限度地说：

$$
W_1
$$

把 residual representation投影到一个更宽：

> feature-detection / intermediate computation space。

每一列：

$$
w_j
$$

定义某种 linear direction。

---

## 127. W₂ 可以怎样直觉理解？

$$
W_2
$$

把 activation pattern：

$$
r
$$

重新投影到：

$$
d_{\text{model}}
$$

residual space。

所以：

> 哪些 hidden features被激活，以及它们怎样写回 output dimensions，

共同决定 FFN update。

---

## 128. 为什么 Bias 也重要？

第一层：

$$
xW_1+b_1
$$

bias会改变：

> ReLU gate threshold。

例如没有 bias：

$$
w^\top x>0
$$

才激活。

有 bias：

$$
w^\top x+b>0
$$

decision boundary可以平移。

第二层 bias：

$$
b_2
$$

也允许输出有 learned offset。

---

## 129. ReLU Decision Boundary 是什么？

某个 hidden neuron：

$$
h_j=w_j^\top x+b_j
$$

其开关边界：

$$
w_j^\top x+b_j=0
$$

这是 hidden space 中一个 hyperplane。

一侧：

$$
h_j>0
$$

unit active。

另一侧：

$$
h_j<0
$$

unit output 0。

大量 neurons形成很多分区，

让 FFN成为 piecewise-linear function。

---

## 130. 为什么 Width 增加会增加 Piecewise Capacity？

更多 hidden units意味着更多：

$$
w_j^\top x+b_j
$$

gating boundaries。

因此函数可以形成更丰富的 input-dependent regions和 output combinations。

但：

> width更大不保证实际任务表现一定更好。

仍受：

- data；
- optimization；
- regularization；
- overfitting；
- compute budget；

影响。

---

## 131. 为什么 ACT 只有 10 分钟 Demonstrations还敢用 3200-D FFN？

ACT的成功来自整个系统：

- pretrained ResNet backbone / image representation；
- strong architecture priors；
- action chunking；
- CVAE；
- regularization；
- task structure。

不能从：

$$
d_{ff}=3200
$$

单独推断：

> “小数据绝不会过拟合。”

模型容量和有效泛化之间关系需要实验。

---

## 132. FFN Width 和 Chunk Size 是不同 Axis

ACT：

$$
d_{ff}=3200
$$

是：

> feature width。

chunk：

$$
k=100
$$

是：

> output sequence length。

不要混淆：

```text
3200
≠
3200 action slots
```

FFN不会把：

$$
100
$$

个 action positions扩成：

$$
3200
$$

个 positions。

---

## 133. 一个完整的 ACT Decoder FFN Shape

输入：

$$
[B,100,512]
$$

flatten概念上可以看成：

$$
(B\times100)
$$

个 512-D vectors。

第一 Linear：

$$
[B,100,3200]
$$

activation：

$$
[B,100,3200]
$$

第二 Linear：

$$
[B,100,512]
$$

所以：

> 100 个 action slots始终是 100 个。

---

## 134. 一个完整的 ACT Policy Encoder FFN Shape

输入：

$$
[B,1202,512]
$$

第一 Linear：

$$
[B,1202,3200]
$$

第二 Linear：

$$
[B,1202,512]
$$

所以：

> 1202 memory positions始终是 1202。

---

## 135. 为什么 FFN 不需要 Padding Mask？

FFN逐 token处理。

对于 padding token，

它仍然可以计算一个 output。

真正重要的是：

> Padding token不要通过 Attention影响真实 tokens。

所以 padding mask主要用于 Attention。

后续如果 padding representation自己发生变化通常无所谓，

只要：

- loss不使用它；
- Attention持续正确屏蔽。

具体实现可能还有其他 masking策略。

---

## 136. Causal Mask 也和 FFN 无关

Causal Mask限制：

> token-to-token visibility。

FFN不看其他 positions。

所以 FFN没有：

$$
T\times T
$$

causal mask。

如果输入 token representation已经 obey causality，

position-wise FFN不会创建 future leakage。

---

## 137. 为什么 FFN 不会破坏 Causality？

假设：

$$
h_t
$$

只依赖：

$$
x_{\le t}
$$

那么：

$$
FFN(h_t)
$$

只是：

> $h_t$ 的函数。

它不会凭空访问：

$$
x_{>t}
$$

所以：

$$
FFN(h_t)
$$

仍只依赖：

$$
x_{\le t}
$$

因此 causal property保留。

---

## 138. 为什么 FFN 不会破坏 Permutation Equivariance？

如果没有 positional information，

同一个 FFN独立作用每一 row。

输入 permutation：

$$
PX
$$

得到：

$$
FFN(PX)
=
PFFN(X)
$$

因为只是重新排列相同函数的输入 rows。

所以 position-wise FFN本身也是：

> permutation equivariant。

真正位置结构需要 position information / masks提供。

---

## 139. 这个性质为什么重要？

它进一步说明：

> FFN本身不会告诉 Transformer顺序。

它只是对每个 position统一处理。

所以：

- Self-Attention content-only；
- Position-wise FFN；

组合起来如果都没有 PE/mask，

整个网络仍保留强 permutation symmetry。

---

## 140. FFN 能不能处理不同长度 Sequence？

可以。

因为它根本不依赖：

$$
N
$$

固定值。

同一个：

$$
Linear(512,3200)
$$

可以应用到：

- N=10；
- N=100；
- N=1202；

只要最后 feature dimension：

$$
512
$$

一致。

这是 parameter sharing带来的自然能力。

---

## 141. 为什么 Transformer 的 FFN 对文本和图像都能复用？

因为它只关心：

> hidden vector。

只要 upstream把：

- word；
- image patch；
- joint；
- action slot；

都表示成：

$$
D
$$

维，

同样的 FFN architecture就可以使用。

这就是 Transformer模块化能力的重要来源。

---

## 142. 但文本模型和 ACT 的 FFN 参数会共享吗？

当然不会。

这里只是：

> architecture form相同。

不同模型：

> 不同 trained parameters。

就像两个 CNN都用 3×3 convolution，

不代表 convolution weights一样。

---

## 143. 为什么 FFN 是每层不同参数，而不是全模型一个共享 MLP？

这样每一层都能形成：

> 不同 depth-specific computation。

例如：

$$
x^{(1)}
$$

和：

$$
x^{(10)}
$$

representations分布和语义可能完全不同。

独立 FFN能适应这些不同层级。

---

## 144. FFN 和 Output Head 不是一回事

ACT FFN：

$$
512\rightarrow3200\rightarrow512
$$

是 Transformer内部 computation。

Action Head：

$$
512\rightarrow14
$$

是最终 task output projection。

所以：

$$
\boxed{
FFN
\neq
Action\ Head
}
$$

---

## 145. Language Transformer 也一样

Transformer FFN：

$$
512\rightarrow2048\rightarrow512
$$

最后 vocabulary head：

$$
512\rightarrow|Vocab|
$$

是不同模块。

一个负责：

> internal feature computation。

一个负责：

> task output。

---

## 146. 为什么 FFN 最终不直接输出动作/词？

因为每一层 FFN只是：

> representation refinement。

后面还有：

- 更多 Transformer layers；
- task head。

让中间 block保持：

$$
d_{\text{model}}
$$

统一，有利于深层堆叠和 residual。

---

## 147. 为什么“MLP”这个词有时让人低估它？

因为我们常把 MLP理解成：

> 初学机器学习里最简单的小网络。

但一个：

$$
512\rightarrow3200\rightarrow512
$$

的 MLP拥有数百万参数，

又在每层重复。

在大型 Transformer中，

FFN往往是极重要的 compute / parameter component。

所以：

> 简单结构 ≠ 小作用。

---

## 148. 为什么 Dense FFN 容易扩展成 Mixture-of-Experts？

Dense FFN对每个 token：

> 都使用同一组大 MLP parameters。

MoE Transformer则尝试：

> 为不同 tokens动态路由到不同 FFN experts。

这可以在不让每个 token都使用全部参数的情况下：

> 增加总模型容量。

所以 MoE 的一个重要切入点正是：

> 替换 Transformer FFN sub-layer。

这进一步说明 FFN是模型容量核心位置。

---

## 149. MoE 和 Multi-Head Attention 不一样

Multi-Head：

> 同一 Attention layer中的多个 parallel representation heads，通常全部计算。

MoE：

> 多个 FFN experts，常通过 router选择少数 experts。

所以：

$$
\boxed{
\text{attention head}
\neq
\text{FFN expert}
}
$$

以后可以单独学习 MoE。

---

## 150. 现代 LLM 为什么常把 FFN 叫 MLP Block？

因为本质就是：

> token-wise MLP。

不同代码可能叫：

- FFN；
- MLP；
- FeedForward；
- MLPBlock；
- SwiGLU block。

读代码时应该看：

> 它是不是在 hidden dimension 上做 expansion/gating/projection。

而不是只看类名。

---

## 151. 一个最小 PyTorch FFN

原始 Transformer风格：

```python
class FFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
```

输入：

$$
[B,N,512]
$$

输出：

$$
[B,N,512]
$$

---

## 152. 为什么这段代码自动逐 Position 运行？

PyTorch：

```python
nn.Linear(in_features, out_features)
```

只把 input的：

> 最后一个 dimension

当 features。

所以：

$$
[B,N,512]
$$

进入：

```python
Linear(512,2048)
```

自动输出：

$$
[B,N,2048]
$$

前面：

$$
B,N
$$

只是 batch-like dimensions。

不需要显式 loop。

---

## 153. 一个 ACT 风格最小 FFN

```python
class ACTFFN(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_ff=3200,
        dropout=0.1
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

这就是官方 Transformer Layer中 FFN核心部分的简化版。

Residual / Norm在外层处理。

---

## 154. 常见误解一：FFN 是为了让 Token 互相交流

**错误。**

标准 Position-wise FFN不做 token mixing。

---

## 155. 常见误解二：FFN 只是一个 Linear Layer

**错误。**

原始：

$$
Linear
\rightarrow
ReLU
\rightarrow
Linear
$$

---

## 156. 常见误解三：两层 Linear 天然比一层强

**错误。**

如果没有非线性，

两层可以合并成一层 affine transform。

---

## 157. 常见误解四：FFN 的唯一作用是“加非线性”

**不完整。**

Attention本身已有 Softmax非线性。

FFN更完整的作用是：

> 提供宽维、逐 token 的 nonlinear feature transformation capacity。

---

## 158. 常见误解五：2048 是 Token 数量

**错误。**

它是：

$$
d_{ff}
$$

feature width。

token count：

$$
N
$$

不变。

---

## 159. 常见误解六：3200 是 ACT Action Chunk 长度

**错误。**

ACT：

$$
d_{ff}=3200
$$

chunk：

$$
k=100
$$

是两个完全不同 axis。

---

## 160. 常见误解七：每个 Token 有自己独立的 FFN 参数

**错误。**

同一 layer所有 positions共享同一 FFN。

---

## 161. 常见误解八：所有 Transformer Layers 又共享同一个 FFN

**错误。**

原始 Transformer不同 layers使用不同 FFN参数。

---

## 162. 常见误解九：FFN 参数很少，所以可以忽略

**错误。**

原始 FFN约：

$$
2.10M
$$

参数 per block。

ACT FFN约：

$$
3.28M
$$

参数 per block。

---

## 163. 常见误解十：Attention 的参数一定比 FFN 多

**不一定，而且经典配置恰恰相反。**

在 $d=512$ 下，

MHA Q/K/V/O weight约：

$$
1.05M
$$

ACT FFN weights约：

$$
3.28M
$$

---

## 164. 常见误解十一：FFN 会把 1202 Tokens 压成一个 Vector

**错误。**

$$
[B,1202,512]
\rightarrow
[B,1202,3200]
\rightarrow
[B,1202,512]
$$

token count始终 1202。

---

## 165. 常见误解十二：FFN 会破坏 Causal Mask

**不会。**

它不跨 positions访问数据。

---

## 166. 常见误解十三：FFN 自己需要 Causal Mask

**不需要。**

Mask是 token-to-token connectivity问题。

---

## 167. 常见误解十四：FFN 自己决定 Position

**错误。**

同一 FFN应用所有 positions。

Position information来自其他 mechanism。

---

## 168. 常见误解十五：Transformer 必须使用 ReLU FFN

**错误。**

ReLU是原始 Transformer选择。

现代模型常用 GELU / gated FFNs等。

---

## 169. 常见误解十六：ACT 使用 GELU，因为现代 Transformer 都用 GELU

**错误。**

ACT official architecture默认 activation：

$$
\boxed{
ReLU
}
$$

---

## 170. 常见误解十七：ACT 的 dim_feedforward 默认 2048，所以论文也是 2048

**错误。**

通用 class default是 2048。

ACT官方训练命令和论文 Table III：

$$
\boxed{
3200
}
$$

---

## 171. 常见误解十八：FFN 输出就是最终机器人动作

**错误。**

FFN仍输出：

$$
512
$$

维 hidden representation。

最终 action head才：

$$
512\rightarrow14
$$

---

## 172. 常见误解十九：FFN 扩维后 Residual 直接把 3200-D 加回 512-D

**错误。**

第二 Linear先：

$$
3200\rightarrow512
$$

然后才能 residual：

$$
512+512
$$

---

## 173. 常见误解二十：Attention 已经能看全局，所以 FFN 没意义

**错误。**

“能访问全局信息”和“有足够的 nonlinear local computation”不是同一件事。

---

## 174. 用两个轴记住整个 Transformer

输入：

$$
X\in\mathbb R^{N\times D}
$$

---

### Attention

重点处理：

$$
\boxed{
N\text{ 这个 token axis}
}
$$

建立：

$$
N\times N
$$

dynamic relations。

---

### FFN

保持：

$$
N
$$

不变，

暂时把：

$$
D
$$

扩成：

$$
D_{ff}
$$

再压回：

$$
D
$$

重点处理：

$$
\boxed{
\text{feature axis}
}
$$

---

## 175. 一张最有用的图

```text
Token 1 ─┐
Token 2 ─┼─→ Attention ─→ Token 1'
Token 3 ─┤                Token 2'
Token 4 ─┘                Token 3'
                          Token 4'
                              │
                              ▼

                     same FFN applied
                     independently:

Token 1' ─→ 512 → 2048/3200 → 512
Token 2' ─→ 512 → 2048/3200 → 512
Token 3' ─→ 512 → 2048/3200 → 512
Token 4' ─→ 512 → 2048/3200 → 512
```

所以：

> Attention 横向交流。

> FFN 纵向加工每个 token自己的 features。

---

## 176. 一句话真正理解 FFN

> **Transformer 的 Position-wise Feed-Forward Network 是一个对每个 token 独立、但在同一 layer 的所有 positions 共享参数的两层 MLP：它先把 $d_{\text{model}}$ 维 contextual representation投影到更宽的 $d_{\text{ff}}$ feature space，通过非线性激活形成 input-dependent intermediate feature pattern，再投影回 $d_{\text{model}}$ residual space，从而在 Attention完成跨 token 信息交换之后，为每个 token提供强大的局部 nonlinear computation。**

---

## 177. 一句话理解 Attention + FFN

> **Attention 决定当前 token 应该从其他 positions 读取哪些信息，而 FFN 决定这个已经获得 context 的 token 接下来怎样在自己的 feature space 中重新组合、门控和加工这些信息；前者提供动态通信，后者提供宽维的 per-token computation。**

---

## 178. 一句话连接 ACT

> **ACT 的 Transformer 将每个 512-D visual/joint/latent/action representation在 Attention后送入 $512\rightarrow3200\rightarrow512$ 的 ReLU FFN；这不会改变 1202 个 observation-memory tokens 或 $k$ 个 action slots 的数量，而是在每一个 contextualized token内部提供约 3.28M 参数规模的 nonlinear feature transformation，使 observation encoding和action decoding不仅能够“互相读信息”，还能够对读到的信息进行丰富的局部计算。**

---

## 179. 下一篇：Residual Connection

现在我们已经知道：

```text
Attention
→ 信息交换

FFN
→ feature computation
```

但 Transformer 并不是直接：

$$
x\rightarrow Attention(x)\rightarrow FFN(\cdot)
$$

每个 sub-layer 都有一条非常关键的：

$$
\boxed{
x+F(x)
}
$$

这就是：

> **Residual Connection。**

下一篇：

> **[Residual Connection：为什么网络要把输入直接加回来？](./residual-connection.md)**

会从 ResNet 原论文开始解释：

- 为什么深层网络不是“层越多训练误差一定越低”；
- degradation problem是什么；
- 为什么让网络学：
  $$
  F(x)=H(x)-x
  $$
  可能比直接学 $H(x)$ 更容易；
- identity path怎样帮助 gradient传播；
- 为什么 residual不是“防止信息丢失”这么简单；
- $x+F(x)$ 为什么要求 shape一致；
- Transformer中的 Attention residual / FFN residual分别在做什么；
- Pre-LN为什么会让 residual stream更加直接；
- ACT 的 Encoder/Decoder每一层里 residual具体加在哪里。

---

### Primary Source：Transformer FFN

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin.

**Attention Is All You Need.**  
NeurIPS 2017.

- arXiv: https://arxiv.org/abs/1706.03762
- HTML: https://arxiv.org/html/1706.03762

Section 3.3 明确定义：

$$
\boxed{
FFN(x)
=
\max(
0,
xW_1+b_1
)
W_2+b_2
}
$$

并说明：

- Encoder / Decoder每层都有 FFN；
- 每个 position独立且相同地应用；
- 同一 layer不同 positions共享 Linear transformations；
- 不同 layers参数不同；
- 也可以描述成两个 kernel size 1 convolutions；
- input/output：
  $$
  d_{\text{model}}=512
  $$
- hidden：
  $$
  d_{\text{ff}}=2048
  $$

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
\#encoder\ layers=4
}
$$

$$
\boxed{
\#decoder\ layers=7
}
$$

$$
\boxed{
feedforward\ dimension=3200
}
$$

$$
\boxed{
hidden\ dimension=512
}
$$

$$
\boxed{
\#heads=8
}
$$

$$
\boxed{
dropout=0.1
}
$$

因此 ACT Transformer FFN 典型 shape：

$$
\boxed{
512
\rightarrow
3200
\rightarrow
512
}
$$

---

### Official ACT Implementation

Repository:

https://github.com/tonyzhaozh/act

`detr/models/transformer.py` 的 Encoder Layer：

```python
self.linear1 =
    nn.Linear(
        d_model,
        dim_feedforward
    )

self.dropout =
    nn.Dropout(dropout)

self.linear2 =
    nn.Linear(
        dim_feedforward,
        d_model
    )
```

Post-LN forward：

```python
src2 =
    self.linear2(
        self.dropout(
            self.activation(
                self.linear1(src)
            )
        )
    )

src =
    src +
    self.dropout2(src2)

src =
    self.norm2(src)
```

Decoder 同样具有：

```python
linear1
activation
dropout
linear2
```

FFN。

---

#### Activation

当前 `_get_activation_fn` 支持：

```text
relu
gelu
glu
```

Transformer constructor默认：

```python
activation="relu"
```

所以 canonical ACT configuration使用：

$$
\boxed{
ReLU
}
$$

---

#### Config vs Class Default

通用 Transformer class 的：

```python
dim_feedforward=2048
```

只是默认参数。

真正 ACT builder 使用：

```python
dim_feedforward=args.dim_feedforward
```

官方 README训练示例：

```text
--hidden_dim 512
--dim_feedforward 3200
```

与论文 Table III 一致。

因此：

$$
\boxed{
\text{ACT canonical }d_{ff}=3200
}
$$

---

### Modern Activation Background

#### GELU

Dan Hendrycks, Kevin Gimpel.

**Gaussian Error Linear Units (GELUs).**

- arXiv: https://arxiv.org/abs/1606.08415

常见定义：

$$
\boxed{
GELU(x)=x\Phi(x)
}
$$

GELU 是后续 Transformer模型中常见的平滑 activation。

它不是原始 2017 Transformer 的 activation。

---

#### Gated FFN / SwiGLU

Noam Shazeer.

**GLU Variants Improve Transformer.**

- arXiv: https://arxiv.org/abs/2002.05202

现代 Transformer常使用 gated FFN variants，例如：

$$
\boxed{
Swish(xW_g)
\odot
(xW_v)
}
$$

再通过 output projection回到 model dimension。

这属于 Transformer FFN 的后续演化，

不应和原始：

$$
Linear\rightarrow ReLU\rightarrow Linear
$$

混为一谈。

---

### 本文知识连接

#### 前置

- [Transformer](./transformer.md)
- [Transformer Encoder](./transformer-encoder.md)
- [Transformer Decoder](./transformer-decoder.md)
- [Self-Attention](./self-attention.md)

#### 神经网络基础

- [Linear Layer](./linear-layer.md)
- [MLP](./mlp.md)
- [ReLU](./relu.md)
- GELU
- GLU
- SwiGLU

#### Transformer Components

- [Residual Connection](./residual-connection.md)
- [Layer Normalization](./layer-normalization.md)
- [Dropout](./dropout.md)
- [Multi-Head Attention](./multi-head-attention.md)

#### 数学

- Affine Transformation
- Piecewise Linear Function
- Hyperplane

#### 后续扩展

- Mixture of Experts

#### Robot Learning

- [ACT Architecture](../robot-learning/act/architecture.md)
- [ACT Training](../robot-learning/act/training.md)
- [ACT Complete Data Flow](../robot-learning/act/complete-data-flow.md)

#### 下一步

- [Residual Connection](./residual-connection.md)
