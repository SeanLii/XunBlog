# XunBlog Content Design Standard v6

本文件是 XunBlog 内容创作与维护的强制标准。它同时约束 **知识归属（ownership）**、**正文完整性（completeness）**、**正文组织（exposition）** 与 **写作正式性（formality）**。

---

# 1. 最终目标

XunBlog 是一套结构化、可递归导航的 AI 技术知识库。每个页面都应该像一本高质量现代技术教材中的独立章节：

- 技术与数学准确；
- 结构清楚；
- 语言直接、易读；
- 不依赖故事化或聊天式教学；
- 不因为服务某条学习路径而压缩自身知识；
- 不因为追求“内容丰富”而人为复杂化本来简单的概念。

核心原则是：

> **知识难度由知识本身决定；语言和结构只负责降低不必要的理解成本。**

## 1.1 v6 的三个不可替代质量轴

所有页面必须同时满足 **逻辑严谨、数学性、解释性**。三者不能互相替代。

### 逻辑严谨

- 定义、假设、结论与适用范围必须分清；
- 每个结论都应能追溯到前面的定义、公式、机制或来源；
- 不能把“通常如此”写成无条件成立；
- 极限、等价、最优、唯一、收敛等强结论必须写出必要条件；
- paper-level description 与 released implementation 不得混为同一个事实。

### 数学性

数学性不是公式数量，而是**该用数学时必须用足够准确的数学建立关系**。

- 数学对象要给正式定义；
- 模型 objective、概率关系、tensor transformation、gradient、ODE / optimization 等核心关系应给出必要公式；
- 公式中的变量、条件与索引方向必须明确；
- 不为了显得技术化给纯结构性内容硬塞公式；
- 不允许只有直觉而跳过决定机制本身的数学。

### 解释性

解释性不是把内容改写成故事，而是让正式知识形成闭环。重要定义、公式或机制出现后，读者应该知道：

1. 它在描述什么对象；
2. 为什么此处需要它；
3. 每个关键项如何影响结果；
4. 它与前后知识如何连接；
5. 它成立到什么边界。

最终目标不是“公式多”或“说得简单”，而是：

> **定义准确 → 数学建立关系 → 解释说明意义 → 条件与边界收口。**

---

# 2. 两种独立必须同时成立

一个 topic 拥有独立 URL，不代表它已经真正独立。

必须同时满足：

1. **Canonical Ownership 独立**：它被放在真实的知识来源与学术谱系中；
2. **Canonical Exposition 独立**：正文首先服务于标题知识本身，而不是服务于 ACT、Transformer、π0 或其他阅读路径。

例如 Dot Product 不能写成：

```text
公式
→ QK^T
→ sqrt(d_k)
→ Attention
```

而应先完整建立：

```text
algebraic definition
→ geometric meaning
→ norm / angle
→ orthogonality
→ projection / similarity
→ matrix relation
→ applications
```

Attention 只出现在后续 application / connection 中。

---

# 3. Canonical Ownership / Provenance

知识归属首先看 **这个具体知识以什么身份进入文献**，而不是看后来哪些模型使用了它。

## 3.1 模型内部提出的具体机制

如果一个具体 mechanism / component 在某个模型或算法中被正式提出、命名并定义为该模型的一部分，它保留在该模型下面。

例如：

```text
Transformer
├── Scaled Dot-Product Attention
├── Multi-Head Attention
└── Position-Wise Feed-Forward Network
```

```text
BERT
└── CLS Token
```

```text
DETR
└── Object Query
```

```text
ResNet
└── Residual Connection
```

```text
ACT
├── Action Chunking
└── Temporal Ensemble
```

```text
π0
├── Action Expert
└── Blockwise Causal Attention Mask
```

后来被其他模型复用，不会自动改变 canonical home。

## 3.2 在目标模型之前已经独立存在的知识

如果概念在该模型之前已经作为独立理论、算法或机制存在，则不能因为后来的模型大量使用它就改变 ownership。

例如：

- Attention、Self-Attention 早于 Transformer；
- Layer Normalization 早于 Transformer；
- positional representations 早于 Transformer；
- Flow Matching 早于 π0；
- CVAE 是独立提出的 conditional generative model；
- DAgger 是独立 imitation-learning algorithm。

## 3.3 来源说明在正文中只占必要位置

Provenance 用于确定 Tree，不用于把学习文章写成知识史。

学习正文中若来源会影响概念边界，通常一句话说明即可。详细历史判定、论文谱系和 ownership 论证放在 `CANONICAL_INDEPENDENCE_AUDIT.md` 与 `SOURCE_MAP.md`。

不要为了证明归属而插入长篇历史叙述。

---

# 4. Navigation Graph 不迁移 Ownership

一个知识被多个模型使用，只增加 graph edges：

```text
π0 → Action Chunking
ACT → Transformer
ACT → Object Query
Transformer → Residual Connection
```

它不会因为使用次数增加而搬家。

> Tree 决定 home；Graph 决定跳转。

---

# 5. 独立 Topic 判定

一个概念是否值得拥有自己的 canonical page，检查：

- 是否有稳定、明确、脱离当前模型仍成立的定义；
- 是否有值得独立建立的机制、数学性质、算法流程或学术身份；
- 是否本身就是公认模型 / 算法 / 数学对象；
- 独立后是否能减少其他页面承担不属于自己的教学责任。

不要因为出现一个专业词就机械拆页。

同样，一个模型内部 component 也可以拥有完整独立页面；“独立页面”不等于“离开父模型”。

---

# 6. 内容完整性只由知识本身决定

**禁止使用字符数、阅读时间、页数、section 数量或公式数量作为内容质量代理。**

判断一篇文章是否完整，唯一依据是：

> **这个 topic 本身重要的定义、结构、机制、数学、性质、假设、边界与连接是否已经被正确建立。**

因此：

- VAE、Variational Inference、Entropy、BERT、DETR 等知识容量较大的主题，不得压缩成“够理解下一个模型”的摘要；
- CLS Token、Object Query、Causal Mask、Vector Norm 等范围较窄的主题，如果定义、机制与边界已经完整，就不应为了统一篇幅加入无关内容；
- “内容丰富”不等于“文章越长越好”；
- “概念简单”也不等于只给一句定义而跳过必要性质。

每一页必须被真正阅读和判断，不能通过长度规则批量判定。

---

# 7. 正式教材风格

XunBlog 的目标不是科普文章、博客随笔、课堂聊天或论文翻译，而是：

> **正式、清楚、可理解的技术教材。**

正文应直接进入知识对象本身，并按知识结构展开。

优先写法：

```text
定义 / 对象
→ 结构或输入输出
→ mechanism
→ mathematics
→ properties / assumptions
→ limitations
→ connections / applications
```

避免把文章组织成：

```text
先讲故事
→ 提一个问题
→ 再举一个类比
→ 纠正常见误解
→ 很晚才给正式定义
```

## 7.1 可读性不依赖故事化

可以使用直觉和例子，但它们只是 supporting devices。

- 例子用于验证、具体化已经建立的知识；
- 直觉用于解释困难的数学或机制；
- 类比只能短暂辅助，不得成为正式定义；
- 不用生活故事承担文章主线。

## 7.2 避免课堂式口吻

普通 canonical page 中尽量避免：

- “如果只记住一句话”；
- “先不要管……”；
- “想象一下……”；
- “可以先把它理解成……”；
- “真正要记住的是……”；
- 连续的“为什么 / 那怎么办 / 接下来呢”；
- 以纠错作为文章主体。

可以直接写清正式关系，而不是模拟老师与学生对话。

## 7.3 正式不等于晦涩

不要为了显得正式而使用不必要的复杂词。

技术术语必须保留，但第一次出现时用简单、准确的语言解释。

---

# 8. Model / Architecture 页面

总览页首先建立 whole-system mental model：

```text
是什么
→ 解决什么问题
→ 输入 / 输出
→ 最小完整数据流
→ 核心设计
→ 主要组件
→ mathematics / training / inference
→ limitations / sources
```

读者在进入局部公式前，必须已经知道自己正在看什么系统。

Paper grounding 负责事实正确，不负责决定文章叙述顺序。

---

# 9. Mechanism / Component 页面

例如 Attention、QKV、Temporal Ensemble、Object Query：

```text
formal object / role
→ input / output
→ complete computation or data flow
→ mathematics
→ properties / assumptions
→ boundary
→ connections / applications
```

直觉可以辅助解释，但不替代正式定义。

不能只写到“够理解父模型”为止。

---

# 10. Mathematics 页面

数学页先建立数学对象，而不是 AI 使用场景：

```text
formal definition
→ interpretation
→ core properties
→ derivations / identities when important
→ examples
→ relationships to neighboring concepts
→ applications
```

数学页不能以“为了看懂 Transformer / VAE”起笔，也不能因为主要应用在 AI 中就跳过数学对象自身的重要性质。

---

# 11. Algorithms / Learning Paradigms 页面

算法必须拥有自己的：

- problem setting；
- inputs / data assumptions；
- objective；
- procedure；
- theoretical or statistical motivation（如果属于该算法核心）；
- limitations / costs；
- relationship to neighboring algorithms。

例如 DAgger 不能只在 Behavior Cloning 的一个 section 中带过。

---

# 12. Model-Specific 页面

`CVAE in ACT`、`ACT Training`、`Flow Matching in π0` 等可以明确依赖父模型。

它们的职责是解释：

- 父模型如何使用已有知识；或
- 父模型自己的结构 / training / inference / data flow。

它们不能重新拥有通用 CVAE、Flow Matching、Transformer 等理论。

---

# 13. 问题型标题

问题可以在知识自然发展到某一步时出现，但不作为默认 H2/H3 组织方式。

优先：

```text
Posterior Collapse
Distribution Shift
KV Cache Memory Cost
```

而不是连续：

```text
为什么会……
那怎么办……
为什么还要……
```

例外是页面本身就是 model-specific question，例如：

`为什么 ACT 推理时令 z = 0？`

---

# 14. 数学写作

每个重要公式必须让读者知道：

1. variables 是什么；
2. 公式描述什么关系；
3. 为什么此处需要它；
4. 它和前文什么概念连接；
5. 它对模型 / 算法 / 数学对象意味着什么。

禁止“公式 → 显然 → 下一个公式”。

直觉与正式定义必须分开。

---

# 15. Paper / Code Grounding

- 模型、算法、架构：原论文是主要依据；
- released implementation detail：官方代码是主要依据；
- paper / code 冲突：明确区分 `Paper description` 与 `Released implementation`；
- 博客、README、二手文章不能替代原始依据；
- 不为了叙述整齐隐藏冲突；
- 不把未经来源支持的解释写成论文事实。

---

# 16. Application 的位置

通用 canonical page 必须先完成自身知识，再进入应用。

删除 ACT / Transformer / π0 / VAE 等 application 段落后，通用页面仍然应该能够独立成立。

模型应用不应反过来决定数学或通用机制页面讲多少。

---

# 17. 页面内部不要讨论网站维护规则

Ownership 规则用于组织知识库，不应成为正文里的产品说明。

正文可以写：

> Attention 早于 Transformer。

不要写：

> XunBlog 因此把 Attention 的 canonical home 放到……

正文始终保持技术教材身份。

---

# 18. Final Self-Test

每篇 canonical page 发布前逐篇检查：

1. 标题是否就是知识名称；
2. ownership 是否有来源依据；
3. breadcrumb 是否反映真实 Tree；
4. 首段是否直接进入知识对象；
5. 对于模型，是否先建立清楚的整体系统、输入与输出；
6. 对于数学 / mechanism / algorithm，是否完整建立其自身核心结构；
7. 是否遗漏了这个 topic 本身重要的理论、性质、假设或限制；
8. 是否因为服务 ACT / Transformer / π0 等应用而提前停止；
9. 是否为了显得丰富而给简单概念加入无关复杂度；
10. 是否存在故事化、科普式或课堂聊天式组织；
11. examples / intuition 是否只承担辅助作用；
12. 是否把其他独立知识抢进来重复讲；
13. paper / code discrepancy 是否准确披露；
14. provenance 是否只保留理解 ownership 所需的最小说明；
15. 是否有断链、重复 canonical、多个 H1、title/H1 mismatch、损坏公式或非法控制字符；
16. 是否为了“新版”而重写已经合格的内容；
17. 逻辑是否存在条件缺失、概念偷换、过强结论或索引方向不明；
18. 该数学的地方是否给出了足够正式的数学，而不是只剩语言直觉；
19. 每个核心公式 / 机制是否解释了变量、作用、意义与边界，而不是“公式 → 下一个公式”。

最后一条同样重要：

> **不为了重新而重新。只有内容、结构、正式性、归属或准确性存在实质问题时才修改。**
