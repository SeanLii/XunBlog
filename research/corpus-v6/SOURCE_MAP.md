# XunBlog v6 Source Map

本文件记录 v6 中需要历史来源或原始实现依据的主要页面。  
数学基础页以标准定义为主，不强行为每个基础定理指定“唯一原论文”。

原则：

- **theory / model / algorithm**：优先原论文；
- **released behavior / fixed implementation detail**：优先官方代码；
- **historical ownership**：优先能证明该机制在目标模型之前已存在，或确实在该模型中被正式提出的原始文献；
- **paper / code discrepancy**：两个版本同时保留。

---

# Attention lineage

## Attention / Cross-Attention

- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate*. 2014.  
  https://arxiv.org/abs/1409.0473

该工作在 Transformer 之前已经让 decoder 根据当前状态对 encoder representations 做动态 soft alignment，是 modern encoder–decoder attention 的关键来源之一。

## Query / Key / Value lineage

- Miller et al. *Key-Value Memory Networks for Directly Reading Documents*. EMNLP 2016.  
  https://arxiv.org/abs/1606.03126  
  https://aclanthology.org/D16-1147/

该工作明确区分 memory addressing 的 keys 与用于读取的 values，并由 input/query 去访问 memory，因此 Q/K/V 式 retrieval mental model 不能被当成 Transformer 才出现的思想。

## Self-Attention before Transformer

- Cheng, Dong, Lapata. *Long Short-Term Memory-Networks for Machine Reading*. EMNLP 2016.  
  https://arxiv.org/abs/1601.06733
- Lin et al. *A Structured Self-attentive Sentence Embedding*. ICLR 2017.  
  https://arxiv.org/abs/1703.03130

这些工作证明 sequence 内部 attention / self-attentive mechanism 在 Transformer 之前已经存在。

---

# Transformer

- Vaswani et al. *Attention Is All You Need*. 2017.  
  https://arxiv.org/abs/1706.03762

用于：

- Transformer overall architecture；
- Scaled Dot-Product Attention；
- Multi-Head Attention；
- Position-Wise Feed-Forward Network；
- Transformer Encoder；
- Transformer Decoder；
- sinusoidal positional encoding 的 Transformer-specific implementation；
- decoder causal masking 的 Transformer-specific use。

## Positional representation before Transformer

- Gehring et al. *Convolutional Sequence to Sequence Learning*. ICML 2017.  
  https://proceedings.mlr.press/v70/gehring17a.html

用于说明 learned positional representations 在 Transformer 之前已经用于 sequence modeling，因此 generic `Positional Encoding` 不归 Transformer 私有。

## Layer Normalization

- Ba, Kiros, Hinton. *Layer Normalization*. 2016.  
  https://arxiv.org/abs/1607.06450

LayerNorm 在 Transformer 之前独立提出。

## KV Cache

KV Cache 页以 Transformer autoregressive inference 的计算复用为主，不把它描述成 2017 Transformer paper 的原始贡献。π0 中的 prefix cache 细节以官方 openpi 实现为依据：

- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py

---

# BERT

- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. 2018.  
  https://arxiv.org/abs/1810.04805

用于：

- BERT overview；
- `[CLS]` / `[SEP]` 等 BERT-specific input design；
- masked language modeling；
- encoder-only bidirectional pretraining。

`CLS Token` 因此放在 BERT 下，而不是 Transformer 下。

---

# DETR

- Carion et al. *End-to-End Object Detection with Transformers*. 2020.  
  https://arxiv.org/abs/2005.12872
- Official repository:  
  https://github.com/facebookresearch/detr

用于：

- DETR overview；
- learned object queries；
- direct set prediction；
- bipartite matching；
- ACT action-query lineage。

`Object Query` 属于 DETR，而不是原始 Transformer 的标准组件。

v6 precision note：原始 DETR official implementation 中 `tgt = torch.zeros_like(query_embed)`，learned object queries 以 `query_pos=query_embed` 传入 decoder。因此正文区分：object-query embedding、decoder content state 与 attention projection 中的 \(Q\)。

---

# CNN / ResNet

## Convolutional Neural Networks

- LeCun et al. *Gradient-Based Learning Applied to Document Recognition*. 1998.
- Krizhevsky, Sutskever, Hinton. *ImageNet Classification with Deep Convolutional Neural Networks*. 2012.
- Dumoulin, Visin. *A Guide to Convolution Arithmetic for Deep Learning*. 2016.
  https://arxiv.org/abs/1603.07285

用于 CNN architecture、convolution arithmetic、stride / padding / output-size 与 spatial processing 的技术依据。

## ResNet / Residual Connection

- He et al. *Deep Residual Learning for Image Recognition*. 2015/2016.  
  https://arxiv.org/abs/1512.03385

ResNet 的关键贡献是 residual learning formulation：

\[
y=x+F(x).
\]

v6 不声称 ResNet 发明了历史上一切形式的 skip connection；`Residual Connection` 页面讨论的是 ResNet residual-learning lineage，以及后来被 Transformer 等架构复用的形式。

---

# Representation Learning

## Autoencoder

- Hinton & Salakhutdinov. *Reducing the Dimensionality of Data with Neural Networks*. Science, 2006.  
  https://www.science.org/doi/10.1126/science.1127647

Autoencoder 属于 representation learning，不作为 VAE 的附属 background page。

---

# Variational Inference / VAE family

## Variational Inference

- Blei, Kucukelbir, McAuliffe. *Variational Inference: A Review for Statisticians*. 2017.  
  https://arxiv.org/abs/1601.00670

## VAE / AEVB

- Kingma & Welling. *Auto-Encoding Variational Bayes*. 2013/2014.  
  https://arxiv.org/abs/1312.6114
- Rezende, Mohamed, Wierstra. *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. ICML 2014.  
  https://proceedings.mlr.press/v32/rezende14.html

## Reparameterization Trick

现代 deep variational inference 中的 reparameterized / pathwise gradient 以 Kingma & Welling 和 Rezende et al. 为关键来源；更广泛的 pathwise-gradient ideas 早于 VAE。因此 v6 把它放在：

```text
Probability
└── Variational Inference
    └── Reparameterization Trick
```

而不是 VAE 私有子页。

补充：
- Jankowiak & Obermeyer. *Pathwise Derivatives Beyond the Reparameterization Trick*. ICML 2018.  
  https://proceedings.mlr.press/v80/jankowiak18a.html

## Evidence Lower Bound

ELBO 属于 variational inference 的一般目标，不定义为 VAE 专属的 `reconstruction + KL` 固定模板。

## Conditional Variational Autoencoder

- Sohn, Lee, Yan. *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS 2015.  
  https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html

CVAE 是单独论文提出的 conditional generative model，因此在 Generative Models 下独立存在；`CVAE in ACT` 只解释 ACT 怎样使用它。

## Posterior Collapse

- He et al. *Lagging Inference Networks and Posterior Collapse in Variational Autoencoders*. ICLR 2019.  
  https://arxiv.org/abs/1901.05534

Posterior Collapse 作为 variational latent-variable modeling 的独立 failure mode 存在，不作为 VAE architecture 的“私有 component”。

## Flow Matching

- Lipman et al. *Flow Matching for Generative Modeling*. 2022 / ICLR 2023.  
  https://arxiv.org/abs/2210.02747

Flow Matching 明显早于 π0，因此保持独立 Generative Models 页面；π0 只拥有 `Flow Matching in π0`。

v6 precision note：正文显式区分 sample-level trajectory velocity、conditional vector field 与 marginal vector field，并用原论文 Gaussian OT-style conditional path 说明 \(\sigma_{min}>0\) 时的 path / velocity。

---

# Imitation Learning

## Distribution shift / sequential imitation

- Ross & Bagnell. *Efficient Reductions for Imitation Learning*. AISTATS 2010.  
  https://proceedings.mlr.press/v9/ross10a.html

## DAgger

- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011.  
  https://proceedings.mlr.press/v15/ross11a.html

DAgger 是独立算法，因此拥有自己的 canonical page，而不是 Behavior Cloning 内的一小节。

---

# Vision-Language / VLA / Cross-Embodiment

## Vision-Language Model

- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. 2021.  
  https://arxiv.org/abs/2103.00020
- Alayrac et al. *Flamingo: a Visual Language Model for Few-Shot Learning*. 2022.  
  https://arxiv.org/abs/2204.14198

VLM 页面按 model family 写，不用单一 architecture 定义全部 VLM。

## VLA

- Brohan et al. / Zitkovich et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. 2023.  
  https://arxiv.org/abs/2307.15818
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. 2024.  
  https://arxiv.org/abs/2406.09246

## Cross-Embodiment Learning

- Open X-Embodiment Collaboration et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*. 2023/2024.  
  https://arxiv.org/abs/2310.08864

---

# ACT

## Primary paper

- Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. 2023.  
  https://arxiv.org/abs/2304.13705

## Official implementation

- https://github.com/tonyzhaozh/act
- policy / model implementation:  
  https://github.com/tonyzhaozh/act/blob/main/policy.py  
  https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
- evaluation / temporal aggregation:  
  https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py

## Ownership notes

### Action Chunking

ACT paper explicitly says its design is **inspired by action chunking, a neuroscience concept**. Therefore v6 does not claim ACT invented all meanings of action chunking.

The page `ACT / Action Chunking` is scoped to ACT's robot-policy formulation:

\[
\pi_\theta(a_{t:t+k-1}\mid o_t),
\]

where a policy predicts a future action sequence to reduce effective horizon.

### Temporal Ensemble

The specific overlapping-chunk temporal aggregation used by ACT remains under ACT.

v6 precision note：ACT paper 明确说明 \(w_0\) 是 oldest action prediction 的权重；official evaluation code 以 query time 从旧到新保留当前时刻的 populated predictions，并使用 `exp(-0.01 * arange(...))`。正文因此直接写明时间方向，不再保留模糊表述。

### Existing components reused by ACT

ACT links outward instead of re-owning:

- ResNet → CNN / ResNet
- Transformer → Deep Learning / Transformer
- CLS Token → BERT
- Object Query → DETR
- CVAE → Generative Models / CVAE

## Paper / released-code discrepancies retained

The ACT discrepancy page keeps distinctions such as:

- reconstruction-loss wording / implementation；
- optimizer details；
- latent defaults；
- released code behavior that differs from paper-level description.

---

# π0

## Primary paper

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*.  
  https://www.pi.website/download/pi0.pdf
- RSS 2025 paper page:  
  https://roboticsconference.org/2025/program/papers/10/

## Official implementation

- https://github.com/Physical-Intelligence/openpi
- π0 JAX model:  
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
- π0 PyTorch model:  
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/pi0_pytorch.py

## Ownership

π0 owns its own model-specific designs:

- Action Expert
- Blockwise Causal Attention Mask **in π0**
- π0 Architecture
- π0 Training / Inference / Complete Data Flow

π0 does not own:

- VLM
- VLA
- Flow Matching
- ODE
- Euler Method
- KV Cache
- Action Chunking as an ACT-scoped robot policy mechanism

These are linked through the graph.

## Paper / code difference

Official `openpi` code explicitly notes that its flow-time convention is opposite to the π0 paper:

```text
paper:  noise → data with increasing τ
openpi: t=1 noise → t=0 data
```

v6 keeps both conventions visible instead of silently normalizing them.

The implementation also fills a prefix KV cache once and reuses it during the repeated action-flow integration steps.

---

# Mathematics

Mathematics pages are not grounded in AI papers merely because AI later uses them.

Current independent math tree includes:

- Linear Algebra: Vector, Vector Norm, Matrix, Linear Transformation, Dot Product
- Analysis: Convolution
- Probability: Random Variable, Distribution, Conditional Probability, Bayes' Theorem, Expectation, Variance, Covariance, Normal / Multivariate Normal, Latent Variable, Variational Inference
- Information Theory: Entropy, Cross-Entropy, KL Divergence
- Calculus: Ordinary Differential Equation
- Numerical Methods: Euler Method

AI models appear only as later applications / connections.
