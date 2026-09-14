# Source Map

模型、算法与架构页面以原论文为主要依据；涉及 ACT 具体行为时以官方 released implementation 作为实现依据。下面列出正文中实际使用的主要原始来源。

## Primary Papers

- **ACT / ALOHA** — Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (2023): https://arxiv.org/abs/2304.13705
- **Transformer** — Vaswani et al., *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762
- **DETR / learned queries** — Carion et al., *End-to-End Object Detection with Transformers* (2020): https://arxiv.org/abs/2005.12872
- **BERT / CLS token precedent** — Devlin et al., *BERT* (2018): https://arxiv.org/abs/1810.04805
- **VAE** — Kingma & Welling, *Auto-Encoding Variational Bayes*: https://arxiv.org/abs/1312.6114
- **CVAE** — Sohn, Lee & Yan, *Learning Structured Output Representation using Deep Conditional Generative Models* (2015): https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
- **Behavior Cloning / distribution shift / DAgger** — Ross, Gordon & Bagnell (2011): https://proceedings.mlr.press/v15/ross11a.html
- **ResNet** — He et al., *Deep Residual Learning for Image Recognition*: https://arxiv.org/abs/1512.03385
- **CNN historical reference** — LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (1998).

## Official Implementations

- **ACT official repository**: https://github.com/tonyzhaozh/act
- ACT policy / loss: https://github.com/tonyzhaozh/act/blob/main/policy.py
- ACT DETR-VAE architecture: https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py
- ACT rollout / temporal aggregation: https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py

## Grounding Policy

- 论文负责定义研究方法、架构与实验事实。
- 官方代码负责确认 tensor flow、loss、fixed constants、padding、optimizer 与 rollout 等实现细节。
- 如果 paper description 与 released implementation 不完全一致，在 `Paper and Released Implementation` 中分别记录，不进行静默统一。
- 数学基础页面采用标准数学定义与推导，不为了贴近某个应用页面而改变 canonical 定义。
