# XunBlog v6 Canonical Path Migration

本文件列出当前 v6 canonical routes 相对于更早网站结构发生过的 ownership-driven 路径迁移。v6 本轮内容重写不额外改变这些 canonical routes。

> 建议 Codex：以 `CONTENT_INDEX.md` 为最终 canonical truth；本表用于迁移旧链接。

| Old / Previous Path | v6 Canonical Path | Reason |
|---|---|---|
| `/deep-learning/transformer/attention/` | `/deep-learning/attention/` | Attention predates Transformer |
| `/deep-learning/transformer/attention/qkv/` | `/deep-learning/attention/qkv/` | Q/K/V retrieval lineage predates Transformer |
| `/deep-learning/transformer/attention/self-attention/` | `/deep-learning/attention/self-attention/` | Self-attention predates Transformer |
| `/deep-learning/transformer/attention/cross-attention/` | `/deep-learning/attention/cross-attention/` | Encoder–decoder attention predates Transformer |
| `/deep-learning/transformer/attention/scaled-dot-product-attention/` | `/deep-learning/transformer/scaled-dot-product-attention/` | Transformer-specific formulation |
| `/deep-learning/transformer/attention/multi-head-attention/` | `/deep-learning/transformer/multi-head-attention/` | Transformer-specific formulation |
| `/deep-learning/transformer/positional-encoding/` | `/deep-learning/sequence-modeling/positional-encoding/` | Generic positional representation predates Transformer |
| `/deep-learning/transformer/causal-mask/` | `/deep-learning/sequence-modeling/causal-mask/` | Generic autoregressive masking predates Transformer |
| `/deep-learning/transformer/feed-forward-network/` | `/deep-learning/transformer/position-wise-feed-forward-network/` | Old title was too generic; page is Transformer position-wise FFN |
| `/deep-learning/transformer/cls-token/` | `/deep-learning/bert/cls-token/` | `[CLS]` is BERT-specific |
| `/deep-learning/transformer/learnable-query-embedding/` | `/deep-learning/detr/object-query/` | ACT query-slot lineage is DETR Object Query |
| `/deep-learning/core/residual-connection/` | `/deep-learning/cnn/resnet/residual-connection/` | Residual-learning formulation belongs to ResNet lineage |
| `/deep-learning/cnn/convolution/` | `/mathematics/analysis/convolution/` | Convolution is a mathematical operation |
| `/generative-models/autoencoder/` | `/deep-learning/representation-learning/autoencoder/` | Autoencoder is broader representation-learning architecture |
| `/generative-models/latent-variable/` | `/mathematics/probability/latent-variable/` | Latent variable is probabilistic modeling concept |
| `/generative-models/variational-inference/` | `/mathematics/probability/variational-inference/` | VI is general approximate inference |
| `/generative-models/evidence-lower-bound/` | `/mathematics/probability/variational-inference/evidence-lower-bound/` | ELBO belongs to VI |
| `/generative-models/variational-autoencoder/reparameterization-trick/` | `/mathematics/probability/variational-inference/reparameterization-trick/` | Reparameterization is general pathwise-gradient / VI technique |
| `/generative-models/variational-autoencoder/conditional-variational-autoencoder/` | `/generative-models/conditional-variational-autoencoder/` | CVAE is an independently proposed model |
| `/generative-models/variational-autoencoder/posterior-collapse/` | `/generative-models/posterior-collapse/` | Posterior Collapse is an independent failure mode |

## New canonical pages

These pages did not exist in the earlier compact corpus, or are now promoted to explicit canonical identities:

- `/deep-learning/core/multilayer-perceptron/`
- `/deep-learning/bert/`
- `/deep-learning/detr/`
- `/deep-learning/transformer/kv-cache/`
- `/mathematics/linear-algebra/vector-norm/`
- `/mathematics/linear-algebra/linear-transformation/`
- `/mathematics/probability/bayes-theorem/`
- `/mathematics/probability/covariance/`
- `/mathematics/information-theory/entropy/`
- `/mathematics/information-theory/cross-entropy/`
- `/deep-learning/core/activation-function/`
- `/robot-learning/dagger/`

## Model-specific paths intentionally retained

The following stay model-scoped:

- `/robot-learning/act/action-chunking/`
- `/robot-learning/act/temporal-ensemble/`
- `/robot-learning/act/cvae-in-act/`
- `/robot-learning/pi0/action-expert/`
- `/robot-learning/pi0/blockwise-causal-attention-mask/`
- `/robot-learning/pi0/flow-matching-in-pi0/`

Do not “promote” them merely because another model later uses a similar mechanism.
