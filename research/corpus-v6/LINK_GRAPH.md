# XunBlog v6 Navigation Graph

Tree 决定归属；本表记录页面之间实际存在的 prerequisite / related / body-link 导航边。

Total nodes: **82**  
Total unique edges: **441**

| Source | Relation | Target |
|---|---|---|
| [Attention](/deep-learning/attention/) | body-link | [Cross-Attention](/deep-learning/attention/cross-attention/) |
| [Attention](/deep-learning/attention/) | body-link | [Query / Key / Value](/deep-learning/attention/qkv/) |
| [Attention](/deep-learning/attention/) | body-link | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Attention](/deep-learning/attention/) | body-link | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [Attention](/deep-learning/attention/) | body-link | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [Attention](/deep-learning/attention/) | body-link | [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) |
| [Attention](/deep-learning/attention/) | body-link | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Attention](/deep-learning/attention/) | prerequisites | [Softmax](/deep-learning/core/softmax/) |
| [Attention](/deep-learning/attention/) | prerequisites | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Attention](/deep-learning/attention/) | related | [Transformer](/deep-learning/transformer/) |
| [Cross-Attention](/deep-learning/attention/cross-attention/) | prerequisites | [Attention](/deep-learning/attention/) |
| [Cross-Attention](/deep-learning/attention/cross-attention/) | prerequisites | [Query / Key / Value](/deep-learning/attention/qkv/) |
| [Cross-Attention](/deep-learning/attention/cross-attention/) | related | [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) |
| [Query / Key / Value](/deep-learning/attention/qkv/) | body-link | [Object Query](/deep-learning/detr/object-query/) |
| [Query / Key / Value](/deep-learning/attention/qkv/) | prerequisites | [Attention](/deep-learning/attention/) |
| [Query / Key / Value](/deep-learning/attention/qkv/) | related | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Self-Attention](/deep-learning/attention/self-attention/) | body-link | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [Self-Attention](/deep-learning/attention/self-attention/) | body-link | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [Self-Attention](/deep-learning/attention/self-attention/) | prerequisites | [Attention](/deep-learning/attention/) |
| [Self-Attention](/deep-learning/attention/self-attention/) | prerequisites | [Query / Key / Value](/deep-learning/attention/qkv/) |
| [Self-Attention](/deep-learning/attention/self-attention/) | related | [Transformer](/deep-learning/transformer/) |
| [BERT](/deep-learning/bert/) | body-link | [CLS Token](/deep-learning/bert/cls-token/) |
| [BERT](/deep-learning/bert/) | prerequisites | [Self-Attention](/deep-learning/attention/self-attention/) |
| [BERT](/deep-learning/bert/) | prerequisites | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [BERT](/deep-learning/bert/) | prerequisites | [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) |
| [BERT](/deep-learning/bert/) | related | [CLS Token](/deep-learning/bert/cls-token/) |
| [CLS Token](/deep-learning/bert/cls-token/) | prerequisites | [BERT](/deep-learning/bert/) |
| [CLS Token](/deep-learning/bert/cls-token/) | related | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) | body-link | [ResNet](/deep-learning/cnn/resnet/) |
| [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) | body-link | [Activation Function](/deep-learning/core/activation-function/) |
| [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) | body-link | [Convolution](/mathematics/analysis/convolution/) |
| [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) | prerequisites | [Activation Function](/deep-learning/core/activation-function/) |
| [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) | prerequisites | [Convolution](/mathematics/analysis/convolution/) |
| [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) | related | [ResNet](/deep-learning/cnn/resnet/) |
| [ResNet](/deep-learning/cnn/resnet/) | body-link | [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) |
| [ResNet](/deep-learning/cnn/resnet/) | prerequisites | [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) |
| [ResNet](/deep-learning/cnn/resnet/) | related | [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) |
| [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) | prerequisites | [ResNet](/deep-learning/cnn/resnet/) |
| [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) | related | [Transformer](/deep-learning/transformer/) |
| [Activation Function](/deep-learning/core/activation-function/) | body-link | [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) |
| [Activation Function](/deep-learning/core/activation-function/) | body-link | [Softmax](/deep-learning/core/softmax/) |
| [Activation Function](/deep-learning/core/activation-function/) | body-link | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Activation Function](/deep-learning/core/activation-function/) | prerequisites | [Linear Layer](/deep-learning/core/linear-layer/) |
| [Activation Function](/deep-learning/core/activation-function/) | related | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Embedding](/deep-learning/core/embedding/) | prerequisites | [Matrix](/mathematics/linear-algebra/matrix/) |
| [Embedding](/deep-learning/core/embedding/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Embedding](/deep-learning/core/embedding/) | related | [Linear Layer](/deep-learning/core/linear-layer/) |
| [Embedding](/deep-learning/core/embedding/) | related | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [Layer Normalization](/deep-learning/core/layer-normalization/) | prerequisites | [Expectation](/mathematics/probability/expectation/) |
| [Layer Normalization](/deep-learning/core/layer-normalization/) | prerequisites | [Variance](/mathematics/probability/variance/) |
| [Layer Normalization](/deep-learning/core/layer-normalization/) | related | [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) |
| [Layer Normalization](/deep-learning/core/layer-normalization/) | related | [Transformer](/deep-learning/transformer/) |
| [Linear Layer](/deep-learning/core/linear-layer/) | body-link | [Activation Function](/deep-learning/core/activation-function/) |
| [Linear Layer](/deep-learning/core/linear-layer/) | body-link | [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) |
| [Linear Layer](/deep-learning/core/linear-layer/) | prerequisites | [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) |
| [Linear Layer](/deep-learning/core/linear-layer/) | prerequisites | [Matrix](/mathematics/linear-algebra/matrix/) |
| [Linear Layer](/deep-learning/core/linear-layer/) | related | [Activation Function](/deep-learning/core/activation-function/) |
| [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) | body-link | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) | prerequisites | [Activation Function](/deep-learning/core/activation-function/) |
| [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) | prerequisites | [Linear Layer](/deep-learning/core/linear-layer/) |
| [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) | related | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Softmax](/deep-learning/core/softmax/) | body-link | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Softmax](/deep-learning/core/softmax/) | body-link | [Cross-Entropy](/mathematics/information-theory/cross-entropy/) |
| [Softmax](/deep-learning/core/softmax/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Softmax](/deep-learning/core/softmax/) | related | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Softmax](/deep-learning/core/softmax/) | related | [Cross-Entropy](/mathematics/information-theory/cross-entropy/) |
| [DETR](/deep-learning/detr/) | body-link | [Object Query](/deep-learning/detr/object-query/) |
| [DETR](/deep-learning/detr/) | prerequisites | [ResNet](/deep-learning/cnn/resnet/) |
| [DETR](/deep-learning/detr/) | prerequisites | [Transformer](/deep-learning/transformer/) |
| [DETR](/deep-learning/detr/) | related | [Object Query](/deep-learning/detr/object-query/) |
| [DETR](/deep-learning/detr/) | related | [Architecture](/robot-learning/act/architecture/) |
| [Object Query](/deep-learning/detr/object-query/) | prerequisites | [Cross-Attention](/deep-learning/attention/cross-attention/) |
| [Object Query](/deep-learning/detr/object-query/) | prerequisites | [DETR](/deep-learning/detr/) |
| [Object Query](/deep-learning/detr/object-query/) | related | [Architecture](/robot-learning/act/architecture/) |
| [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) | body-link | [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) |
| [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) | prerequisites | [Embedding](/deep-learning/core/embedding/) |
| [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) | prerequisites | [Transformer](/deep-learning/transformer/) |
| [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) | related | [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) |
| [Autoencoder](/deep-learning/representation-learning/autoencoder/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Autoencoder](/deep-learning/representation-learning/autoencoder/) | prerequisites | [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) |
| [Autoencoder](/deep-learning/representation-learning/autoencoder/) | related | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) | prerequisites | [Attention](/deep-learning/attention/) |
| [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) | related | [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) |
| [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) | prerequisites | [Embedding](/deep-learning/core/embedding/) |
| [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) | related | [Transformer](/deep-learning/transformer/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Query / Key / Value](/deep-learning/attention/qkv/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Layer Normalization](/deep-learning/core/layer-normalization/) |
| [Transformer](/deep-learning/transformer/) | body-link | [DETR](/deep-learning/detr/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) |
| [Transformer](/deep-learning/transformer/) | body-link | [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) |
| [Transformer](/deep-learning/transformer/) | prerequisites | [Attention](/deep-learning/attention/) |
| [Transformer](/deep-learning/transformer/) | prerequisites | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [Transformer](/deep-learning/transformer/) | related | [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) |
| [Transformer](/deep-learning/transformer/) | related | [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) |
| [Transformer](/deep-learning/transformer/) | related | [Architecture](/robot-learning/act/architecture/) |
| [KV Cache](/deep-learning/transformer/kv-cache/) | body-link | [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) |
| [KV Cache](/deep-learning/transformer/kv-cache/) | body-link | [Inference](/robot-learning/pi0/inference/) |
| [KV Cache](/deep-learning/transformer/kv-cache/) | prerequisites | [Query / Key / Value](/deep-learning/attention/qkv/) |
| [KV Cache](/deep-learning/transformer/kv-cache/) | prerequisites | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [KV Cache](/deep-learning/transformer/kv-cache/) | related | [Inference](/robot-learning/pi0/inference/) |
| [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) | prerequisites | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) | related | [Cross-Attention](/deep-learning/attention/cross-attention/) |
| [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) | related | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) | body-link | [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) |
| [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) | prerequisites | [Multilayer Perceptron](/deep-learning/core/multilayer-perceptron/) |
| [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) | related | [Transformer](/deep-learning/transformer/) |
| [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) | body-link | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) | prerequisites | [Query / Key / Value](/deep-learning/attention/qkv/) |
| [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) | prerequisites | [Softmax](/deep-learning/core/softmax/) |
| [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) | prerequisites | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) | related | [Multi-Head Attention](/deep-learning/transformer/multi-head-attention/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | body-link | [DETR](/deep-learning/detr/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | body-link | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | body-link | [ACT](/robot-learning/act/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | prerequisites | [Cross-Attention](/deep-learning/attention/cross-attention/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | prerequisites | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | prerequisites | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) | related | [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | body-link | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | body-link | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | prerequisites | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | prerequisites | [Residual Connection](/deep-learning/cnn/resnet/residual-connection/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | prerequisites | [Layer Normalization](/deep-learning/core/layer-normalization/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | prerequisites | [Position-Wise Feed-Forward Network](/deep-learning/transformer/position-wise-feed-forward-network/) |
| [Transformer Encoder](/deep-learning/transformer/transformer-encoder/) | related | [Transformer Decoder](/deep-learning/transformer/transformer-decoder/) |
| [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) | body-link | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) | prerequisites | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) | related | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [Flow Matching](/generative-models/flow-matching/) | body-link | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Flow Matching](/generative-models/flow-matching/) | body-link | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [Flow Matching](/generative-models/flow-matching/) | prerequisites | [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) |
| [Flow Matching](/generative-models/flow-matching/) | prerequisites | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Flow Matching](/generative-models/flow-matching/) | related | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Flow Matching](/generative-models/flow-matching/) | related | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [Posterior Collapse](/generative-models/posterior-collapse/) | prerequisites | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | body-link | [Posterior Collapse](/generative-models/posterior-collapse/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | body-link | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | body-link | [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | prerequisites | [Autoencoder](/deep-learning/representation-learning/autoencoder/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | prerequisites | [Latent Variable](/mathematics/probability/latent-variable/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | prerequisites | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | related | [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | related | [Posterior Collapse](/generative-models/posterior-collapse/) |
| [Variational Autoencoder](/generative-models/variational-autoencoder/) | related | [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) |
| [Convolution](/mathematics/analysis/convolution/) | body-link | [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) |
| [Convolution](/mathematics/analysis/convolution/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Convolution](/mathematics/analysis/convolution/) | related | [Convolutional Neural Network](/deep-learning/cnn/convolutional-neural-network/) |
| [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) | body-link | [Flow Matching](/generative-models/flow-matching/) |
| [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) | body-link | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) | related | [Flow Matching](/generative-models/flow-matching/) |
| [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) | related | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Cross-Entropy](/mathematics/information-theory/cross-entropy/) | body-link | [Softmax](/deep-learning/core/softmax/) |
| [Cross-Entropy](/mathematics/information-theory/cross-entropy/) | body-link | [Entropy](/mathematics/information-theory/entropy/) |
| [Cross-Entropy](/mathematics/information-theory/cross-entropy/) | body-link | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [Cross-Entropy](/mathematics/information-theory/cross-entropy/) | prerequisites | [Entropy](/mathematics/information-theory/entropy/) |
| [Cross-Entropy](/mathematics/information-theory/cross-entropy/) | related | [Softmax](/deep-learning/core/softmax/) |
| [Cross-Entropy](/mathematics/information-theory/cross-entropy/) | related | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [Entropy](/mathematics/information-theory/entropy/) | body-link | [Cross-Entropy](/mathematics/information-theory/cross-entropy/) |
| [Entropy](/mathematics/information-theory/entropy/) | body-link | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [Entropy](/mathematics/information-theory/entropy/) | prerequisites | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Entropy](/mathematics/information-theory/entropy/) | related | [Cross-Entropy](/mathematics/information-theory/cross-entropy/) |
| [Entropy](/mathematics/information-theory/entropy/) | related | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | body-link | [Cross-Entropy](/mathematics/information-theory/cross-entropy/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | body-link | [Entropy](/mathematics/information-theory/entropy/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | body-link | [Variational Inference](/mathematics/probability/variational-inference/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | body-link | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | prerequisites | [Cross-Entropy](/mathematics/information-theory/cross-entropy/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | prerequisites | [Entropy](/mathematics/information-theory/entropy/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | related | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [KL Divergence](/mathematics/information-theory/kl-divergence/) | related | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Dot Product](/mathematics/linear-algebra/dot-product/) | body-link | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Dot Product](/mathematics/linear-algebra/dot-product/) | body-link | [Matrix](/mathematics/linear-algebra/matrix/) |
| [Dot Product](/mathematics/linear-algebra/dot-product/) | body-link | [Vector Norm](/mathematics/linear-algebra/vector-norm/) |
| [Dot Product](/mathematics/linear-algebra/dot-product/) | prerequisites | [Vector Norm](/mathematics/linear-algebra/vector-norm/) |
| [Dot Product](/mathematics/linear-algebra/dot-product/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Dot Product](/mathematics/linear-algebra/dot-product/) | related | [Scaled Dot-Product Attention](/deep-learning/transformer/scaled-dot-product-attention/) |
| [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) | body-link | [Linear Layer](/deep-learning/core/linear-layer/) |
| [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) | prerequisites | [Matrix](/mathematics/linear-algebra/matrix/) |
| [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) | related | [Linear Layer](/deep-learning/core/linear-layer/) |
| [Matrix](/mathematics/linear-algebra/matrix/) | body-link | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Matrix](/mathematics/linear-algebra/matrix/) | body-link | [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) |
| [Matrix](/mathematics/linear-algebra/matrix/) | body-link | [Vector](/mathematics/linear-algebra/vector/) |
| [Matrix](/mathematics/linear-algebra/matrix/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Matrix](/mathematics/linear-algebra/matrix/) | related | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Matrix](/mathematics/linear-algebra/matrix/) | related | [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) |
| [Vector Norm](/mathematics/linear-algebra/vector-norm/) | body-link | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Vector Norm](/mathematics/linear-algebra/vector-norm/) | prerequisites | [Vector](/mathematics/linear-algebra/vector/) |
| [Vector Norm](/mathematics/linear-algebra/vector-norm/) | related | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Vector](/mathematics/linear-algebra/vector/) | body-link | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Vector](/mathematics/linear-algebra/vector/) | body-link | [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) |
| [Vector](/mathematics/linear-algebra/vector/) | body-link | [Vector Norm](/mathematics/linear-algebra/vector-norm/) |
| [Vector](/mathematics/linear-algebra/vector/) | related | [Dot Product](/mathematics/linear-algebra/dot-product/) |
| [Vector](/mathematics/linear-algebra/vector/) | related | [Matrix](/mathematics/linear-algebra/matrix/) |
| [Vector](/mathematics/linear-algebra/vector/) | related | [Vector Norm](/mathematics/linear-algebra/vector-norm/) |
| [Euler Method](/mathematics/numerical-methods/euler-method/) | body-link | [π0](/robot-learning/pi0/) |
| [Euler Method](/mathematics/numerical-methods/euler-method/) | prerequisites | [Ordinary Differential Equation](/mathematics/calculus/ordinary-differential-equation/) |
| [Euler Method](/mathematics/numerical-methods/euler-method/) | related | [Flow Matching](/generative-models/flow-matching/) |
| [Bayes' Theorem](/mathematics/probability/bayes-theorem/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Bayes' Theorem](/mathematics/probability/bayes-theorem/) | body-link | [Conditional Probability](/mathematics/probability/conditional-probability/) |
| [Bayes' Theorem](/mathematics/probability/bayes-theorem/) | body-link | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Bayes' Theorem](/mathematics/probability/bayes-theorem/) | prerequisites | [Conditional Probability](/mathematics/probability/conditional-probability/) |
| [Bayes' Theorem](/mathematics/probability/bayes-theorem/) | related | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Conditional Probability](/mathematics/probability/conditional-probability/) | body-link | [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) |
| [Conditional Probability](/mathematics/probability/conditional-probability/) | body-link | [Bayes' Theorem](/mathematics/probability/bayes-theorem/) |
| [Conditional Probability](/mathematics/probability/conditional-probability/) | body-link | [Latent Variable](/mathematics/probability/latent-variable/) |
| [Conditional Probability](/mathematics/probability/conditional-probability/) | prerequisites | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Conditional Probability](/mathematics/probability/conditional-probability/) | related | [Bayes' Theorem](/mathematics/probability/bayes-theorem/) |
| [Covariance](/mathematics/probability/covariance/) | body-link | [Linear Transformation](/mathematics/linear-algebra/linear-transformation/) |
| [Covariance](/mathematics/probability/covariance/) | body-link | [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) |
| [Covariance](/mathematics/probability/covariance/) | body-link | [Variance](/mathematics/probability/variance/) |
| [Covariance](/mathematics/probability/covariance/) | prerequisites | [Expectation](/mathematics/probability/expectation/) |
| [Covariance](/mathematics/probability/covariance/) | prerequisites | [Variance](/mathematics/probability/variance/) |
| [Covariance](/mathematics/probability/covariance/) | related | [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) |
| [Expectation](/mathematics/probability/expectation/) | body-link | [Entropy](/mathematics/information-theory/entropy/) |
| [Expectation](/mathematics/probability/expectation/) | body-link | [Variance](/mathematics/probability/variance/) |
| [Expectation](/mathematics/probability/expectation/) | body-link | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [Expectation](/mathematics/probability/expectation/) | prerequisites | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Expectation](/mathematics/probability/expectation/) | prerequisites | [Random Variable](/mathematics/probability/random-variable/) |
| [Expectation](/mathematics/probability/expectation/) | related | [Variance](/mathematics/probability/variance/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | body-link | [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | body-link | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | prerequisites | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | prerequisites | [Random Variable](/mathematics/probability/random-variable/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | related | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Latent Variable](/mathematics/probability/latent-variable/) | related | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | body-link | [Covariance](/mathematics/probability/covariance/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | body-link | [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | prerequisites | [Matrix](/mathematics/linear-algebra/matrix/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | prerequisites | [Covariance](/mathematics/probability/covariance/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | prerequisites | [Normal Distribution](/mathematics/probability/normal-distribution/) |
| [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) | related | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | body-link | [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | body-link | [Variance](/mathematics/probability/variance/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | prerequisites | [Expectation](/mathematics/probability/expectation/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | prerequisites | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | prerequisites | [Variance](/mathematics/probability/variance/) |
| [Normal Distribution](/mathematics/probability/normal-distribution/) | related | [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) |
| [Probability Distribution](/mathematics/probability/probability-distribution/) | body-link | [Conditional Probability](/mathematics/probability/conditional-probability/) |
| [Probability Distribution](/mathematics/probability/probability-distribution/) | body-link | [Normal Distribution](/mathematics/probability/normal-distribution/) |
| [Probability Distribution](/mathematics/probability/probability-distribution/) | body-link | [Random Variable](/mathematics/probability/random-variable/) |
| [Probability Distribution](/mathematics/probability/probability-distribution/) | prerequisites | [Random Variable](/mathematics/probability/random-variable/) |
| [Probability Distribution](/mathematics/probability/probability-distribution/) | related | [Conditional Probability](/mathematics/probability/conditional-probability/) |
| [Probability Distribution](/mathematics/probability/probability-distribution/) | related | [Normal Distribution](/mathematics/probability/normal-distribution/) |
| [Random Variable](/mathematics/probability/random-variable/) | body-link | [Expectation](/mathematics/probability/expectation/) |
| [Random Variable](/mathematics/probability/random-variable/) | body-link | [Latent Variable](/mathematics/probability/latent-variable/) |
| [Random Variable](/mathematics/probability/random-variable/) | body-link | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Random Variable](/mathematics/probability/random-variable/) | body-link | [Variance](/mathematics/probability/variance/) |
| [Random Variable](/mathematics/probability/random-variable/) | related | [Expectation](/mathematics/probability/expectation/) |
| [Random Variable](/mathematics/probability/random-variable/) | related | [Probability Distribution](/mathematics/probability/probability-distribution/) |
| [Random Variable](/mathematics/probability/random-variable/) | related | [Variance](/mathematics/probability/variance/) |
| [Variance](/mathematics/probability/variance/) | body-link | [Covariance](/mathematics/probability/covariance/) |
| [Variance](/mathematics/probability/variance/) | body-link | [Expectation](/mathematics/probability/expectation/) |
| [Variance](/mathematics/probability/variance/) | body-link | [Normal Distribution](/mathematics/probability/normal-distribution/) |
| [Variance](/mathematics/probability/variance/) | prerequisites | [Expectation](/mathematics/probability/expectation/) |
| [Variance](/mathematics/probability/variance/) | related | [Covariance](/mathematics/probability/covariance/) |
| [Variance](/mathematics/probability/variance/) | related | [Normal Distribution](/mathematics/probability/normal-distribution/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | body-link | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | body-link | [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | prerequisites | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | prerequisites | [Bayes' Theorem](/mathematics/probability/bayes-theorem/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | prerequisites | [Latent Variable](/mathematics/probability/latent-variable/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | related | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Variational Inference](/mathematics/probability/variational-inference/) | related | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) | body-link | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) | body-link | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) | prerequisites | [KL Divergence](/mathematics/information-theory/kl-divergence/) |
| [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) | prerequisites | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) | related | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) | body-link | [Variational Autoencoder](/generative-models/variational-autoencoder/) |
| [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) | body-link | [Multivariate Normal Distribution](/mathematics/probability/multivariate-normal-distribution/) |
| [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) | body-link | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) | prerequisites | [Variational Inference](/mathematics/probability/variational-inference/) |
| [Reparameterization Trick](/mathematics/probability/variational-inference/reparameterization-trick/) | related | [Evidence Lower Bound](/mathematics/probability/variational-inference/evidence-lower-bound/) |
| [ACT](/robot-learning/act/) | body-link | [ResNet](/deep-learning/cnn/resnet/) |
| [ACT](/robot-learning/act/) | body-link | [DETR](/deep-learning/detr/) |
| [ACT](/robot-learning/act/) | body-link | [Transformer](/deep-learning/transformer/) |
| [ACT](/robot-learning/act/) | body-link | [Action Chunking](/robot-learning/act/action-chunking/) |
| [ACT](/robot-learning/act/) | body-link | [Architecture](/robot-learning/act/architecture/) |
| [ACT](/robot-learning/act/) | body-link | [Complete Data Flow](/robot-learning/act/complete-data-flow/) |
| [ACT](/robot-learning/act/) | body-link | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [ACT](/robot-learning/act/) | body-link | [Inference](/robot-learning/act/inference/) |
| [ACT](/robot-learning/act/) | body-link | [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/) |
| [ACT](/robot-learning/act/) | body-link | [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) |
| [ACT](/robot-learning/act/) | body-link | [Training](/robot-learning/act/training/) |
| [ACT](/robot-learning/act/) | body-link | [Vision Pipeline](/robot-learning/act/vision-pipeline/) |
| [ACT](/robot-learning/act/) | body-link | [Behavior Cloning](/robot-learning/behavior-cloning/) |
| [ACT](/robot-learning/act/) | prerequisites | [Behavior Cloning](/robot-learning/behavior-cloning/) |
| [ACT](/robot-learning/act/) | related | [Action Chunking](/robot-learning/act/action-chunking/) |
| [ACT](/robot-learning/act/) | related | [Architecture](/robot-learning/act/architecture/) |
| [ACT](/robot-learning/act/) | related | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [ACT](/robot-learning/act/) | related | [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) |
| [Action Chunking](/robot-learning/act/action-chunking/) | body-link | [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) |
| [Action Chunking](/robot-learning/act/action-chunking/) | prerequisites | [Behavior Cloning](/robot-learning/behavior-cloning/) |
| [Action Chunking](/robot-learning/act/action-chunking/) | related | [Inference](/robot-learning/act/inference/) |
| [Action Chunking](/robot-learning/act/action-chunking/) | related | [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) |
| [Architecture](/robot-learning/act/architecture/) | body-link | [CLS Token](/deep-learning/bert/cls-token/) |
| [Architecture](/robot-learning/act/architecture/) | body-link | [ResNet](/deep-learning/cnn/resnet/) |
| [Architecture](/robot-learning/act/architecture/) | body-link | [DETR](/deep-learning/detr/) |
| [Architecture](/robot-learning/act/architecture/) | body-link | [Vision Pipeline](/robot-learning/act/vision-pipeline/) |
| [Architecture](/robot-learning/act/architecture/) | prerequisites | [ACT](/robot-learning/act/) |
| [Architecture](/robot-learning/act/architecture/) | related | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [Architecture](/robot-learning/act/architecture/) | related | [Training](/robot-learning/act/training/) |
| [Architecture](/robot-learning/act/architecture/) | related | [Vision Pipeline](/robot-learning/act/vision-pipeline/) |
| [Complete Data Flow](/robot-learning/act/complete-data-flow/) | prerequisites | [Architecture](/robot-learning/act/architecture/) |
| [Complete Data Flow](/robot-learning/act/complete-data-flow/) | prerequisites | [Inference](/robot-learning/act/inference/) |
| [Complete Data Flow](/robot-learning/act/complete-data-flow/) | prerequisites | [Training](/robot-learning/act/training/) |
| [Complete Data Flow](/robot-learning/act/complete-data-flow/) | related | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [Complete Data Flow](/robot-learning/act/complete-data-flow/) | related | [Vision Pipeline](/robot-learning/act/vision-pipeline/) |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | body-link | [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | body-link | [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | prerequisites | [Conditional Variational Autoencoder](/generative-models/conditional-variational-autoencoder/) |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | related | [Training](/robot-learning/act/training/) |
| [CVAE in ACT](/robot-learning/act/cvae-in-act/) | related | [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) |
| [Inference](/robot-learning/act/inference/) | body-link | [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) |
| [Inference](/robot-learning/act/inference/) | prerequisites | [Action Chunking](/robot-learning/act/action-chunking/) |
| [Inference](/robot-learning/act/inference/) | prerequisites | [Architecture](/robot-learning/act/architecture/) |
| [Inference](/robot-learning/act/inference/) | related | [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) |
| [Inference](/robot-learning/act/inference/) | related | [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) |
| [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/) | prerequisites | [ACT](/robot-learning/act/) |
| [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/) | related | [Architecture](/robot-learning/act/architecture/) |
| [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/) | related | [Training](/robot-learning/act/training/) |
| [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) | prerequisites | [Action Chunking](/robot-learning/act/action-chunking/) |
| [Temporal Ensemble](/robot-learning/act/temporal-ensemble/) | related | [Inference](/robot-learning/act/inference/) |
| [Training](/robot-learning/act/training/) | body-link | [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/) |
| [Training](/robot-learning/act/training/) | prerequisites | [Architecture](/robot-learning/act/architecture/) |
| [Training](/robot-learning/act/training/) | prerequisites | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [Training](/robot-learning/act/training/) | related | [Inference](/robot-learning/act/inference/) |
| [Training](/robot-learning/act/training/) | related | [Paper and Released Implementation](/robot-learning/act/paper-and-released-implementation/) |
| [Vision Pipeline](/robot-learning/act/vision-pipeline/) | body-link | [ResNet](/deep-learning/cnn/resnet/) |
| [Vision Pipeline](/robot-learning/act/vision-pipeline/) | prerequisites | [ResNet](/deep-learning/cnn/resnet/) |
| [Vision Pipeline](/robot-learning/act/vision-pipeline/) | prerequisites | [Positional Encoding](/deep-learning/sequence-modeling/positional-encoding/) |
| [Vision Pipeline](/robot-learning/act/vision-pipeline/) | related | [Architecture](/robot-learning/act/architecture/) |
| [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) | prerequisites | [Normal Distribution](/mathematics/probability/normal-distribution/) |
| [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) | prerequisites | [CVAE in ACT](/robot-learning/act/cvae-in-act/) |
| [为什么 ACT 推理时令 z = 0？](/robot-learning/act/why-z-zero-at-inference/) | related | [Inference](/robot-learning/act/inference/) |
| [Behavior Cloning](/robot-learning/behavior-cloning/) | body-link | [ACT](/robot-learning/act/) |
| [Behavior Cloning](/robot-learning/behavior-cloning/) | body-link | [DAgger](/robot-learning/dagger/) |
| [Behavior Cloning](/robot-learning/behavior-cloning/) | body-link | [Imitation Learning](/robot-learning/imitation-learning/) |
| [Behavior Cloning](/robot-learning/behavior-cloning/) | prerequisites | [Imitation Learning](/robot-learning/imitation-learning/) |
| [Behavior Cloning](/robot-learning/behavior-cloning/) | related | [ACT](/robot-learning/act/) |
| [Behavior Cloning](/robot-learning/behavior-cloning/) | related | [DAgger](/robot-learning/dagger/) |
| [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) | body-link | [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) |
| [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) | prerequisites | [Imitation Learning](/robot-learning/imitation-learning/) |
| [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) | related | [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) |
| [DAgger](/robot-learning/dagger/) | prerequisites | [Behavior Cloning](/robot-learning/behavior-cloning/) |
| [DAgger](/robot-learning/dagger/) | related | [Imitation Learning](/robot-learning/imitation-learning/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | body-link | [ACT](/robot-learning/act/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | body-link | [Behavior Cloning](/robot-learning/behavior-cloning/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | body-link | [DAgger](/robot-learning/dagger/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | body-link | [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | related | [ACT](/robot-learning/act/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | related | [Behavior Cloning](/robot-learning/behavior-cloning/) |
| [Imitation Learning](/robot-learning/imitation-learning/) | related | [DAgger](/robot-learning/dagger/) |
| [π0](/robot-learning/pi0/) | body-link | [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) |
| [π0](/robot-learning/pi0/) | body-link | [Flow Matching](/generative-models/flow-matching/) |
| [π0](/robot-learning/pi0/) | body-link | [Action Expert](/robot-learning/pi0/action-expert/) |
| [π0](/robot-learning/pi0/) | body-link | [Architecture](/robot-learning/pi0/architecture/) |
| [π0](/robot-learning/pi0/) | body-link | [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) |
| [π0](/robot-learning/pi0/) | body-link | [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) |
| [π0](/robot-learning/pi0/) | body-link | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [π0](/robot-learning/pi0/) | body-link | [Inference](/robot-learning/pi0/inference/) |
| [π0](/robot-learning/pi0/) | body-link | [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) |
| [π0](/robot-learning/pi0/) | body-link | [Training](/robot-learning/pi0/training/) |
| [π0](/robot-learning/pi0/) | prerequisites | [Flow Matching](/generative-models/flow-matching/) |
| [π0](/robot-learning/pi0/) | prerequisites | [Action Chunking](/robot-learning/act/action-chunking/) |
| [π0](/robot-learning/pi0/) | prerequisites | [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) |
| [π0](/robot-learning/pi0/) | related | [Action Expert](/robot-learning/pi0/action-expert/) |
| [π0](/robot-learning/pi0/) | related | [Architecture](/robot-learning/pi0/architecture/) |
| [π0](/robot-learning/pi0/) | related | [Inference](/robot-learning/pi0/inference/) |
| [π0](/robot-learning/pi0/) | related | [Training](/robot-learning/pi0/training/) |
| [Action Expert](/robot-learning/pi0/action-expert/) | prerequisites | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Action Expert](/robot-learning/pi0/action-expert/) | prerequisites | [Architecture](/robot-learning/pi0/architecture/) |
| [Action Expert](/robot-learning/pi0/action-expert/) | related | [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) |
| [Action Expert](/robot-learning/pi0/action-expert/) | related | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [Architecture](/robot-learning/pi0/architecture/) | body-link | [Action Expert](/robot-learning/pi0/action-expert/) |
| [Architecture](/robot-learning/pi0/architecture/) | body-link | [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) |
| [Architecture](/robot-learning/pi0/architecture/) | prerequisites | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Architecture](/robot-learning/pi0/architecture/) | prerequisites | [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) |
| [Architecture](/robot-learning/pi0/architecture/) | prerequisites | [π0](/robot-learning/pi0/) |
| [Architecture](/robot-learning/pi0/architecture/) | related | [Action Expert](/robot-learning/pi0/action-expert/) |
| [Architecture](/robot-learning/pi0/architecture/) | related | [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) |
| [Architecture](/robot-learning/pi0/architecture/) | related | [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | body-link | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | body-link | [KV Cache](/deep-learning/transformer/kv-cache/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | prerequisites | [Self-Attention](/deep-learning/attention/self-attention/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | prerequisites | [Causal Mask](/deep-learning/sequence-modeling/causal-mask/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | prerequisites | [Architecture](/robot-learning/pi0/architecture/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | related | [KV Cache](/deep-learning/transformer/kv-cache/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | related | [Action Expert](/robot-learning/pi0/action-expert/) |
| [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) | related | [Inference](/robot-learning/pi0/inference/) |
| [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) | prerequisites | [Architecture](/robot-learning/pi0/architecture/) |
| [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) | prerequisites | [Inference](/robot-learning/pi0/inference/) |
| [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) | prerequisites | [Training](/robot-learning/pi0/training/) |
| [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) | related | [Action Expert](/robot-learning/pi0/action-expert/) |
| [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) | related | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) | body-link | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) | prerequisites | [Flow Matching](/generative-models/flow-matching/) |
| [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) | prerequisites | [Architecture](/robot-learning/pi0/architecture/) |
| [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) | related | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) | related | [Inference](/robot-learning/pi0/inference/) |
| [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) | related | [Training](/robot-learning/pi0/training/) |
| [Inference](/robot-learning/pi0/inference/) | body-link | [KV Cache](/deep-learning/transformer/kv-cache/) |
| [Inference](/robot-learning/pi0/inference/) | prerequisites | [KV Cache](/deep-learning/transformer/kv-cache/) |
| [Inference](/robot-learning/pi0/inference/) | prerequisites | [Euler Method](/mathematics/numerical-methods/euler-method/) |
| [Inference](/robot-learning/pi0/inference/) | prerequisites | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [Inference](/robot-learning/pi0/inference/) | related | [Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) |
| [Inference](/robot-learning/pi0/inference/) | related | [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) |
| [Paper and Released Implementation](/robot-learning/pi0/paper-and-released-implementation/) | prerequisites | [π0](/robot-learning/pi0/) |
| [Paper and Released Implementation](/robot-learning/pi0/paper-and-released-implementation/) | related | [Action Expert](/robot-learning/pi0/action-expert/) |
| [Paper and Released Implementation](/robot-learning/pi0/paper-and-released-implementation/) | related | [Inference](/robot-learning/pi0/inference/) |
| [Paper and Released Implementation](/robot-learning/pi0/paper-and-released-implementation/) | related | [Training](/robot-learning/pi0/training/) |
| [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) | body-link | [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) |
| [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) | prerequisites | [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) |
| [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) | prerequisites | [π0](/robot-learning/pi0/) |
| [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) | related | [Training](/robot-learning/pi0/training/) |
| [Training](/robot-learning/pi0/training/) | prerequisites | [Architecture](/robot-learning/pi0/architecture/) |
| [Training](/robot-learning/pi0/training/) | prerequisites | [Flow Matching in π0](/robot-learning/pi0/flow-matching-in-pi0/) |
| [Training](/robot-learning/pi0/training/) | related | [Complete Data Flow](/robot-learning/pi0/complete-data-flow/) |
| [Training](/robot-learning/pi0/training/) | related | [Inference](/robot-learning/pi0/inference/) |
| [Training](/robot-learning/pi0/training/) | related | [Pre-training and Post-training in π0](/robot-learning/pi0/pretraining-and-posttraining/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | body-link | [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | body-link | [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | body-link | [Imitation Learning](/robot-learning/imitation-learning/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | body-link | [π0](/robot-learning/pi0/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | prerequisites | [Vision-Language Model](/deep-learning/multimodal/vision-language-model/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | prerequisites | [Imitation Learning](/robot-learning/imitation-learning/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | related | [Cross-Embodiment Learning](/robot-learning/cross-embodiment-learning/) |
| [Vision-Language-Action Model](/robot-learning/vision-language-action-model/) | related | [π0](/robot-learning/pi0/) |
