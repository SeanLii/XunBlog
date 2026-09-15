---
title: "Blockwise Causal Attention Mask in π0"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/blockwise-causal-attention-mask/"
prerequisites:
  - "/deep-learning/sequence-modeling/causal-mask/"
  - "/deep-learning/attention/self-attention/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/deep-learning/transformer/kv-cache/"
  - "/robot-learning/pi0/inference/"
  - "/robot-learning/pi0/action-expert/"
---

# Blockwise Causal Attention Mask in π0

π0 partitions the input sequence into three functional blocks and uses a blockwise mask to control information flow:

```text
Block 1              Block 2       Block 3
images + language | robot state | noisy actions
```

The intended visibility is

```text
                    visible source
               B1       B2       B3
B1 image/lang   ✓        ✗        ✗
B2 state        ✓        ✓        ✗
B3 actions      ✓        ✓        ✓
```

Within each block, tokens use full bidirectional attention. The causal structure applies between blocks rather than forcing token-by-token autoregressive action generation.

## Relation to Causal Masking

General causal masking is defined separately in [Causal Mask](/deep-learning/sequence-modeling/causal-mask/). π0's contribution is the specific grouping and visibility pattern for VLM prefix, robot state and noisy actions.

This mask serves both representation structure and inference efficiency.

## VLM Prefix

Image and language tokens form the first block. They do not attend to later robot-state or action tokens.

This preserves a prefix computation compatible with the pretrained vision-language pathway: semantic representations are formed without depending on flow-step-specific action noise.

## Robot-State Block

Robot state can attend to the VLM prefix and itself, but not to action tokens.

Since robot state remains fixed during the internal flow integration of one policy call, preventing dependence on changing action tokens allows its attention representation to remain cacheable together with the preceding prefix.

## Action Block

Action tokens can read image/language, robot state and other action tokens. Within the action block the interaction is bidirectional:

\[
a_t^\tau
\leftrightarrow
\cdots
\leftrightarrow
a_{t+H-1}^\tau.
\]

π0 therefore models the action chunk jointly rather than autoregressively. A later action position can interact directly with an earlier action position during the same flow step.

## KV-Cache Consequence

For a fixed observation, the image/language/state prefix does not depend on the changing noisy action suffix. Its keys and values can be computed once and stored in a [KV Cache](/deep-learning/transformer/kv-cache/).

Each subsequent flow step only needs to update the action-side computation against the cached prefix. This is important because action generation uses multiple integration steps.

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi attention-mask implementation. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
