---
title: "Architecture"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/architecture/"
prerequisites:
  - "/robot-learning/pi0/"
  - "/deep-learning/multimodal/vision-language-model/"
  - "/deep-learning/attention/self-attention/"
related:
  - "/robot-learning/pi0/action-expert/"
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
  - "/robot-learning/pi0/complete-data-flow/"
---

# Architecture

π0 combines a pretrained vision-language backbone with a smaller robotics-specific Action Expert inside a shared attention computation. Image and language tokens use VLM parameters; robot state and noisy action tokens use Action Expert parameters. The token groups interact through a structured attention mask.

整体结构为

```text
RGB images ─→ vision encoder ─┐
                              │
language tokens ──────────────┼→ VLM expert ───────┐
                              │                     │
robot state ──────────────────┼→ Action Expert ────┤
                              │                     │
noisy action chunk + time ────┘→ Action Expert ────┤
                                                    ↓
                                      shared attention interaction
                                                    ↓
                                       action-token hidden states
                                                    ↓
                                           vector-field output
```

## Image Inputs

Observation can contain multiple RGB camera views:

\[
I_t^1,\ldots,I_t^n.
\]

Each image is processed by the visual encoder associated with the VLM backbone, producing a sequence of visual embeddings. The Transformer does not directly operate on raw pixels; it operates on these encoded visual tokens.

The π0 paper builds on PaliGemma. The released openpi implementation exposes the corresponding vision module through the PaliGemma/SigLIP stack.

## Language Inputs

Language instruction $\ell_t$ is tokenized and embedded in the VLM hidden space. Visual and language tokens together form the semantic prefix representing the task and scene.

This prefix preserves the main interface of the pretrained VLM, allowing π0 to reuse visual-language representations rather than relearning semantics solely from robot trajectories.

## Robot State

Robot proprioceptive state $q_t$ is continuous and does not naturally share the VLM token interface. π0 projects state through robotics-specific parameters into the Action Expert hidden dimension.

In the original π0 formulation, state is part of the suffix-side robotics input. Current openpi later added π0.5-related alternatives, so implementation branches for newer models should not be confused with the original π0 paper definition.

## Noisy Action Chunk

At Flow Matching time $\tau$, the model receives

\[
A_t^\tau
=
[a_t^\tau,\ldots,a_{t+H-1}^\tau].
\]

Each action vector becomes one action token after projection. The paper uses

\[
H=50.
\]

Action tokens therefore preserve the temporal positions of the complete chunk rather than compressing the entire trajectory into a single latent vector.

## Flow-Time Conditioning

The same action value has different meaning at different points along the flow path. π0 therefore conditions action representations on flow timestep $\tau$.

The paper combines projected action features with sinusoidal time embeddings and MLP transformations. Abstractly:

\[
e(a^\tau,\tau)
=
g_\phi(W_a a^\tau,\phi(\tau)).
\]

This gives Action Expert access to both current noisy action state and its position along the generative flow.

## Token Blocks

The complete sequence is partitioned into functional blocks:

```text
Block 1              Block 2       Block 3
images + language | robot state | noisy actions
```

[Blockwise Causal Attention Mask in π0](/robot-learning/pi0/blockwise-causal-attention-mask/) controls which blocks can read which other blocks.

Action positions can attend bidirectionally to one another, allowing the entire future chunk to be modeled jointly instead of autoregressively.

## VLM Expert and Action Expert

The architecture is not a serial pipeline in which a VLM first emits a semantic output and a separate controller consumes it. Both experts participate in Transformer layers, with token routing determining which parameter set processes each modality.

The [Action Expert](/robot-learning/pi0/action-expert/) is smaller than the VLM expert. This choice reduces repeated action-side compute during iterative Flow Matching inference.

## Output Representation

The network does not directly emit final actions during a flow step. It takes the hidden states corresponding to action positions and projects them to a vector field:

\[
v_\theta(A_t^\tau,o_t)
\in
\mathbb R^{H\times d_a}.
\]

A numerical ODE step then updates the current noisy action chunk. Repeating this process produces the final action sample.

## Prefix Caching

The attention dependency structure prevents observation prefix tokens from depending on action tokens. During inference, observation-side keys and values can therefore be cached once while action tokens are repeatedly updated.

This separates compute into:

- fixed observation prefix computation；
- repeated Action Expert / action-suffix computation。

The resulting KV-cache structure is important because flow generation requires multiple forward evaluations per robot policy call.

## Model Scale

The paper describes a PaliGemma-based VLM with a smaller approximately 300M-parameter Action Expert. The released openpi configuration uses variants named `gemma_2b` and `gemma_300m` for the two expert parameter sets.

These dimensions are implementation/model-scale choices rather than mathematical requirements of the π0 architecture.

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, especially Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi implementation. https://github.com/Physical-Intelligence/openpi
