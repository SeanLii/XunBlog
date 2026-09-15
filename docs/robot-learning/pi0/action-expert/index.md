---
title: "Action Expert"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/action-expert/"
prerequisites:
  - "/robot-learning/pi0/architecture/"
  - "/deep-learning/attention/self-attention/"
related:
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
---

# Action Expert

Action Expert is π0's robotics-specific Transformer parameter set for robot state and continuous action tokens. It extends a pretrained VLM with parameters specialized for modalities that were not part of the VLM's original image-language pretraining interface.

## Expert Routing

π0 routes token groups to two parameter sets:

- **VLM expert**: image and language tokens；
- **Action Expert**: robot-state and noisy-action tokens。

The two groups still participate in shared attention interaction. Action Expert is therefore not a standalone controller placed after a finished VLM output; it is one expert inside the same Transformer computation.

## Robotics-Specific Inputs

Robot state

\[
q_t
\]

and noisy action chunk

\[
A_t^\tau
\]

are continuous vectors rather than language-token IDs. They require dedicated projections and transformations before entering the Transformer representation space.

The Action Expert supplies these robotics-specific weights while allowing them to interact with semantic representations produced by the VLM expert.

## Smaller Expert Scale

Flow Matching inference repeatedly reevaluates action-side tokens while observation prefix information remains fixed and can be cached. The π0 paper therefore uses a smaller Action Expert, reported at roughly 300M parameters, with hidden width around 1024 and MLP dimension around 4096.

The architecture choice targets inference efficiency: the repeatedly changing suffix is processed by the smaller expert rather than the full VLM parameter set alone.

## Flow-Time Conditioning

Action tokens represent intermediate states along the Flow Matching trajectory, so the Action Expert must know the current flow timestep.

The paper combines projected action vectors with sinusoidal timestep embedding through learned transformations. A representative form is

\[
e(a^\tau,\tau)
=
g_\phi(W_a a^\tau,\phi(\tau)).
\]

The resulting token encodes both the current action value and its location along the noise-to-data path.

## Vector-Field Output

After Transformer interaction, π0 selects the hidden states for the $H$ action positions and applies an output projection:

\[
h^{action}
\in
\mathbb R^{H\times d_h},
\]

\[
v_\theta
=
W_{out}h^{action}
\in
\mathbb R^{H\times d_a}.
\]

This output is the conditional vector field used by the numerical integration step. It is not a class probability and is not yet the final action chunk.

## Released openpi Interface

Current openpi's default original-π0 configuration exposes values such as

```text
VLM variant:       gemma_2b
Action Expert:     gemma_300m
action horizon:    50
action dimension:  32
```

The 32-dimensional action interface is a padded/unified model interface and does not imply that every robot in the training mixture has 32 physical actuators. Dataset transforms adapt embodiment-specific state/action vectors to the model representation.

The repository also contains later π0-family variants. Their additional conditioning or tokenization logic should not be treated as part of the original 2024 π0 architecture unless explicitly stated.

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi `pi0.py` and `pi0_config.py`. https://github.com/Physical-Intelligence/openpi
