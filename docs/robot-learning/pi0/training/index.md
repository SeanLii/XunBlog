---
title: "Training"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/training/"
prerequisites:
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/robot-learning/pi0/pretraining-and-posttraining/"
  - "/robot-learning/pi0/inference/"
  - "/robot-learning/pi0/complete-data-flow/"
---

# Training

π0 training converts a demonstrated future action chunk into a conditional Flow Matching regression problem. Each sample contains current observation

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t]
\]

and future action chunk

\[
A_t=[a_t,\ldots,a_{t+H-1}].
\]

The model does not train by directly regressing final actions from observation. Instead it constructs noisy intermediate actions and predicts the velocity field that transports them toward the demonstrated action distribution.

## Action-Chunk Target

From a robot trajectory, training selects a current timestep and future horizon

\[
A_t\in\mathbb R^{H\times d_a}.
\]

The π0 paper uses

\[
H=50.
\]

All action positions are treated jointly as one sample from the conditional action distribution.

## Gaussian Noise

Sample

\[
\epsilon\sim\mathcal N(0,I)
\]

with the same shape as $A_t$. This supplies the base-distribution sample for the flow path.

## Flow Timestep

Sample a scalar flow time

\[
\tau\in[0,1].
\]

The π0 paper uses a shifted beta distribution rather than uniform timestep sampling, placing more mass in the noisier part of the path. This is a π0-specific training choice, not a general requirement of Flow Matching.

## Noisy Intermediate Action

Under the paper convention,

\[
A_t^\tau
=
\tau A_t+(1-\tau)\epsilon.
\]

At $\tau=0$, the sample is pure noise; at $\tau=1$, it equals the demonstrated action chunk.

The derivative of this linear path is

\[
u_t=A_t-\epsilon.
\]

This velocity has the same shape as the action chunk.

## Observation Prefix Encoding

Images are converted to visual tokens and language is embedded by the VLM pathway. Robot state is projected through robotics-specific parameters.

The observation-side inputs form the fixed conditioning context for the vector field:

\[
v_\theta(A_t^\tau,o_t).
\]

## Action and Time Embedding

Each noisy action vector is projected into Action Expert hidden space and combined with an embedding of $\tau$. The resulting $H$ action tokens enter the Transformer under π0's blockwise attention mask.

The network therefore receives both:

- the current point $A_t^\tau$ on the action path；
- the flow-time coordinate $\tau$。

## Vector-Field Prediction

After Transformer interaction, π0 selects the hidden states corresponding to the $H$ action positions and projects them back to action dimension:

\[
v_\theta
\in
\mathbb R^{H\times d_a}.
\]

The output represents velocity, not final action values.

## Flow Matching Objective

The paper objective is

\[
\mathcal L(\theta)
=
\mathbb E
\left[
\lVert
v_\theta(A_t^\tau,o_t)
-(A_t-\epsilon)
\rVert_2^2
\right].
\]

Training can evaluate this objective in one network forward for a sampled $\tau$. It does not need to numerically integrate ten flow steps during each training example.

## Released openpi Time Convention

Current openpi parameterizes the same path in the opposite direction:

```text
t = 1 : noise
t = 0 : data
```

and constructs

\[
x_t=t\epsilon+(1-t)A,
\]

with target

\[
u_t=\epsilon-A.
\]

The squared vector-field regression objective is unchanged up to path orientation. Paper equations and code equations should therefore not be mixed without first fixing the time convention.

## Training versus Inference

Training has access to the demonstrated $A_t$, so it can sample any intermediate point directly from $(A_t,\epsilon,\tau)$.

Inference has no target action chunk. It must begin from noise and repeatedly apply the learned vector field through numerical integration until it reaches the data end of the path.

This difference explains why training uses a single sampled flow time per example while inference uses multiple sequential network evaluations.

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Section IV. https://arxiv.org/abs/2410.24164
- Official openpi `Pi0.compute_loss`. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
