---
title: "Flow Matching in π0"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/flow-matching-in-pi0/"
prerequisites:
  - "/generative-models/flow-matching/"
  - "/robot-learning/pi0/architecture/"
related:
  - "/robot-learning/pi0/training/"
  - "/robot-learning/pi0/inference/"
  - "/mathematics/numerical-methods/euler-method/"
standard: "XunBlog Content & Knowledge Architecture v1.0"
rebuilt: "2026-09-15"
---
# Flow Matching in π0

> **知识边界**：本文的 canonical 对象是 **Flow Matching in π0**。依赖机制由 [Flow Matching](/generative-models/flow-matching/)、[Architecture](/robot-learning/pi0/architecture/) 的 canonical page 定义；本文只在当前语境中调用其接口。


π0 applies conditional Flow Matching to the entire future action chunk. The modeled object is

\[
A_t=[a_t,\ldots,a_{t+H-1}]
\in\mathbb R^{H\times d_a},
\]

conditioned on observation $o_t$. The policy therefore learns a conditional action distribution

\[
p(A_t\mid o_t)
\]

through a learned vector field rather than direct one-pass action regression.

## Linear Conditional Path

Let

\[
\epsilon\sim\mathcal N(0,I)
\]

be an action-shaped Gaussian sample and $A_t$ a demonstrated action chunk. Under the paper convention, define

\[
A_t^\tau
=
\tau A_t+(1-\tau)\epsilon,
\qquad \tau\in[0,1].
\]

The endpoints are

\[
A_t^0=\epsilon,
\qquad
A_t^1=A_t.
\]

Thus the path interpolates between the base noise sample and the demonstrated action sample.

## Target Vector Field

Differentiating the path gives

\[
\frac{dA_t^\tau}{d\tau}
=A_t-\epsilon.
\]

π0 predicts

\[
v_\theta(A_t^\tau,o_t)
\]

and minimizes

\[
\mathcal L(\theta)
=
\mathbb E
\left[
\left\|
 v_\theta(A_t^\tau,o_t)
-(A_t-\epsilon)
\right\|_2^2
\right].
\]

The target is a velocity field, not the final action itself.

## Observation Conditioning

Without observation, the vector field would model the marginal distribution of robot action chunks. A policy requires the conditional distribution for the current visual scene, instruction and robot state.

π0 therefore evaluates

\[
v_\theta(A_t^\tau,o_t),
\]

so the same noisy action state can receive different update directions under different observations.

## Chunk-Level Generation

The paper uses

\[
H=50.
\]

All action positions are noised, represented and denoised jointly. The action block uses bidirectional attention, so future positions can interact inside the same flow step.

Flow Matching therefore generates a structured $H\times d_a$ trajectory sample, not 50 independent scalar/vector generations.

## Timestep Sampling

π0 does not sample $\tau$ uniformly. The paper uses a shifted beta distribution that emphasizes noisier regions and avoids sampling exactly at the final endpoint.

This changes the training distribution over vector-field states, placing more optimization weight on regions where the model must infer plausible actions from relatively corrupted action inputs.

The sampling choice is π0-specific; standard Flow Matching does not require this exact beta distribution.

## Numerical Sampling

At inference, ground-truth $A_t$ is unavailable. The model starts from Gaussian noise and solves the learned ODE numerically:

\[
\frac{dA^\tau}{d\tau}
=v_\theta(A^\tau,o_t).
\]

π0 uses [Euler Method](/mathematics/numerical-methods/euler-method/) with ten steps in the paper experiments. Each step evaluates the vector field on the current action sample and advances it along the flow trajectory.

## Paper and openpi Time Directions

The paper uses

```text
τ = 0 : noise
τ = 1 : data
```

whereas current openpi uses

```text
t = 1 : noise
t = 0 : data
```

and constructs

\[
x_t=t\epsilon+(1-t)A,
\qquad
u_t=\epsilon-A.
\]

These are opposite parameterizations of the same path. When comparing equations, both interpolation formula and velocity sign must be converted together.

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Section IV and Appendix B. https://arxiv.org/abs/2410.24164
- Official openpi `pi0.py`. https://github.com/Physical-Intelligence/openpi
