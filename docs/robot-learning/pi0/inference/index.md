---
title: "Inference"
kind: "pi0-topic"
domain: "Robot Learning / π0"
parent: "π0"
canonical: "/robot-learning/pi0/inference/"
prerequisites:
  - "/deep-learning/transformer/kv-cache/"
  - "/robot-learning/pi0/flow-matching-in-pi0/"
  - "/mathematics/numerical-methods/euler-method/"
related:
  - "/robot-learning/pi0/blockwise-causal-attention-mask/"
  - "/robot-learning/pi0/complete-data-flow/"
---

# Inference

π0 inference generates an action chunk by numerically integrating a learned conditional vector field from Gaussian noise toward the action-data distribution. A single robot policy call therefore contains multiple internal flow steps.

```text
current observation
      ↓
encode and cache fixed prefix
      ↓
Gaussian action noise
      ↓
vector-field prediction
      ↓
Euler update
      ↓
repeat for multiple flow steps
      ↓
final action chunk
```

## Robot Time and Flow Time

Two distinct time variables are involved.

**Robot timestep** $t$ indexes the physical control process:

\[
t,t+1,t+2,\ldots
\]

**Flow timestep** $\tau$ indexes the internal generative trajectory used to produce one action chunk.

The observation remains fixed during the internal flow integration of one policy call. Only the current noisy action sample and flow time change across integration steps.

## Observation Encoding

At robot timestep $t$, observation is

\[
o_t=[I_t^1,\ldots,I_t^n,\ell_t,q_t].
\]

Image and language tokens form the semantic prefix; state provides the current robot condition. These inputs are fixed while the model generates $A_t$.

## KV Cache

Because π0's blockwise attention mask prevents prefix representations from depending on later action tokens, observation-side keys and values can be computed once and stored in a [KV Cache](/deep-learning/transformer/kv-cache/).

Later flow steps reuse this cache while recomputing the action suffix for updated noisy actions. Current openpi's PyTorch implementation explicitly performs a prefix forward with caching before the iterative denoising loop.

## Initial Action Noise

Under the paper convention,

\[
A_t^0\sim\mathcal N(0,I).
\]

The tensor already has final action-chunk shape

\[
H\times d_a,
\]

but its values are sampled from the base Gaussian distribution.

## Vector-Field Evaluation

At flow time $\tau$, the model evaluates

\[
v_\theta(A_t^\tau,o_t).
\]

This output has the same action-chunk shape and specifies the local velocity of the ODE trajectory.

## Euler Integration

The paper uses forward Euler integration:

\[
A_t^{\tau+\delta}
=
A_t^\tau+
\delta v_\theta(A_t^\tau,o_t).
\]

With

\[
\delta=0.1,
\]

inference performs 10 steps from the noise end to the data end.

Each new vector-field evaluation conditions on the updated action chunk; the ten steps are therefore sequential rather than ten independent predictions that can simply be averaged.

## Final Action Chunk

After integration, π0 obtains

\[
A_t=[a_t,\ldots,a_{t+H-1}],
\qquad H=50.
\]

Generating 50 future actions does not imply that all 50 must be executed before new perception. The controller can execute a prefix of the chunk and then replan from a new observation.

## Chunk Execution and Replanning

The π0 paper reports that ACT-style Temporal Ensemble was tested but reduced performance. The final evaluation therefore uses open-loop execution of part of each generated chunk before querying the policy again.

Reported settings include approximately:

- 16 actions at 20 Hz for UR5e / Franka setups；
- 25 actions at 50 Hz for other robots。

The resulting control loop is

```text
observe
  ↓
generate 50-action chunk
  ↓
execute configured prefix
  ↓
observe again
  ↓
generate new chunk
```

This differs from ACT's every-timestep query + temporal aggregation strategy.

## Released openpi Convention

Current openpi uses the opposite time direction:

```text
t = 1 : noise
t = 0 : data
```

with default

\[
dt=-\frac{1}{10}.
\]

The update is

\[
x_{t+dt}=x_t+dt\,v_t.
\]

This is the same flow path with reversed parameterization. Paper and code formulas must be compared after accounting for the sign convention.

## Inference Cost

Flow generation requires repeated action-side network evaluation. Prefix caching reduces redundant observation computation, and the smaller Action Expert reduces the cost of the changing suffix.

The π0 paper reports an example 3-camera RTX 4090 timing breakdown in which image encoding, observation forward and ten action-flow forwards contribute separate portions of total policy latency. Exact latency depends on hardware and implementation, so these numbers are experimental measurements rather than architecture constants.

## Sources

- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. 2024, Appendix D. https://arxiv.org/abs/2410.24164
- Official openpi `Pi0.sample_actions`. https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
