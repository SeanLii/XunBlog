# XunBlog v6 Quality Audit

Final audit target: `XunBlog_Knowledge_v6`

v6 is a precision release over the fully re-audited v5 corpus. The purpose is not to generate another wholesale rewrite, but to close the remaining places where a mathematically correct-looking page could still be logically over-broad, implementation-ambiguous or insufficiently explained.

---

# 1. Acceptance dimensions

Every page is judged on three independent dimensions.

## Logical rigor

Definitions, assumptions, index directions, implementation conventions and conclusion strength must agree. Statements involving uniqueness, limits, equivalence, optimality or convergence must include the conditions that make them true.

## Mathematical sufficiency

A page uses as much mathematics as the knowledge itself requires. Mathematical pages and model objectives must state the relations that define the object; structural pages are not required to add decorative equations.

## Explanatory completeness

Important formulas and mechanisms must state what the objects are, what relation is being expressed, why the relation is needed and what its boundary is. Formality must not become unexplained symbolic compression.

The resulting standard is:

> **accurate definition → mathematical relation → explanatory meaning → assumptions / boundary**

---

# 2. v6 precision revisions

Six content pages changed.

| Page | v6 correction |
|---|---|
| Softmax | Added unique-argmax condition for the \(T\to0^+\) one-hot limit and the tied-maxima limit. |
| Cross-Entropy | Added continuous density definition, differential entropy / KL decomposition and support / absolute-continuity boundary. |
| Object Query | Separated learned object-query embeddings (`query_pos`), zero-initialized decoder content state and attention-layer \(Q\). |
| Temporal Ensemble | Made the ACT paper's oldest-prediction \(w_0\) convention explicit and aligned it with released code ordering. |
| Flow Matching | Separated sample trajectory velocity, conditional target field and marginal field; added the original Gaussian OT-style path formulas. |
| Vector Norm | Explained what finite-dimensional norm equivalence actually guarantees for convergence / topology and what it does not guarantee for optimization. |

No canonical route, page ownership or navigation-tree change was required.

---

# 3. Structural validation

Automated validation after the v6 edits produced:

| Check | Result |
|---|---:|
| Content pages | **82** |
| Unique canonical URLs | **82** |
| Duplicate canonical URLs | **0** |
| Frontmatter prerequisite / related links | **273** |
| Markdown body internal-link occurrences | **200** |
| Broken internal targets | **0** |
| Unique typed navigation relations | **441** |
| Pages with multiple / missing H1 | **0** |
| H1 / frontmatter title mismatch | **0** |
| Question-style H2/H3 | **0** |
| Illegal control characters | **0** |
| Unbalanced fenced code blocks | **0** |
| Unbalanced `\[` / `\]` display-math delimiters | **0** |

`441` counts relation type as part of the navigation edge: frontmatter relations and body-link relations remain distinct even when they connect the same source and target.

---

# 4. Source / implementation precision

The v6 revisions preserve the existing source policy:

- original papers are primary for model / algorithm claims;
- official released code is primary for implementation behavior;
- paper and code differences are shown rather than silently reconciled.

Two implementation-sensitive v6 corrections are especially important:

- Original DETR initializes decoder content with zeros and passes learned object queries as `query_pos`; therefore object-query embedding and attention \(Q\) are not the same object.
- ACT's temporal-ensemble paper states that \(w_0\) weights the oldest prediction; the official evaluation buffer preserves the corresponding oldest-to-newest ordering.

---

# 5. Mathematical precision examples

The v6 changes enforce the difference between a formula and a mathematically closed claim.

For Softmax,

\[
T\to0^+
\]

only yields a unique one-hot limit when the maximum logit is unique.

For continuous Cross-Entropy,

\[
H(p,q)=-\int p(x)\log q(x)\,dx
\]

uses densities, not point probabilities, and its relation to KL is

\[
H(p,q)=h(p)+D_{KL}(p\|q).
\]

For Flow Matching, the velocity of one sampled interpolation trajectory is not automatically the same mathematical object as the marginal vector field used by the generative ODE.

---

# 6. Explanatory precision examples

The v6 edits also close explanation gaps without turning pages into story-based exposition.

- Vector Norm now explains why norm equivalence implies the same finite-dimensional convergence notion rather than merely stating an inequality.
- Temporal Ensemble now explains how the decay parameter changes the rate at which new observations influence execution.
- Object Query now shows how learned slot identity, decoder content and projected attention queries relate in the actual DETR computation.

---

# 7. Final acceptance

v6 satisfies the current XunBlog standard:

- formal textbook-style exposition;
- logical conditions and boundaries stated where required;
- mathematical relations used when they define the knowledge;
- formulas connected to explanatory meaning;
- original-paper and official-code behavior kept distinguishable;
- independent canonical topics remain independent;
- no route or structural regression introduced by the precision edits.

The release rule remains:

> **Do not rewrite because the version changed. Rewrite only when accuracy, logic, mathematics, explanation or ownership materially requires it.**
