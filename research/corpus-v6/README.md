# XunBlog Knowledge v6

This package is the precision-refined XunBlog knowledge corpus for the current ACT / Transformer / π0 scope and all canonical prerequisites currently included in the tree.

## v6 objective

v6 is built on the full 82-page v5 re-audit and focuses on one stricter acceptance rule:

> **Every page must simultaneously satisfy logical rigor, mathematical sufficiency and explanatory completeness.**

The three dimensions are not interchangeable. A page does not become rigorous merely because it contains equations; it does not become readable by replacing formal definitions with intuition; and it does not become mathematically complete by adding formulas that do not establish the mechanism being taught.

The v6 acceptance loop is:

```text
accurate definition
→ mathematics establishes the relation
→ explanation establishes the meaning
→ assumptions / boundary close the claim
```

## Corpus status

- **82** content pages
- **82** unique canonical URLs
- **0** duplicate canonical URLs
- **0** broken internal targets
- **273** frontmatter prerequisite / related links
- **200** body internal-link occurrences
- **441** unique typed navigation relations (frontmatter relation + body-link relation)
- **0** H1/title mismatches
- **0** question-style H2/H3
- **0** illegal control characters
- **0** unbalanced fenced-code or display-math delimiters detected

## v6 content changes

Six pages received targeted precision revisions; canonical routes and knowledge ownership did not change.

1. `Softmax` — low-temperature limit now states the unique-maximum condition and tie case.
2. `Cross-Entropy` — continuous-density definition, differential entropy / KL relation and support conditions added.
3. `Object Query` — original DETR `query_pos` behavior is separated from zero-initialized decoder content and attention-projection \(Q\).
4. `Temporal Ensemble` — ACT's oldest-to-newest indexing and \(w_0\) convention are made explicit.
5. `Flow Matching` — sample velocity, conditional vector field and marginal vector field are separated; the original Gaussian OT-style path is stated explicitly.
6. `Vector Norm` — finite-dimensional norm equivalence is connected to convergence / topology and separated from optimization geometry.

Detailed changes are in `V6_REVISION_NOTES.md`.

## Content standard

`CONTENT_DESIGN_STANDARD.md` is the mandatory writing standard. v6 adds explicit checks for:

- missing assumptions behind strong mathematical claims;
- equations that are present but unexplained;
- intuitive explanations that replace rather than support formal mathematics;
- paper-level and released-code facts being silently merged;
- sample-level, conditional and marginal mathematical objects being conflated.

## Important files

- `CONTENT_DESIGN_STANDARD.md` — mandatory v6 writing standard.
- `V6_REVISION_NOTES.md` — exact v6 precision changes.
- `CONTENT_PAGE_AUDIT.md` — cumulative all-82-page audit plus v6 delta.
- `KNOWLEDGE_TREE.md` — canonical ownership tree for Sidebar / Breadcrumb.
- `CONTENT_INDEX.md` — every title, canonical URL and source file.
- `LINK_GRAPH.md` — prerequisite / related / body-link navigation graph.
- `CANONICAL_INDEPENDENCE_AUDIT.md` — topic independence and provenance decisions.
- `CANONICAL_PATH_MIGRATION.md` — route migration from earlier site structures.
- `SOURCE_MAP.md` — paper / official-code grounding.
- `QUALITY_AUDIT.md` — v6 manual precision + automated structural checks.

## Import rule for Codex

When importing this package:

1. Treat `CONTENT_INDEX.md` as canonical route truth.
2. Treat `KNOWLEDGE_TREE.md` as Sidebar / Breadcrumb ownership truth.
3. Apply `CANONICAL_PATH_MIGRATION.md` to old routes where needed.
4. Preserve page frontmatter, H1, equations and internal canonical links.
5. Do not flatten the tree.
6. Do not create duplicate explanations for an existing canonical topic.
7. Do not automatically shorten large articles or expand small ones.
8. Do not rewrite prose during import unless required to fix rendering.

## Final principle

> **Definition accuracy, mathematics and explanation must form one closed reasoning chain.**

The amount of content is determined by the knowledge itself, not by page length or equation count.
