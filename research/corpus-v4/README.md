# XunBlog Knowledge v4

This is the final v4 content package for the current XunBlog knowledge scope.

## What changed in v4

v4 combines two requirements:

1. **Every genuine independent topic must be taught as an independent topic.**
2. **Ownership follows the origin / academic lineage of the specific concept, not later reuse.**

The result is not simply “more pages”. The tree itself was corrected.

Key examples:

```text
Attention          → independent from Transformer
Self-Attention     → independent from Transformer
CLS Token          → BERT
Object Query       → DETR
Residual Connection→ ResNet
Flow Matching      → independent from π0
CVAE               → independent Generative Model
Reparameterization → Variational Inference
Action Chunking    → ACT-scoped robot-policy formulation
Action Expert      → π0
```

## Content count

- 82 content pages
- 82 unique canonical URLs
- 0 broken internal links

## Important files

- `CONTENT_DESIGN_STANDARD.md`  
  Final writing and ownership rules.

- `KNOWLEDGE_TREE.md`  
  Canonical ownership tree for sidebar and breadcrumbs.

- `CONTENT_INDEX.md`  
  Every page, canonical URL and file path.

- `LINK_GRAPH.md`  
  Cross-page navigation graph.

- `CANONICAL_INDEPENDENCE_AUDIT.md`  
  Detailed reasoning for what was moved, split, retained or rewritten.

- `CANONICAL_PATH_MIGRATION.md`  
  Old / previous route → v4 route mapping for Codex.

- `SOURCE_MAP.md`  
  Paper / code grounding and provenance references.

- `QUALITY_AUDIT.md`  
  Final automated and manual checks.

## Implementation rule for Codex

When importing this package:

1. Do not flatten the tree.
2. Use `CONTENT_INDEX.md` for canonical routes.
3. Use `KNOWLEDGE_TREE.md` for breadcrumb ownership.
4. Update old internal routes using `CANONICAL_PATH_MIGRATION.md`.
5. Do not create duplicate pages for concepts already present.
6. Keep model-specific pages model-specific.
7. Do not rewrite article prose during import unless required for rendering.
8. Preserve equations, headings, frontmatter and canonical links.

## Final writing principle

A page can be short or long, but it must be complete for its own title.

A later model using a mechanism creates a navigation edge, not a new ownership claim.
