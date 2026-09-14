import fs from 'node:fs'
import path from 'node:path'
import { createMarkdownRenderer } from 'vitepress'
import { useKatexMath } from '../docs/.vitepress/markdown/katex.mjs'

const docsRoot = path.resolve('docs')
const markdown = await createMarkdownRenderer(docsRoot, {
  config(instance) {
    useKatexMath(instance)
  }
})

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(directory, entry.name)
    if (entry.isDirectory() && entry.name === '.vitepress') return []
    if (entry.isDirectory()) return walk(fullPath)
    return entry.isFile() && entry.name.endsWith('.md') ? [fullPath] : []
  })
}

const files = walk(docsRoot)
const anchorCache = new Map()

function anchorsFor(file) {
  if (anchorCache.has(file)) return anchorCache.get(file)
  const html = markdown.render(fs.readFileSync(file, 'utf8'), { path: file })
  const anchors = new Set()
  for (const match of html.matchAll(/<h[1-6][^>]*\sid="([^"]+)"/g)) {
    anchors.add(match[1])
  }
  for (const match of html.matchAll(/<[^>]+\sid="([^"]+)"/g)) {
    anchors.add(match[1])
  }
  anchorCache.set(file, anchors)
  return anchors
}

function resolveTarget(source, rawPath) {
  const decodedPath = decodeURI(rawPath)
  const base = decodedPath.startsWith('/')
    ? path.join(docsRoot, decodedPath.slice(1))
    : path.resolve(path.dirname(source), decodedPath)

  const candidates = path.extname(base)
    ? [base]
    : [`${base}.md`, path.join(base, 'index.md')]
  return candidates.find((candidate) => fs.existsSync(candidate))
}

const errors = []
let checkedLinks = 0
let checkedAnchors = 0
const outgoingLinks = new Map()

for (const file of files) {
  const source = fs.readFileSync(file, 'utf8')
  const lines = source.split('\n')
  let inFence = false
  let h1Count = 0
  let displayDelimiterCount = 0

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const trimmed = line.trimStart()
    if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
      inFence = !inFence
      continue
    }
    if (inFence) continue

    if (/^#\s+/.test(line)) h1Count += 1
    if (/\\\[|\\\]|\\\(|\\\)/.test(line)) {
      errors.push(`${path.relative(process.cwd(), file)}:${index + 1} uses an unsupported escaped math delimiter`)
    }
    displayDelimiterCount += (line.match(/\$\$/g) ?? []).length

    for (const match of line.matchAll(/(?<!!)\[[^\]\n]+\]\(([^)\s]+)\)/g)) {
      const rawTarget = match[1]
      if (/^(?:https?:|mailto:|tel:)/.test(rawTarget)) continue

      const hashIndex = rawTarget.indexOf('#')
      const pathPart = hashIndex >= 0 ? rawTarget.slice(0, hashIndex) : rawTarget
      const anchor = hashIndex >= 0 ? decodeURIComponent(rawTarget.slice(hashIndex + 1)) : ''
      const target = pathPart ? resolveTarget(file, pathPart) : file
      checkedLinks += 1

      if (!target) {
        errors.push(`${path.relative(process.cwd(), file)}:${index + 1} missing ${rawTarget}`)
        continue
      }

      const sourceName = path.relative(docsRoot, file).split(path.sep).join('/')
      const targetName = path.relative(docsRoot, target).split(path.sep).join('/')
      if (!outgoingLinks.has(sourceName)) outgoingLinks.set(sourceName, new Set())
      outgoingLinks.get(sourceName).add(targetName)

      if (anchor) {
        checkedAnchors += 1
        if (!anchorsFor(target).has(anchor)) {
          errors.push(`${path.relative(process.cwd(), file)}:${index + 1} missing anchor ${rawTarget}`)
        }
      }
    }
  }

  if (inFence) errors.push(`${path.relative(process.cwd(), file)} has an unclosed code fence`)
  const expectedH1Count = source.includes('<HomePage />') ? 0 : 1
  if (h1Count !== expectedH1Count) {
    errors.push(`${path.relative(process.cwd(), file)} has ${h1Count} top-level headings; expected ${expectedH1Count}`)
  }
  if (displayDelimiterCount % 2 !== 0) {
    errors.push(`${path.relative(process.cwd(), file)} has an unbalanced display-math delimiter`)
  }
}

const requiredKnowledgeLinks = {
  'robot-learning/act/act-what-problem-does-it-solve.md': [
    'robot-learning/imitation-learning/behavior-cloning-distribution-shift.md',
    'robot-learning/act/action-chunking.md',
    'robot-learning/act/temporal-ensemble.md',
    'generative-models/cvae.md',
    'robot-learning/act/cvae-in-act.md',
    'deep-learning/transformer.md',
    'robot-learning/act/architecture.md',
    'robot-learning/act/training.md',
    'robot-learning/act/inference.md'
  ],
  'robot-learning/act/architecture.md': [
    'robot-learning/act/vision-pipeline.md',
    'robot-learning/act/detr-to-act.md',
    'deep-learning/transformer.md',
    'deep-learning/transformer-encoder.md',
    'deep-learning/transformer-decoder.md',
    'deep-learning/cross-attention.md',
    'deep-learning/multi-head-attention.md',
    'deep-learning/positional-encoding.md',
    'generative-models/cvae.md',
    'robot-learning/act/cvae-in-act.md',
    'generative-models/latent-variable.md',
    'robot-learning/act/action-chunking.md'
  ],
  'robot-learning/act/complete-data-flow.md': [
    'robot-learning/act/action-chunking.md',
    'robot-learning/act/temporal-ensemble.md',
    'robot-learning/act/cvae-in-act.md',
    'deep-learning/transformer-encoder.md',
    'deep-learning/transformer-decoder.md',
    'robot-learning/act/vision-pipeline.md',
    'robot-learning/act/training.md',
    'robot-learning/act/inference.md',
    'robot-learning/act/why-z-zero-at-inference.md'
  ],
  'deep-learning/transformer.md': [
    'deep-learning/attention.md',
    'deep-learning/self-attention.md',
    'deep-learning/cross-attention.md',
    'deep-learning/qkv.md',
    'deep-learning/dot-product.md',
    'deep-learning/softmax.md',
    'deep-learning/multi-head-attention.md',
    'deep-learning/positional-encoding.md',
    'deep-learning/causal-mask.md',
    'deep-learning/transformer-encoder.md',
    'deep-learning/transformer-decoder.md',
    'deep-learning/feed-forward-network.md',
    'deep-learning/residual-connection.md',
    'deep-learning/layer-normalization.md',
    'deep-learning/dropout.md'
  ],
  'robot-learning/act/cvae-in-act.md': [
    'generative-models/cvae.md',
    'generative-models/vae.md',
    'generative-models/latent-variable.md',
    'generative-models/reparameterization-trick.md',
    'generative-models/posterior-collapse.md',
    'robot-learning/act/why-z-zero-at-inference.md'
  ],
  'robot-learning/act/why-z-zero-at-inference.md': [
    'robot-learning/act/cvae-in-act.md',
    'generative-models/cvae.md',
    'generative-models/latent-variable.md',
    'generative-models/posterior-collapse.md'
  ],
  'robot-learning/act/detr-to-act.md': [
    'deep-learning/transformer-decoder.md',
    'deep-learning/cross-attention.md',
    'deep-learning/self-attention.md',
    'deep-learning/qkv.md',
    'deep-learning/positional-encoding.md',
    'deep-learning/causal-mask.md',
    'robot-learning/act/architecture.md'
  ],
  'robot-learning/act/vision-pipeline.md': [
    'robot-learning/act/architecture.md',
    'deep-learning/transformer.md',
    'deep-learning/self-attention.md',
    'deep-learning/cross-attention.md',
    'deep-learning/positional-encoding.md',
    'deep-learning/residual-connection.md'
  ],
  'robot-learning/act/training.md': [
    'robot-learning/act/cvae-in-act.md',
    'generative-models/reparameterization-trick.md',
    'deep-learning/backpropagation.md',
    'robot-learning/act/action-chunking.md',
    'generative-models/posterior-collapse.md'
  ],
  'robot-learning/act/inference.md': [
    'robot-learning/act/temporal-ensemble.md',
    'robot-learning/act/why-z-zero-at-inference.md',
    'robot-learning/act/architecture.md',
    'robot-learning/act/complete-data-flow.md'
  ]
}

for (const [source, targets] of Object.entries(requiredKnowledgeLinks)) {
  const actual = outgoingLinks.get(source) ?? new Set()
  for (const target of targets) {
    if (!actual.has(target)) errors.push(`${source} missing required knowledge link to ${target}`)
  }
}

if (errors.length) {
  console.error(errors.join('\n'))
  process.exitCode = 1
} else {
  console.log(`Checked ${checkedLinks} internal links across ${files.length} Markdown files (${checkedAnchors} anchored links); no broken targets or required knowledge-graph edges.`)
}
