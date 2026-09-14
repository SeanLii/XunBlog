import fs from 'node:fs'
import path from 'node:path'
import { createMarkdownRenderer } from 'vitepress'
import { useKatexMath } from '../docs/.vitepress/markdown/katex.mjs'
import { navigationGraph } from '../docs/.vitepress/knowledge-graph.mjs'

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
const canonicalOwners = new Map()
let canonicalPages = 0

for (const file of files) {
  const source = fs.readFileSync(file, 'utf8')
  const lines = source.split('\n')
  const sourceName = path.relative(docsRoot, file).split(path.sep).join('/')
  let inFence = false
  let h1Count = 0
  let displayDelimiterCount = 0
  let bracketOpenCount = 0
  let bracketCloseCount = 0

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const trimmed = line.trimStart()
    if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
      inFence = !inFence
      continue
    }
    if (inFence) continue

    if (/^#\s+/.test(line)) h1Count += 1
    if (trimmed === '\\[') bracketOpenCount += 1
    if (trimmed === '\\]') bracketCloseCount += 1
    if (/\\\(|\\\)/.test(line)) {
      errors.push(`${path.relative(process.cwd(), file)}:${index + 1} uses an unsupported inline escaped math delimiter`)
    }
    if ((/\\\[|\\\]/.test(line)) && trimmed !== '\\[' && trimmed !== '\\]') {
      errors.push(`${path.relative(process.cwd(), file)}:${index + 1} must place a bracket math delimiter on its own line`)
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

  const frontmatterEnd = source.startsWith('---\n') ? source.indexOf('\n---', 4) : -1
  if (frontmatterEnd >= 0) {
    const frontmatter = source.slice(4, frontmatterEnd)
    for (const match of frontmatter.matchAll(/^\s*-\s*["'](\/[^"']+)["']\s*$/gm)) {
      const rawTarget = match[1]
      const target = resolveTarget(file, rawTarget)
      checkedLinks += 1
      if (!target) {
        errors.push(`${sourceName} frontmatter missing ${rawTarget}`)
        continue
      }
      const targetName = path.relative(docsRoot, target).split(path.sep).join('/')
      if (!outgoingLinks.has(sourceName)) outgoingLinks.set(sourceName, new Set())
      outgoingLinks.get(sourceName).add(targetName)
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
  if (bracketOpenCount !== bracketCloseCount) {
    errors.push(`${path.relative(process.cwd(), file)} has unbalanced bracket math delimiters`)
  }

  const canonicalMatch = source.match(/^canonical:\s*["']([^"']+)["']\s*$/m)
  if (canonicalMatch) {
    canonicalPages += 1
    const canonical = canonicalMatch[1]
    const relativeFile = path.relative(docsRoot, file).split(path.sep).join('/')
    const expectedCanonical = relativeFile.endsWith('/index.md')
      ? `/${relativeFile.slice(0, -'index.md'.length)}`
      : `/${relativeFile.slice(0, -'.md'.length)}/`
    if (canonical !== expectedCanonical) {
      errors.push(`${relativeFile} declares ${canonical}; expected ${expectedCanonical}`)
    }
    if (canonicalOwners.has(canonical)) {
      errors.push(`${relativeFile} duplicates canonical URL owned by ${canonicalOwners.get(canonical)}`)
    } else {
      canonicalOwners.set(canonical, relativeFile)
    }
  }
}

for (const [source, targets] of Object.entries(navigationGraph)) {
  if (!canonicalOwners.has(source)) errors.push(`knowledge graph source does not exist: ${source}`)
  for (const target of targets) {
    if (!canonicalOwners.has(target)) errors.push(`knowledge graph target does not exist: ${target}`)
  }
}

if (canonicalPages !== 53) {
  errors.push(`found ${canonicalPages} canonical knowledge pages; expected 53`)
}

if (errors.length) {
  console.error(errors.join('\n'))
  process.exitCode = 1
} else {
  console.log(`Checked ${canonicalPages} canonical pages and ${checkedLinks} internal links across ${files.length} Markdown files (${checkedAnchors} anchored links); no structural, target, or knowledge-graph errors.`)
}
