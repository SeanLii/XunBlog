import fs from 'node:fs'
import path from 'node:path'
import { createMarkdownRenderer } from 'vitepress'
import { useKatexMath } from '../docs/.vitepress/markdown/katex.mjs'

const docsRoot = path.resolve('docs')
const outputRoot = path.join(docsRoot, '.vitepress', 'dist')
if (!fs.existsSync(outputRoot)) {
  throw new Error('Missing build output. Run npm run docs:build first.')
}

function walk(directory, extension, ignoredDirectory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(directory, entry.name)
    if (entry.isDirectory() && entry.name === ignoredDirectory) return []
    if (entry.isDirectory()) return walk(fullPath, extension, ignoredDirectory)
    return entry.isFile() && entry.name.endsWith(extension) ? [fullPath] : []
  })
}

const markdown = await createMarkdownRenderer(docsRoot, {
  config(instance) {
    useKatexMath(instance)
  }
})

let expectedDisplayMath = 0
let expectedInlineMath = 0
let expectedMermaid = 0

for (const file of walk(docsRoot, '.md', '.vitepress')) {
  const tokens = markdown.parse(fs.readFileSync(file, 'utf8'), { path: file })
  for (const token of tokens) {
    if (token.type === 'math_block') expectedDisplayMath += 1
    if (token.type === 'fence' && token.info.trim() === 'mermaid') expectedMermaid += 1
    for (const child of token.children ?? []) {
      if (child.type === 'math_inline') expectedInlineMath += 1
    }
  }
}

let renderedDisplayMath = 0
let renderedMath = 0
let renderedMermaid = 0
let renderErrors = 0

const htmlFiles = walk(outputRoot, '.html')
for (const file of htmlFiles) {
  const html = fs.readFileSync(file, 'utf8')
  renderedDisplayMath += (html.match(/class="katex-display"/g) ?? []).length
  renderedMath += (html.match(/class="katex"/g) ?? []).length
  renderedMermaid += (html.match(/class="mermaid"/g) ?? []).length
  renderErrors += (html.match(/math-render-error|katex-error/g) ?? []).length
}

const bundledCss = walk(outputRoot, '.css')
  .map((file) => fs.readFileSync(file, 'utf8'))
  .join('\n')
const hasKatexCss = bundledCss.includes('.katex-display') && bundledCss.includes('KaTeX_Main')

const renderedInlineMath = renderedMath - renderedDisplayMath
const mismatches = [
  ['display formulas', expectedDisplayMath, renderedDisplayMath],
  ['inline formulas', expectedInlineMath, renderedInlineMath],
  ['Mermaid diagrams', expectedMermaid, renderedMermaid]
].filter(([, expected, actual]) => expected !== actual)

if (renderErrors > 0) mismatches.push(['render errors', 0, renderErrors])
if (expectedDisplayMath + expectedInlineMath > 0 && !hasKatexCss) {
  mismatches.push(['bundled KaTeX CSS', 'present', 'missing'])
}

if (mismatches.length) {
  for (const [label, expected, actual] of mismatches) {
    console.error(`${label}: expected ${expected}, rendered ${actual}`)
  }
  process.exitCode = 1
} else {
  console.log(
    `Verified ${renderedDisplayMath} display formulas, ${renderedInlineMath} inline formulas, ` +
    `${renderedMermaid} Mermaid diagrams, and KaTeX CSS across ${htmlFiles.length} built pages.`
  )
}
