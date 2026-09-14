import fs from 'node:fs'
import path from 'node:path'
import { createMarkdownRenderer } from 'vitepress'
import { useKatexMath } from '../docs/.vitepress/markdown/katex.mjs'

const docsRoot = path.resolve('docs')
const failures = []
let currentFile = ''

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

const markdown = await createMarkdownRenderer(docsRoot, {
  config(instance) {
    useKatexMath(instance, {
      onError(error, content, displayMode) {
        failures.push({
          file: path.relative(process.cwd(), currentFile),
          content: content.replace(/\s+/g, ' ').trim(),
          displayMode,
          message: error instanceof Error ? error.message : String(error)
        })
        return `<code class="math-render-error">${escapeHtml(content)}</code>`
      }
    })
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
for (const file of files) {
  currentFile = file
  markdown.render(fs.readFileSync(file, 'utf8'), { path: file })
}

if (failures.length) {
  const unique = [...new Map(
    failures.map((failure) => [
      `${failure.file}\n${failure.content}\n${failure.message}`,
      failure
    ])
  ).values()]
  for (const failure of unique) {
    console.error(`${failure.file}: ${failure.message}\n  ${failure.content}`)
  }
  console.error(`Found ${failures.length} failed formulas (${unique.length} unique).`)
  process.exitCode = 1
} else {
  console.log(`Rendered every formula across ${files.length} Markdown files without errors.`)
}
