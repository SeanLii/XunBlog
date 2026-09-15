import fs from 'node:fs'
import path from 'node:path'

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name)
    if (entry.isDirectory() && entry.name === '.vitepress') return []
    if (entry.isDirectory()) return walk(file)
    return entry.isFile() && entry.name === 'index.md' ? [file] : []
  })
}

const errors = []
let pages = 0
for (const file of walk('docs')) {
  const source = fs.readFileSync(file, 'utf8')
  if (!/^canonical:/m.test(source)) continue
  pages += 1
  const end = source.indexOf('\n---', 4)
  const body = source.slice(end + 4)
  const chinese = (body.match(/[\u3400-\u9fff]/g) || []).length
  const sections = (body.match(/^## /gm) || []).length
  const links = (body.match(/\[[^\]]+\]\(\/[^)]+\)/g) || []).length
  const displayMath = (body.match(/^\\\[$/gm) || []).length + Math.floor((body.match(/\$\$/g) || []).length / 2)

  if (chinese < 150) errors.push(`${file}: only ${chinese} Chinese characters; explanatory prose is not substantial`)
  if (sections < 3) errors.push(`${file}: only ${sections} progressive sections`)
  if (/知识边界|canonical(?: page| topics?)|归属树|依赖图|ΔK|内容规范|创作规范/i.test(body)) {
    errors.push(`${file}: exposes internal content-architecture language in the article body`)
  }
  if (/^#{2,4} .*?[？?]$/m.test(body)) {
    errors.push(`${file}: uses question-led section organization`)
  }
  if (/如果你|你可以先|先别|让我们|接下来怎么办|这一篇会|这篇会/.test(body)) {
    errors.push(`${file}: uses classroom-style reader prompting`)
  }
  if (/^## (?:优点|缺点|优缺点|Limitations|Scope and Limitations|Applications and Limitations)$/m.test(body)) {
    errors.push(`${file}: isolates evaluation from the mechanism that causes it`)
  }
  if (links === 0 && !/mathematics\/(?:linear-algebra\/vector|probability\/random-variable)\/index\.md$/.test(file)) {
    errors.push(`${file}: has no recursive knowledge link`)
  }
  if (displayMath === 0) errors.push(`${file}: has no mathematical definition or mechanism expression`)
  if (/docs\/robot-learning\/(?:act|pi0)\//.test(file) && !/^## Sources$/m.test(body)) {
    errors.push(`${file}: model-specific claims have no source boundary`)
  }
}

if (pages !== 82) errors.push(`expected 82 canonical pages, found ${pages}`)
if (errors.length) {
  console.error(errors.join('\n'))
  process.exitCode = 1
} else {
  console.log(`Content-standard audit: ${pages} pages have substantial Chinese exposition, progressive structure, mathematical responsibility, recursive links, and no exposed editorial terminology.`)
}
