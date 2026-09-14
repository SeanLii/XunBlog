import katex from 'katex'

function validDelimiter(state, position) {
  const previous = position > 0 ? state.src.charCodeAt(position - 1) : -1
  const next = position + 1 <= state.posMax ? state.src.charCodeAt(position + 1) : -1
  return {
    canOpen: next !== 0x20 && next !== 0x09,
    canClose:
      previous !== 0x20 &&
      previous !== 0x09 &&
      !(next >= 0x30 && next <= 0x39)
  }
}

function mathInline(state, silent) {
  if (state.src[state.pos] !== '$') return false

  const opening = validDelimiter(state, state.pos)
  if (!opening.canOpen) {
    if (!silent) state.pending += '$'
    state.pos += 1
    return true
  }

  const start = state.pos + 1
  let match = start
  while ((match = state.src.indexOf('$', match)) !== -1) {
    let preceding = match - 1
    while (state.src[preceding] === '\\') preceding -= 1
    if ((match - preceding) % 2 === 1) break
    match += 1
  }

  if (match === -1) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (match === start) {
    if (!silent) state.pending += '$$'
    state.pos = start + 1
    return true
  }

  if (!validDelimiter(state, match).canClose) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(start, match)
  }
  state.pos = match + 1
  return true
}

function mathBlock(state, startLine, endLine, silent) {
  let position = state.bMarks[startLine] + state.tShift[startLine]
  let maximum = state.eMarks[startLine]
  if (position + 2 > maximum || state.src.slice(position, position + 2) !== '$$') return false
  if (silent) return true

  position += 2
  let firstLine = state.src.slice(position, maximum)
  let lastLine = ''
  let found = false
  let nextLine = startLine

  if (firstLine.trim().endsWith('$$')) {
    firstLine = firstLine.trim().slice(0, -2)
    found = true
  }

  while (!found) {
    nextLine += 1
    if (nextLine >= endLine) break
    position = state.bMarks[nextLine] + state.tShift[nextLine]
    maximum = state.eMarks[nextLine]
    if (position < maximum && state.tShift[nextLine] < state.blkIndent) break

    const currentLine = state.src.slice(position, maximum)
    if (currentLine.trim().endsWith('$$')) {
      const closingPosition = currentLine.lastIndexOf('$$')
      lastLine = currentLine.slice(0, closingPosition)
      found = true
    }
  }

  if (!found) return false

  state.line = nextLine + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content =
    (firstLine.trim() ? `${firstLine}\n` : '') +
    state.getLines(startLine + 1, nextLine, state.tShift[startLine], true) +
    (lastLine.trim() ? lastLine : '')
  token.map = [startLine, state.line]
  token.markup = '$$'
  return true
}

function bracketMathBlock(state, startLine, endLine, silent) {
  const startPosition = state.bMarks[startLine] + state.tShift[startLine]
  const startMaximum = state.eMarks[startLine]
  if (state.src.slice(startPosition, startMaximum).trim() !== '\\[') return false
  if (silent) return true

  let nextLine = startLine + 1
  let found = false
  while (nextLine < endLine) {
    const position = state.bMarks[nextLine] + state.tShift[nextLine]
    const maximum = state.eMarks[nextLine]
    if (state.src.slice(position, maximum).trim() === '\\]') {
      found = true
      break
    }
    nextLine += 1
  }

  if (!found) return false

  state.line = nextLine + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content = state.getLines(startLine + 1, nextLine, state.tShift[startLine], true)
  token.map = [startLine, state.line]
  token.markup = '\\[\\]'
  return true
}

function render(content, displayMode) {
  return katex.renderToString(content.trim(), {
    displayMode,
    throwOnError: true,
    strict: false,
    trust: false,
    output: 'htmlAndMathml',
    macros: {
      '\\logvar': '\\operatorname{logvar}'
    }
  })
}

export function useKatexMath(markdown, options = {}) {
  const renderToken = (content, displayMode) => {
    try {
      return render(content, displayMode)
    } catch (error) {
      if (!options.onError) throw error
      return options.onError(error, content, displayMode)
    }
  }

  markdown.inline.ruler.after('escape', 'math_inline', mathInline)
  markdown.block.ruler.before('fence', 'math_block_bracket', bracketMathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  markdown.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  markdown.renderer.rules.math_inline = (tokens, index) => renderToken(tokens[index].content, false)
  markdown.renderer.rules.math_block = (tokens, index) => `${renderToken(tokens[index].content, true)}\n`
}
