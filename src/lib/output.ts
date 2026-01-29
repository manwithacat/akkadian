/**
 * Output handling for Akkadian CLI
 */

import type { CommandError, CommandOutput, OutputMeta, OutputOptions, ProgressEvent } from '../types/output'

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m',
}

function c(color: keyof typeof colors, text: string, useColor: boolean): string {
  return useColor ? `${colors[color]}${text}${colors.reset}` : text
}

/**
 * Determine if output should be JSON based on environment
 */
export function detectOutputFormat(): 'json' | 'human' {
  if (!process.stdout.isTTY) return 'json'
  if (process.env.AKK_JSON === '1') return 'json'
  return 'human'
}

/**
 * Create default output options
 */
export function defaultOutputOptions(): OutputOptions {
  const format = detectOutputFormat()
  return {
    format,
    verbose: process.env.AKK_VERBOSE === '1',
    quiet: process.env.AKK_QUIET === '1',
    color: process.stdout.isTTY && !process.env.NO_COLOR,
  }
}

/**
 * Create a success result
 */
export function success<T>(data: T, meta?: Partial<OutputMeta>): CommandOutput<T> {
  return {
    success: true,
    data,
    meta: meta as OutputMeta,
  }
}

/**
 * Create an error result with agent hint
 */
export function error(
  code: string,
  message: string,
  agentHint?: string,
  context?: Record<string, unknown>
): CommandOutput<never> {
  const err: CommandError = {
    code,
    message,
  }
  if (agentHint) err.__agent_hint = agentHint
  if (context) err.context = context
  return {
    success: false,
    error: err,
  }
}

/**
 * Format output for display
 */
export function format<T>(output: CommandOutput<T>, options: OutputOptions): string {
  if (options.format === 'json' || (options.format === 'auto' && !process.stdout.isTTY)) {
    return JSON.stringify(output, null, 2)
  }
  return formatHuman(output, options)
}

/**
 * Format output as human-readable text
 */
function formatHuman<T>(output: CommandOutput<T>, options: OutputOptions): string {
  const lines: string[] = []
  const useColor = options.color

  if (output.success) {
    if (output.data !== undefined) {
      lines.push(formatData(output.data, useColor))
    }
    if (output.meta?.truncated) {
      lines.push('')
      lines.push(c('yellow', `... ${output.meta.remaining} more items`, useColor))
    }
  } else if (output.error) {
    lines.push(c('red', `Error: ${output.error.message}`, useColor))
    if (options.verbose && output.error.stack) {
      lines.push('')
      lines.push(c('gray', output.error.stack, useColor))
    }
    if (output.error.__agent_hint) {
      lines.push('')
      lines.push(c('cyan', `Hint: ${output.error.__agent_hint}`, useColor))
    }
  }

  if (output.meta?.duration_ms !== undefined && options.verbose) {
    lines.push('')
    lines.push(c('dim', `Completed in ${output.meta.duration_ms}ms`, useColor))
  }

  return lines.join('\n')
}

/**
 * Format data based on type
 */
function formatData(data: unknown, useColor: boolean): string {
  if (data === null || data === undefined) return ''
  if (typeof data === 'string') return data
  if (typeof data === 'number' || typeof data === 'boolean') return String(data)
  if (Array.isArray(data)) return formatArray(data, useColor)
  if (typeof data === 'object') return formatObject(data as Record<string, unknown>, useColor)
  return String(data)
}

function formatArray(arr: unknown[], useColor: boolean): string {
  if (arr.length === 0) return c('dim', '(empty)', useColor)

  if (arr.every((item) => typeof item !== 'object' || item === null)) {
    return arr.map((item) => `  • ${item}`).join('\n')
  }

  return arr.map((item, i) => `${c('dim', `[${i}]`, useColor)} ${formatData(item, useColor)}`).join('\n\n')
}

function formatObject(obj: Record<string, unknown>, useColor: boolean): string {
  const entries = Object.entries(obj)
  if (entries.length === 0) return c('dim', '(empty)', useColor)

  const maxKeyLen = Math.max(...entries.map(([k]) => k.length))
  return entries
    .map(([key, value]) => {
      const paddedKey = key.padEnd(maxKeyLen)
      const formattedValue =
        typeof value === 'object' && value !== null
          ? '\n' +
            formatData(value, useColor)
              .split('\n')
              .map((l) => '  ' + l)
              .join('\n')
          : String(value)
      return `${c('cyan', paddedKey, useColor)}  ${formattedValue}`
    })
    .join('\n')
}

/**
 * Write output to stdout
 */
export function write<T>(output: CommandOutput<T>, options: OutputOptions): void {
  if (options.quiet && output.success) return
  console.log(format(output, options))
}

/**
 * Write a progress event (numeric progress bar)
 */
export function progress(event: ProgressEvent, options: OutputOptions): void {
  if (options.quiet) return

  if (options.format === 'json') {
    console.log(JSON.stringify(event))
  } else {
    const pct = Math.round((event.step / event.total) * 100)
    const bar = '█'.repeat(Math.floor(pct / 5)) + '░'.repeat(20 - Math.floor(pct / 5))
    const msg = options.color ? `${colors.cyan}[${bar}]${colors.reset} ${event.message}` : `[${bar}] ${event.message}`
    process.stderr.write(`\r${msg}\x1b[K`)
    if (event.step === event.total) {
      process.stderr.write('\n')
    }
  }
}

/**
 * Step event for named workflow steps
 */
export interface StepEvent {
  step: string
  message: string
  details?: Record<string, unknown>
}

/**
 * Write a step event (named step update)
 */
export function logStep(event: StepEvent, options: OutputOptions): void {
  if (options.quiet) return

  if (options.format === 'json') {
    console.log(JSON.stringify(event))
  } else {
    const msg = options.color
      ? `${colors.cyan}[${event.step}]${colors.reset} ${event.message}`
      : `[${event.step}] ${event.message}`
    console.log(msg)
  }
}

/**
 * Write a warning message
 */
export function warn(message: string, options: OutputOptions): void {
  if (options.quiet) return

  if (options.format === 'json') {
    console.log(JSON.stringify({ type: 'warning', message }))
  } else {
    const msg = options.color ? `${colors.yellow}Warning: ${message}${colors.reset}` : `Warning: ${message}`
    console.log(msg)
  }
}

/**
 * Write an info message
 */
export function info(message: string, options: OutputOptions): void {
  if (options.quiet) return

  if (options.format === 'json') {
    console.log(JSON.stringify({ type: 'info', message }))
  } else {
    const msg = options.color ? `${colors.green}✓${colors.reset} ${message}` : `✓ ${message}`
    console.log(msg)
  }
}

// ============================================
// Simple print helpers for command output
// ============================================

/**
 * Print an error message to stderr
 */
export function printError(message: string): void {
  const useColor = process.stdout.isTTY && !process.env.NO_COLOR
  console.error(useColor ? `${colors.red}Error: ${message}${colors.reset}` : `Error: ${message}`)
}

/**
 * Print an info message
 */
export function printInfo(message: string): void {
  const useColor = process.stdout.isTTY && !process.env.NO_COLOR
  console.log(useColor ? `${colors.cyan}${message}${colors.reset}` : message)
}

/**
 * Print a success message
 */
export function printSuccess(message: string): void {
  const useColor = process.stdout.isTTY && !process.env.NO_COLOR
  console.log(useColor ? `${colors.green}✓ ${message}${colors.reset}` : `✓ ${message}`)
}

/**
 * Print a warning message
 */
export function printWarning(message: string): void {
  const useColor = process.stdout.isTTY && !process.env.NO_COLOR
  console.log(useColor ? `${colors.yellow}Warning: ${message}${colors.reset}` : `Warning: ${message}`)
}

/**
 * Print data as JSON
 */
export function printJson(data: unknown): void {
  console.log(JSON.stringify(data, null, 2))
}

/**
 * Print a simple table (array of rows)
 */
export function printTable(headers: string[], rows: string[][]): void {
  const useColor = process.stdout.isTTY && !process.env.NO_COLOR

  // Calculate column widths
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => String(r[i] || '').length)))

  // Print header
  const headerLine = headers.map((h, i) => h.padEnd(widths[i])).join(' | ')
  const separator = widths.map((w) => '-'.repeat(w)).join('-+-')

  console.log(useColor ? `${colors.bold}${headerLine}${colors.reset}` : headerLine)
  console.log(separator)

  // Print rows
  for (const row of rows) {
    console.log(row.map((cell, i) => String(cell || '').padEnd(widths[i])).join(' | '))
  }
}
