/**
 * Domain Knowledge Loader
 *
 * Provides progressive disclosure of domain knowledge
 * to minimize token usage while maximizing utility.
 */

import { akkadianDK } from './akkadian-dk'
import type { LoadLevel, QuickReference } from './types'

export * from './types'
export { akkadianDK }

/**
 * Get quick reference (minimal tokens, ~200-300)
 */
export function getQuickReference(): QuickReference {
  return {
    name: akkadianDK.name,
    purpose: akkadianDK.quick.purpose,
    commands: akkadianDK.quick.commands,
    workflows: akkadianDK.workflows.map((w) => `${w.id}: ${w.name}`),
  }
}

/**
 * Get command reference for specific command
 */
export function getCommandHelp(command: string): string | null {
  const cmd = akkadianDK.commands[command]
  if (!cmd) return null

  let help = `## ${cmd.name}\n\n${cmd.description}\n\n`
  help += `Usage: ${cmd.usage}\n\n`

  if (cmd.options) {
    help += 'Options:\n'
    for (const [opt, desc] of Object.entries(cmd.options)) {
      help += `  ${opt}: ${desc}\n`
    }
    help += '\n'
  }

  if (cmd.examples) {
    help += 'Examples:\n'
    for (const ex of cmd.examples) {
      help += `  ${ex}\n`
    }
  }

  return help
}

/**
 * Get workflow guide
 */
export function getWorkflow(id: string): string | null {
  const workflow = akkadianDK.workflows.find((w) => w.id === id)
  if (!workflow) return null

  let guide = `## ${workflow.name}\n\n`
  guide += `${workflow.description}\n\n`
  guide += `When: ${workflow.when}\n\n`
  guide += 'Steps:\n'

  for (let i = 0; i < workflow.steps.length; i++) {
    const step = workflow.steps[i]
    guide += `${i + 1}. ${step.action}\n`
    if (step.command) {
      guide += `   $ ${step.command}\n`
    }
    if (step.notes) {
      guide += `   Note: ${step.notes}\n`
    }
  }

  return guide
}

/**
 * Get pattern/recipe
 */
export function getPattern(id: string): string | null {
  const pattern = akkadianDK.patterns.find((p) => p.id === id)
  if (!pattern) return null

  let guide = `## ${pattern.name}\n\n`
  guide += `Problem: ${pattern.problem}\n\n`
  guide += `Solution: ${pattern.solution}\n\n`

  if (pattern.commands) {
    guide += 'Commands:\n'
    for (const cmd of pattern.commands) {
      guide += `  $ ${cmd}\n`
    }
  }

  return guide
}

/**
 * Get error fix
 */
export function getErrorFix(code: string): string | null {
  const error = akkadianDK.errors.find((e) => e.code === code)
  if (!error) return null

  return `## ${error.code}: ${error.message}\n\nCause: ${error.cause}\n\nFix: ${error.fix}`
}

/**
 * Get platform info
 */
export function getPlatformInfo(platform: string): string | null {
  const plat = akkadianDK.platforms[platform]
  if (!plat) return null

  let info = `## ${plat.name}\n\nLimits:\n`
  for (const [key, value] of Object.entries(plat.limits)) {
    info += `  ${key}: ${value}\n`
  }

  info += '\nTips:\n'
  for (const tip of plat.tips) {
    info += `  - ${tip}\n`
  }

  return info
}

/**
 * Get knowledge at specified level
 */
export function getKnowledge(level: LoadLevel): unknown {
  switch (level) {
    case 'quick':
      return getQuickReference()

    case 'commands':
      return {
        ...getQuickReference(),
        commands: akkadianDK.commands,
      }

    case 'workflows':
      return {
        ...getQuickReference(),
        workflows: akkadianDK.workflows,
      }

    case 'full':
      return akkadianDK

    default:
      return getQuickReference()
  }
}

/**
 * Search knowledge base
 */
export function searchKnowledge(query: string): string[] {
  const results: string[] = []
  const q = query.toLowerCase()

  // Search commands
  for (const [name, cmd] of Object.entries(akkadianDK.commands)) {
    if (name.includes(q) || cmd.description.toLowerCase().includes(q)) {
      results.push(`command:${name}`)
    }
  }

  // Search workflows
  for (const workflow of akkadianDK.workflows) {
    if (workflow.name.toLowerCase().includes(q) || workflow.description.toLowerCase().includes(q)) {
      results.push(`workflow:${workflow.id}`)
    }
  }

  // Search patterns
  for (const pattern of akkadianDK.patterns) {
    if (pattern.name.toLowerCase().includes(q) || pattern.problem.toLowerCase().includes(q)) {
      results.push(`pattern:${pattern.id}`)
    }
  }

  // Search errors
  for (const error of akkadianDK.errors) {
    if (error.code.toLowerCase().includes(q) || error.message.toLowerCase().includes(q)) {
      results.push(`error:${error.code}`)
    }
  }

  return results
}

/**
 * Format knowledge for display
 */
export function formatQuickReference(): string {
  const ref = getQuickReference()

  let output = `# ${ref.name}\n\n`
  output += `${ref.purpose}\n\n`
  output += '## Commands\n\n'

  for (const cmd of ref.commands) {
    output += `- ${cmd}\n`
  }

  output += '\n## Workflows\n\n'
  for (const wf of ref.workflows) {
    output += `- ${wf}\n`
  }

  return output
}
