/**
 * Domain Knowledge Types
 *
 * Structured schema for domain knowledge that can be loaded
 * by any LLM agent via MCP resources.
 */

/**
 * A single CLI command reference
 */
export interface CommandRef {
  name: string
  description: string
  usage: string
  options?: Record<string, string>
  examples?: string[]
}

/**
 * A workflow step
 */
export interface WorkflowStep {
  action: string
  command?: string
  notes?: string
}

/**
 * A complete workflow
 */
export interface Workflow {
  id: string
  name: string
  description: string
  when: string // When to use this workflow
  steps: WorkflowStep[]
  benefits?: string[] // Why use this workflow
  why?: string[] // Explanation of the approach
}

/**
 * A common pattern or recipe
 */
export interface Pattern {
  id: string
  name: string
  problem: string
  solution: string
  commands?: string[]
  config?: string // Example configuration
  metadata?: Record<string, string> // Metadata field descriptions
  checks?: string[] // Validation checks performed
}

/**
 * Error/troubleshooting entry
 */
export interface ErrorEntry {
  code: string
  message: string
  cause: string
  fix: string
}

/**
 * Configuration reference
 */
export interface ConfigRef {
  file: string
  description: string
  schema: Record<
    string,
    {
      type: string
      description: string
      default?: string
    }
  >
}

/**
 * Complete Domain Knowledge document
 */
export interface DomainKnowledge {
  version: string
  name: string
  description: string

  // Quick reference - minimal tokens
  quick: {
    purpose: string
    commands: string[] // One-liner list: "akk doctor - Check environment"
  }

  // Full command reference
  commands: Record<string, CommandRef>

  // Workflows for common tasks
  workflows: Workflow[]

  // Patterns and recipes
  patterns: Pattern[]

  // Error troubleshooting
  errors: ErrorEntry[]

  // Configuration files
  config: ConfigRef[]

  // Platform-specific notes
  platforms: Record<
    string,
    {
      name: string
      limits: Record<string, string>
      tips: string[]
    }
  >
}

/**
 * Subset for quick loading (minimal tokens)
 */
export interface QuickReference {
  name: string
  purpose: string
  commands: string[]
  workflows: string[] // Just workflow names
}

/**
 * Load levels for progressive disclosure
 */
export type LoadLevel = 'quick' | 'commands' | 'workflows' | 'full'
