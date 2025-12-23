/**
 * Preflight Validate Command
 *
 * Validates notebook structure, syntax, and readiness for operation.
 */

import { existsSync, readFileSync } from 'fs'
import { basename, extname } from 'path'
import { z } from 'zod'
import { getValidationSummary, type ValidationOptions, validateNotebook } from '../../lib/notebook-validator'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import type { PlatformId } from '../../types/platform'
import type { NotebookContent } from '../../types/template'
import { PLATFORMS } from './platforms'

const ValidateArgs = z.object({
  path: z.string().describe('Path to notebook (.ipynb) or script (.py)'),
  platform: z.string().optional().describe('Target platform for platform-specific checks'),
  strict: z.boolean().default(false).describe('Treat warnings as errors'),
  skipSyntax: z.boolean().default(false).describe('Skip Python syntax checks'),
  skipSecurity: z.boolean().default(false).describe('Skip security checks'),
  json: z.boolean().default(false).describe('Output raw JSON results'),
})

export const validate: CommandDefinition<typeof ValidateArgs> = {
  name: 'preflight validate',
  description: 'Validate notebook structure and readiness for operation',
  help: `
Comprehensive validation for ML notebooks.

Checks:
  - Structure: Notebook format, cell types, metadata
  - Syntax: Python code validity, template variable substitution
  - Config: Required configuration keys, placeholder values
  - Dependencies: Package installations match imports
  - Platform: Platform-specific paths and requirements
  - Tools: ML tool integrations (sacrebleu, qlora, etc.)
  - Security: Hardcoded secrets, injection risks

Use --platform to enable platform-specific validation.
Use --strict to treat warnings as errors (useful for CI).

Returns a readiness score (0-100) and detailed breakdown.
`,
  examples: [
    'akk preflight validate notebook.ipynb',
    'akk preflight validate notebook.ipynb --platform kaggle-p100',
    'akk preflight validate notebook.ipynb --strict',
    'akk preflight validate notebook.py --platform colab-pro',
  ],
  args: ValidateArgs,

  async run(args, ctx) {
    // Validate file exists
    if (!existsSync(args.path)) {
      return error(
        'FILE_NOT_FOUND',
        `File not found: ${args.path}`,
        'Provide a valid path to a notebook (.ipynb) or script (.py)'
      )
    }

    // Validate platform if provided
    let platform: PlatformId | undefined
    if (args.platform) {
      if (!PLATFORMS[args.platform]) {
        return error(
          'INVALID_PLATFORM',
          `Unknown platform: ${args.platform}`,
          `Use 'akk preflight platforms' to see available options`,
          { available: Object.keys(PLATFORMS) }
        )
      }
      platform = args.platform as PlatformId
    }

    // Read and parse file
    const ext = extname(args.path).toLowerCase()
    let notebook: NotebookContent

    if (ext === '.ipynb') {
      try {
        notebook = JSON.parse(readFileSync(args.path, 'utf-8'))
      } catch (e) {
        return error(
          'PARSE_ERROR',
          `Failed to parse notebook: ${e instanceof Error ? e.message : 'Unknown error'}`,
          'Ensure the file is a valid Jupyter notebook'
        )
      }
    } else if (ext === '.py') {
      // Convert Python script to notebook format for validation
      const content = readFileSync(args.path, 'utf-8')
      notebook = scriptToNotebook(content)
    } else {
      return error('INVALID_FORMAT', `Unsupported file format: ${ext}`, 'Provide a .ipynb or .py file')
    }

    // Run validation
    const options: ValidationOptions = {
      platform,
      checkSyntax: !args.skipSyntax,
      checkSecurity: !args.skipSecurity,
      strict: args.strict,
    }

    const result = validateNotebook(notebook, options)

    // Generate summary
    const summary = getValidationSummary(result)

    // Determine overall status
    const status = result.ready ? 'ready' : result.valid ? 'warnings' : 'failed'

    if (args.json) {
      return success({
        file: basename(args.path),
        platform: platform || 'any',
        status,
        ready: result.ready,
        score: result.score,
        valid: result.valid,
        categories: result.categories,
        errors: result.errors,
        warnings: result.warnings,
      })
    }

    return success({
      file: basename(args.path),
      platform: platform ? PLATFORMS[platform].name : 'Any platform',
      status,
      ready: result.ready,
      score: `${result.score}/100`,
      summary: {
        errors: result.errors.length,
        warnings: result.warnings.length,
      },
      categories: Object.fromEntries(
        Object.entries(result.categories).map(([cat, stats]) => [
          cat,
          stats.failed > 0 ? 'FAIL' : stats.warnings > 0 ? 'WARN' : 'PASS',
        ])
      ),
      errors:
        result.errors.length > 0
          ? result.errors.slice(0, 10).map((e) => ({
              code: e.code,
              message: e.message,
              suggestion: e.suggestion,
            }))
          : undefined,
      warnings:
        result.warnings.length > 0
          ? result.warnings.slice(0, 5).map((w) => ({
              code: w.code,
              message: w.message,
            }))
          : undefined,
      moreErrors: result.errors.length > 10 ? `... and ${result.errors.length - 10} more` : undefined,
      moreWarnings: result.warnings.length > 5 ? `... and ${result.warnings.length - 5} more` : undefined,
    })
  },
}

/**
 * Convert Python script to notebook format for validation
 */
function scriptToNotebook(content: string): NotebookContent {
  const cells: NotebookContent['cells'] = []

  // Split by cell markers
  const sections = content.split(/^# %%|^# In\[\d*\]:/m)

  for (const section of sections) {
    if (!section.trim()) continue

    const lines = section.trim().split('\n')

    // Check for markdown markers
    if (lines[0]?.trim() === '[markdown]') {
      cells.push({
        cell_type: 'markdown',
        source: lines.slice(1).map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
        metadata: {},
      })
    } else {
      cells.push({
        cell_type: 'code',
        source: lines.map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
        metadata: {},
        execution_count: null,
        outputs: [],
      })
    }
  }

  // If no cell markers, treat as single code cell
  if (cells.length === 0) {
    cells.push({
      cell_type: 'code',
      source: content.split('\n').map((l, i, arr) => (i < arr.length - 1 ? l + '\n' : l)),
      metadata: {},
      execution_count: null,
      outputs: [],
    })
  }

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
    },
    cells,
  }
}
