/**
 * Retrieve and display Kaggle kernel logs
 */

import { mkdtemp, rm } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { z } from 'zod'
import { downloadKernelOutput, getKernelStatus } from '../../lib/kaggle'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

interface LogEntry {
  stream_name: 'stdout' | 'stderr'
  time: number
  data: string
}

const LogsArgs = z.object({
  path: z.string().describe('Kernel slug (user/kernel-name)'),
  stream: z.enum(['all', 'stdout', 'stderr']).default('all').describe('Filter by stream'),
  errors: z.boolean().default(false).describe('Show only error-related lines'),
  raw: z.boolean().default(false).describe('Show raw JSON output'),
  tail: z.number().optional().describe('Show only last N lines'),
  save: z.string().optional().describe('Save parsed log to file'),
})

export const logs: CommandDefinition<typeof LogsArgs> = {
  name: 'kaggle logs',
  description: 'Retrieve and display kernel execution logs',
  help: `
Downloads and displays the execution log from a Kaggle kernel.
Automatically parses the JSON log format into readable output.

Options:
  --stream   Filter by stream: all, stdout, stderr (default: all)
  --errors   Show only error-related lines (exceptions, tracebacks)
  --raw      Show raw JSON format instead of parsed
  --tail N   Show only last N lines
  --save     Save parsed log to file
`,
  examples: [
    'akk kaggle logs manwithacat/nllb-train',
    'akk kaggle logs manwithacat/nllb-train --errors',
    'akk kaggle logs manwithacat/nllb-train --stream stderr --tail 50',
    'akk kaggle logs manwithacat/nllb-train --save ./logs/run.log',
  ],
  args: LogsArgs,

  async run(args, ctx) {
    const { path: slug, stream, errors, raw, tail, save } = args

    if (!slug.includes('/')) {
      return error(
        'INVALID_SLUG',
        'Kernel slug must be in format "user/kernel-name"',
        'Example: manwithacat/nllb-train',
        { slug }
      )
    }

    // Check kernel status first
    logStep({ step: 'status', message: `Checking kernel: ${slug}...` }, ctx.output)

    try {
      const status = await getKernelStatus(slug)

      // Create temp directory for download
      const tempDir = await mkdtemp(join(tmpdir(), 'akk-logs-'))

      try {
        logStep({ step: 'download', message: 'Downloading logs...' }, ctx.output)

        const result = await downloadKernelOutput(slug, tempDir)

        if (!result.success) {
          return error(
            'DOWNLOAD_FAILED',
            `Failed to download logs: ${result.message}`,
            'Kernel may still be running or have no output',
            { slug }
          )
        }

        // Find log file
        const kernelName = slug.split('/')[1]
        const logPath = join(tempDir, `${kernelName}.log`)
        const logFile = Bun.file(logPath)

        if (!(await logFile.exists())) {
          return error(
            'NO_LOG_FILE',
            'No log file found in kernel output',
            'Kernel may not have produced any output yet',
            { slug, tempDir }
          )
        }

        const logContent = await logFile.text()

        // Return raw if requested
        if (raw) {
          if (save) {
            await Bun.write(save, logContent)
          }
          return success({
            slug,
            status: status.logStep,
            format: 'raw',
            content: logContent,
            saved: save || null,
          })
        }

        // Parse JSON log entries
        let entries: LogEntry[] = []
        try {
          // Log format is a JSON array
          const parsed = JSON.parse(logContent)
          entries = Array.isArray(parsed) ? parsed : [parsed]
        } catch {
          // Try line-by-line parsing (some logs may have trailing commas)
          const cleaned = logContent
            .replace(/^\[/, '')
            .replace(/\]$/, '')
            .split('\n')
            .filter((line) => line.trim() && line.trim() !== ',')
            .map((line) => line.replace(/^,/, '').replace(/,$/, ''))

          for (const line of cleaned) {
            try {
              entries.push(JSON.parse(line))
            } catch {
              // Skip unparseable lines
            }
          }
        }

        // Filter by stream
        if (stream !== 'all') {
          entries = entries.filter((e) => e.stream_name === stream)
        }

        // Filter for errors if requested
        if (errors) {
          const errorPatterns = [
            /error/i,
            /exception/i,
            /traceback/i,
            /failed/i,
            /--->/,
            /RuntimeError/,
            /ValueError/,
            /TypeError/,
            /AttributeError/,
            /ImportError/,
            /ModuleNotFoundError/,
          ]
          entries = entries.filter((e) => errorPatterns.some((pattern) => pattern.test(e.data)))
        }

        // Apply tail limit
        if (tail && tail > 0) {
          entries = entries.slice(-tail)
        }

        // Format output
        const lines = entries.map((e) => e.data.replace(/\n$/, ''))
        const output = lines.join('\n')

        // Save if requested
        if (save) {
          await Bun.write(save, output)
        }

        // Build summary
        const summary = {
          slug,
          status: status.logStep,
          totalEntries: entries.length,
          streams: {
            stdout: entries.filter((e) => e.stream_name === 'stdout').length,
            stderr: entries.filter((e) => e.stream_name === 'stderr').length,
          },
          filter: stream !== 'all' ? stream : null,
          errorsOnly: errors,
          tail: tail || null,
          saved: save || null,
        }

        return success({
          ...summary,
          log: output,
        })
      } finally {
        // Cleanup temp directory
        await rm(tempDir, { recursive: true, force: true })
      }
    } catch (err) {
      return error('LOGS_ERROR', err instanceof Error ? err.message : String(err), 'Check kernel slug is correct', {
        slug,
      })
    }
  },
}
