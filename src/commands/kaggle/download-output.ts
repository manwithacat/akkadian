/**
 * Download kernel output files
 */

import { join } from 'path'
import { z } from 'zod'
import { downloadKernelOutput, getKernelStatus } from '../../lib/kaggle'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const DownloadOutputArgs = z.object({
  path: z.string().describe('Kernel slug (user/kernel-name)'), // 'path' is first positional arg
  output: z.string().optional().describe('Output directory (default: ./output/{kernel})'),
  force: z.boolean().default(false).describe('Overwrite existing files'),
})

export const downloadOutput: CommandDefinition<typeof DownloadOutputArgs> = {
  name: 'kaggle download-output',
  description: 'Download kernel output files',
  help: `
Downloads the output files from a completed Kaggle kernel.

Options:
  --output   Output directory (default: ./output/{kernel-name})
  --force    Overwrite existing files
`,
  examples: [
    'akk kaggle download-output manwithacat/nllb-train',
    'akk kaggle download-output manwithacat/nllb-train --output ./models/nllb-v1',
  ],
  args: DownloadOutputArgs,

  async run(args, ctx) {
    const { path: slug, output: outputPath, force } = args

    if (!slug.includes('/')) {
      return error(
        'INVALID_SLUG',
        'Kernel slug must be in format "user/kernel-name"',
        'Example: manwithacat/nllb-train',
        {
          slug,
        }
      )
    }

    // Check kernel status first
    logStep({ step: 'status', message: `Checking kernel status: ${slug}...` }, ctx.output)

    try {
      const status = await getKernelStatus(slug)

      if (status.status !== 'complete') {
        return error(
          'KERNEL_NOT_COMPLETE',
          `Kernel status is "${status.status}", not complete`,
          'Wait for kernel to complete or use "akk kaggle run-kernel --wait"',
          { slug, logStep }
        )
      }

      // Determine output directory
      const kernelName = slug.split('/')[1]
      const outDir = outputPath || join(ctx.cwd, 'output', kernelName)

      // Check if directory exists and has files
      if (!force) {
        const dirCheck = Bun.spawn(['test', '-d', outDir])
        if ((await dirCheck.exited) === 0) {
          const files = Bun.spawn(['ls', '-A', outDir], { stdout: 'pipe' })
          const output = await new Response(files.stdout).text()
          if (output.trim()) {
            return error(
              'DIR_NOT_EMPTY',
              `Output directory is not empty: ${outDir}`,
              'Use --force to overwrite or specify a different --output path',
              { path: outDir }
            )
          }
        }
      }

      logStep({ step: 'download', message: `Downloading to ${outDir}...` }, ctx.output)

      const result = await downloadKernelOutput(slug, outDir)

      if (!result.success) {
        return error('DOWNLOAD_FAILED', `Failed to download: ${result.message}`, 'Check kernel has output files', {
          slug,
          path: outDir,
        })
      }

      // List downloaded files
      const lsProc = Bun.spawn(['ls', '-la', outDir], { stdout: 'pipe' })
      const lsOutput = await new Response(lsProc.stdout).text()

      return success({
        slug,
        outputDir: outDir,
        files: lsOutput
          .trim()
          .split('\n')
          .slice(1)
          .map((line) => {
            const parts = line.split(/\s+/)
            return parts.slice(-1)[0] // filename
          })
          .filter((f) => f && f !== '.' && f !== '..'),
        message: result.message.trim(),
      })
    } catch (err) {
      return error(
        'DOWNLOAD_ERROR',
        err instanceof Error ? err.message : String(err),
        'Check kernel slug is correct and kernel has output',
        { slug }
      )
    }
  },
}
