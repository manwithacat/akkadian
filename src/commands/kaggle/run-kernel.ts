/**
 * Run and monitor a Kaggle kernel
 */

import { z } from 'zod'
import { downloadKernelOutput, getKernelStatus, waitForKernel } from '../../lib/kaggle'
import { error, logStep, progress, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const RunKernelArgs = z.object({
  path: z.string().describe('Kernel slug (user/kernel-name)'), // 'path' is first positional arg
  wait: z.boolean().default(false).describe('Wait for completion'),
  interval: z.number().default(30).describe('Poll interval in seconds'),
  timeout: z.number().default(7200).describe('Timeout in seconds'),
  download: z.boolean().default(false).describe('Download output after completion'),
  output: z.string().optional().describe('Output directory for downloads'),
})

export const runKernel: CommandDefinition<typeof RunKernelArgs> = {
  name: 'kaggle run-kernel',
  description: 'Run and monitor a Kaggle kernel',
  help: `
Checks the status of a Kaggle kernel and optionally waits for completion.
Use --wait to poll until the kernel finishes.
Use --download to automatically download outputs after completion.

Options:
  --wait      Wait for kernel completion
  --interval  Poll interval in seconds (default: 30)
  --timeout   Maximum wait time in seconds (default: 7200 = 2 hours)
  --download  Download output after completion
  --output    Output directory for downloads (default: ./output)
`,
  examples: [
    'akk kaggle run-kernel manwithacat/nllb-train',
    'akk kaggle run-kernel manwithacat/nllb-train --wait',
    'akk kaggle run-kernel manwithacat/nllb-train --wait --download --output ./outputs/nllb',
  ],
  args: RunKernelArgs,

  async run(args, ctx) {
    const { path: slug, wait, interval, timeout, download, output: outputDir } = args

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

    logStep({ step: 'status', message: `Checking kernel status: ${slug}...` }, ctx.output)

    try {
      const initialStatus = await getKernelStatus(slug)

      if (!wait) {
        return success({
          slug,
          status: initialStatus.status,
          failureMessage: initialStatus.failureMessage,
        })
      }

      // Wait for completion
      if (
        initialStatus.status === 'complete' ||
        initialStatus.status === 'error' ||
        initialStatus.status === 'cancelled'
      ) {
        if (download && initialStatus.status === 'complete') {
          const outDir = outputDir || './output'
          logStep({ step: 'download', message: `Downloading output to ${outDir}...` }, ctx.output)
          await downloadKernelOutput(slug, outDir)
        }

        return success({
          slug,
          status: initialStatus.status,
          failureMessage: initialStatus.failureMessage,
          downloaded: download && initialStatus.status === 'complete',
        })
      }

      logStep(
        {
          step: 'wait',
          message: `Waiting for kernel completion (polling every ${interval}s, timeout ${timeout}s)...`,
        },
        ctx.output
      )

      const finalStatus = await waitForKernel(slug, {
        interval: interval * 1000,
        timeout: timeout * 1000,
        onStatus: (status) => {
          logStep({ step: 'poll', message: `Status: ${status.status}` }, ctx.output)
        },
      })

      if (download && finalStatus.status === 'complete') {
        const outDir = outputDir || './output'
        logStep({ step: 'download', message: `Downloading output to ${outDir}...` }, ctx.output)
        await downloadKernelOutput(slug, outDir)
      }

      if (finalStatus.status === 'error') {
        return error(
          'KERNEL_FAILED',
          `Kernel failed: ${finalStatus.failureMessage || 'Unknown error'}`,
          'Check kernel logs',
          {
            slug,
            status: finalStatus,
          }
        )
      }

      return success({
        slug,
        status: finalStatus.status,
        failureMessage: finalStatus.failureMessage,
        downloaded: download && finalStatus.status === 'complete',
      })
    } catch (err) {
      return error(
        'STATUS_ERROR',
        err instanceof Error ? err.message : String(err),
        'Check kernel slug is correct and kernel exists',
        { slug }
      )
    }
  },
}
