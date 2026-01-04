/**
 * Check Kaggle kernel status
 */

import { z } from 'zod'
import { getKernelStatus, normalizeKernelInput } from '../../lib/kaggle'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const StatusArgs = z.object({
  path: z.string().describe('Kernel slug (user/kernel-name) or Kaggle URL'),
})

export const status: CommandDefinition<typeof StatusArgs> = {
  name: 'kaggle status',
  description: 'Check kernel execution status',
  help: `
Checks the current status of a Kaggle kernel.

Accepts either:
  - Kernel slug: manwithacat/nllb-train
  - Kaggle URL: https://www.kaggle.com/code/manwithacat/nllb-train

Possible statuses:
  - queued: Waiting to run
  - running: Currently executing
  - complete: Finished successfully
  - error: Failed with error
  - cancelled: Manually cancelled
`,
  examples: [
    'akk kaggle status manwithacat/nllb-train',
    'akk kaggle status https://www.kaggle.com/code/manwithacat/nllb-train?scriptVersionId=123',
  ],
  args: StatusArgs,

  async run(args, ctx) {
    const { path: input } = args

    // Parse URL or slug
    const parsed = normalizeKernelInput(input)
    if (!parsed) {
      return error(
        'INVALID_INPUT',
        'Could not parse kernel identifier',
        'Provide a kernel slug (user/kernel-name) or a Kaggle URL',
        { input }
      )
    }

    const { slug, versionId } = parsed

    try {
      const status = await getKernelStatus(slug)

      return success({
        slug,
        versionId: versionId || null,
        status: status.status,
        failureMessage: status.failureMessage || null,
        url: `https://www.kaggle.com/code/${slug}${versionId ? `?scriptVersionId=${versionId}` : ''}`,
      })
    } catch (err) {
      return error('STATUS_ERROR', err instanceof Error ? err.message : String(err), 'Check kernel slug is correct', {
        slug,
      })
    }
  },
}
