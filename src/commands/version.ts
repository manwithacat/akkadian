/**
 * Version command
 */

import { z } from 'zod'
import { success } from '../lib/output'
import type { CommandDefinition } from '../types/commands'

const VersionArgs = z.object({})

export const version: CommandDefinition<typeof VersionArgs> = {
  name: 'version',
  description: 'Show version information',
  examples: ['akk version'],
  args: VersionArgs,

  async run(_args, _ctx) {
    const pkg = await import('../../package.json')

    return success({
      name: pkg.name,
      version: pkg.version,
      bun: Bun.version,
      platform: process.platform,
      arch: process.arch,
    })
  },
}
