/**
 * Competition Init Command
 *
 * Initialize a new competition directory with competition.toml
 * and standard folder structure.
 */

import { z } from 'zod'
import { existsSync } from 'fs'
import { join, resolve } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error } from '../../lib/output'
import { initCompetitionDirectory, findCompetitionConfig } from '../../lib/config'

const InitArgs = z.object({
  path: z.string().optional().describe('Directory to initialize (default: current directory)'),
  name: z.string().optional().describe('Competition name'),
  slug: z.string().optional().describe('Competition slug (Kaggle competition ID)'),
  username: z.string().optional().describe('Kaggle username'),
  force: z.boolean().default(false).describe('Overwrite existing competition.toml'),
})

export const init: CommandDefinition<typeof InitArgs> = {
  name: 'competition init',
  description: 'Initialize a new competition directory',
  help: `
Initialize a new competition directory with:
- competition.toml configuration file
- Standard folder structure (notebooks, models, submissions, datasets, artifacts)

If run without arguments, uses interactive prompts to gather information.
If run in a directory with an existing competition.toml, use --force to overwrite.

The competition slug should match the Kaggle competition ID (e.g., "babylonian-engine-efficiency-challenge").
`,
  examples: [
    'akk competition init',
    'akk competition init ./my-competition',
    'akk competition init --slug babylonian-engine-efficiency-challenge --username myuser',
    'akk competition init --force',
  ],
  args: InitArgs,

  async run(args, ctx) {
    const targetDir = resolve(args.path || process.cwd())

    // Check for existing competition.toml
    const existingConfig = await findCompetitionConfig(targetDir)
    if (existingConfig && !args.force) {
      return error(
        'ALREADY_INITIALIZED',
        `Competition already initialized at ${existingConfig}`,
        'Use --force to overwrite the existing configuration'
      )
    }

    // Get competition details
    const name = args.name || args.slug || 'my-competition'
    const slug = args.slug || name.toLowerCase().replace(/\s+/g, '-')
    const username = args.username || ctx.config?.kaggle?.username || 'unknown'

    // Initialize directory
    try {
      const competitionDir = await initCompetitionDirectory(targetDir, name, slug, username)

      return success({
        message: 'Competition initialized successfully',
        directory: competitionDir.root,
        config: competitionDir.configPath,
        structure: {
          notebooks: competitionDir.notebooks,
          models: competitionDir.models,
          submissions: competitionDir.submissions,
          datasets: competitionDir.datasets,
          artifacts: competitionDir.artifacts,
        },
        next_steps: [
          `cd ${competitionDir.root}`,
          'Edit competition.toml to configure your competition',
          'akk competition status  # View current status',
          'akk template generate training --platform kaggle-p100  # Generate training notebook',
        ],
      })
    } catch (err) {
      return error(
        'INIT_FAILED',
        `Failed to initialize competition: ${err instanceof Error ? err.message : 'Unknown error'}`,
        'Check directory permissions and try again'
      )
    }
  },
}
