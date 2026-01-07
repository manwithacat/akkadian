/**
 * Prepare notebook for Colab - copy to Downloads with proper setup
 */

import { copyFileSync, existsSync } from 'fs'
import { basename, join } from 'path'
import { z } from 'zod'
import { error, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const PrepareArgs = z.object({
  notebook: z.string().describe('Notebook path (.py or .ipynb)'),
  dest: z.string().optional().describe('Destination directory (default: ~/Downloads)'),
  sync: z.boolean().default(true).describe('Sync .py to .ipynb first'),
})

export const prepare: CommandDefinition<typeof PrepareArgs> = {
  name: 'workflow prepare',
  description: 'Prepare notebook for Colab upload (copy to Downloads)',
  help: `
Prepares a training notebook for manual Colab upload:

1. Syncs .py to .ipynb (if --sync)
2. Copies .ipynb to ~/Downloads (or custom dest)
3. Displays ready-to-run message

Examples:
  akk workflow prepare --notebook notebooks/colab/nllb_train_v5.py
  akk workflow prepare --notebook nllb_train_v5.py --dest /tmp
`,
  examples: [
    'akk workflow prepare --notebook notebooks/colab/nllb_train_v5.py',
    'akk workflow prepare --notebook nllb_train_v5_enriched.py',
  ],
  args: PrepareArgs,

  async run(args, ctx) {
    const notebookPath = args.notebook.startsWith('/') ? args.notebook : join(ctx.cwd, args.notebook)

    // Determine paths
    const isPy = notebookPath.endsWith('.py')
    const pyPath = isPy ? notebookPath : notebookPath.replace('.ipynb', '.py')
    const ipynbPath = isPy ? notebookPath.replace('.py', '.ipynb') : notebookPath

    // Check source exists
    if (!existsSync(isPy ? pyPath : ipynbPath)) {
      return error('NOT_FOUND', `Notebook not found: ${isPy ? pyPath : ipynbPath}`, 'Check path', {})
    }

    // Sync if needed
    if (args.sync && isPy) {
      console.log('Syncing .py to .ipynb...')
      const proc = Bun.spawn(['jupytext', '--sync', pyPath], {
        stdout: 'pipe',
        stderr: 'pipe',
      })
      await proc.exited

      if (proc.exitCode !== 0) {
        const stderr = await new Response(proc.stderr).text()
        return error('SYNC_FAILED', 'jupytext sync failed', stderr, {})
      }
    }

    // Copy to destination
    const dest = args.dest || join(process.env.HOME || '/tmp', 'Downloads')
    const destPath = join(dest, basename(ipynbPath))

    if (!existsSync(dest)) {
      return error('DEST_NOT_FOUND', `Destination not found: ${dest}`, 'Create directory first', {})
    }

    copyFileSync(ipynbPath, destPath)

    const notebookName = basename(ipynbPath, '.ipynb')

    return success({
      notebook: destPath,
      message: `Ready for Colab upload`,
      instructions: [
        `1. Upload ${basename(destPath)} to Google Colab`,
        `2. Select GPU runtime (A100 recommended)`,
        `3. Run all cells`,
        `4. Monitor: akk colab status --run ${notebookName}`,
      ],
    })
  },
}
