/**
 * Download model from GCS
 */

import { basename, join } from 'path'
import { z } from 'zod'
import { download, getSize, listFiles, rsync } from '../../lib/gcs'
import { error, logStep, progress, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const DownloadModelArgs = z.object({
  path: z.string().describe('GCS path (gs://bucket/path)'),
  output: z.string().optional().describe('Local output directory'),
  sync: z.boolean().default(false).describe('Use rsync instead of cp'),
  force: z.boolean().default(false).describe('Overwrite existing files'),
})

export const downloadModel: CommandDefinition<typeof DownloadModelArgs> = {
  name: 'colab download-model',
  description: 'Download model from GCS',
  help: `
Downloads a model from Google Cloud Storage.
Supports both single files and directories.

Options:
  --output   Local output directory (default: ./models/{name})
  --sync     Use rsync for efficient syncing (good for large models)
  --force    Overwrite existing files
`,
  examples: [
    'akk colab download-model gs://akkadian-byt5-train/output/final',
    'akk colab download-model gs://my-bucket/model --output ./models/nllb-v2',
    'akk colab download-model gs://my-bucket/model --sync',
  ],
  args: DownloadModelArgs,

  async run(args, ctx) {
    const { path: gcsPath, output: outputPath, sync, force } = args
    const config = ctx.config

    // Validate GCS path
    if (!gcsPath.startsWith('gs://')) {
      return error('INVALID_PATH', 'GCS path must start with gs://', 'Example: gs://bucket/path/to/model', {
        path: gcsPath,
      })
    }

    // Determine output directory
    const modelName = basename(gcsPath.replace(/\/$/, ''))
    const modelsDir = config?.paths?.models || 'models'
    const outDir = outputPath || join(ctx.cwd, modelsDir, modelName)

    // Check if output exists
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

    // Get size info
    logStep({ step: 'size', message: 'Checking model size...' }, ctx.output)
    const size = await getSize(gcsPath)
    const sizeStr = size ? `${(size / 1024 / 1024 / 1024).toFixed(2)} GB` : 'unknown'

    // List files
    logStep({ step: 'list', message: 'Listing files...' }, ctx.output)
    const files = await listFiles(gcsPath)

    if (files.length === 0) {
      return error('NO_FILES', `No files found at ${gcsPath}`, 'Check the GCS path is correct', { path: gcsPath })
    }

    // Create output directory
    await Bun.spawn(['mkdir', '-p', outDir]).exited

    // Download
    logStep(
      {
        step: 'download',
        message: `Downloading ${files.length} files (${sizeStr}) to ${outDir}...`,
      },
      ctx.output
    )

    let result
    if (sync) {
      result = await rsync(gcsPath, outDir, { delete: force })
    } else {
      result = await download(gcsPath + '/*', outDir, { recursive: true })
    }

    if (!result.success) {
      return error('DOWNLOAD_FAILED', `Failed to download: ${result.message}`, 'Check GCS permissions and path', {
        path: gcsPath,
        output: outDir,
      })
    }

    // List downloaded files
    const lsProc = Bun.spawn(['ls', '-la', outDir], { stdout: 'pipe' })
    const lsOutput = await new Response(lsProc.stdout).text()
    const downloadedFiles = lsOutput
      .trim()
      .split('\n')
      .slice(1)
      .map((line) => {
        const parts = line.split(/\s+/)
        return parts.slice(-1)[0]
      })
      .filter((f) => f && f !== '.' && f !== '..')

    return success({
      source: gcsPath,
      destination: outDir,
      size: sizeStr,
      files: downloadedFiles,
      method: sync ? 'rsync' : 'cp',
    })
  },
}
