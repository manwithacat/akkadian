/**
 * Upload notebook to Kaggle kernels
 */

import { z } from 'zod'
import { join, basename, dirname } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'
import { convertToNotebook, createKernelMetadata, pushKernel } from '../../lib/kaggle'

const UploadNotebookArgs = z.object({
  path: z.string().describe('Path to .py or .ipynb file'),
  title: z.string().optional().describe('Kernel title (default: filename)'),
  gpu: z.boolean().default(true).describe('Enable GPU'),
  internet: z.boolean().default(true).describe('Enable internet'),
  competition: z.string().optional().describe('Competition slug'),
  datasets: z.string().optional().describe('Dataset sources (comma-separated)'),
  models: z.string().optional().describe('Model sources (comma-separated)'),
})

export const uploadNotebook: CommandDefinition<typeof UploadNotebookArgs> = {
  name: 'kaggle upload-notebook',
  description: 'Upload notebook to Kaggle kernels',
  help: `
Uploads a Python script or Jupyter notebook to Kaggle kernels.
If a .py file is provided, it will be converted to .ipynb using jupytext.

Options:
  --title      Kernel title (default: filename without extension)
  --gpu        Enable GPU (default: true)
  --internet   Enable internet access (default: true)
  --competition  Competition slug for data access
  --datasets   Dataset sources (comma-separated)
  --models     Model sources (comma-separated)
`,
  examples: [
    'akk kaggle upload-notebook notebooks/colab/nllb_train.py',
    'akk kaggle upload-notebook train.py --title "NLLB Training v2" --competition babylonian-engine-efficiency-challenge',
    'akk kaggle upload-notebook train.ipynb --datasets manwithacat/akkadian-data --models facebook/nllb-200-600M',
  ],
  args: UploadNotebookArgs,

  async run(args, ctx) {
    const { path: notebookPath, title, gpu, internet, competition, datasets, models } = args
    const config = ctx.config

    // Get username from config
    const username = config?.kaggle?.username
    if (!username) {
      return error(
        'NO_KAGGLE_USERNAME',
        'Kaggle username not configured',
        'Add kaggle.username to akk.toml or set KAGGLE_USERNAME',
        { configPath: ctx.configPath }
      )
    }

    // Resolve path
    const fullPath = notebookPath.startsWith('/') ? notebookPath : join(ctx.cwd, notebookPath)

    // Check file exists
    const file = Bun.file(fullPath)
    if (!(await file.exists())) {
      return error('FILE_NOT_FOUND', `File not found: ${fullPath}`, 'Check the file path', { path: fullPath })
    }

    // Determine file type and notebook path
    const fileName = basename(fullPath)
    const ext = fileName.slice(fileName.lastIndexOf('.'))
    const baseName = fileName.slice(0, fileName.lastIndexOf('.'))
    const dir = dirname(fullPath)

    let ipynbPath = fullPath
    let codeFile = fileName

    // Convert .py to .ipynb if needed
    if (ext === '.py') {
      progress({ step: 'convert', message: 'Converting .py to .ipynb with jupytext...' }, ctx.output)

      ipynbPath = join(dir, `${baseName}.ipynb`)
      const result = await convertToNotebook(fullPath, ipynbPath)

      if (!result.success) {
        return error('CONVERT_FAILED', `Failed to convert to notebook: ${result.message}`, 'Check jupytext is installed', {
          path: fullPath,
        })
      }

      codeFile = `${baseName}.ipynb`
    }

    // Create metadata
    const kernelTitle = title || baseName.replace(/[-_]/g, ' ')
    const metadata = createKernelMetadata({
      username,
      title: kernelTitle,
      codeFile,
      enableGpu: gpu,
      enableInternet: internet,
      competition: competition || config?.kaggle?.competition,
      datasets: datasets?.split(',').map((s) => s.trim()),
      models: models?.split(',').map((s) => s.trim()),
    })

    // Write metadata file
    const metadataPath = join(dir, 'kernel-metadata.json')
    await Bun.write(metadataPath, JSON.stringify(metadata, null, 2))

    progress({ step: 'upload', message: `Uploading to kaggle kernels...` }, ctx.output)

    // Push to Kaggle
    const result = await pushKernel(dir)

    if (!result.success) {
      return error('UPLOAD_FAILED', `Failed to upload kernel: ${result.message}`, 'Check kaggle credentials and API quota', {
        metadata,
      })
    }

    return success({
      kernel: metadata.id,
      title: kernelTitle,
      codeFile,
      gpu,
      internet,
      competition: metadata.competition_sources?.[0],
      message: result.message.trim(),
    })
  },
}
