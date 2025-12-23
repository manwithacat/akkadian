/**
 * Upload notebook to GCS for Colab execution
 */

import { basename, join } from 'path'
import { z } from 'zod'
import { bucketExists, upload } from '../../lib/gcs'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const UploadNotebookArgs = z.object({
  path: z.string().describe('Path to notebook (.ipynb or .py)'),
  name: z.string().optional().describe('Name in GCS (default: filename)'),
  bucket: z.string().optional().describe('GCS bucket (default: from config)'),
  prefix: z.string().default('notebooks').describe('GCS prefix path'),
  version: z.string().optional().describe('Version tag (e.g., v1, v2)'),
})

export const uploadNotebook: CommandDefinition<typeof UploadNotebookArgs> = {
  name: 'colab upload-notebook',
  description: 'Upload notebook to GCS for Colab execution',
  help: `
Uploads a Jupyter notebook or Python script to GCS for use in Google Colab.

The notebook will be uploaded to: gs://{bucket}/{prefix}/{name}

Options:
  --path      Local path to notebook (.ipynb) or Python script (.py)
  --name      Name in GCS (default: same as local filename)
  --bucket    GCS bucket (default: from akk.toml colab.gcs_bucket)
  --prefix    GCS prefix path (default: 'notebooks')
  --version   Version tag (appended to name, e.g., notebook_v2.ipynb)

After uploading, open in Colab:
  https://colab.research.google.com/github or via GCS integration
`,
  examples: [
    'akk colab upload-notebook notebooks/colab/nllb_train.ipynb',
    'akk colab upload-notebook nllb_train.py --version v4',
    'akk colab upload-notebook train.ipynb --bucket my-bucket --prefix training',
  ],
  args: UploadNotebookArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucketName = args.bucket || config?.colab?.gcs_bucket

    if (!bucketName) {
      return error('NO_BUCKET', 'No GCS bucket configured', 'Set colab.gcs_bucket in akk.toml or use --bucket', {})
    }

    // Resolve local path
    const localPath = args.path.startsWith('/') ? args.path : join(ctx.cwd, args.path)
    const file = Bun.file(localPath)

    if (!(await file.exists())) {
      return error('FILE_NOT_FOUND', `File not found: ${localPath}`, 'Check the path', { path: localPath })
    }

    // Build GCS path
    let filename = args.name || basename(localPath)
    if (args.version) {
      const ext = filename.includes('.') ? filename.slice(filename.lastIndexOf('.')) : ''
      const base = filename.includes('.') ? filename.slice(0, filename.lastIndexOf('.')) : filename
      filename = `${base}_${args.version}${ext}`
    }

    const gcsPath = `gs://${bucketName}/${args.prefix}/${filename}`

    // Check bucket exists
    logStep({ step: 'check', message: `Checking bucket: ${bucketName}...` }, ctx.output)
    const exists = await bucketExists(bucketName)
    if (!exists) {
      return error('BUCKET_NOT_FOUND', `Bucket not found: ${bucketName}`, 'Run: akk colab configure --create', {})
    }

    // Upload
    logStep({ step: 'upload', message: `Uploading to ${gcsPath}...` }, ctx.output)
    const result = await upload(localPath, gcsPath)

    if (!result.success) {
      return error('UPLOAD_FAILED', `Upload failed: ${result.message}`, 'Check GCS permissions', {})
    }

    // Generate Colab URL
    const colabUrl = `https://colab.research.google.com/drive` // User needs to open from GCS

    return success({
      uploaded: gcsPath,
      size: file.size,
      colabInstructions: [
        '1. Open Google Colab',
        '2. File > Open notebook > Google Cloud Storage tab',
        `3. Navigate to: ${bucketName}/${args.prefix}/${filename}`,
        '4. Or mount GCS in Colab and open directly',
      ],
      gcsUri: gcsPath,
    })
  },
}
