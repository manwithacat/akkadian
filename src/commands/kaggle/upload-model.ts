/**
 * Upload model to Kaggle Models
 */

import { basename, join } from 'path'
import { z } from 'zod'
import { createModel, createModelInstance, initModel, type ModelMetadata } from '../../lib/kaggle'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const UploadModelArgs = z.object({
  path: z.string().describe('Path to model directory'),
  slug: z.string().optional().describe('Model slug (default: directory name)'),
  title: z.string().optional().describe('Model title'),
  notes: z.string().optional().describe('Version notes'),
  private: z.boolean().default(true).describe('Make model private'),
  new: z.boolean().default(false).describe('Create new model (vs new version)'),
})

export const uploadModel: CommandDefinition<typeof UploadModelArgs> = {
  name: 'kaggle upload-model',
  description: 'Upload model to Kaggle Models',
  help: `
Uploads a model directory to Kaggle Models.
If the model already exists, creates a new version.
Use --new to create a new model.

Options:
  --slug     Model slug (default: directory name)
  --title    Model title (required for new models)
  --notes    Version notes
  --private  Make model private (default: true)
  --new      Create new model instead of new version
`,
  examples: [
    'akk kaggle upload-model ./models/nllb-akkadian-v1',
    'akk kaggle upload-model ./models/nllb --slug nllb-akkadian --title "NLLB Akkadian v1" --new',
    'akk kaggle upload-model ./models/nllb --notes "Improved training with dynamic length"',
  ],
  args: UploadModelArgs,

  async run(args, ctx) {
    const { path: modelPath, slug, title, notes, private: isPrivate, new: isNew } = args
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
    const fullPath = modelPath.startsWith('/') ? modelPath : join(ctx.cwd, modelPath)

    // Check directory exists
    const stat = await Bun.file(fullPath).exists()
    // For directories, we check differently
    const proc = Bun.spawn(['test', '-d', fullPath])
    await proc.exited
    const isDir = (await proc.exited) === 0

    if (!isDir) {
      return error('DIR_NOT_FOUND', `Directory not found: ${fullPath}`, 'Provide a valid model directory', {
        path: fullPath,
      })
    }

    const modelSlug = slug || basename(fullPath)
    const modelTitle = title || modelSlug.replace(/[-_]/g, ' ')

    // Check if metadata exists
    const metadataPath = join(fullPath, 'dataset-metadata.json')
    const metadataExists = await Bun.file(metadataPath).exists()

    if (isNew || !metadataExists) {
      // Initialize model metadata
      logStep({ step: 'init', message: 'Initializing model metadata...' }, ctx.output)

      const metadata: ModelMetadata = {
        ownerSlug: username,
        slug: modelSlug,
        title: modelTitle,
        subtitle: notes || `${modelTitle} for Akkadian translation`,
        isPrivate,
        description: `Model trained for the Babylonian Engine Efficiency Challenge`,
      }

      await Bun.write(metadataPath, JSON.stringify(metadata, null, 2))

      if (isNew) {
        logStep({ step: 'create', message: 'Creating new model on Kaggle...' }, ctx.output)

        const result = await createModel(fullPath)
        if (!result.success) {
          return error('CREATE_FAILED', `Failed to create model: ${result.message}`, 'Check if model slug is unique', {
            metadata,
          })
        }

        return success({
          model: `${username}/${modelSlug}`,
          title: modelTitle,
          isNew: true,
          message: result.message.trim(),
        })
      }
    }

    // Create new version
    logStep({ step: 'upload', message: 'Creating new model version...' }, ctx.output)

    const result = await createModelInstance(fullPath, notes)
    if (!result.success) {
      return error(
        'UPLOAD_FAILED',
        `Failed to create model version: ${result.message}`,
        'Check model exists on Kaggle',
        {
          slug: modelSlug,
        }
      )
    }

    return success({
      model: `${username}/${modelSlug}`,
      title: modelTitle,
      notes,
      message: result.message.trim(),
    })
  },
}
