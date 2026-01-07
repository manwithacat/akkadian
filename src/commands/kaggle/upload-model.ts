/**
 * Upload model to Kaggle Models (proper 3-tier structure)
 *
 * Kaggle Models hierarchy:
 * - Model: Top-level container (e.g., "nllb-akkadian")
 * - Instance: Framework-specific variant (e.g., "pytorch/annotated-v1")
 * - Version: Actual model files
 */

import { basename, join } from 'path'
import { z } from 'zod'
import {
  createModel,
  createModelInstance,
  createModelVersion,
  listModels,
  MODEL_GUIDANCE,
  type ModelInstanceMetadata,
  type ModelMetadata,
} from '../../lib/kaggle'
import { error, info, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const UploadModelArgs = z.object({
  path: z.string().describe('Path to model directory'),
  model: z.string().optional().describe('Model slug (e.g., nllb-akkadian)'),
  instance: z.string().optional().describe('Instance slug (e.g., annotated-v1)'),
  framework: z.enum(['PyTorch', 'TensorFlow', 'JAX', 'Flax', 'Other']).default('PyTorch').describe('ML framework'),
  title: z.string().optional().describe('Model title (for new models)'),
  notes: z.string().optional().describe('Version notes'),
  private: z.boolean().default(false).describe('Make model private'),
  baseModelUrl: z.string().optional().describe('Base model URL (e.g., huggingface.co/facebook/nllb-200-1.3B)'),
  init: z.boolean().default(false).describe('Initialize metadata templates only'),
  guidance: z.boolean().default(false).describe('Show model structure guidance'),
})

export const uploadModel: CommandDefinition<typeof UploadModelArgs> = {
  name: 'kaggle upload-model',
  description: 'Upload model to Kaggle Models (3-tier: Model → Instance → Version)',
  help: `
Uploads a model to Kaggle Models using the proper 3-tier structure:

  Model (container) → Instance (framework) → Version (files)

This is the CORRECT way to upload ML models to Kaggle. Do NOT use
datasets for model weights.

Options:
  --model      Model slug (default: directory name)
  --instance   Instance slug (default: "default")
  --framework  Framework: PyTorch, TensorFlow, JAX, Flax, Other
  --title      Model title (required for new models)
  --notes      Version notes
  --private    Make model private
  --base-model-url  URL to base model (for fine-tuned models)
  --init       Initialize metadata templates without uploading
  --guidance   Show detailed guidance on model structure
`,
  examples: [
    'akk kaggle upload-model ./models/nllb-v1 --model nllb-akkadian --instance annotated-v1',
    'akk kaggle upload-model ./models/nllb-v1 --init  # Create metadata templates',
    'akk kaggle upload-model ./models/nllb-v1 --notes "Improved training"  # New version',
    'akk kaggle upload-model --guidance  # Show model structure guide',
  ],
  args: UploadModelArgs,

  async run(args, ctx) {
    // Show guidance if requested
    if (args.guidance) {
      return success({
        guidance: MODEL_GUIDANCE,
        message: 'Use --init to create metadata templates, then edit and upload',
      })
    }

    const { path: modelPath, framework, notes, private: isPrivate, baseModelUrl, init: initOnly } = args
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
    const proc = Bun.spawn(['test', '-d', fullPath])
    await proc.exited
    const isDir = (await proc.exited) === 0

    if (!isDir) {
      return error('DIR_NOT_FOUND', `Directory not found: ${fullPath}`, 'Provide a valid model directory', {
        path: fullPath,
      })
    }

    const modelSlug =
      args.model ||
      basename(fullPath)
        .replace(/[^a-z0-9-]/gi, '-')
        .toLowerCase()
    const instanceSlug = args.instance || 'default'
    const modelTitle = args.title || modelSlug.replace(/[-_]/g, ' ')

    // Check for existing metadata files
    const modelMetadataPath = join(fullPath, 'model-metadata.json')
    const instanceMetadataPath = join(fullPath, 'model-instance-metadata.json')
    const hasModelMetadata = await Bun.file(modelMetadataPath).exists()
    const hasInstanceMetadata = await Bun.file(instanceMetadataPath).exists()

    // Initialize templates if requested or missing
    if (initOnly || (!hasModelMetadata && !hasInstanceMetadata)) {
      logStep({ step: 'init', message: 'Creating metadata templates...' }, ctx.output)

      // Create model metadata
      const modelMetadata: ModelMetadata = {
        ownerSlug: username,
        slug: modelSlug,
        title: modelTitle,
        subtitle: `Fine-tuned model for Akkadian translation`,
        isPrivate,
        description: `# Model Summary\n\nFine-tuned model for the Deep Past Initiative Machine Translation competition.\n\n# Training Details\n\n- Base Model: ${baseModelUrl || 'N/A'}\n- Framework: ${framework}`,
        provenanceSources: baseModelUrl,
      }
      await Bun.write(modelMetadataPath, JSON.stringify(modelMetadata, null, 2))

      // Create instance metadata
      const instanceMetadata: ModelInstanceMetadata = {
        ownerSlug: username,
        modelSlug: modelSlug,
        instanceSlug: instanceSlug,
        framework: framework,
        overview: `${modelTitle} - ${framework} implementation`,
        usage: `# Model Format\n\nHuggingFace Transformers (SafeTensors format)\n\n# Model Usage\n\n\`\`\`python\nfrom transformers import AutoModelForSeq2SeqLM, AutoTokenizer\n\nmodel = AutoModelForSeq2SeqLM.from_pretrained("path/to/model")\ntokenizer = AutoTokenizer.from_pretrained("path/to/model")\n\`\`\``,
        licenseName: 'CC BY 4.0',
        fineTunable: true,
        trainingData: [],
        modelInstanceType: baseModelUrl ? 'external' : 'unspecified',
        externalBaseModelUrl: baseModelUrl,
      }
      await Bun.write(instanceMetadataPath, JSON.stringify(instanceMetadata, null, 2))

      if (initOnly) {
        return success({
          model: modelSlug,
          instance: instanceSlug,
          framework,
          files: [modelMetadataPath, instanceMetadataPath],
          message: 'Metadata templates created. Edit them, add model files, then run without --init to upload.',
          nextStep: `akk kaggle upload-model ${modelPath}`,
        })
      }
    }

    // Determine upload action based on what exists
    const instancePath = `${username}/${modelSlug}/${framework.toLowerCase()}/${instanceSlug}`

    // Check if model exists
    logStep({ step: 'check', message: 'Checking existing model...' }, ctx.output)
    const modelsResult = await listModels(username)
    const modelExists = modelsResult.models?.includes(`${username}/${modelSlug}`)

    if (!modelExists) {
      // Create new model first
      logStep({ step: 'create-model', message: `Creating model: ${modelSlug}...` }, ctx.output)

      const createResult = await createModel(fullPath)
      if (!createResult.success) {
        return error('CREATE_MODEL_FAILED', `Failed to create model: ${createResult.message}`, 'Check model metadata', {
          modelSlug,
          metadata: modelMetadataPath,
        })
      }

      info(`Model created: https://www.kaggle.com/models/${username}/${modelSlug}`, ctx.output)
    }

    // Create or update instance
    logStep({ step: 'upload', message: `Uploading instance: ${instancePath}...` }, ctx.output)

    // Try to create instance (will fail if exists, then we create version)
    const instanceResult = await createModelInstance(fullPath, notes)

    if (!instanceResult.success) {
      // Instance might exist, try creating a new version
      if (instanceResult.message.includes('already exists') || instanceResult.message.includes('duplicate')) {
        logStep({ step: 'version', message: 'Instance exists, creating new version...' }, ctx.output)

        const versionResult = await createModelVersion(instancePath, fullPath, notes)
        if (!versionResult.success) {
          return error(
            'VERSION_FAILED',
            `Failed to create version: ${versionResult.message}`,
            'Check instance path and files',
            { instancePath }
          )
        }

        return success({
          model: `${username}/${modelSlug}`,
          instance: instancePath,
          framework,
          action: 'new_version',
          notes,
          message: versionResult.message.trim(),
          url: `https://www.kaggle.com/models/${username}/${modelSlug}`,
        })
      }

      return error(
        'INSTANCE_FAILED',
        `Failed to create instance: ${instanceResult.message}`,
        'Check instance metadata',
        {
          instancePath,
        }
      )
    }

    return success({
      model: `${username}/${modelSlug}`,
      instance: instancePath,
      framework,
      action: 'new_instance',
      notes,
      message: instanceResult.message.trim(),
      url: `https://www.kaggle.com/models/${username}/${modelSlug}`,
      kernelReference: {
        model_sources: [instancePath],
      },
    })
  },
}
