/**
 * Upload notebook to Kaggle kernels with versioning support
 */

import { basename, dirname, join } from 'path'
import { z } from 'zod'
import { findCompetitionConfig, loadCompetitionConfig } from '../../lib/config'
import { convertToNotebook, createKernelMetadata, pushKernel } from '../../lib/kaggle'
import { generateVersionedSlug, registerKernelVersion, type VersionedKernel } from '../../lib/kernel-registry'
import { checkServer, logKernelUpload } from '../../lib/mlflow'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'
import type { KernelMetadata } from '../../types/competition'
import { generateVersionedKernelId, incrementVersion, KernelMetadataSchema } from '../../types/competition'

const UploadNotebookArgs = z.object({
  path: z.string().describe('Path to .py or .ipynb file'),
  title: z.string().optional().describe('Kernel base title (default: filename)'),
  gpu: z.boolean().default(true).describe('Enable GPU'),
  internet: z.boolean().optional().describe('Enable internet (default: from config or true)'),
  competition: z.string().optional().describe('Competition slug'),
  datasets: z.string().optional().describe('Dataset sources (comma-separated)'),
  models: z.string().optional().describe('Model sources (comma-separated)'),
  strategy: z
    .enum(['timestamp', 'semver', 'experiment', 'overwrite'])
    .optional()
    .describe('Versioning strategy (default: from config or semver)'),
  noVersion: z.boolean().default(false).describe('Skip versioning, use exact title'),
  model: z.string().optional().describe('Associated model name (for tracking)'),
  notes: z.string().optional().describe('Version notes'),
  dryRun: z.boolean().default(false).describe('Preview version without uploading'),
})

export const uploadNotebook: CommandDefinition<typeof UploadNotebookArgs> = {
  name: 'kaggle upload-notebook',
  description: 'Upload notebook to Kaggle kernels with versioning',
  help: `
Uploads a Python script or Jupyter notebook to Kaggle kernels.
If a .py file is provided, it will be converted to .ipynb using jupytext.

By default, each upload creates a uniquely-named kernel version,
tracked locally in competition.toml and optionally in MLflow.

Versioning Options:
  --strategy   Versioning strategy:
               - semver: nllb-train-v1, nllb-train-v2 (default)
               - timestamp: nllb-train-20241222-143022
               - experiment: nllb-train-exp-01, nllb-train-exp-02
               - overwrite: Use exact title (Kaggle default behavior)
  --no-version Skip versioning, use exact title
  --model      Associated model name (for version tracking)
  --notes      Version notes
  --dry-run    Preview the versioned name without uploading

Other Options:
  --title      Kernel base title (default: filename without extension)
  --gpu        Enable GPU (default: true)
  --internet   Enable internet access (default: from akk.toml kaggle.enable_internet or true)
  --competition  Competition slug for data access
  --datasets   Dataset sources (comma-separated)
  --models     Model sources (comma-separated)
`,
  examples: [
    'akk kaggle upload-notebook train.py  # Creates train-v1, train-v2, etc.',
    'akk kaggle upload-notebook train.py --strategy timestamp  # train-20241222-143022',
    'akk kaggle upload-notebook train.py --dry-run  # Preview next version',
    'akk kaggle upload-notebook train.py --model nllb-v4 --notes "Fixed BLEU eval"',
    'akk kaggle upload-notebook train.py --no-version --title "NLLB Training"',
    'akk kaggle upload-notebook train.ipynb --datasets manwithacat/akkadian-data',
  ],
  args: UploadNotebookArgs,

  async run(args, ctx) {
    const {
      path: notebookPath,
      title,
      gpu,
      internet: internetArg,
      competition,
      datasets,
      models,
      strategy,
      noVersion,
      model,
      notes,
      dryRun,
    } = args
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
      logStep({ step: 'convert', message: 'Converting .py to .ipynb with jupytext...' }, ctx.output)

      ipynbPath = join(dir, `${baseName}.ipynb`)
      const result = await convertToNotebook(fullPath, ipynbPath)

      if (!result.success) {
        return error(
          'CONVERT_FAILED',
          `Failed to convert to notebook: ${result.message}`,
          'Check jupytext is installed',
          {
            path: fullPath,
          }
        )
      }

      codeFile = `${baseName}.ipynb`
    }

    // Determine kernel title with versioning
    let kernelTitle: string
    let versionedKernel: VersionedKernel | null = null
    const kernelBaseName = title || baseName.replace(/[-_]/g, ' ')

    if (noVersion || strategy === 'overwrite') {
      // Use exact title without versioning
      kernelTitle = kernelBaseName
    } else {
      // Check for competition.toml for versioning
      const compConfigPath = await findCompetitionConfig(ctx.cwd)

      if (compConfigPath) {
        // Use kernel registry for versioning
        const versioningStrategy = strategy || 'semver'

        if (dryRun) {
          // Preview mode - show what would be created
          const compConfig = await loadCompetitionConfig(compConfigPath)
          if (compConfig) {
            const normalizedName = kernelBaseName.toLowerCase().replace(/[^a-z0-9]+/g, '-')
            const existingConfig = compConfig.kernels[normalizedName]
            const currentVersion = existingConfig?.current_version || 0
            const nextVersion = currentVersion + 1
            const separator = compConfig.competition.kaggle?.kernel_versioning?.prefix_separator || '-'

            const previewSlug = generateVersionedSlug(kernelBaseName, nextVersion, versioningStrategy, separator)

            return success({
              dryRun: true,
              baseName: kernelBaseName,
              currentVersion,
              nextVersion,
              previewSlug,
              previewKernelId: `${username}/${previewSlug}`,
              strategy: versioningStrategy,
              message: `Would create: ${username}/${previewSlug}`,
            })
          }
        }

        logStep({ step: 'version', message: 'Registering kernel version...' }, ctx.output)

        try {
          const registration = await registerKernelVersion(
            {
              baseName: kernelBaseName,
              strategy: versioningStrategy,
              username,
              model,
              notes,
            },
            ctx.cwd
          )

          versionedKernel = registration.kernel
          kernelTitle = versionedKernel.slug.replace(/-/g, ' ')
        } catch (_err) {
          // Fallback to simple versioning if registry fails
          logStep(
            {
              step: 'version',
              message: 'Registry unavailable, using simple versioning',
            },
            ctx.output
          )
          kernelTitle = kernelBaseName
        }
      } else {
        // No competition.toml - use simple title
        kernelTitle = kernelBaseName
      }
    }

    // Check for existing metadata file with versioning config
    const existingMetadataPath = fullPath.replace(/\.(py|ipynb)$/, '-metadata.json')
    const existingMetadataFile = Bun.file(existingMetadataPath)
    let existingMetadata: KernelMetadata | null = null

    if (await existingMetadataFile.exists()) {
      try {
        const rawMetadata = await existingMetadataFile.json()
        const parsed = KernelMetadataSchema.safeParse(rawMetadata)
        if (parsed.success) {
          existingMetadata = parsed.data
          logStep({ step: 'metadata', message: `Found metadata: ${existingMetadataPath}` }, ctx.output)
        }
      } catch {
        // Ignore parse errors, will create new metadata
      }
    }

    // Determine kernel ID based on versioning config
    let finalKernelId: string
    let finalTitle: string

    if (existingMetadata?.versioning && !existingMetadata.versioning.use_kaggle_versioning) {
      // Use metadata versioning - embed version in kernel ID
      const baseName =
        existingMetadata.versioning.base_name || existingMetadata.title.replace(/\s*v?\d+\.\d+\.\d+\s*$/i, '').trim()
      const { id, slug } = generateVersionedKernelId(username, baseName, existingMetadata.versioning)
      finalKernelId = id
      finalTitle = slug

      logStep(
        {
          step: 'version',
          message: `Using metadata versioning: ${id} (v${existingMetadata.versioning.current_version})`,
        },
        ctx.output
      )
    } else if (versionedKernel) {
      // Use registry versioning
      finalKernelId = versionedKernel.kernelId
      finalTitle = versionedKernel.slug
    } else {
      // Fallback to simple naming
      finalKernelId = `${username}/${kernelTitle.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
      finalTitle = kernelTitle
    }

    // Create metadata using determined kernel ID
    // Priority for competition: command arg > existing metadata > config
    const competitionSlug = competition || existingMetadata?.competition_sources?.[0] || config?.kaggle?.competition

    // Priority for internet: command arg > existing metadata > config > default true
    const effectiveInternet =
      internetArg !== undefined
        ? internetArg
        : existingMetadata?.enable_internet !== undefined
          ? existingMetadata.enable_internet
          : (config?.kaggle?.enable_internet ?? true)

    const metadata = createKernelMetadata({
      username,
      title: finalTitle,
      codeFile,
      enableGpu: gpu,
      enableInternet: effectiveInternet,
      competition: competitionSlug,
      datasets: existingMetadata?.dataset_sources || datasets?.split(',').map((s) => s.trim()),
      models: existingMetadata?.model_sources || models?.split(',').map((s) => s.trim()),
    })

    // Override ID if we computed it from versioning
    if (existingMetadata?.versioning && !existingMetadata.versioning.use_kaggle_versioning) {
      metadata.id = finalKernelId
    }

    // Write metadata file
    const metadataPath = join(dir, 'kernel-metadata.json')
    await Bun.write(metadataPath, JSON.stringify(metadata, null, 2))

    logStep({ step: 'upload', message: `Uploading to Kaggle kernels...` }, ctx.output)

    // Push to Kaggle
    const result = await pushKernel(dir)

    if (!result.success) {
      return error(
        'UPLOAD_FAILED',
        `Failed to upload kernel: ${result.message}`,
        'Check kaggle credentials and API quota',
        {
          metadata,
        }
      )
    }

    // Auto-increment version in metadata file after successful upload
    if (existingMetadata?.versioning && !existingMetadata.versioning.use_kaggle_versioning) {
      const currentVersion = existingMetadata.versioning.current_version
      const nextVersion = incrementVersion(currentVersion, existingMetadata.versioning.strategy, 'patch')

      const updatedMetadata = {
        ...existingMetadata,
        versioning: {
          ...existingMetadata.versioning,
          current_version: nextVersion,
        },
      }

      await Bun.write(existingMetadataPath, JSON.stringify(updatedMetadata, null, 2))
      logStep(
        {
          step: 'version',
          message: `Auto-incremented version: ${currentVersion} -> ${nextVersion}`,
        },
        ctx.output
      )
    }

    // Build response with version info
    const response: Record<string, unknown> = {
      kernel: metadata.id,
      title: versionedKernel?.slug || kernelTitle,
      codeFile,
      gpu,
      internet: effectiveInternet,
      competition: metadata.competition_sources?.[0],
      message: result.message.trim(),
    }

    // Add version tracking info if available
    if (versionedKernel) {
      response.versioning = {
        baseName: versionedKernel.baseName,
        version: versionedKernel.version,
        strategy: strategy || 'semver',
        timestamp: versionedKernel.timestamp,
      }
      if (model) {
        response.versioning = { ...(response.versioning as object), model }
      }
      if (notes) {
        response.versioning = { ...(response.versioning as object), notes }
      }

      // Check for competition config to see if MLflow tracking is enabled
      const compConfigPath = await findCompetitionConfig(ctx.cwd)
      if (compConfigPath) {
        const compConfig = await loadCompetitionConfig(compConfigPath)
        const trackInMlflow = compConfig?.competition.kaggle?.kernel_versioning?.track_in_mlflow ?? true

        if (trackInMlflow) {
          // Check if MLflow server is running
          const mlflowPort = compConfig?.mlflow?.port || config?.mlflow?.port || 5001
          const mlflowRunning = await checkServer(mlflowPort)

          if (mlflowRunning) {
            logStep({ step: 'mlflow', message: 'Logging kernel to MLflow...' }, ctx.output)

            const mlflowRun = await logKernelUpload(mlflowPort, {
              kernelId: versionedKernel.kernelId,
              baseName: versionedKernel.baseName,
              version: versionedKernel.version,
              strategy: strategy || 'semver',
              timestamp: versionedKernel.timestamp,
              model,
              notes,
              gpu,
              internet: effectiveInternet,
              competition: competition || config?.kaggle?.competition,
            })

            if (mlflowRun) {
              response.mlflow = {
                experimentId: mlflowRun.experimentId,
                runId: mlflowRun.runId,
                runName: mlflowRun.runName,
              }
            }
          }
        }
      }
    }

    return success(response)
  },
}
