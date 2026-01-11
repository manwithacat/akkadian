/**
 * Register a new prediction job
 */

import { findCompetitionConfig } from '../../lib/config'
import { printError, printInfo, printJson, printSuccess } from '../../lib/output'
import {
  type CreatePredictionJobOptions,
  createPredictionJob,
  formatPredictionJob,
  registerPredictionJob,
} from '../../lib/prediction'
import type { CommandContext } from '../../types/commands'
import type { ModelSource } from '../../types/competition'

export interface RegisterPredictionOptions {
  model: string
  modelSource?: string
  modelName?: string
  dataset: string
  datasetVersion?: string
  datasetSize?: number
  platform?: string
  trainingRun?: string
  predictions?: string
  metrics?: string
  notes?: string
  json?: boolean
}

export async function register(options: RegisterPredictionOptions, _ctx: CommandContext): Promise<void> {
  // Validate required options
  if (!options.model) {
    printError('--model is required')
    return
  }

  if (!options.dataset) {
    printError('--dataset is required')
    return
  }

  // Find competition config
  const configPath = await findCompetitionConfig()
  if (!configPath) {
    printError("No competition.toml found. Run 'akk competition init' first.")
    return
  }

  // Determine model source
  let modelSource: ModelSource = 'local'
  if (options.modelSource) {
    const validSources = ['kaggle', 'gcs', 'local', 'mlflow']
    if (!validSources.includes(options.modelSource)) {
      printError(`Invalid model source: ${options.modelSource}. Valid: ${validSources.join(', ')}`)
      return
    }
    modelSource = options.modelSource as ModelSource
  } else if (options.model.includes('/') && !options.model.startsWith('/')) {
    // Looks like a Kaggle model path (e.g., "user/model-name")
    modelSource = 'kaggle'
  } else if (options.model.startsWith('gs://')) {
    modelSource = 'gcs'
  } else if (options.model.startsWith('mlflow:')) {
    modelSource = 'mlflow'
  }

  // Create prediction job options
  const jobOptions: CreatePredictionJobOptions = {
    modelPath: options.model,
    modelSource,
    modelName: options.modelName,
    datasetName: options.dataset,
    datasetVersion: options.datasetVersion,
    datasetSize: options.datasetSize,
    platform: options.platform,
    trainingRunId: options.trainingRun,
    notes: options.notes,
  }

  // Create and register the job
  const job = createPredictionJob(jobOptions)

  // Add artifacts if provided
  if (options.predictions || options.metrics) {
    job.artifacts = {
      predictions: options.predictions || '',
      metrics: options.metrics,
    }
  }

  try {
    await registerPredictionJob(job, configPath)

    if (options.json) {
      printJson(job)
    } else {
      printSuccess(`Registered prediction job: ${job.prediction_id}`)
      console.log('\n' + formatPredictionJob(job))
      printInfo('\nUpdate status with: akk predict update <id> --status running|complete|error')
    }
  } catch (err) {
    printError(`Failed to register prediction: ${err instanceof Error ? err.message : String(err)}`)
  }
}
