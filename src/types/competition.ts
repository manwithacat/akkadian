/**
 * Competition Configuration Types
 *
 * Defines the schema for competition.toml - the directory-based
 * competition configuration file.
 */

import { z } from 'zod'
import type { PlatformId } from './platform'

/**
 * Competition platform types
 */
export type CompetitionPlatform = 'kaggle' | 'drivendata' | 'other'

/**
 * Model status in the competition
 */
export type ModelStatus = 'active' | 'testing' | 'archived'

/**
 * Kernel run status
 */
export type KernelStatus = 'queued' | 'running' | 'complete' | 'error' | 'cancelled'

/**
 * Kaggle-specific competition settings
 */
export const KaggleSettingsSchema = z.object({
  username: z.string(),
  team: z.string().optional(),
  data_sources: z.array(z.string()).default([]),
  model_sources: z.array(z.string()).optional(),
})

export type KaggleSettings = z.infer<typeof KaggleSettingsSchema>

/**
 * Competition metadata
 */
export const CompetitionMetaSchema = z.object({
  name: z.string(),
  slug: z.string(),
  platform: z.enum(['kaggle', 'drivendata', 'other']).default('kaggle'),
  deadline: z.string().optional(),
  metric: z.string().default('bleu'),
  metric_direction: z.enum(['maximize', 'minimize']).default('maximize'),
  kaggle: KaggleSettingsSchema.optional(),
})

export type CompetitionMeta = z.infer<typeof CompetitionMetaSchema>

/**
 * Project metadata
 */
export const ProjectMetaSchema = z.object({
  name: z.string(),
  version: z.string().default('0.1.0'),
})

export type ProjectMeta = z.infer<typeof ProjectMetaSchema>

/**
 * Active model configuration
 */
export const ActiveModelSchema = z.object({
  name: z.string(),
  path: z.string(),
  base: z.string(),
  current_score: z.number().optional(),
  best_score: z.number().optional(),
  best_submission: z.string().optional(),
})

export type ActiveModel = z.infer<typeof ActiveModelSchema>

/**
 * Model configuration in the registry
 */
export const ModelConfigSchema = z.object({
  base: z.string(),
  path: z.string(),
  best_score: z.number().optional(),
  submissions: z.number().default(0),
  status: z.enum(['active', 'testing', 'archived']).default('testing'),
  notes: z.string().optional(),
})

export type ModelConfig = z.infer<typeof ModelConfigSchema>

/**
 * Kernel configuration
 */
export const KernelConfigSchema = z.object({
  slug: z.string(),
  current_version: z.number().default(1),
  last_run: z.string().optional(),
  last_status: z.enum(['queued', 'running', 'complete', 'error', 'cancelled']).optional(),
})

export type KernelConfig = z.infer<typeof KernelConfigSchema>

/**
 * Submission history record
 */
export const SubmissionRecordSchema = z.object({
  id: z.string(),
  timestamp: z.string(),
  model: z.string(),
  kernel_version: z.number().optional(),
  public_score: z.number().optional(),
  private_score: z.number().optional(),
  notes: z.string().optional(),
})

export type SubmissionRecord = z.infer<typeof SubmissionRecordSchema>

/**
 * Submission state
 */
export const SubmissionStateSchema = z.object({
  total: z.number().default(0),
  remaining_today: z.number().optional(),
  best_score: z.number().optional(),
  best_submission_id: z.string().optional(),
  history: z.array(SubmissionRecordSchema).default([]),
})

export type SubmissionState = z.infer<typeof SubmissionStateSchema>

/**
 * Training defaults
 */
export const TrainingDefaultsSchema = z.object({
  default_platform: z.string().default('kaggle-p100'),
  default_epochs: z.number().default(10),
  default_batch_size: z.number().default(2),
  gradient_accumulation_steps: z.number().optional(),
  max_src_len: z.number().optional(),
  max_tgt_len: z.number().optional(),
  learning_rate: z.number().optional(),
  fp16: z.boolean().default(true),
})

export type TrainingDefaults = z.infer<typeof TrainingDefaultsSchema>

/**
 * GCS configuration for Colab/Vertex
 */
export const GCSConfigSchema = z.object({
  bucket: z.string(),
  project: z.string(),
  run_prefix: z.string().default('mlflow/runs'),
  cross_account: z.boolean().default(false),
  service_account_key: z.string().optional(),
})

export type GCSConfig = z.infer<typeof GCSConfigSchema>

/**
 * MLFlow configuration
 */
export const MLFlowConfigSchema = z.object({
  experiment: z.string(),
  tracking_uri: z.string().default('sqlite:///mlflow/mlflow.db'),
  artifact_location: z.string().default('./mlflow/artifacts'),
  port: z.number().default(5001),
})

export type MLFlowConfig = z.infer<typeof MLFlowConfigSchema>

/**
 * Path configuration
 */
export const PathConfigSchema = z.object({
  notebooks: z.string().default('notebooks'),
  models: z.string().default('models'),
  submissions: z.string().default('submissions'),
  datasets: z.string().default('datasets'),
  artifacts: z.string().default('artifacts'),
})

export type PathConfig = z.infer<typeof PathConfigSchema>

/**
 * Full competition configuration schema
 */
export const CompetitionConfigSchema = z.object({
  competition: CompetitionMetaSchema,
  project: ProjectMetaSchema,
  active_model: ActiveModelSchema.optional(),
  models: z.record(z.string(), ModelConfigSchema).default({}),
  kernels: z.record(z.string(), KernelConfigSchema).default({}),
  submissions: SubmissionStateSchema.default({ total: 0, history: [] }),
  training: TrainingDefaultsSchema.default({}),
  gcs: GCSConfigSchema.optional(),
  mlflow: MLFlowConfigSchema.optional(),
  paths: PathConfigSchema.default({}),
})

export type CompetitionConfig = z.infer<typeof CompetitionConfigSchema>

/**
 * Competition directory structure
 */
export interface CompetitionDirectory {
  root: string
  configPath: string
  config: CompetitionConfig
  notebooks: string
  models: string
  submissions: string
  datasets: string
  artifacts: string
}

/**
 * Helper to get absolute paths from competition config
 */
export function getCompetitionPaths(root: string, config: CompetitionConfig): CompetitionDirectory {
  const { paths } = config
  return {
    root,
    configPath: `${root}/competition.toml`,
    config,
    notebooks: `${root}/${paths.notebooks}`,
    models: `${root}/${paths.models}`,
    submissions: `${root}/${paths.submissions}`,
    datasets: `${root}/${paths.datasets}`,
    artifacts: `${root}/${paths.artifacts}`,
  }
}

/**
 * Default competition configuration template
 */
export function createDefaultCompetitionConfig(
  name: string,
  slug: string,
  username: string
): CompetitionConfig {
  return {
    competition: {
      name,
      slug,
      platform: 'kaggle',
      metric: 'bleu',
      metric_direction: 'maximize',
      kaggle: {
        username,
        data_sources: [slug],
      },
    },
    project: {
      name: slug,
      version: '0.1.0',
    },
    models: {},
    kernels: {},
    submissions: {
      total: 0,
      history: [],
    },
    training: {
      default_platform: 'kaggle-p100',
      default_epochs: 10,
      default_batch_size: 2,
      fp16: true,
    },
    paths: {
      notebooks: 'notebooks',
      models: 'models',
      submissions: 'submissions',
      datasets: 'datasets',
      artifacts: 'artifacts',
    },
  }
}
