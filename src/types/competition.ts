/**
 * Competition Configuration Types
 *
 * Defines the schema for competition.toml - the directory-based
 * competition configuration file.
 */

import { z } from 'zod'

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
 * Kernel versioning strategy
 */
export type VersioningStrategy =
  | 'timestamp' // nllb-train-20241222-143022
  | 'semver' // nllb-train-v1, nllb-train-v2
  | 'experiment' // nllb-train-exp-01, nllb-train-exp-02
  | 'overwrite' // default Kaggle behavior (no versioning)

/**
 * Kernel versioning configuration (for competition.toml)
 */
export const KernelVersioningSchema = z.object({
  strategy: z.enum(['timestamp', 'semver', 'experiment', 'overwrite']).default('semver'),
  prefix_separator: z.string().default('-'),
  track_in_mlflow: z.boolean().default(true),
  auto_increment: z.boolean().default(true),
})

export type KernelVersioning = z.infer<typeof KernelVersioningSchema>

/**
 * Notebook-level versioning configuration (for kernel-metadata.json)
 *
 * When use_kaggle_versioning is false, each version creates a unique kernel
 * with the version embedded in the ID (e.g., nllb-inference-v1-0-0).
 * This avoids Kaggle's confusing internal version system.
 */
export const NotebookVersioningSchema = z.object({
  /** If false, embed version in kernel ID instead of using Kaggle's versioning */
  use_kaggle_versioning: z.boolean().default(true),
  /** Versioning strategy: semver embeds major-minor-patch in ID */
  strategy: z.enum(['timestamp', 'semver', 'experiment']).default('semver'),
  /** Current version (semver format: "1.0.0" or number for other strategies) */
  current_version: z.union([z.string(), z.number()]).default('1.0.0'),
  /** Base kernel name (without version suffix) */
  base_name: z.string().optional(),
})

export type NotebookVersioning = z.infer<typeof NotebookVersioningSchema>

/**
 * Extended kernel metadata with versioning support
 */
export const KernelMetadataSchema = z.object({
  id: z.string(),
  title: z.string(),
  code_file: z.string(),
  language: z.string().default('python'),
  kernel_type: z.string().default('script'),
  is_private: z.boolean().default(false),
  enable_gpu: z.boolean().default(true),
  enable_tpu: z.boolean().default(false),
  enable_internet: z.boolean().default(false),
  dataset_sources: z.array(z.string()).default([]),
  competition_sources: z.array(z.string()).default([]),
  kernel_sources: z.array(z.string()).default([]),
  model_sources: z.array(z.string()).default([]),
  /** Versioning configuration - when present, controls how kernel IDs are generated */
  versioning: NotebookVersioningSchema.optional(),
})

export type KernelMetadata = z.infer<typeof KernelMetadataSchema>

/**
 * Generate a versioned kernel ID based on metadata versioning config
 */
export function generateVersionedKernelId(
  username: string,
  baseName: string,
  versioning: NotebookVersioning
): { id: string; slug: string } {
  const baseSlug = baseName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')

  if (versioning.use_kaggle_versioning) {
    // Use simple slug, let Kaggle handle versioning
    return {
      id: `${username}/${baseSlug}`,
      slug: baseSlug,
    }
  }

  // Embed version in the kernel ID
  let versionSuffix: string

  switch (versioning.strategy) {
    case 'semver': {
      // Convert "1.0.0" to "v1-0-0"
      const version = String(versioning.current_version)
      versionSuffix = `v${version.replace(/\./g, '-')}`
      break
    }
    case 'timestamp': {
      const now = new Date()
      versionSuffix = now
        .toISOString()
        .slice(0, 19)
        .replace(/[-:T]/g, '')
        .replace(/(\d{8})(\d{6})/, '$1-$2')
      break
    }
    case 'experiment': {
      const expNum =
        typeof versioning.current_version === 'number'
          ? versioning.current_version
          : parseInt(String(versioning.current_version).split('.')[0], 10) || 1
      versionSuffix = `exp-${String(expNum).padStart(2, '0')}`
      break
    }
    default:
      versionSuffix = `v${String(versioning.current_version).replace(/\./g, '-')}`
  }

  const slug = `${baseSlug}-${versionSuffix}`
  return {
    id: `${username}/${slug}`,
    slug,
  }
}

/**
 * Increment version based on strategy and bump type
 */
export function incrementVersion(
  current: string | number,
  strategy: NotebookVersioning['strategy'],
  bump: 'major' | 'minor' | 'patch' = 'patch'
): string | number {
  if (strategy === 'semver') {
    const parts = String(current).split('.').map(Number)
    const [major = 1, minor = 0, patch = 0] = parts

    switch (bump) {
      case 'major':
        return `${major + 1}.0.0`
      case 'minor':
        return `${major}.${minor + 1}.0`
      default:
        return `${major}.${minor}.${patch + 1}`
    }
  }

  // For timestamp/experiment, just increment the number
  const num = typeof current === 'number' ? current : parseInt(String(current), 10) || 0
  return num + 1
}

/**
 * Kaggle-specific competition settings
 */
export const KaggleSettingsSchema = z.object({
  username: z.string(),
  team: z.string().optional(),
  data_sources: z.array(z.string()).default([]),
  model_sources: z.array(z.string()).optional(),
  kernel_versioning: KernelVersioningSchema.optional(),
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
 * Individual kernel version record
 */
export const KernelVersionRecordSchema = z.object({
  version: z.number(),
  kaggle_slug: z.string(),
  timestamp: z.string(),
  status: z.enum(['queued', 'running', 'complete', 'error', 'cancelled']).optional(),
  mlflow_run_id: z.string().optional(),
  model: z.string().optional(),
  notes: z.string().optional(),
})

export type KernelVersionRecord = z.infer<typeof KernelVersionRecordSchema>

/**
 * Kernel configuration
 */
export const KernelConfigSchema = z.object({
  base_name: z.string(),
  slug: z.string(),
  current_version: z.number().default(1),
  last_run: z.string().optional(),
  last_status: z.enum(['queued', 'running', 'complete', 'error', 'cancelled']).optional(),
  versioning_strategy: z.enum(['timestamp', 'semver', 'experiment', 'overwrite']).optional(),
  versions: z.array(KernelVersionRecordSchema).default([]),
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
export function createDefaultCompetitionConfig(name: string, slug: string, username: string): CompetitionConfig {
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
