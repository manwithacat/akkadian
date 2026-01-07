/**
 * Kernel Registry
 *
 * Manages local versioning for Kaggle kernels, tracking versions
 * in competition.toml and providing unique naming strategies.
 */

import type { CompetitionConfig, KernelConfig, KernelVersionRecord, VersioningStrategy } from '../types/competition'
import { findCompetitionConfig, loadCompetitionConfig, saveCompetitionConfig } from './config'

/**
 * Options for generating a versioned kernel name
 */
export interface VersionOptions {
  baseName: string
  strategy?: VersioningStrategy
  username: string
  model?: string
  notes?: string
}

/**
 * Result of registering a new kernel version
 */
export interface VersionedKernel {
  kernelId: string // Full Kaggle kernel ID: username/slug
  slug: string // Just the slug part
  version: number // Version number
  baseName: string // Base name without version
  timestamp: string // ISO timestamp
}

/**
 * Format timestamp for kernel names
 */
function formatTimestamp(date: Date = new Date()): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')
  const seconds = String(date.getSeconds()).padStart(2, '0')
  return `${year}${month}${day}-${hours}${minutes}${seconds}`
}

/**
 * Generate a versioned kernel slug based on strategy
 */
export function generateVersionedSlug(
  baseName: string,
  version: number,
  strategy: VersioningStrategy,
  separator: string = '-'
): string {
  // Normalize base name to slug format
  const baseSlug = baseName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')

  switch (strategy) {
    case 'timestamp':
      return `${baseSlug}${separator}${formatTimestamp()}`

    case 'semver':
      return `${baseSlug}${separator}v${version}`

    case 'experiment':
      return `${baseSlug}${separator}exp${separator}${String(version).padStart(2, '0')}`
    default:
      return baseSlug
  }
}

/**
 * Get kernel configuration by base name, creating if needed
 */
export function getOrCreateKernelConfig(config: CompetitionConfig, baseName: string): KernelConfig {
  const normalizedName = baseName.toLowerCase().replace(/[^a-z0-9]+/g, '-')

  if (config.kernels[normalizedName]) {
    return config.kernels[normalizedName]
  }

  // Create new kernel entry
  return {
    base_name: baseName,
    slug: normalizedName,
    current_version: 0,
    versions: [],
  }
}

/**
 * Register a new kernel version
 *
 * Creates a versioned kernel name, records it in the registry,
 * and returns the full kernel information.
 */
export async function registerKernelVersion(
  options: VersionOptions,
  projectDir?: string
): Promise<{
  kernel: VersionedKernel
  config: CompetitionConfig
  configPath: string
}> {
  const { baseName, username, model, notes } = options

  // Find competition config
  const configPath = await findCompetitionConfig(projectDir || process.cwd())
  if (!configPath) {
    throw new Error('No competition.toml found. Run "akk competition init" first.')
  }

  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    throw new Error('Failed to load competition.toml')
  }

  // Get versioning strategy
  const strategy = options.strategy || config.competition.kaggle?.kernel_versioning?.strategy || 'semver'

  const separator = config.competition.kaggle?.kernel_versioning?.prefix_separator || '-'

  // Get or create kernel config
  const normalizedName = baseName.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  const kernelConfig = getOrCreateKernelConfig(config, baseName)

  // Increment version
  const newVersion = kernelConfig.current_version + 1
  const timestamp = new Date().toISOString()

  // Generate versioned slug
  const versionedSlug = generateVersionedSlug(baseName, newVersion, strategy, separator)

  // Create version record
  const versionRecord: KernelVersionRecord = {
    version: newVersion,
    kaggle_slug: `${username}/${versionedSlug}`,
    timestamp,
    model,
    notes,
  }

  // Update kernel config
  kernelConfig.current_version = newVersion
  kernelConfig.last_run = timestamp
  kernelConfig.versioning_strategy = strategy
  kernelConfig.versions.push(versionRecord)

  // Save to competition config
  config.kernels[normalizedName] = kernelConfig

  // Persist
  const projectRoot = configPath.replace('/competition.toml', '')
  await saveCompetitionConfig(config, projectRoot)

  return {
    kernel: {
      kernelId: `${username}/${versionedSlug}`,
      slug: versionedSlug,
      version: newVersion,
      baseName,
      timestamp,
    },
    config,
    configPath,
  }
}

/**
 * Get kernel version history
 */
export async function getKernelHistory(baseName: string, projectDir?: string): Promise<KernelVersionRecord[]> {
  const configPath = await findCompetitionConfig(projectDir || process.cwd())
  if (!configPath) {
    return []
  }

  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    return []
  }

  const normalizedName = baseName.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  const kernelConfig = config.kernels[normalizedName]

  return kernelConfig?.versions || []
}

/**
 * Update kernel version status (e.g., after run completes)
 */
export async function updateKernelVersionStatus(
  baseName: string,
  version: number,
  status: KernelVersionRecord['status'],
  mlflowRunId?: string,
  projectDir?: string
): Promise<void> {
  const configPath = await findCompetitionConfig(projectDir || process.cwd())
  if (!configPath) {
    throw new Error('No competition.toml found')
  }

  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    throw new Error('Failed to load competition.toml')
  }

  const normalizedName = baseName.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  const kernelConfig = config.kernels[normalizedName]

  if (!kernelConfig) {
    throw new Error(`Kernel "${baseName}" not found in registry`)
  }

  // Find and update version record
  const versionRecord = kernelConfig.versions.find((v) => v.version === version)
  if (versionRecord) {
    versionRecord.status = status
    if (mlflowRunId) {
      versionRecord.mlflow_run_id = mlflowRunId
    }
  }

  // Update last status
  kernelConfig.last_status = status

  // Persist
  const projectRoot = configPath.replace('/competition.toml', '')
  await saveCompetitionConfig(config, projectRoot)
}

/**
 * List all registered kernels with their current versions
 */
export async function listRegisteredKernels(projectDir?: string): Promise<
  Array<{
    name: string
    currentVersion: number
    strategy?: VersioningStrategy
    lastRun?: string
    lastStatus?: string
  }>
> {
  const configPath = await findCompetitionConfig(projectDir || process.cwd())
  if (!configPath) {
    return []
  }

  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    return []
  }

  return Object.entries(config.kernels).map(([_name, kernel]) => ({
    name: kernel.base_name,
    currentVersion: kernel.current_version,
    strategy: kernel.versioning_strategy,
    lastRun: kernel.last_run,
    lastStatus: kernel.last_status,
  }))
}

/**
 * Get the latest version of a kernel
 */
export async function getLatestKernelVersion(
  baseName: string,
  projectDir?: string
): Promise<KernelVersionRecord | null> {
  const history = await getKernelHistory(baseName, projectDir)
  return history.length > 0 ? history[history.length - 1] : null
}

/**
 * Preview what the next version slug would be (without registering)
 */
export function previewNextVersion(
  baseName: string,
  currentVersion: number,
  username: string,
  strategy: VersioningStrategy = 'semver',
  separator: string = '-'
): VersionedKernel {
  const nextVersion = currentVersion + 1
  const slug = generateVersionedSlug(baseName, nextVersion, strategy, separator)

  return {
    kernelId: `${username}/${slug}`,
    slug,
    version: nextVersion,
    baseName,
    timestamp: new Date().toISOString(),
  }
}
