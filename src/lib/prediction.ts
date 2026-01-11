/**
 * Prediction Job Management
 *
 * Handles registration, tracking, and analysis of prediction jobs
 * for weight optimization and model comparison.
 */

import { existsSync, readFileSync, writeFileSync } from 'fs'
import type { ModelSource, PredictionJobRecord, PredictionStatus } from '../types/competition'
import { loadCompetitionConfig, saveCompetitionConfig } from './config'

/**
 * Options for creating a new prediction job
 */
export interface CreatePredictionJobOptions {
  modelPath: string
  modelSource: ModelSource
  modelName?: string
  datasetName: string
  datasetVersion?: string
  datasetSize?: number
  platform?: string
  trainingRunId?: string
  notes?: string
}

/**
 * Options for listing prediction jobs
 */
export interface ListPredictionJobsOptions {
  status?: PredictionStatus
  modelName?: string
  dataset?: string
  trainingRunId?: string
  limit?: number
}

/**
 * Generate a unique prediction job ID
 */
export function generatePredictionId(modelName: string, datasetName: string): string {
  const timestamp = new Date().toISOString().replace(/[:-]/g, '').slice(0, 15)
  const slug = modelName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .slice(0, 20)
  return `pred-${slug}-${datasetName}-${timestamp}`
}

/**
 * Create a new prediction job record
 */
export function createPredictionJob(options: CreatePredictionJobOptions): PredictionJobRecord {
  const modelName = options.modelName || options.modelPath.split('/').pop() || 'unknown'

  return {
    prediction_id: generatePredictionId(modelName, options.datasetName),
    job_type: 'prediction',
    model_source: options.modelSource,
    model_path: options.modelPath,
    model_name: modelName,
    dataset_name: options.datasetName,
    dataset_version: options.datasetVersion,
    dataset_size: options.datasetSize,
    platform: options.platform || 'local',
    timestamp: new Date().toISOString(),
    status: 'queued',
    training_run_id: options.trainingRunId,
    notes: options.notes,
  }
}

/**
 * Register a prediction job in competition.toml
 */
export async function registerPredictionJob(job: PredictionJobRecord, configPath: string): Promise<void> {
  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    throw new Error(`Could not load competition config from ${configPath}`)
  }

  // Initialize predictions if not present
  if (!config.predictions) {
    config.predictions = { total: 0, jobs: {} }
  }

  // Add or update the job
  config.predictions.jobs[job.prediction_id] = job
  config.predictions.total = Object.keys(config.predictions.jobs).length

  // Save back to competition.toml
  const configDir = configPath.replace('/competition.toml', '')
  await saveCompetitionConfig(config, configDir)
}

/**
 * Update prediction job status
 */
export async function updatePredictionJobStatus(
  predictionId: string,
  status: PredictionStatus,
  configPath: string,
  updates?: Partial<PredictionJobRecord>
): Promise<void> {
  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    throw new Error(`Could not load competition config from ${configPath}`)
  }

  const job = config.predictions?.jobs[predictionId]
  if (!job) {
    throw new Error(`Prediction job ${predictionId} not found`)
  }

  // Update status and any additional fields
  job.status = status
  if (updates) {
    Object.assign(job, updates)
  }

  // Calculate duration if completing
  if (status === 'complete' || status === 'error') {
    const startTime = new Date(job.timestamp).getTime()
    const endTime = Date.now()
    job.duration_seconds = Math.round((endTime - startTime) / 1000)
  }

  // Save back
  const configDir = configPath.replace('/competition.toml', '')
  await saveCompetitionConfig(config, configDir)
}

/**
 * List prediction jobs with optional filtering
 */
export async function listPredictionJobs(
  configPath: string,
  options: ListPredictionJobsOptions = {}
): Promise<PredictionJobRecord[]> {
  const config = await loadCompetitionConfig(configPath)
  if (!config || !config.predictions?.jobs) {
    return []
  }

  let jobs = Object.values(config.predictions.jobs)

  // Apply filters
  if (options.status) {
    jobs = jobs.filter((j) => j.status === options.status)
  }
  if (options.modelName) {
    jobs = jobs.filter((j) => j.model_name?.toLowerCase().includes(options.modelName!.toLowerCase()))
  }
  if (options.dataset) {
    jobs = jobs.filter((j) => j.dataset_name.toLowerCase().includes(options.dataset!.toLowerCase()))
  }
  if (options.trainingRunId) {
    jobs = jobs.filter((j) => j.training_run_id === options.trainingRunId)
  }

  // Sort by timestamp (newest first)
  jobs.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())

  // Apply limit
  if (options.limit && options.limit > 0) {
    jobs = jobs.slice(0, options.limit)
  }

  return jobs
}

/**
 * Get a specific prediction job by ID
 */
export async function getPredictionJob(predictionId: string, configPath: string): Promise<PredictionJobRecord | null> {
  const config = await loadCompetitionConfig(configPath)
  return config?.predictions?.jobs[predictionId] || null
}

/**
 * Delete a prediction job
 */
export async function deletePredictionJob(predictionId: string, configPath: string): Promise<void> {
  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    throw new Error(`Could not load competition config from ${configPath}`)
  }

  if (!config.predictions?.jobs[predictionId]) {
    throw new Error(`Prediction job ${predictionId} not found`)
  }

  delete config.predictions.jobs[predictionId]
  config.predictions.total = Object.keys(config.predictions.jobs).length

  const configDir = configPath.replace('/competition.toml', '')
  await saveCompetitionConfig(config, configDir)
}

/**
 * Import prediction results from a JSONL file
 */
export interface PredictionResult {
  id: string | number
  source: string
  prediction: string
  reference?: string
}

export async function importPredictions(filePath: string): Promise<PredictionResult[]> {
  if (!existsSync(filePath)) {
    throw new Error(`Prediction file not found: ${filePath}`)
  }

  const content = readFileSync(filePath, 'utf-8')
  const lines = content.trim().split('\n')

  return lines.map((line, idx) => {
    try {
      return JSON.parse(line) as PredictionResult
    } catch {
      throw new Error(`Invalid JSON at line ${idx + 1}: ${line.slice(0, 50)}...`)
    }
  })
}

/**
 * Save predictions to a JSONL file
 */
export function savePredictions(predictions: PredictionResult[], outputPath: string): void {
  const content = predictions.map((p) => JSON.stringify(p)).join('\n')
  writeFileSync(outputPath, content, 'utf-8')
}

/**
 * Calculate metrics from predictions
 */
export interface MetricOptions {
  compute_bleu?: boolean
  compute_chrf?: boolean
}

/**
 * Summary of prediction jobs for MCP context
 */
export interface PredictionSummary {
  total: number
  by_status: Record<PredictionStatus, number>
  by_model: Record<string, PredictionJobRecord[]>
  recent: PredictionJobRecord[]
}

/**
 * Get prediction summary for MCP status resource
 */
export async function getPredictionSummary(configPath: string): Promise<PredictionSummary> {
  const jobs = await listPredictionJobs(configPath)

  const summary: PredictionSummary = {
    total: jobs.length,
    by_status: {
      queued: 0,
      running: 0,
      complete: 0,
      error: 0,
      cancelled: 0,
    },
    by_model: {},
    recent: jobs.slice(0, 5),
  }

  for (const job of jobs) {
    // Count by status
    summary.by_status[job.status]++

    // Group by model
    const modelKey = job.model_name || 'unknown'
    if (!summary.by_model[modelKey]) {
      summary.by_model[modelKey] = []
    }
    summary.by_model[modelKey].push(job)
  }

  return summary
}

/**
 * Format prediction job for display
 */
export function formatPredictionJob(job: PredictionJobRecord): string {
  const lines: string[] = [
    `ID: ${job.prediction_id}`,
    `Model: ${job.model_name || job.model_path}`,
    `Source: ${job.model_source}`,
    `Dataset: ${job.dataset_name}${job.dataset_version ? ` (${job.dataset_version})` : ''}`,
    `Status: ${job.status}`,
    `Platform: ${job.platform}`,
    `Started: ${job.timestamp}`,
  ]

  if (job.duration_seconds) {
    lines.push(`Duration: ${job.duration_seconds}s`)
  }

  if (job.metric_summary) {
    const metrics = Object.entries(job.metric_summary)
      .filter(([_, v]) => v !== undefined)
      .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(2) : v}`)
      .join(', ')
    if (metrics) {
      lines.push(`Metrics: ${metrics}`)
    }
  }

  if (job.artifacts?.predictions) {
    lines.push(`Predictions: ${job.artifacts.predictions}`)
  }

  if (job.notes) {
    lines.push(`Notes: ${job.notes}`)
  }

  return lines.join('\n')
}

/**
 * Format multiple prediction jobs as a table
 */
export function formatPredictionTable(jobs: PredictionJobRecord[]): string {
  if (jobs.length === 0) {
    return 'No prediction jobs found.'
  }

  const headers = ['ID', 'Model', 'Dataset', 'Status', 'BLEU', 'Started']
  const rows = jobs.map((job) => [
    job.prediction_id.slice(0, 30),
    (job.model_name || 'unknown').slice(0, 20),
    job.dataset_name.slice(0, 15),
    job.status,
    job.metric_summary?.bleu?.toFixed(2) || '-',
    new Date(job.timestamp).toLocaleDateString(),
  ])

  // Calculate column widths
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => String(r[i]).length)))

  // Format header
  const headerLine = headers.map((h, i) => h.padEnd(widths[i])).join(' | ')
  const separator = widths.map((w) => '-'.repeat(w)).join('-+-')

  // Format rows
  const rowLines = rows.map((row) => row.map((cell, i) => String(cell).padEnd(widths[i])).join(' | '))

  return [headerLine, separator, ...rowLines].join('\n')
}
