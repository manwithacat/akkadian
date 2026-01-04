/**
 * Kaggle CLI Wrapper
 */

import { type CommandResult, cli } from './process'

export interface KernelMetadata {
  id: string
  title: string
  code_file: string
  language: 'python'
  kernel_type: 'notebook' | 'script'
  is_private: boolean
  enable_gpu: boolean
  enable_internet: boolean
  competition_sources?: string[]
  dataset_sources?: string[]
  model_sources?: string[]
}

export interface KernelStatus {
  status: 'queued' | 'running' | 'complete' | 'error' | 'cancelled'
  failureMessage?: string
}

export interface ModelMetadata {
  ownerSlug: string
  slug: string
  title: string
  subtitle: string
  isPrivate: boolean
  description: string
  publishTime?: string
  provenanceSources?: string
}

export interface ModelInstanceMetadata {
  ownerSlug: string
  modelSlug: string
  instanceSlug: string
  framework: 'PyTorch' | 'TensorFlow' | 'JAX' | 'Flax' | 'Other'
  overview: string
  usage: string
  licenseName: string
  fineTunable: boolean
  trainingData: string[]
  // Valid Kaggle API values: unspecified, baseModel, external (NOT finetuned - use external with baseModelUrl)
  modelInstanceType: 'unspecified' | 'baseModel' | 'external'
  baseModelInstanceId?: number
  externalBaseModelUrl?: string
}

/**
 * Kaggle Model structure guidance for LLM agents
 */
export const MODEL_GUIDANCE = `
## Kaggle Models vs Datasets

Use **Kaggle Models** (not Datasets) for ML model weights:

### Model Structure (3-tier hierarchy)
1. **Model** - Top-level container (e.g., "nllb-akkadian")
   - Metadata: title, subtitle, description, license
   - Create once, contains multiple instances

2. **Model Instance** - Framework-specific variant (e.g., "nllb-akkadian/pytorch/annotated-v1")
   - Metadata: framework, overview, usage, license
   - One per framework (PyTorch, TensorFlow, etc.)

3. **Version** - Actual model files
   - Upload new versions as you iterate
   - Version notes for changelog

### When to Use Models vs Datasets
- **Models**: ML model weights, fine-tuned models, pre-trained models
- **Datasets**: Training data, evaluation data, raw files

### Kernel Reference
\`\`\`json
{
  "model_sources": ["owner/model-name/framework/instance-slug"]
}
\`\`\`

### Benefits of Models
- Better discoverability (appears in Models section)
- Proper versioning with notes
- Framework metadata (PyTorch, TensorFlow, etc.)
- License tracking
- Provenance chain (base model references)
`

/**
 * Run kaggle CLI command (using shared process utility)
 */
async function runKaggle(args: string[]): Promise<CommandResult> {
  return cli.kaggle(args)
}

/**
 * Push a kernel to Kaggle
 */
export async function pushKernel(metadataPath: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['kernels', 'push', '-p', metadataPath])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Get kernel status
 */
export async function getKernelStatus(slug: string): Promise<KernelStatus> {
  const { stdout, stderr, exitCode } = await runKaggle(['kernels', 'status', slug])

  if (exitCode !== 0) {
    throw new Error(`Failed to get kernel status: ${stderr}`)
  }

  // Parse status from output like:
  // - "manwithacat/nllb-train has status "complete""
  // - "manwithacat/nllb-train has status "KernelWorkerStatus.RUNNING""
  const statusMatch = stdout.match(/status "(?:KernelWorkerStatus\.)?(\w+)"/i)
  const rawStatus = statusMatch?.[1]?.toLowerCase() || 'error'

  // Map Kaggle status values to our simplified status
  const statusMap: Record<string, KernelStatus['status']> = {
    queued: 'queued',
    running: 'running',
    complete: 'complete',
    error: 'error',
    cancelled: 'cancelled',
    cancelacknowledged: 'cancelled',
  }

  const status = statusMap[rawStatus] || 'error'

  // Check for failure message
  const failureMatch = stdout.match(/failureMessage:\s*"([^"]+)"/)

  return {
    status,
    failureMessage: failureMatch?.[1],
  }
}

/**
 * Download kernel output files
 */
export async function downloadKernelOutput(
  slug: string,
  outputDir: string
): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['kernels', 'output', slug, '-p', outputDir])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * List user's kernels
 * @param user - Kaggle username
 * @param sortBy - Sort order: 'dateRun' for most recent, 'hotness' for popular (default: 'dateRun')
 * @param limit - Maximum number of kernels to return (default: 20)
 */
export async function listKernels(
  user: string,
  sortBy: 'dateRun' | 'hotness' | 'dateCreated' = 'dateRun',
  limit = 20
): Promise<string[]> {
  const { stdout, exitCode } = await runKaggle([
    'kernels',
    'list',
    '-m',
    '--user',
    user,
    '--sort-by',
    sortBy,
    '--page-size',
    String(Math.min(limit, 200)), // Kaggle max is 200
  ])

  if (exitCode !== 0) {
    return []
  }

  // Parse kernel slugs from output
  const lines = stdout.trim().split('\n').slice(1) // Skip header
  return lines
    .map((line) => line.split(/\s+/)[0])
    .filter(Boolean)
    .slice(0, limit)
}

/**
 * Initialize model metadata
 */
export async function initModel(folder: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['models', 'init', '-p', folder])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Create a new model on Kaggle
 */
export async function createModel(folder: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['models', 'create', '-p', folder])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Create a new model instance (with initial version)
 * Used when creating the first version of an instance
 */
export async function createModelInstance(
  folder: string,
  notes?: string
): Promise<{ success: boolean; message: string }> {
  const args = ['models', 'instances', 'create', '-p', folder]
  if (notes) {
    args.push('-n', notes)
  }

  const { stdout, stderr, exitCode } = await runKaggle(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Create a new version of an existing model instance
 * Format: owner/model-name/framework/instance-slug
 */
export async function createModelVersion(
  instancePath: string,
  folder: string,
  notes?: string
): Promise<{ success: boolean; message: string }> {
  const args = ['models', 'instances', 'versions', 'create', instancePath, '-p', folder]
  if (notes) {
    args.push('-n', notes)
  }

  const { stdout, stderr, exitCode } = await runKaggle(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Initialize model instance metadata template
 */
export async function initModelInstance(folder: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['models', 'instances', 'init', '-p', folder])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * List user's models
 */
export async function listModels(user: string): Promise<{ success: boolean; message: string; models?: string[] }> {
  const { stdout, stderr, exitCode } = await runKaggle(['models', 'list', '--owner', user])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  // Parse model slugs from output
  const lines = stdout.trim().split('\n').slice(1) // Skip header
  const models = lines.map((line) => line.split(/\s+/)[0]).filter(Boolean)

  return { success: true, message: stdout, models }
}

/**
 * Get model instance files
 */
export async function getModelInstanceFiles(instancePath: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['models', 'instances', 'files', instancePath])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Convert Python script to Jupyter notebook using jupytext
 */
export async function convertToNotebook(
  pyPath: string,
  ipynbPath: string
): Promise<{ success: boolean; message: string }> {
  const result = await cli.jupytext(['--to', 'notebook', '-o', ipynbPath, pyPath])

  if (!result.success) {
    return { success: false, message: result.stderr || result.stdout }
  }

  return { success: true, message: result.stdout }
}

/**
 * Create kernel metadata JSON
 */
export function createKernelMetadata(options: {
  username: string
  title: string
  codeFile: string
  enableGpu?: boolean
  enableInternet?: boolean
  competition?: string
  datasets?: string[]
  models?: string[]
}): KernelMetadata {
  const slug = options.title.toLowerCase().replace(/[^a-z0-9]+/g, '-')

  return {
    id: `${options.username}/${slug}`,
    title: options.title,
    code_file: options.codeFile,
    language: 'python',
    kernel_type: 'notebook',
    is_private: true,
    enable_gpu: options.enableGpu ?? true,
    enable_internet: options.enableInternet ?? true,
    competition_sources: options.competition ? [options.competition] : undefined,
    dataset_sources: options.datasets,
    model_sources: options.models,
  }
}

/**
 * Poll kernel status until completion
 */
export async function waitForKernel(
  slug: string,
  options: {
    interval?: number // milliseconds
    timeout?: number // milliseconds
    onStatus?: (status: KernelStatus) => void
  } = {}
): Promise<KernelStatus> {
  const interval = options.interval ?? 30000 // 30 seconds
  const timeout = options.timeout ?? 7200000 // 2 hours
  const startTime = Date.now()

  while (Date.now() - startTime < timeout) {
    const status = await getKernelStatus(slug)
    options.onStatus?.(status)

    if (status.status === 'complete' || status.status === 'error' || status.status === 'cancelled') {
      return status
    }

    await Bun.sleep(interval)
  }

  return {
    status: 'error',
    failureMessage: 'Timeout waiting for kernel completion',
  }
}

/**
 * Competition submission status
 */
export interface CompetitionSubmission {
  fileName: string
  date: string
  description: string
  status: 'pending' | 'complete' | 'error'
  publicScore?: number
  privateScore?: number
}

/**
 * Parse a CSV line handling quoted fields
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]

    if (char === '"') {
      inQuotes = !inQuotes
    } else if (char === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += char
    }
  }
  result.push(current)

  return result
}

/**
 * Get competition submissions
 */
export async function getCompetitionSubmissions(competition: string): Promise<CompetitionSubmission[]> {
  const { stdout, stderr, exitCode } = await runKaggle(['competitions', 'submissions', '-c', competition, '-v'])

  if (exitCode !== 0) {
    throw new Error(`Failed to get submissions: ${stderr}`)
  }

  // Parse CSV output
  const lines = stdout.trim().split('\n')
  if (lines.length < 2) {
    return []
  }

  // Skip header line
  const submissions: CompetitionSubmission[] = []
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i]
    // CSV format: fileName,date,description,logStep,publicScore,privateScore
    const parts = parseCSVLine(line)
    if (parts.length >= 4) {
      const rawStatus = parts[3].toLowerCase()
      let status: CompetitionSubmission['status'] = 'error'
      if (rawStatus.includes('pending')) {
        status = 'pending'
      } else if (rawStatus.includes('complete')) {
        status = 'complete'
      }

      submissions.push({
        fileName: parts[0],
        date: parts[1],
        description: parts[2] || '',
        status,
        publicScore: parts[4] ? parseFloat(parts[4]) : undefined,
        privateScore: parts[5] ? parseFloat(parts[5]) : undefined,
      })
    }
  }

  return submissions
}

/**
 * Parse a Kaggle URL to extract kernel slug
 *
 * Supported formats:
 * - https://www.kaggle.com/code/username/kernel-name
 * - https://www.kaggle.com/code/username/kernel-name?scriptVersionId=123
 * - https://kaggle.com/code/username/kernel-name
 * - kaggle.com/code/username/kernel-name
 */
export function parseKaggleUrl(url: string): { slug: string; versionId?: string } | null {
  // Normalize URL
  let normalized = url.trim()
  if (!normalized.startsWith('http')) {
    normalized = `https://${normalized}`
  }

  try {
    const parsed = new URL(normalized)

    // Check if it's a kaggle.com URL
    if (!parsed.hostname.includes('kaggle.com')) {
      return null
    }

    // Extract path parts: /code/username/kernel-name
    const pathParts = parsed.pathname.split('/').filter(Boolean)

    if (pathParts.length >= 3 && pathParts[0] === 'code') {
      const slug = `${pathParts[1]}/${pathParts[2]}`
      const versionId = parsed.searchParams.get('scriptVersionId') || undefined

      return { slug, versionId }
    }

    return null
  } catch {
    return null
  }
}

/**
 * Check if a string looks like a Kaggle URL
 */
export function isKaggleUrl(input: string): boolean {
  return input.includes('kaggle.com/code/') || input.includes('kaggle.com/code/')
}

/**
 * Normalize input to kernel slug - handles both URLs and direct slugs
 */
export function normalizeKernelInput(input: string): { slug: string; versionId?: string } | null {
  // If it looks like a URL, parse it
  if (isKaggleUrl(input)) {
    return parseKaggleUrl(input)
  }

  // If it's a direct slug (contains /)
  if (input.includes('/') && !input.includes(' ')) {
    return { slug: input }
  }

  return null
}

/**
 * Running kernel info
 */
export interface RunningKernel {
  slug: string
  status: KernelStatus['status']
  failureMessage?: string
}

/**
 * List kernels with running status
 * Checks status of recent kernels and returns those that are running/queued
 *
 * Note: Uses 'dateRun' sorting to find recently active kernels (most likely to be running)
 * Checks up to 20 kernels by default for better coverage
 */
export async function listRunningKernels(user: string, limit = 20): Promise<RunningKernel[]> {
  // Get recent kernels sorted by last run time
  const kernelSlugs = await listKernels(user, 'dateRun', limit)

  if (kernelSlugs.length === 0) {
    return []
  }

  // Check status of each (in parallel for speed)
  const results = await Promise.allSettled(
    kernelSlugs.map(async (slug) => {
      const status = await getKernelStatus(slug)
      return { slug, ...status }
    })
  )

  // Filter to running/queued
  const running: RunningKernel[] = []
  for (const result of results) {
    if (result.status === 'fulfilled') {
      const { slug, status, failureMessage } = result.value
      if (status === 'running' || status === 'queued') {
        running.push({ slug, status, failureMessage })
      }
    }
  }

  return running
}

/**
 * Get status of multiple kernels in parallel
 */
export async function getKernelStatuses(slugs: string[]): Promise<
  Array<{
    slug: string
    status: KernelStatus['status']
    failureMessage?: string
  }>
> {
  const results = await Promise.allSettled(
    slugs.map(async (slug) => {
      const status = await getKernelStatus(slug)
      return { slug, ...status }
    })
  )

  return results
    .filter(
      (
        r
      ): r is PromiseFulfilledResult<{
        slug: string
        status: KernelStatus['status']
        failureMessage?: string
      }> => r.status === 'fulfilled'
    )
    .map((r) => r.value)
}
