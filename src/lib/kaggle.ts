/**
 * Kaggle CLI Wrapper
 */

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

/**
 * Run kaggle CLI command
 */
async function runKaggle(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const proc = Bun.spawn(['kaggle', ...args], {
    stdout: 'pipe',
    stderr: 'pipe',
  })

  const stdout = await new Response(proc.stdout).text()
  const stderr = await new Response(proc.stderr).text()
  const exitCode = await proc.exited

  return { stdout, stderr, exitCode }
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
    'queued': 'queued',
    'running': 'running',
    'complete': 'complete',
    'error': 'error',
    'cancelled': 'cancelled',
    'cancelacknowledged': 'cancelled',
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
export async function downloadKernelOutput(slug: string, outputDir: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runKaggle(['kernels', 'output', slug, '-p', outputDir])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * List user's kernels
 */
export async function listKernels(user: string): Promise<string[]> {
  const { stdout, exitCode } = await runKaggle(['kernels', 'list', '-m', '--user', user])

  if (exitCode !== 0) {
    return []
  }

  // Parse kernel slugs from output
  const lines = stdout.trim().split('\n').slice(1) // Skip header
  return lines.map((line) => line.split(/\s+/)[0]).filter(Boolean)
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
 * Create a new model instance (version)
 */
export async function createModelInstance(folder: string, notes?: string): Promise<{ success: boolean; message: string }> {
  const args = ['models', 'instance', 'create', '-p', folder]
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
 * Convert Python script to Jupyter notebook using jupytext
 */
export async function convertToNotebook(pyPath: string, ipynbPath: string): Promise<{ success: boolean; message: string }> {
  const proc = Bun.spawn(['jupytext', '--to', 'notebook', '-o', ipynbPath, pyPath], {
    stdout: 'pipe',
    stderr: 'pipe',
  })

  const stdout = await new Response(proc.stdout).text()
  const stderr = await new Response(proc.stderr).text()
  const exitCode = await proc.exited

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
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

  return { status: 'error', failureMessage: 'Timeout waiting for kernel completion' }
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
    // CSV format: fileName,date,description,status,publicScore,privateScore
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
