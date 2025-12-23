/**
 * MLFlow Client Wrapper
 */

export interface MLFlowConfig {
  trackingUri: string
  artifactLocation: string
  port: number
}

export interface Experiment {
  experiment_id: string
  name: string
  artifact_location: string
  lifecycle_stage: string
}

export interface Run {
  runId: string
  experimentId: string
  status: string
  startTime: number
  endTime?: number
  artifactUri: string
}

export interface Metric {
  key: string
  value: number
  timestamp: number
  step: number
}

/**
 * Run mlflow CLI command
 */
async function runMlflow(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const proc = Bun.spawn(['mlflow', ...args], {
    stdout: 'pipe',
    stderr: 'pipe',
  })

  const stdout = await new Response(proc.stdout).text()
  const stderr = await new Response(proc.stderr).text()
  const exitCode = await proc.exited

  return { stdout, stderr, exitCode }
}

/**
 * Check if MLFlow is installed
 */
export async function checkInstalled(): Promise<{ installed: boolean; version?: string }> {
  const { stdout, exitCode } = await runMlflow(['--version'])

  if (exitCode !== 0) {
    return { installed: false }
  }

  const versionMatch = stdout.match(/([\d.]+)/)
  return { installed: true, version: versionMatch?.[1] }
}

/**
 * Check if MLFlow server is running
 */
export async function checkServer(port: number): Promise<boolean> {
  try {
    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/experiments/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_results: 1 }),
    })
    return response.ok
  } catch {
    return false
  }
}

/**
 * Start MLFlow server
 */
export async function startServer(config: MLFlowConfig): Promise<{
  success: boolean
  pid?: number
  message: string
}> {
  // Check if already running
  if (await checkServer(config.port)) {
    return { success: true, message: `MLFlow already running on port ${config.port}` }
  }

  // Start server in background
  const proc = Bun.spawn(
    [
      'mlflow',
      'server',
      '--backend-store-uri',
      config.trackingUri,
      '--default-artifact-root',
      config.artifactLocation,
      '--host',
      '0.0.0.0',
      '--port',
      String(config.port),
    ],
    {
      stdout: 'ignore',
      stderr: 'ignore',
    }
  )

  // Wait for server to start (up to 10 seconds)
  for (let i = 0; i < 10; i++) {
    await Bun.sleep(1000)
    if (await checkServer(config.port)) {
      return { success: true, pid: proc.pid, message: `MLFlow server started on port ${config.port}` }
    }
  }

  return { success: false, message: 'Failed to start MLFlow server (timeout)' }
}

/**
 * Stop MLFlow server
 */
export async function stopServer(port: number): Promise<{ success: boolean; message: string }> {
  // Find process by port
  const proc = Bun.spawn(['lsof', '-t', `-i:${port}`], {
    stdout: 'pipe',
  })

  const stdout = await new Response(proc.stdout).text()
  const pids = stdout.trim().split('\n').filter(Boolean)

  if (pids.length === 0) {
    return { success: true, message: 'No MLFlow server running' }
  }

  // Kill processes
  for (const pid of pids) {
    await Bun.spawn(['kill', pid]).exited
  }

  return { success: true, message: `Stopped ${pids.length} process(es)` }
}

/**
 * List experiments via REST API
 */
export async function listExperiments(port: number): Promise<Experiment[]> {
  try {
    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/experiments/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_results: 100 }),
    })
    if (!response.ok) return []

    const data = (await response.json()) as { experiments?: Experiment[] }
    return data.experiments || []
  } catch {
    return []
  }
}

/**
 * Create or get experiment
 */
export async function getOrCreateExperiment(port: number, name: string): Promise<string | null> {
  try {
    // Try to get existing experiment
    const experiments = await listExperiments(port)
    const existing = experiments.find((e) => e.name === name)
    if (existing) return existing.experiment_id

    // Create new experiment
    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/experiments/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    })

    if (!response.ok) return null

    const data = (await response.json()) as { experiment_id?: string }
    return data.experiment_id || null
  } catch {
    return null
  }
}

/**
 * Create a new run
 */
export async function createRun(
  port: number,
  experimentId: string,
  runName?: string
): Promise<{ runId: string; artifactUri: string } | null> {
  try {
    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        experiment_id: experimentId,
        run_name: runName,
        start_time: Date.now(),
      }),
    })

    if (!response.ok) return null

    const data = (await response.json()) as { run?: { info?: { run_id?: string; artifact_uri?: string } } }
    const info = data.run?.info
    if (!info?.run_id) return null

    return { runId: info.run_id, artifactUri: info.artifact_uri || '' }
  } catch {
    return null
  }
}

/**
 * Log parameters
 */
export async function logParams(port: number, runId: string, params: Record<string, string>): Promise<boolean> {
  try {
    for (const [key, value] of Object.entries(params)) {
      const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/log-parameter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, key, value: String(value) }),
      })

      if (!response.ok) return false
    }
    return true
  } catch {
    return false
  }
}

/**
 * Log metrics
 */
export async function logMetrics(port: number, runId: string, metrics: Record<string, number>): Promise<boolean> {
  try {
    const timestamp = Date.now()
    for (const [key, value] of Object.entries(metrics)) {
      const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/log-metric`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, key, value, timestamp, step: 0 }),
      })

      if (!response.ok) return false
    }
    return true
  } catch {
    return false
  }
}

/**
 * Set run status
 */
export async function setRunStatus(
  port: number,
  runId: string,
  status: 'RUNNING' | 'FINISHED' | 'FAILED' | 'KILLED'
): Promise<boolean> {
  try {
    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        run_id: runId,
        status,
        end_time: status !== 'RUNNING' ? Date.now() : undefined,
      }),
    })

    return response.ok
  } catch {
    return false
  }
}

/**
 * Get run info
 */
export async function getRun(port: number, runId: string): Promise<Run | null> {
  try {
    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/get?run_id=${runId}`)
    if (!response.ok) return null

    const data = (await response.json()) as { run?: { info?: Run } }
    return data.run?.info || null
  } catch {
    return null
  }
}

/**
 * Set tags on a run
 */
export async function setRunTags(port: number, runId: string, tags: Record<string, string>): Promise<boolean> {
  try {
    for (const [key, value] of Object.entries(tags)) {
      const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/set-tag`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, key, value: String(value) }),
      })

      if (!response.ok) return false
    }
    return true
  } catch {
    return false
  }
}

// ============================================
// Kernel Tracking for Kaggle Notebooks
// ============================================

export interface KernelRunInfo {
  experimentId: string
  runId: string
  runName: string
  kernelId: string
  version: number
  baseName: string
  strategy: string
  timestamp: string
  model?: string
  notes?: string
}

/**
 * Log a Kaggle kernel upload to MLflow
 *
 * Creates a new run in the "kaggle-kernels" experiment to track
 * kernel versions, associated models, and eventual run results.
 */
export async function logKernelUpload(
  port: number,
  kernelInfo: {
    kernelId: string
    baseName: string
    version: number
    strategy: string
    timestamp: string
    model?: string
    notes?: string
    gpu?: boolean
    internet?: boolean
    competition?: string
  }
): Promise<KernelRunInfo | null> {
  try {
    // Get or create "kaggle-kernels" experiment
    const experimentId = await getOrCreateExperiment(port, 'kaggle-kernels')
    if (!experimentId) return null

    // Create run with descriptive name
    const runName = `${kernelInfo.baseName}-v${kernelInfo.version}`
    const run = await createRun(port, experimentId, runName)
    if (!run) return null

    // Log parameters
    const params: Record<string, string> = {
      kernel_id: kernelInfo.kernelId,
      base_name: kernelInfo.baseName,
      version: String(kernelInfo.version),
      strategy: kernelInfo.strategy,
      upload_timestamp: kernelInfo.timestamp,
    }

    if (kernelInfo.model) params.model = kernelInfo.model
    if (kernelInfo.competition) params.competition = kernelInfo.competition
    if (kernelInfo.gpu !== undefined) params.gpu = String(kernelInfo.gpu)
    if (kernelInfo.internet !== undefined) params.internet = String(kernelInfo.internet)

    await logParams(port, run.runId, params)

    // Set tags for filtering
    const tags: Record<string, string> = {
      'mlflow.runName': runName,
      kernel_type: 'kaggle',
      kernel_base: kernelInfo.baseName,
    }

    if (kernelInfo.model) tags.model = kernelInfo.model
    if (kernelInfo.notes) tags.notes = kernelInfo.notes

    await setRunTags(port, run.runId, tags)

    // Keep run in RUNNING state (will update when kernel completes)

    return {
      experimentId,
      runId: run.runId,
      runName,
      kernelId: kernelInfo.kernelId,
      version: kernelInfo.version,
      baseName: kernelInfo.baseName,
      strategy: kernelInfo.strategy,
      timestamp: kernelInfo.timestamp,
      model: kernelInfo.model,
      notes: kernelInfo.notes,
    }
  } catch {
    return null
  }
}

/**
 * Update kernel run with final status and metrics
 */
export async function updateKernelRun(
  port: number,
  runId: string,
  status: 'complete' | 'error' | 'cancelled',
  metrics?: Record<string, number>
): Promise<boolean> {
  try {
    // Map Kaggle status to MLflow status
    const mlflowStatus: 'FINISHED' | 'FAILED' | 'KILLED' =
      status === 'complete' ? 'FINISHED' : status === 'cancelled' ? 'KILLED' : 'FAILED'

    // Log metrics if provided
    if (metrics && Object.keys(metrics).length > 0) {
      await logMetrics(port, runId, metrics)
    }

    // Update run status
    await setRunStatus(port, runId, mlflowStatus)

    // Add completion tag
    await setRunTags(port, runId, {
      kernel_status: status,
      completion_time: new Date().toISOString(),
    })

    return true
  } catch {
    return false
  }
}

/**
 * List all kernel runs from MLflow
 */
export async function listKernelRuns(
  port: number,
  filter?: {
    baseName?: string
    model?: string
    status?: string
  }
): Promise<
  Array<{
    runId: string
    runName: string
    kernelId: string
    version: number
    status: string
    timestamp: string
    model?: string
  }>
> {
  try {
    // Build filter string
    let filterStr = 'tags.kernel_type = "kaggle"'
    if (filter?.baseName) {
      filterStr += ` AND tags.kernel_base = "${filter.baseName}"`
    }
    if (filter?.model) {
      filterStr += ` AND tags.model = "${filter.model}"`
    }

    const response = await fetch(`http://localhost:${port}/api/2.0/mlflow/runs/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        experiment_ids: [],
        filter: filterStr,
        max_results: 100,
        order_by: ['start_time DESC'],
      }),
    })

    if (!response.ok) return []

    const data = (await response.json()) as {
      runs?: Array<{
        info?: {
          run_id?: string
          status?: string
          start_time?: number
        }
        data?: {
          params?: Array<{ key: string; value: string }>
          tags?: Array<{ key: string; value: string }>
        }
      }>
    }

    const runs = data.runs || []

    return runs
      .map((run) => {
        const params = new Map(run.data?.params?.map((p) => [p.key, p.value]) || [])
        const tags = new Map(run.data?.tags?.map((t) => [t.key, t.value]) || [])

        return {
          runId: run.info?.run_id || '',
          runName: tags.get('mlflow.runName') || '',
          kernelId: params.get('kernel_id') || '',
          version: parseInt(params.get('version') || '0', 10),
          status: run.info?.status || 'UNKNOWN',
          timestamp: params.get('upload_timestamp') || '',
          model: params.get('model'),
        }
      })
      .filter((r) => r.runId)
  } catch {
    return []
  }
}
