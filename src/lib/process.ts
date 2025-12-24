/**
 * Process execution utilities
 *
 * Centralizes all Bun.spawn patterns to reduce duplication and provide
 * consistent error handling, logging, and output capture.
 */

export interface CommandResult {
  stdout: string
  stderr: string
  exitCode: number
  success: boolean
}

export interface CommandOptions {
  /** Working directory for the command */
  cwd?: string
  /** Environment variables to add/override */
  env?: Record<string, string>
  /** Timeout in milliseconds (0 = no timeout) */
  timeout?: number
  /** Whether to throw on non-zero exit code */
  throwOnError?: boolean
  /** Inherit stdin from parent process */
  inheritStdin?: boolean
}

/**
 * Run a command and capture output
 *
 * @example
 * // Simple command
 * const result = await runCommand('ls', ['-la'])
 *
 * // With options
 * const result = await runCommand('python3', ['script.py'], {
 *   cwd: '/path/to/dir',
 *   timeout: 30000
 * })
 *
 * // Throw on error
 * const result = await runCommand('kaggle', ['kernels', 'push'], {
 *   throwOnError: true
 * })
 */
export async function runCommand(
  command: string,
  args: string[] = [],
  options: CommandOptions = {}
): Promise<CommandResult> {
  const { cwd, env, timeout = 0, throwOnError = false, inheritStdin = false } = options

  const proc = Bun.spawn([command, ...args], {
    stdout: 'pipe',
    stderr: 'pipe',
    stdin: inheritStdin ? 'inherit' : 'ignore',
    cwd,
    env: env ? { ...process.env, ...env } : undefined,
  })

  let stdout: string
  let stderr: string
  let exitCode: number

  if (timeout > 0) {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        proc.kill()
        reject(new Error(`Command timed out after ${timeout}ms: ${command} ${args.join(' ')}`))
      }, timeout)
    })

    const resultPromise = Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited])

    const [stdoutResult, stderrResult, exitCodeResult] = (await Promise.race([resultPromise, timeoutPromise])) as [
      string,
      string,
      number,
    ]

    stdout = stdoutResult
    stderr = stderrResult
    exitCode = exitCodeResult
  } else {
    ;[stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ])
  }

  const result: CommandResult = {
    stdout: stdout.trim(),
    stderr: stderr.trim(),
    exitCode,
    success: exitCode === 0,
  }

  if (throwOnError && !result.success) {
    const error = new CommandError(`Command failed with exit code ${exitCode}: ${command} ${args.join(' ')}`, result)
    throw error
  }

  return result
}

/**
 * Run a command and return success/failure only (for simple checks)
 */
export async function runCheck(command: string, args: string[] = []): Promise<boolean> {
  const result = await runCommand(command, args)
  return result.success
}

/**
 * Check if a command/binary exists
 */
export async function commandExists(command: string): Promise<boolean> {
  return runCheck('which', [command])
}

/**
 * Check if a directory exists
 */
export async function directoryExists(path: string): Promise<boolean> {
  return runCheck('test', ['-d', path])
}

/**
 * Check if a file exists
 */
export async function fileExists(path: string): Promise<boolean> {
  return runCheck('test', ['-f', path])
}

/**
 * Create a directory (mkdir -p)
 */
export async function ensureDir(path: string): Promise<void> {
  await runCommand('mkdir', ['-p', path])
}

/**
 * Custom error class with command result details
 */
export class CommandError extends Error {
  constructor(
    message: string,
    public readonly result: CommandResult
  ) {
    super(message)
    this.name = 'CommandError'
  }

  get stdout(): string {
    return this.result.stdout
  }

  get stderr(): string {
    return this.result.stderr
  }

  get exitCode(): number {
    return this.result.exitCode
  }
}

/**
 * Specialized runners for common CLI tools
 */
export const cli = {
  /**
   * Run kaggle CLI command
   */
  async kaggle(args: string[], options?: CommandOptions): Promise<CommandResult> {
    return runCommand('kaggle', args, options)
  },

  /**
   * Run gsutil command
   */
  async gsutil(args: string[], options?: CommandOptions): Promise<CommandResult> {
    return runCommand('gsutil', args, options)
  },

  /**
   * Run gcloud command
   */
  async gcloud(args: string[], options?: CommandOptions): Promise<CommandResult> {
    return runCommand('gcloud', args, options)
  },

  /**
   * Run python3 command
   */
  async python(args: string[], options?: CommandOptions): Promise<CommandResult> {
    return runCommand('python3', args, options)
  },

  /**
   * Run python3 with inline code
   */
  async pythonCode(code: string, options?: CommandOptions): Promise<CommandResult> {
    return runCommand('python3', ['-c', code], options)
  },

  /**
   * Run mlflow CLI command
   */
  async mlflow(args: string[], options?: CommandOptions): Promise<CommandResult> {
    return runCommand('mlflow', args, options)
  },

  /**
   * Run jupytext command
   */
  async jupytext(args: string[], options?: CommandOptions): Promise<CommandResult> {
    return runCommand('jupytext', args, options)
  },
}
