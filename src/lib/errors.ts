/**
 * Standardized error handling for Akkadian CLI
 *
 * Provides a consistent error class that integrates with
 * the CLI output system and includes agent hints for LLM assistance.
 */

import type { CommandError, CommandOutput } from '../types/output'

/**
 * Standard error codes used throughout the CLI
 */
export const ErrorCode = {
  // Configuration errors
  NO_CONFIG: 'NO_CONFIG',
  INVALID_CONFIG: 'INVALID_CONFIG',
  NO_PROJECT: 'NO_PROJECT',

  // Authentication errors
  KAGGLE_AUTH: 'KAGGLE_AUTH',
  GCS_AUTH: 'GCS_AUTH',
  MLFLOW_AUTH: 'MLFLOW_AUTH',

  // Resource errors
  NOT_FOUND: 'NOT_FOUND',
  ALREADY_EXISTS: 'ALREADY_EXISTS',
  PERMISSION_DENIED: 'PERMISSION_DENIED',

  // Validation errors
  VALIDATION: 'VALIDATION',
  INVALID_INPUT: 'INVALID_INPUT',
  INVALID_FORMAT: 'INVALID_FORMAT',

  // Network/service errors
  NETWORK: 'NETWORK',
  TIMEOUT: 'TIMEOUT',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',

  // Command execution errors
  COMMAND_FAILED: 'COMMAND_FAILED',
  PROCESS_ERROR: 'PROCESS_ERROR',

  // MLFlow specific
  MLFLOW_NOT_RUNNING: 'MLFLOW_NOT_RUNNING',
  EXPERIMENT_NOT_FOUND: 'EXPERIMENT_NOT_FOUND',
  RUN_NOT_FOUND: 'RUN_NOT_FOUND',

  // Kaggle specific
  KERNEL_ERROR: 'KERNEL_ERROR',
  INVALID_SLUG: 'INVALID_SLUG',
  UPLOAD_FAILED: 'UPLOAD_FAILED',

  // GCS specific
  BUCKET_NOT_FOUND: 'BUCKET_NOT_FOUND',
  DOWNLOAD_FAILED: 'DOWNLOAD_FAILED',

  // Internal errors
  INTERNAL: 'INTERNAL',
  NOT_IMPLEMENTED: 'NOT_IMPLEMENTED',
} as const

export type ErrorCode = (typeof ErrorCode)[keyof typeof ErrorCode]

/**
 * Default hints for common error codes
 */
const DefaultHints: Record<string, string> = {
  [ErrorCode.NO_CONFIG]: 'Create an akk.toml file in your project root',
  [ErrorCode.NO_PROJECT]: 'Run from the Akkadian project root or use --cwd',
  [ErrorCode.KAGGLE_AUTH]: 'Ensure ~/.kaggle/kaggle.json exists with valid credentials',
  [ErrorCode.GCS_AUTH]: 'Run `gcloud auth login` to authenticate with Google Cloud',
  [ErrorCode.MLFLOW_NOT_RUNNING]: 'Start MLFlow server with `akk mlflow start`',
  [ErrorCode.NOT_FOUND]: 'Check the path or name spelling',
  [ErrorCode.TIMEOUT]: 'Try again or increase the timeout value',
  [ErrorCode.NETWORK]: 'Check your internet connection',
  [ErrorCode.INVALID_SLUG]: 'Kernel slug must be in format "user/kernel-name"',
}

export interface AkkErrorOptions {
  /** Error code for categorization */
  code: ErrorCode | string
  /** Human-readable error message */
  message: string
  /** Hint for LLM agents on how to resolve */
  hint?: string
  /** Additional context data */
  context?: Record<string, unknown>
  /** Original error that caused this */
  cause?: Error
}

/**
 * Custom error class for Akkadian CLI
 *
 * Provides structured error information that can be:
 * - Thrown and caught like regular errors
 * - Converted to CommandOutput for CLI response
 * - Serialized to JSON with agent hints
 *
 * @example
 * throw new AkkError({
 *   code: ErrorCode.NOT_FOUND,
 *   message: 'Kernel not found',
 *   hint: 'Check the kernel slug is correct',
 *   context: { slug: 'user/kernel-name' }
 * })
 */
export class AkkError extends Error {
  readonly code: string
  readonly hint?: string
  readonly context?: Record<string, unknown>
  override readonly cause?: Error

  constructor(options: AkkErrorOptions) {
    super(options.message)
    this.name = 'AkkError'
    this.code = options.code
    this.hint = options.hint || DefaultHints[options.code]
    this.context = options.context
    this.cause = options.cause

    // Capture stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, AkkError)
    }
  }

  /**
   * Create from an unknown error (e.g., from catch block)
   */
  static from(err: unknown, code: ErrorCode | string = ErrorCode.INTERNAL): AkkError {
    if (err instanceof AkkError) {
      return err
    }

    if (err instanceof Error) {
      return new AkkError({
        code,
        message: err.message,
        cause: err,
      })
    }

    return new AkkError({
      code,
      message: String(err),
    })
  }

  /**
   * Create a NOT_FOUND error
   */
  static notFound(resource: string, context?: Record<string, unknown>): AkkError {
    return new AkkError({
      code: ErrorCode.NOT_FOUND,
      message: `${resource} not found`,
      context,
    })
  }

  /**
   * Create a VALIDATION error
   */
  static validation(message: string, context?: Record<string, unknown>): AkkError {
    return new AkkError({
      code: ErrorCode.VALIDATION,
      message,
      context,
    })
  }

  /**
   * Create an INVALID_INPUT error
   */
  static invalidInput(message: string, hint?: string): AkkError {
    return new AkkError({
      code: ErrorCode.INVALID_INPUT,
      message,
      hint,
    })
  }

  /**
   * Convert to CommandError format
   */
  toCommandError(): CommandError {
    const err: CommandError = {
      code: this.code,
      message: this.message,
    }
    if (this.hint) {
      err.__agent_hint = this.hint
    }
    if (this.context) {
      err.context = this.context
    }
    if (this.stack) {
      err.stack = this.stack
    }
    return err
  }

  /**
   * Convert to CommandOutput format
   */
  toOutput(): CommandOutput<never> {
    return {
      success: false,
      error: this.toCommandError(),
    }
  }

  /**
   * Serialize to JSON (for logging/debugging)
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      hint: this.hint,
      context: this.context,
      stack: this.stack,
      cause: this.cause?.message,
    }
  }
}

/**
 * Type guard to check if an error is an AkkError
 */
export function isAkkError(err: unknown): err is AkkError {
  return err instanceof AkkError
}

/**
 * Wrap a function to convert thrown errors to AkkErrors
 */
export function wrapError<T>(fn: () => T, code: ErrorCode | string = ErrorCode.INTERNAL): T {
  try {
    return fn()
  } catch (err) {
    throw AkkError.from(err, code)
  }
}

/**
 * Async version of wrapError
 */
export async function wrapErrorAsync<T>(
  fn: () => Promise<T>,
  code: ErrorCode | string = ErrorCode.INTERNAL
): Promise<T> {
  try {
    return await fn()
  } catch (err) {
    throw AkkError.from(err, code)
  }
}
