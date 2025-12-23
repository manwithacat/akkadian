/**
 * Command types for Akkadian CLI
 */

import type { z } from 'zod'
import type { CommandOutput, OutputOptions } from './output'

export interface CommandContext {
  /** Current working directory */
  cwd: string
  /** Output options */
  output: OutputOptions
  /** Path to akk.toml if found */
  configPath?: string
  /** Parsed akk.toml config */
  config?: AkkConfig
}

/** Supported ML frameworks */
export type MLFramework = 'pytorch' | 'tensorflow' | 'jax'

export interface AkkConfig {
  project: {
    name: string
    version: string
  }
  kaggle: {
    username: string
    competition: string
    /** Internet access for Kaggle kernels (false for competition submissions) */
    enable_internet?: boolean
  }
  colab: {
    gcs_bucket: string
    project: string
  }
  mlflow: {
    tracking_uri: string
    artifact_location: string
    port: number
  }
  paths: {
    /** Base directory for competitions */
    competitions?: string
    notebooks: string
    scripts: string
    datasets: string
    models: string
  }
  /** Training configuration */
  training?: {
    /** Primary ML framework: pytorch, tensorflow, or jax */
    framework?: MLFramework
  }
}

export interface CommandDefinition<TArgs extends z.ZodType = z.ZodType> {
  name: string
  description: string
  /** Long description shown in help */
  help?: string
  /** Example usages */
  examples?: string[]
  /** Zod schema for arguments */
  args: TArgs
  /** Command handler */
  run: (args: z.infer<TArgs>, ctx: CommandContext) => Promise<CommandOutput>
}

export type Command = CommandDefinition<z.ZodType>
