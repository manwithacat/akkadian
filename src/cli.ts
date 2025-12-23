/**
 * Akkadian CLI Router
 */

import { z } from 'zod'
// Import commands
import {
  analyze,
  cleanup,
  colabStatus,
  colabUploadNotebook,
  competitionInit,
  competitionStatus,
  configure,
  dataDownload,
  dataExplore,
  dataList,
  dataRegister,
  dataWrangler,
  doctor,
  downloadArtifacts,
  downloadModel,
  downloadOutput,
  evaluate,
  importRun,
  infer,
  kaggleStatus,
  kaggleSubmissions,
  listKernels,
  listRuns,
  log,
  logs,
  mcpServe,
  mlflowRegister,
  modelList,
  modelRegister,
  notebookBuild,
  preflight,
  preflightPlatforms,
  preflightValidate,
  prepare,
  runKernel,
  start,
  sync,
  templateGenerate,
  templateList,
  train,
  uploadModel,
  uploadNotebook,
  version,
  vertexList,
  vertexStatus,
  vertexSubmit,
} from './commands'
import { defaultConfig, findConfigPath, getProjectRoot, loadConfig } from './lib/config'
import { defaultOutputOptions, error, write } from './lib/output'
import type { Command, CommandContext } from './types'

// Command registry
const commands: Record<string, Command> = {
  version,
  doctor,
  // Kaggle commands
  'kaggle upload-notebook': uploadNotebook,
  'kaggle upload-model': uploadModel,
  'kaggle run-kernel': runKernel,
  'kaggle download-output': downloadOutput,
  'kaggle list-kernels': listKernels,
  'kaggle logs': logs,
  'kaggle status': kaggleStatus,
  'kaggle submissions': kaggleSubmissions,
  // Colab commands
  'colab configure': configure,
  'colab download-model': downloadModel,
  'colab upload-notebook': colabUploadNotebook,
  'colab status': colabStatus,
  'colab download-artifacts': downloadArtifacts,
  'colab cleanup': cleanup,
  // MLFlow commands
  'mlflow start': start,
  'mlflow log': log,
  'mlflow sync': sync,
  'mlflow register': mlflowRegister,
  // Model registry commands
  'model list': modelList,
  'model register': modelRegister,
  // Local commands
  'local evaluate': evaluate,
  'local infer': infer,
  'local analyze': analyze,
  // Workflow commands
  'workflow train': train,
  'workflow prepare': prepare,
  'workflow import-run': importRun,
  'workflow list-runs': listRuns,
  // Vertex AI commands
  'vertex submit': vertexSubmit,
  'vertex status': vertexStatus,
  'vertex list': vertexList,
  // Preflight commands
  'preflight check': preflight,
  'preflight platforms': preflightPlatforms,
  'preflight validate': preflightValidate,
  // Competition commands
  'competition init': competitionInit,
  'competition status': competitionStatus,
  // Template commands
  'template generate': templateGenerate,
  'template list': templateList,
  // MCP commands
  'mcp serve': mcpServe,
  // Data management commands
  'data download': dataDownload,
  'data list': dataList,
  'data register': dataRegister,
  'data explore': dataExplore,
  'data wrangler': dataWrangler,
  // Notebook commands
  'notebook build': notebookBuild,
}

// Global options schema
const GlobalOptions = z.object({
  help: z.boolean().default(false),
  json: z.boolean().default(false),
  verbose: z.boolean().default(false),
  quiet: z.boolean().default(false),
})

/**
 * Parse command line arguments
 */
function parseArgs(argv: string[]): {
  command: string | null
  subcommand: string | null
  args: Record<string, unknown>
  globalOpts: z.infer<typeof GlobalOptions>
} {
  const args: Record<string, unknown> = {}
  const globalOpts: Record<string, boolean> = {
    help: false,
    json: false,
    verbose: false,
    quiet: false,
  }

  let command: string | null = null
  let subcommand: string | null = null
  let i = 0

  while (i < argv.length) {
    const arg = argv[i]

    // Global flags
    if (arg === '--help' || arg === '-h') {
      globalOpts.help = true
      i++
      continue
    }
    if (arg === '--json') {
      globalOpts.json = true
      i++
      continue
    }
    if (arg === '--verbose' || arg === '-v') {
      globalOpts.verbose = true
      i++
      continue
    }
    if (arg === '--quiet' || arg === '-q') {
      globalOpts.quiet = true
      i++
      continue
    }

    // Command and subcommand
    if (!arg.startsWith('-')) {
      if (!command) {
        command = arg
        i++
        continue
      }
      if (!subcommand && !arg.startsWith('-')) {
        // Check if this could be a subcommand
        const potentialCmd = `${command} ${arg}`
        if (
          commands[potentialCmd] ||
          [
            'kaggle',
            'colab',
            'local',
            'mlflow',
            'model',
            'workflow',
            'vertex',
            'preflight',
            'competition',
            'template',
            'mcp',
            'data',
            'notebook',
          ].includes(command)
        ) {
          subcommand = arg
          i++
          continue
        }
      }
    }

    // Command arguments
    if (arg.startsWith('--')) {
      // Convert kebab-case to camelCase (e.g., --dry-run -> dryRun)
      const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase())
      const nextArg = argv[i + 1]

      if (!nextArg || nextArg.startsWith('-')) {
        args[key] = true
        i++
      } else if (nextArg === 'true') {
        args[key] = true
        i += 2
      } else if (nextArg === 'false') {
        args[key] = false
        i += 2
      } else {
        const numValue = Number(nextArg)
        args[key] = isNaN(numValue) ? nextArg : numValue
        i += 2
      }
      continue
    }

    // Positional arguments
    if (!arg.startsWith('-')) {
      if (!args.path) {
        args.path = arg
      } else if (!args.name) {
        args.name = arg
      }
      i++
      continue
    }

    i++
  }

  return {
    command,
    subcommand,
    args,
    globalOpts: GlobalOptions.parse(globalOpts),
  }
}

/**
 * Show help message
 */
function showHelp(commandName?: string): void {
  if (commandName && commands[commandName]) {
    const cmd = commands[commandName]
    console.log(`
${cmd.name} - ${cmd.description}
${cmd.help || ''}

Examples:
${(cmd.examples || []).map((e) => `  ${e}`).join('\n')}
`)
    return
  }

  console.log(`
akk - Akkadian ML CLI

Usage: akk <command> [options]

Commands:
  version                      Show version info
  doctor                       Check environment and dependencies

  kaggle upload-notebook       Upload notebook to Kaggle kernels (with versioning)
  kaggle upload-model          Upload model to Kaggle Models
  kaggle run-kernel            Run and monitor a Kaggle kernel
  kaggle download-output       Download kernel outputs
  kaggle list-kernels          List kernel versions from registry
  kaggle logs                  Retrieve and display kernel execution logs
  kaggle status                Check kernel execution status
  kaggle submissions           List competition submissions and scoring status

  colab configure              Set up GCS bucket and auth
  colab upload-notebook        Upload notebook to GCS for Colab
  colab status                 Check training status from GCS
  colab download-model         Download model from GCS
  colab download-artifacts     Download training artifacts from GCS
  colab cleanup                Clean up GCS artifacts to save storage

  local evaluate               Run local evaluation with visualization
  local infer                  Run traced inference on translation model
  local analyze                Analyze training results with recommendations

  mlflow start                 Start MLFlow tracking server
  mlflow log                   Log experiment metrics
  mlflow sync                  Sync runs from GCS to local
  mlflow register              Register model in Model Registry

  model list                   List registered models from Kaggle registry
  model register               Register a model in the local registry

  workflow train               End-to-end training workflow
  workflow prepare             Prepare notebook for Colab upload
  workflow list-runs           List training runs in GCS
  workflow import-run          Import GCS run into local MLFlow

  vertex submit                Submit training job to Vertex AI
  vertex status                Check Vertex AI job status
  vertex list                  List Vertex AI jobs

  preflight check              Check notebook resource requirements
  preflight validate           Validate notebook structure and readiness
  preflight platforms          List available platform profiles

  competition init             Initialize competition directory
  competition status           Show competition status

  template list                List available templates and platforms
  template generate            Generate notebook from template

  mcp serve                    Start MCP server for LLM agents

  data download                Download competition data from Kaggle
  data list                    List registered dataset versions
  data register                Register dataset with lineage tracking
  data explore                 Launch Datasette to explore datasets
  data wrangler                Launch Marimo for rich dataframe exploration

  notebook build               Generate notebook from TOML config

Global Options:
  --help, -h     Show help
  --json         Output as JSON
  --verbose, -v  Verbose output
  --quiet, -q    Suppress output

Examples:
  akk doctor
  akk workflow train --notebook notebooks/colab/nllb_train.ipynb
  akk colab upload-notebook nllb_train.ipynb --version v4
  akk colab status --run nllb-v4 --watch
  akk colab download-artifacts --run nllb-v4
  akk local analyze --run nllb-v4 --compare nllb-v3
  akk kaggle upload-model ./artifacts/nllb-v4/model
  akk mlflow start --port 5001

Run 'akk <command> --help' for command-specific help.
`)
}

/**
 * Create command context
 */
async function createContext(globalOpts: z.infer<typeof GlobalOptions>): Promise<CommandContext> {
  const outputOpts = defaultOutputOptions()

  if (globalOpts.json) outputOpts.format = 'json'
  if (globalOpts.verbose) outputOpts.verbose = true
  if (globalOpts.quiet) outputOpts.quiet = true

  const configPath = await findConfigPath()
  const config = configPath ? await loadConfig(configPath) : defaultConfig()

  return {
    cwd: configPath ? getProjectRoot(configPath) : process.cwd(),
    output: outputOpts,
    configPath: configPath ?? undefined,
    config: config ?? undefined,
  }
}

/**
 * Run the CLI
 */
export async function run(argv: string[] = process.argv.slice(2)): Promise<void> {
  const { command, subcommand, args, globalOpts } = parseArgs(argv)

  // Show help if requested or no command
  if (globalOpts.help || !command) {
    const fullCmd = subcommand && command ? `${command} ${subcommand}` : command
    showHelp(fullCmd ?? undefined)
    process.exit(command ? 0 : 1)
  }

  // Build full command name (for subcommands like "kaggle upload-notebook")
  // At this point, command is guaranteed to be non-null
  const fullCommand = subcommand ? `${command} ${subcommand}` : command

  // Find command
  const cmd = commands[fullCommand!] || commands[command]
  if (!cmd) {
    const ctx = await createContext(globalOpts)
    const output = error(
      'UNKNOWN_COMMAND',
      `Unknown command: ${fullCommand}`,
      `Available commands: ${Object.keys(commands).join(', ')}`,
      { command: fullCommand }
    )
    write(output, ctx.output)
    process.exit(1)
  }

  // Check for --help after command
  if (args.help) {
    showHelp(fullCommand ?? undefined)
    process.exit(0)
  }

  // Create context
  const ctx = await createContext(globalOpts)

  // Special handling for MCP serve - it takes over stdio directly
  if (fullCommand === 'mcp serve') {
    const { main: startMcpServer } = await import('./mcp/server')
    try {
      await startMcpServer()
      // Server exited normally
      process.exit(0)
    } catch (err) {
      // Only write to stderr to avoid breaking MCP protocol
      console.error('MCP server error:', err instanceof Error ? err.message : String(err))
      process.exit(1)
    }
  }

  try {
    // Parse and validate command arguments
    const parsedArgs = cmd.args.safeParse(args)
    if (!parsedArgs.success) {
      const issues = parsedArgs.error.issues
      const output = error(
        'INVALID_ARGS',
        `Invalid arguments: ${issues.map((i: { message: string }) => i.message).join(', ')}`,
        `Run 'akk ${fullCommand} --help' for usage`,
        { issues }
      )
      write(output, ctx.output)
      process.exit(1)
    }

    // Run command
    const result = await cmd.run(parsedArgs.data, ctx)

    // Write output
    write(result, ctx.output)

    // Exit with appropriate code
    process.exit(result.success ? 0 : 1)
  } catch (err) {
    const output = error(
      'INTERNAL_ERROR',
      err instanceof Error ? err.message : String(err),
      'This is a bug in the CLI. Please report it.',
      { stack: err instanceof Error ? err.stack : undefined }
    )
    write(output, ctx.output)
    process.exit(1)
  }
}
