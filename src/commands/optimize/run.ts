/**
 * Run optimization experiment from TOML config
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const RunArgs = z.object({
  config: z.string().describe('Path to TOML config file'),
  mode: z.enum(['evaluate', 'compare', 'ensemble', 'optuna']).default('compare').describe('Execution mode'),
  output: z.string().optional().describe('Output file for ensemble predictions'),
  resume: z.boolean().default(false).describe('Resume existing Optuna study'),
  'dry-run': z.boolean().default(false).describe('Validate config without running'),
})

export const run: CommandDefinition<typeof RunArgs> = {
  name: 'optimize run',
  description: 'Run optimization experiment from config file',
  help: `
Runs model optimization experiments defined in a TOML config file.

Modes:
  evaluate  - Evaluate a single model
  compare   - Compare multiple models side-by-side (default)
  ensemble  - Run ensemble inference with weighted combination
  optuna    - Run hyperparameter optimization study

The config file defines:
  - Models to evaluate (local, HuggingFace, or Kaggle)
  - Generation hyperparameters per model
  - Ensemble weights and method
  - Optuna search space
  - MLflow experiment settings

All runs are logged to MLflow with metrics, predictions, and comparison charts.

Options:
  --config   Path to TOML configuration file (required)
  --mode     Execution mode (default: compare)
  --output   Output file for ensemble predictions (CSV)
  --resume   Resume existing Optuna study from database
  --dry-run  Parse and validate config without running
`,
  examples: [
    'akk optimize run optimization/compare.toml',
    'akk optimize run optimization/compare.toml --mode compare',
    'akk optimize run optimization/tune.toml --mode optuna',
    'akk optimize run optimization/tune.toml --mode optuna --resume',
    'akk optimize run optimization/ensemble.toml --mode ensemble --output submission.csv',
    'akk optimize run optimization/compare.toml --dry-run',
  ],
  args: RunArgs,

  async run(args, ctx) {
    const { config, mode, output, resume } = args
    const dryRun = args['dry-run']

    // Check config file exists
    const configPath = config.startsWith('/') ? config : join(ctx.cwd, config)
    const configExists = await Bun.file(configPath).exists()

    if (!configExists) {
      return error(
        'CONFIG_NOT_FOUND',
        `Config file not found: ${configPath}`,
        'Create a TOML config file with model and experiment settings',
        { path: configPath }
      )
    }

    // Find runner script
    const runnerScript = join(ctx.cwd, 'optimization', 'runner.py')
    const scriptExists = await Bun.file(runnerScript).exists()

    if (!scriptExists) {
      return error(
        'RUNNER_NOT_FOUND',
        `Optimization runner not found: ${runnerScript}`,
        'Ensure optimization module is installed',
        { path: runnerScript }
      )
    }

    // Build command
    const pythonArgs = ['python3', '-m', 'optimization.runner', '--config', configPath, '--mode', mode]

    if (output) {
      pythonArgs.push('--output', output)
    }

    if (resume) {
      pythonArgs.push('--resume')
    }

    if (dryRun) {
      pythonArgs.push('--dry-run')
    }

    logStep({ step: 'optimize', message: `Running ${mode} from ${config}...` }, ctx.output)

    // Run with inherited stdio for interactive progress
    const proc = Bun.spawn(pythonArgs, {
      stdin: 'inherit',
      stdout: 'inherit',
      stderr: 'inherit',
      cwd: ctx.cwd,
      env: {
        ...process.env,
        TOKENIZERS_PARALLELISM: 'false',
      },
    })

    const exitCode = await proc.exited

    if (exitCode !== 0) {
      return error(
        'OPTIMIZE_FAILED',
        `Optimization failed with exit code ${exitCode}`,
        'Check config and model paths',
        {
          exitCode,
          mode,
          config: configPath,
        }
      )
    }

    return success({
      mode,
      config: configPath,
      output: output || null,
      message: dryRun ? 'Config validated successfully' : `${mode} completed successfully`,
    })
  },
}
