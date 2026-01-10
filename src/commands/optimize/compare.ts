/**
 * Quick model comparison without full config
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const CompareArgs = z.object({
  models: z.array(z.string()).optional().describe('Model paths to compare'),
  dataset: z.string().optional().describe('Evaluation dataset path'),
  samples: z.number().default(100).describe('Number of samples to evaluate'),
  metrics: z.array(z.string()).default(['bleu']).describe('Metrics to compute'),
})

export const compare: CommandDefinition<typeof CompareArgs> = {
  name: 'optimize compare',
  description: 'Quick comparison of multiple models',
  help: `
Quick model comparison without requiring a full TOML config file.

Compares multiple models on the same dataset and reports metrics.
For more complex experiments, use 'akk optimize run' with a config file.

Options:
  --models   Model paths to compare (space-separated)
  --dataset  Evaluation dataset (CSV with 'transliteration' and 'translation')
  --samples  Number of samples to evaluate (default: 100)
  --metrics  Metrics to compute (bleu, chrf)

For complex experiments with hyperparameter tuning or ensembles,
use 'akk optimize run' with a TOML configuration file instead.
`,
  examples: [
    'akk optimize compare --models models/byt5-v1 models/nllb-v1',
    'akk optimize compare --models models/byt5-v1 models/nllb-v1 --samples 200',
    'akk optimize compare --models models/byt5-v1 --dataset data/test.csv',
  ],
  args: CompareArgs,

  async run(args, ctx) {
    const { models, dataset, samples, metrics } = args

    if (!models || models.length === 0) {
      return error(
        'NO_MODELS',
        'No models specified',
        'Use --models to specify model paths: akk optimize compare --models models/a models/b',
        {}
      )
    }

    logStep({ step: 'compare', message: `Comparing ${models.length} models...` }, ctx.output)

    // Generate temporary config
    const tempConfig = {
      meta: { name: 'quick-compare', description: 'Quick comparison' },
      experiment: { mlflow_experiment: 'akkadian-optimization' },
      data: {
        test_source: dataset || 'competitions/deep-past-initiative-machine-translation/datasets/test.csv',
        source_column: 'transliteration',
        target_column: 'translation',
        val_samples: samples,
      },
      models: models.map((path, i) => ({
        name: `model-${i + 1}`,
        type: 'local',
        path,
        enabled: true,
        weight: 1.0 / models.length,
      })),
      metrics: {
        primary: 'bleu',
        secondary: metrics.filter((m) => m !== 'bleu'),
      },
      output: { save_predictions: true },
    }

    // Write temp config
    const tempPath = join(ctx.cwd, '.compare-temp.toml')
    const tomlContent = generateToml(tempConfig)
    await Bun.write(tempPath, tomlContent)

    try {
      // Run comparison
      const proc = Bun.spawn(['python3', '-m', 'optimization.runner', '--config', tempPath, '--mode', 'compare'], {
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

      // Clean up temp file
      await Bun.write(tempPath, '')
      const { unlink } = await import('fs/promises')
      await unlink(tempPath).catch(() => {})

      if (exitCode !== 0) {
        return error(
          'COMPARE_FAILED',
          `Comparison failed with exit code ${exitCode}`,
          'Check model paths and dataset',
          {
            exitCode,
            models,
          }
        )
      }

      return success({
        models,
        samples,
        metrics,
        message: 'Comparison completed',
      })
    } catch (err) {
      return error('COMPARE_ERROR', `Comparison error: ${err}`, 'Check Python environment', { error: String(err) })
    }
  },
}

function generateToml(config: Record<string, unknown>): string {
  // Simple TOML generator for the config object
  let toml = ''

  // Meta section
  const meta = config.meta as Record<string, string>
  toml += '[meta]\n'
  toml += `name = "${meta.name}"\n`
  toml += `description = "${meta.description}"\n\n`

  // Experiment section
  const exp = config.experiment as Record<string, string>
  toml += '[experiment]\n'
  toml += `mlflow_experiment = "${exp.mlflow_experiment}"\n\n`

  // Data section
  const data = config.data as Record<string, unknown>
  toml += '[data]\n'
  toml += `test_source = "${data.test_source}"\n`
  toml += `source_column = "${data.source_column}"\n`
  toml += `target_column = "${data.target_column}"\n`
  if (data.val_samples) {
    toml += `val_samples = ${data.val_samples}\n`
  }
  toml += '\n'

  // Models array
  const models = config.models as Array<Record<string, unknown>>
  for (const model of models) {
    toml += '[[models]]\n'
    toml += `name = "${model.name}"\n`
    toml += `type = "${model.type}"\n`
    toml += `path = "${model.path}"\n`
    toml += `enabled = ${model.enabled}\n`
    toml += `weight = ${model.weight}\n\n`
  }

  // Metrics section
  const metrics = config.metrics as Record<string, unknown>
  toml += '[metrics]\n'
  toml += `primary = "${metrics.primary}"\n`
  const secondary = metrics.secondary as string[]
  if (secondary && secondary.length > 0) {
    toml += `secondary = [${secondary.map((s) => `"${s}"`).join(', ')}]\n`
  }
  toml += '\n'

  // Output section
  const output = config.output as Record<string, boolean>
  toml += '[output]\n'
  toml += `save_predictions = ${output.save_predictions}\n`

  return toml
}
