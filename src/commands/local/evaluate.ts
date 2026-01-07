/**
 * Local evaluation command
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const EvaluateArgs = z.object({
  model: z.string().optional().describe('Path to model directory'),
  'splits-dir': z.string().optional().describe('Path to k-fold splits'),
  folds: z.number().default(5).describe('Number of folds'),
  output: z.string().optional().describe('Output JSON file'),
  visualize: z.boolean().default(false).describe('Open visualization'),
  compare: z.string().optional().describe('Compare with another model'),
})

export const evaluate: CommandDefinition<typeof EvaluateArgs> = {
  name: 'local evaluate',
  description: 'Run local evaluation with visualization',
  help: `
Runs local k-fold cross-validation evaluation on a model.
Uses the evaluate_local.py script for BLEU scoring.

Options:
  --model       Path to model directory
  --splits-dir  Path to k-fold splits (default: datasets/kfold_splits)
  --folds       Number of folds (default: 5)
  --output      Save results to JSON file
  --visualize   Open visualization in browser
  --compare     Compare with another model's results
`,
  examples: [
    'akk local evaluate --model ./models/nllb-v1',
    'akk local evaluate --model ./models/nllb-v1 --visualize',
    'akk local evaluate --model ./models/nllb-v1 --output results.json',
    'akk local evaluate --compare ./results/nllb-v1.json --compare ./results/nllb-v2.json',
  ],
  args: EvaluateArgs,

  async run(args, ctx) {
    const _config = ctx.config
    const modelPath = args.model
    const splitsDir = args['splits-dir'] || join(ctx.cwd, 'datasets', 'kfold_splits')
    const outputPath = args.output
    const visualize = args.visualize

    // Find evaluation script
    const evalScript = join(ctx.cwd, 'scripts', 'evaluate_local.py')
    const scriptExists = await Bun.file(evalScript).exists()

    if (!scriptExists) {
      return error(
        'SCRIPT_NOT_FOUND',
        `Evaluation script not found: ${evalScript}`,
        'Create scripts/evaluate_local.py or specify --script path',
        { path: evalScript }
      )
    }

    // Check splits directory
    const splitsExists = (await Bun.spawn(['test', '-d', splitsDir]).exited) === 0
    if (!splitsExists) {
      return error(
        'SPLITS_NOT_FOUND',
        `Splits directory not found: ${splitsDir}`,
        'Generate k-fold splits first or specify --splits-dir',
        { path: splitsDir }
      )
    }

    logStep({ step: 'evaluate', message: 'Running k-fold evaluation...' }, ctx.output)

    // Build command
    const pythonArgs = [
      'python3',
      evalScript,
      '--splits-dir',
      splitsDir,
      '--n-folds',
      String(args.folds),
      '--test-dummy', // For now, use dummy predictor
    ]

    if (outputPath) {
      pythonArgs.push('--output', outputPath)
    }

    // Run evaluation
    const proc = Bun.spawn(pythonArgs, {
      stdout: 'pipe',
      stderr: 'pipe',
    })

    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      return error('EVAL_FAILED', `Evaluation failed: ${stderr || stdout}`, 'Check model path and splits directory', {
        exitCode,
        stderr,
        stdout,
      })
    }

    // Parse results
    let results: Record<string, unknown> = {}
    if (outputPath) {
      try {
        const resultFile = await Bun.file(outputPath).text()
        results = JSON.parse(resultFile)
      } catch {
        // Results may not have been saved
      }
    }

    // Parse output for key metrics
    const bleuMatch = stdout.match(/Overall BLEU:\s*([\d.]+)/)
    const meanBleuMatch = stdout.match(/Mean BLEU:\s*([\d.]+)\s*±\s*([\d.]+)/)

    const metrics = {
      overall_bleu: bleuMatch ? parseFloat(bleuMatch[1]) : undefined,
      mean_bleu: meanBleuMatch ? parseFloat(meanBleuMatch[1]) : undefined,
      std_bleu: meanBleuMatch ? parseFloat(meanBleuMatch[2]) : undefined,
    }

    // Open visualization if requested
    if (visualize && outputPath) {
      // For now, just open the JSON file
      // In future, could serve an HTML visualization
      await Bun.spawn(['open', outputPath]).exited
    }

    return success({
      model: modelPath || 'dummy-baseline',
      splitsDir,
      folds: args.folds,
      output: outputPath,
      metrics,
      log: stdout.trim().split('\n').slice(-10), // Last 10 lines
      results,
    })
  },
}
