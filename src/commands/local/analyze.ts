/**
 * Analyze training results with LLM support
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const AnalyzeArgs = z.object({
  run: z.string().optional().describe('Run name to analyze'),
  path: z.string().optional().describe('Path to artifacts directory'),
  compare: z.string().optional().describe('Compare with another run'),
  'output-file': z.string().optional().describe('Save analysis to file'),
})

interface TrainingMetrics {
  bleu?: number
  loss?: number
  epochs?: number
  samples?: number
  training_time_hours?: number
  [key: string]: number | undefined
}

interface AnalysisResult {
  summary: string
  metrics: TrainingMetrics
  comparison?: {
    baseline: TrainingMetrics
    current: TrainingMetrics
    improvements: Record<string, string>
  }
  recommendations: string[]
  nextSteps: string[]
}

async function loadMetrics(path: string): Promise<TrainingMetrics | null> {
  try {
    const metricsFile = Bun.file(join(path, 'metrics.json'))
    if (await metricsFile.exists()) {
      return JSON.parse(await metricsFile.text())
    }

    // Try eval_results.json
    const evalFile = Bun.file(join(path, 'eval_results.json'))
    if (await evalFile.exists()) {
      return JSON.parse(await evalFile.text())
    }

    return null
  } catch {
    return null
  }
}

async function _loadTrainingLog(path: string): Promise<any[] | null> {
  try {
    const logFile = Bun.file(join(path, 'training_log.json'))
    if (await logFile.exists()) {
      return JSON.parse(await logFile.text())
    }
    return null
  } catch {
    return null
  }
}

function analyzeMetrics(metrics: TrainingMetrics, comparison?: TrainingMetrics): AnalysisResult {
  const recommendations: string[] = []
  const improvements: Record<string, string> = {}

  // BLEU analysis
  if (metrics.bleu !== undefined) {
    if (metrics.bleu < 10) {
      recommendations.push(
        'BLEU < 10 is very low. Consider: more training data, longer training, different model architecture'
      )
    } else if (metrics.bleu < 15) {
      recommendations.push('BLEU 10-15 is below baseline. Try: learning rate tuning, more epochs, data augmentation')
    } else if (metrics.bleu < 20) {
      recommendations.push(
        'BLEU 15-20 is approaching baseline. Consider: fine-tuning hyperparameters, ensemble methods'
      )
    } else if (metrics.bleu < 25) {
      recommendations.push('BLEU 20-25 is competitive. Focus on: inference optimization, domain-specific tuning')
    } else {
      recommendations.push('BLEU > 25 is strong. Consider: model distillation, deployment optimization')
    }

    if (comparison?.bleu !== undefined) {
      const delta = metrics.bleu - comparison.bleu
      improvements.bleu = `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} (${((delta / comparison.bleu) * 100).toFixed(1)}%)`

      if (delta < 0) {
        recommendations.push(
          `BLEU decreased by ${Math.abs(delta).toFixed(2)}. Review changes that may have caused regression.`
        )
      } else if (delta > 0) {
        recommendations.push(`BLEU improved by ${delta.toFixed(2)}! Document what worked.`)
      }
    }
  }

  // Loss analysis
  if (metrics.loss !== undefined) {
    if (metrics.loss > 2) {
      recommendations.push('High loss indicates underfitting. Try: more epochs, larger model, different architecture')
    } else if (metrics.loss > 1) {
      recommendations.push('Moderate loss. Consider: learning rate adjustments, longer training')
    } else if (metrics.loss < 0.5) {
      recommendations.push('Low loss is good but watch for overfitting. Validate on held-out data.')
    }

    if (comparison?.loss !== undefined) {
      const delta = metrics.loss - comparison.loss
      improvements.loss = `${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`
    }
  }

  // Training efficiency
  if (metrics.training_time_hours !== undefined && metrics.epochs !== undefined) {
    const hoursPerEpoch = metrics.training_time_hours / metrics.epochs
    if (hoursPerEpoch > 1) {
      recommendations.push(
        `Training is slow (${hoursPerEpoch.toFixed(1)}h/epoch). Consider: gradient accumulation, mixed precision, smaller batch size`
      )
    }
  }

  // Summary
  let summary = 'Training Analysis Summary:\n'
  if (metrics.bleu !== undefined) {
    summary += `- BLEU: ${metrics.bleu.toFixed(2)}\n`
  }
  if (metrics.loss !== undefined) {
    summary += `- Final Loss: ${metrics.loss.toFixed(4)}\n`
  }
  if (metrics.epochs !== undefined) {
    summary += `- Epochs: ${metrics.epochs}\n`
  }

  // Next steps based on metrics
  const nextSteps: string[] = []

  if (metrics.bleu !== undefined) {
    if (metrics.bleu < 20) {
      nextSteps.push('Try hyperparameter tuning: learning rate, batch size, warmup steps')
      nextSteps.push('Experiment with data augmentation or additional training data')
      nextSteps.push('Consider a larger model variant if compute allows')
    }
    if (metrics.bleu >= 20) {
      nextSteps.push('Submit to Kaggle competition for official scoring')
      nextSteps.push('Try model ensembling with other approaches')
      nextSteps.push('Optimize inference speed for production')
    }
  }

  nextSteps.push('Run error analysis on worst predictions')
  nextSteps.push('Compare with baseline models in MLFlow')

  return {
    summary,
    metrics,
    comparison: comparison
      ? {
          baseline: comparison,
          current: metrics,
          improvements,
        }
      : undefined,
    recommendations,
    nextSteps,
  }
}

export const analyze: CommandDefinition<typeof AnalyzeArgs> = {
  name: 'local analyze',
  description: 'Analyze training results with recommendations',
  help: `
Analyzes training metrics and provides recommendations for improvement.

Analysis includes:
- BLEU score interpretation
- Loss trend analysis
- Comparison with baseline runs
- Specific recommendations for improvement
- Suggested next steps

Options:
  --run          Run name (looks in ./artifacts/{run})
  --path         Direct path to artifacts directory
  --compare      Compare with another run
  --output-file  Save analysis to markdown file
`,
  examples: [
    'akk local analyze --run nllb-v4',
    'akk local analyze --run nllb-v4 --compare nllb-v3',
    'akk local analyze --path ./artifacts/nllb-v4 --output-file analysis.md',
  ],
  args: AnalyzeArgs,

  async run(args, ctx) {
    // Determine artifacts path
    let artifactsPath: string
    if (args.path) {
      artifactsPath = args.path.startsWith('/') ? args.path : join(ctx.cwd, args.path)
    } else if (args.run) {
      artifactsPath = join(ctx.cwd, 'artifacts', args.run)
    } else {
      return error('NO_RUN', 'Specify --run or --path', 'Example: akk local analyze --run nllb-v4', {})
    }

    // Check if path exists
    const metricsFile = Bun.file(join(artifactsPath, 'metrics.json'))
    if (!(await metricsFile.exists())) {
      // Try eval_results.json
      const evalFile = Bun.file(join(artifactsPath, 'eval_results.json'))
      if (!(await evalFile.exists())) {
        return error(
          'METRICS_NOT_FOUND',
          `No metrics found in: ${artifactsPath}`,
          'Download artifacts first: akk colab download-artifacts --run <name>',
          {}
        )
      }
    }

    logStep({ step: 'load', message: 'Loading metrics...' }, ctx.output)

    // Load metrics
    const metrics = await loadMetrics(artifactsPath)
    if (!metrics) {
      return error('LOAD_FAILED', 'Failed to load metrics', 'Check file format', {})
    }

    // Load comparison if requested
    let comparisonMetrics: TrainingMetrics | null = null
    if (args.compare) {
      const comparePath = join(ctx.cwd, 'artifacts', args.compare)
      comparisonMetrics = await loadMetrics(comparePath)
      if (!comparisonMetrics) {
        console.log(`Warning: Could not load comparison metrics from ${comparePath}`)
      }
    }

    logStep({ step: 'analyze', message: 'Analyzing results...' }, ctx.output)

    // Analyze
    const analysis = analyzeMetrics(metrics, comparisonMetrics || undefined)

    // Output to file if requested
    if (args['output-file']) {
      const outputPath = args['output-file'].startsWith('/') ? args['output-file'] : join(ctx.cwd, args['output-file'])

      let markdown = `# Training Analysis: ${args.run || artifactsPath}\n\n`
      markdown += `Generated: ${new Date().toISOString()}\n\n`
      markdown += `## Summary\n\n\`\`\`\n${analysis.summary}\`\`\`\n\n`

      if (analysis.comparison) {
        markdown += `## Comparison with ${args.compare}\n\n`
        markdown += `| Metric | Baseline | Current | Change |\n`
        markdown += `|--------|----------|---------|--------|\n`
        for (const [key, value] of Object.entries(analysis.comparison.improvements)) {
          const baseline = analysis.comparison.baseline[key]
          const current = analysis.comparison.current[key]
          markdown += `| ${key} | ${baseline ?? 'N/A'} | ${current ?? 'N/A'} | ${value} |\n`
        }
        markdown += '\n'
      }

      markdown += `## Recommendations\n\n`
      for (const rec of analysis.recommendations) {
        markdown += `- ${rec}\n`
      }
      markdown += '\n'

      markdown += `## Next Steps\n\n`
      for (const step of analysis.nextSteps) {
        markdown += `- ${step}\n`
      }

      await Bun.write(outputPath, markdown)
      console.log(`Analysis saved to: ${outputPath}`)
    }

    return success({
      run: args.run,
      path: artifactsPath,
      ...analysis,
    })
  },
}
