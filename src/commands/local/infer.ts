/**
 * Local inference with tracing
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const InferArgs = z.object({
  model: z.string().optional().describe('Model path (local, HuggingFace, or Kaggle)'),
  text: z.string().optional().describe('Text to translate'),
  interactive: z.boolean().default(false).describe('Interactive mode'),
  input: z.string().optional().describe('Input file with texts'),
  output: z.string().optional().describe('Output file for translations'),
  port: z.number().default(5001).describe('MLFlow server port for traces'),
  'source-lang': z.string().default('akk_Xsux').describe('Source language'),
  'target-lang': z.string().default('eng_Latn').describe('Target language'),
})

export const infer: CommandDefinition<typeof InferArgs> = {
  name: 'local infer',
  description: 'Run traced inference on translation model',
  help: `
Runs inference on a translation model with MLFlow tracing.
All inference calls are traced and logged to MLFlow.

Options:
  --model        Model path (local dir, HuggingFace hub ID, or Kaggle slug)
  --text         Single text to translate
  --interactive  Interactive translation mode
  --input        Input file with texts (one per line)
  --output       Output file for translations
  --port         MLFlow server port for traces (default: 5001)
  --source-lang  Source language code (default: akk_Xsux)
  --target-lang  Target language code (default: eng_Latn)

Traces are logged to the 'akkadian-inference' experiment in MLFlow.
View traces at: http://localhost:{port}/#/experiments
`,
  examples: [
    'akk local infer --model facebook/nllb-200-distilled-600M --interactive',
    'akk local infer --model ./models/nllb --text "šarrum dannum"',
    'akk local infer --model ./models/nllb --input texts.txt --output translations.txt',
  ],
  args: InferArgs,

  async run(args, ctx) {
    const { model, text, interactive, input, output, port } = args
    const sourceLang = args['source-lang']
    const targetLang = args['target-lang']

    // Find inference script
    const inferScript = join(ctx.cwd, 'mlflow', 'inference', 'traced_translator.py')
    const scriptExists = await Bun.file(inferScript).exists()

    if (!scriptExists) {
      return error(
        'SCRIPT_NOT_FOUND',
        `Inference script not found: ${inferScript}`,
        'Create mlflow/inference/traced_translator.py',
        { path: inferScript }
      )
    }

    // Determine model path
    const modelPath = model || 'facebook/nllb-200-distilled-600M'

    // Build command
    const pythonArgs = [
      'python3',
      inferScript,
      '--model',
      modelPath,
      '--tracking-uri',
      `http://localhost:${port}`,
      '--source-lang',
      sourceLang,
      '--target-lang',
      targetLang,
    ]

    if (interactive) {
      pythonArgs.push('--interactive')

      // Run interactively with stdio inheritance
      logStep({ step: 'load', message: `Loading model: ${modelPath}...` }, ctx.output)

      const proc = Bun.spawn(pythonArgs, {
        stdin: 'inherit',
        stdout: 'inherit',
        stderr: 'inherit',
        cwd: ctx.cwd,
      })

      await proc.exited

      return success({
        model: modelPath,
        mode: 'interactive',
        tracesUrl: `http://localhost:${port}/#/experiments`,
      })
    }

    if (text) {
      pythonArgs.push('--text', text)
    } else if (input) {
      pythonArgs.push('--input', input)
      if (output) {
        pythonArgs.push('--output', output)
      }
    } else {
      return error(
        'NO_INPUT',
        'Specify --text, --input, or --interactive',
        'Example: akk local infer --model ./models/nllb --text "šarrum dannum"',
        {}
      )
    }

    logStep({ step: 'infer', message: `Translating with ${modelPath}...` }, ctx.output)

    const proc = Bun.spawn(pythonArgs, {
      stdout: 'pipe',
      stderr: 'pipe',
      cwd: ctx.cwd,
    })

    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      return error('INFER_FAILED', `Inference failed: ${stderr}`, 'Check model path and dependencies', {
        exitCode,
        stderr: stderr.slice(-500),
      })
    }

    // Parse output
    try {
      const result = JSON.parse(stdout)
      return success({
        ...result,
        tracesUrl: `http://localhost:${port}/#/experiments`,
      })
    } catch {
      // Non-JSON output (batch mode)
      return success({
        model: modelPath,
        output: output || 'stdout',
        lines: stdout.trim().split('\n').length,
        tracesUrl: `http://localhost:${port}/#/experiments`,
      })
    }
  },
}
