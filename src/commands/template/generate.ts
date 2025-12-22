/**
 * Template Generate Command
 *
 * Generates notebooks from templates for a specific platform.
 */

import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success, error } from '../../lib/output'
import { loadCompetitionConfig } from '../../lib/config'
import { createTemplateEngine, AVAILABLE_TEMPLATES, PLATFORM_DISPLAY_NAMES } from '../../templates'
import type { PlatformId } from '../../types/platform'
import type { TemplateContext } from '../../types/template'
import type { ToolId } from '../../types/tools'
import { TOOL_REGISTRY, checkToolCompatibility } from '../../types/tools'

const VALID_PLATFORMS: PlatformId[] = [
  'kaggle-p100', 'kaggle-t4x2', 'kaggle-cpu',
  'colab-free', 'colab-pro',
  'vertex-a100', 'vertex-t4',
  'runpod-a100', 'runpod-3090',
]

const VALID_TOOLS: ToolId[] = ['sacrebleu', 'qlora', 'accelerate', 'onnx', 'streaming', 'hpo']

const GenerateArgs = z.object({
  path: z.string().optional().describe('Template name (training or inference)'),
  platform: z.string().optional().describe('Target platform'),
  model: z.string().optional().describe('Model name'),
  output: z.string().optional().describe('Output file path'),
  epochs: z.number().optional().describe('Number of training epochs'),
  batchSize: z.number().optional().describe('Batch size'),
  srcLang: z.string().optional().describe('Source language code'),
  tgtLang: z.string().optional().describe('Target language code'),
  learningRate: z.number().optional().describe('Learning rate'),
  notebook: z.boolean().default(false).describe('Output as Jupyter notebook'),
  tools: z.string().optional().describe('Comma-separated ML tools to include (sacrebleu,qlora,accelerate,onnx,streaming,hpo)'),
})

export const generate: CommandDefinition<typeof GenerateArgs> = {
  name: 'template generate',
  description: 'Generate a notebook from a template',
  help: `
Generate a notebook from a template for a specific platform.

Available templates:
  training     Base training template for seq2seq models
  inference    Inference template for generating predictions

Available platforms:
  kaggle-p100, kaggle-t4x2, kaggle-cpu
  colab-free, colab-pro
  vertex-a100, vertex-t4
  runpod-a100, runpod-3090

Available ML tools (optional):
  sacrebleu    Reproducible MT evaluation with BLEU, chrF, TER
  qlora        4-bit quantized training with LoRA adapters
  accelerate   Distributed training wrapper for multi-GPU
  onnx         ONNX export for production inference
  streaming    Memory-efficient data loading for large datasets
  hpo          Hyperparameter optimization with Optuna + MLflow

The generated notebook includes:
  - Platform-specific setup (GPU detection, paths, auth)
  - Data loading code tailored to the platform
  - Checkpoint saving with cloud sync where applicable
  - Output saving and model export
  - Optional ML tool integrations

If a competition.toml exists in the current directory, defaults are read from it.
`,
  examples: [
    'akk template generate training --platform kaggle-p100',
    'akk template generate training --platform colab-pro --model facebook/nllb-200-1.3B',
    'akk template generate training --platform vertex-a100 --output train_vertex.py',
    'akk template generate training --platform kaggle-p100 --notebook',
    'akk template generate training --platform colab-pro --tools sacrebleu,qlora,hpo',
  ],
  args: GenerateArgs,

  async run(args, ctx) {
    const templateName = args.path || 'training'

    // Validate template name
    if (!AVAILABLE_TEMPLATES.includes(templateName as typeof AVAILABLE_TEMPLATES[number])) {
      return error(
        'UNKNOWN_TEMPLATE',
        `Unknown template: ${templateName}`,
        `Available templates: ${AVAILABLE_TEMPLATES.join(', ')}`
      )
    }

    // Validate platform
    if (!args.platform) {
      return error(
        'MISSING_PLATFORM',
        'Platform is required',
        `Use --platform <platform>. Available: ${VALID_PLATFORMS.join(', ')}`
      )
    }

    const platform = args.platform as PlatformId
    if (!VALID_PLATFORMS.includes(platform)) {
      return error(
        'UNKNOWN_PLATFORM',
        `Unknown platform: ${platform}`,
        `Available platforms: ${VALID_PLATFORMS.join(', ')}`
      )
    }

    // Parse and validate tools
    let enabledTools: ToolId[] = []
    if (args.tools) {
      const requestedTools = args.tools.split(',').map(t => t.trim().toLowerCase()) as ToolId[]

      // Validate tool names
      for (const tool of requestedTools) {
        if (!VALID_TOOLS.includes(tool)) {
          return error(
            'UNKNOWN_TOOL',
            `Unknown tool: ${tool}`,
            `Available tools: ${VALID_TOOLS.join(', ')}`
          )
        }
      }

      // Check tool compatibility
      const compatibility = checkToolCompatibility(requestedTools)
      if (!compatibility.valid) {
        return error(
          'TOOL_INCOMPATIBILITY',
          `Tool compatibility issue: ${compatibility.errors.join(', ')}`,
          'Some tools cannot be used together'
        )
      }

      enabledTools = requestedTools
    }

    // Try to load competition config for defaults
    let competitionConfig = null
    try {
      competitionConfig = await loadCompetitionConfig()
    } catch {
      // No competition config, use defaults
    }

    // Build template context
    const templateCtx: TemplateContext = {
      platform,
      model: {
        name: args.model || competitionConfig?.active_model?.base || 'facebook/nllb-200-distilled-600M',
        base: args.model || competitionConfig?.active_model?.base || 'facebook/nllb-200-distilled-600M',
      },
      training: {
        epochs: args.epochs || competitionConfig?.training?.default_epochs || 10,
        batch_size: args.batchSize || competitionConfig?.training?.default_batch_size || 2,
        learning_rate: args.learningRate || 5e-5,
        gradient_accumulation_steps: 4,
        max_src_len: 128,
        max_tgt_len: 128,
        warmup_ratio: 0.1,
        weight_decay: 0.01,
        fp16: true,
        save_steps: 500,
        eval_steps: 500,
        logging_steps: 100,
        save_total_limit: 3,
      },
      languages: {
        src: args.srcLang || 'akk_Xsux',
        tgt: args.tgtLang || 'eng_Latn',
      },
      tools: enabledTools.length > 0 ? {
        enabled: enabledTools,
      } : undefined,
    }

    // Add competition context if available
    if (competitionConfig) {
      templateCtx.competition = {
        competition: competitionConfig.competition,
      }
      if (competitionConfig.gcs) {
        templateCtx.gcs = {
          bucket: competitionConfig.gcs.bucket,
          project: competitionConfig.gcs.project,
          cross_account: competitionConfig.gcs.cross_account,
        }
      }
    }

    // Create engine and render
    const engine = createTemplateEngine()

    try {
      const result = engine.render(templateName, templateCtx)

      // Determine output filename
      let outputPath = args.output
      if (!outputPath) {
        const extension = args.notebook ? '.ipynb' : '.py'
        const platformShort = platform.split('-')[0]
        outputPath = `${templateName}_${platformShort}${extension}`
      }

      // Write output
      if (args.notebook) {
        await Bun.write(outputPath, JSON.stringify(result.notebook, null, 2))
      } else {
        await Bun.write(outputPath, result.script)
      }

      // Write metadata for Kaggle if applicable
      let metadataPath: string | undefined
      if (result.metadata && Object.keys(result.metadata).length > 0) {
        metadataPath = outputPath.replace(/\.(py|ipynb)$/, '-metadata.json')
        await Bun.write(metadataPath, JSON.stringify(result.metadata, null, 2))
      }

      return success({
        message: 'Template generated successfully',
        template: templateName,
        platform: PLATFORM_DISPLAY_NAMES[platform],
        output: outputPath,
        format: args.notebook ? 'Jupyter Notebook' : 'Python Script',
        paths: {
          input: result.paths.input,
          output: result.paths.output,
          checkpoint: result.paths.checkpoint,
          model_cache: result.paths.model_cache,
        },
        metadata: metadataPath,
        settings: {
          model: templateCtx.model.name,
          epochs: templateCtx.training.epochs,
          batch_size: templateCtx.training.batch_size,
          learning_rate: templateCtx.training.learning_rate,
        },
        tools: enabledTools.length > 0 ? {
          enabled: enabledTools,
          descriptions: enabledTools.map(t => TOOL_REGISTRY[t].description),
        } : undefined,
        warnings: result.warnings.length > 0 ? result.warnings : undefined,
      })
    } catch (err) {
      return error(
        'GENERATE_FAILED',
        `Failed to generate template: ${err instanceof Error ? err.message : 'Unknown error'}`,
        'Check template name and platform, then try again'
      )
    }
  },
}
