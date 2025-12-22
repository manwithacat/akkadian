/**
 * Template List Command
 *
 * Lists available templates and platforms.
 */

import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success } from '../../lib/output'
import {
  createTemplateEngine,
  AVAILABLE_TEMPLATES,
  TEMPLATE_DESCRIPTIONS,
  PLATFORM_DISPLAY_NAMES,
} from '../../templates'
import { PLATFORMS } from '../../types/platform'
import type { PlatformId } from '../../types/platform'

const ListArgs = z.object({
  platforms: z.boolean().default(false).describe('Show detailed platform information'),
})

export const list: CommandDefinition<typeof ListArgs> = {
  name: 'template list',
  description: 'List available templates and platforms',
  help: `
List available notebook templates and target platforms.

Templates define the structure and logic for training/inference notebooks.
Platforms represent different compute environments with varying GPU, memory,
and time limits.

Use --platforms to see detailed information about each platform including
GPU type, memory, disk space, time limits, and recommended batch sizes.
`,
  examples: [
    'akk template list',
    'akk template list --platforms',
  ],
  args: ListArgs,

  async run(args, ctx) {
    const engine = createTemplateEngine()
    const templates = engine.listTemplates()

    // Build template list
    const templateList = AVAILABLE_TEMPLATES.map(name => {
      const template = templates.find(t => t.name === name)
      return {
        name,
        description: TEMPLATE_DESCRIPTIONS[name],
        version: template?.version || '1.0.0',
        platforms: template?.platforms || [],
      }
    })

    // Build platform list with batch size recommendations
    const platformList = Object.entries(PLATFORMS).map(([id, info]) => {
      const hours = info.time.max_hours === Infinity ? 'Unlimited' : `${info.time.max_hours}h`

      return {
        id,
        displayName: PLATFORM_DISPLAY_NAMES[id as PlatformId],
        gpu: info.gpu.name || 'None (CPU only)',
        gpuMemory: info.gpu.vram_gb ? `${info.gpu.vram_gb}GB` : null,
        systemMemory: `${info.memory.system_gb}GB`,
        disk: `${info.disk.total_gb}GB`,
        timeLimit: hours,
        batchSizes: args.platforms ? getBatchSizeRecommendations(id as PlatformId) : undefined,
      }
    })

    // Group platforms by provider
    const platformGroups = {
      kaggle: platformList.filter(p => p.id.startsWith('kaggle')),
      colab: platformList.filter(p => p.id.startsWith('colab')),
      vertex: platformList.filter(p => p.id.startsWith('vertex')),
      runpod: platformList.filter(p => p.id.startsWith('runpod')),
    }

    return success({
      message: args.platforms
        ? 'Platform details'
        : 'Available templates and platforms',
      templates: templateList,
      platforms: args.platforms ? platformList : undefined,
      platform_summary: !args.platforms ? platformGroups : undefined,
      usage: {
        generate: 'akk template generate <template> --platform <id>',
        example: 'akk template generate training --platform kaggle-p100',
      },
    })
  },
}

function getBatchSizeRecommendations(platformId: PlatformId): Record<string, number> {
  // Model family to batch size mapping
  const modelBatches: Record<string, Record<string, number>> = {
    'kaggle-p100': {
      'nllb-600M': 4,
      'nllb-1.3B': 2,
      'nllb-3.3B': 1,
      'byt5-base': 8,
      'byt5-large': 4,
    },
    'kaggle-t4x2': {
      'nllb-600M': 8,
      'nllb-1.3B': 4,
      'nllb-3.3B': 2,
      'byt5-base': 16,
      'byt5-large': 8,
    },
    'kaggle-cpu': {},
    'colab-free': {
      'nllb-600M': 4,
      'nllb-1.3B': 2,
      'byt5-base': 8,
    },
    'colab-pro': {
      'nllb-600M': 16,
      'nllb-1.3B': 8,
      'nllb-3.3B': 4,
      'byt5-base': 32,
      'byt5-large': 16,
    },
    'vertex-a100': {
      'nllb-600M': 16,
      'nllb-1.3B': 8,
      'nllb-3.3B': 4,
      'byt5-base': 32,
      'byt5-large': 16,
    },
    'vertex-t4': {
      'nllb-600M': 4,
      'nllb-1.3B': 2,
      'byt5-base': 8,
    },
    'runpod-a100': {
      'nllb-600M': 16,
      'nllb-1.3B': 8,
      'nllb-3.3B': 4,
      'byt5-base': 32,
      'byt5-large': 16,
    },
    'runpod-3090': {
      'nllb-600M': 8,
      'nllb-1.3B': 4,
      'nllb-3.3B': 2,
      'byt5-base': 16,
      'byt5-large': 8,
    },
  }

  return modelBatches[platformId] || {}
}
