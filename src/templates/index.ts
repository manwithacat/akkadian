/**
 * Template System Registry
 *
 * Exports all templates, adapters, and the template engine.
 */

import { TemplateEngine } from '../lib/template-engine'
import type { PlatformId } from '../types/platform'
import type { PlatformAdapter } from '../types/template'
import { ColabAdapter } from './adapters/colab'
import { KaggleAdapter } from './adapters/kaggle'
import { RunPodAdapter } from './adapters/runpod'
import { VertexAdapter } from './adapters/vertex'

// Re-export engine
export { TemplateEngine } from '../lib/template-engine'
export type { PlatformId } from '../types/platform'
// Re-export types
export type { PlatformAdapter, TemplateContext, TemplateMetadata, TemplateRenderResult } from '../types/template'
export { ColabAdapter } from './adapters/colab'
// Re-export adapters
export { KaggleAdapter } from './adapters/kaggle'
export { RunPodAdapter } from './adapters/runpod'
export { VertexAdapter } from './adapters/vertex'

/**
 * Get adapter for a platform
 */
export function getAdapter(platformId: PlatformId): PlatformAdapter {
  switch (platformId) {
    case 'kaggle-p100':
    case 'kaggle-t4x2':
    case 'kaggle-cpu':
      return new KaggleAdapter(platformId)
    case 'colab-free':
    case 'colab-pro':
      return new ColabAdapter(platformId)
    case 'vertex-a100':
    case 'vertex-t4':
      return new VertexAdapter(platformId)
    case 'runpod-a100':
    case 'runpod-3090':
      return new RunPodAdapter(platformId)
    default:
      throw new Error(`Unknown platform: ${platformId}`)
  }
}

/**
 * Create a fully configured template engine with all adapters registered
 */
export function createTemplateEngine(): TemplateEngine {
  const engine = new TemplateEngine()

  // Register all platform adapters
  engine.registerAdapter(new KaggleAdapter('kaggle-p100'))
  engine.registerAdapter(new KaggleAdapter('kaggle-t4x2'))
  engine.registerAdapter(new ColabAdapter('colab-free'))
  engine.registerAdapter(new ColabAdapter('colab-pro'))
  engine.registerAdapter(new VertexAdapter('vertex-a100'))
  engine.registerAdapter(new RunPodAdapter('runpod-a100'))

  return engine
}

/**
 * Available template names
 */
export const AVAILABLE_TEMPLATES = ['training', 'inference'] as const
export type TemplateName = (typeof AVAILABLE_TEMPLATES)[number]

/**
 * Template descriptions
 */
export const TEMPLATE_DESCRIPTIONS: Record<TemplateName, string> = {
  training: 'Base training template for sequence-to-sequence translation models',
  inference: 'Inference template for generating predictions',
}

/**
 * Platform display names
 */
export const PLATFORM_DISPLAY_NAMES: Record<PlatformId, string> = {
  'kaggle-p100': 'Kaggle P100 (16GB)',
  'kaggle-t4x2': 'Kaggle T4 x2 (30GB)',
  'kaggle-cpu': 'Kaggle CPU',
  'colab-free': 'Colab Free (T4 15GB)',
  'colab-pro': 'Colab Pro (A100 40GB)',
  'vertex-a100': 'Vertex AI A100 (40GB)',
  'vertex-t4': 'Vertex AI T4 (16GB)',
  'runpod-a100': 'RunPod A100 (40-80GB)',
  'runpod-3090': 'RunPod RTX 3090 (24GB)',
  local: 'Local Machine',
}
