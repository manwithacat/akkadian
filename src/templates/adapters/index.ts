/**
 * Platform Adapters Registry
 *
 * Exports all platform adapters and provides registration utilities.
 */

import type { PlatformId } from '../../types/platform'
import type { PlatformAdapter } from '../../types/template'
import { ColabAdapter } from './colab'
import { KaggleAdapter } from './kaggle'
import { RunPodAdapter } from './runpod'
import { VertexAdapter } from './vertex'

/**
 * All available adapters
 */
export const adapters: PlatformAdapter[] = [
  new KaggleAdapter('kaggle-p100'),
  new KaggleAdapter('kaggle-t4x2'),
  new KaggleAdapter('kaggle-cpu'),
  new ColabAdapter('colab-free'),
  new ColabAdapter('colab-pro'),
  new VertexAdapter('vertex-a100'),
  new VertexAdapter('vertex-t4'),
  new RunPodAdapter('runpod-a100'),
  new RunPodAdapter('runpod-3090'),
]

/**
 * Get adapter by platform ID
 */
export function getAdapter(platform: PlatformId): PlatformAdapter | undefined {
  return adapters.find((a) => a.id === platform)
}

/**
 * Get all adapters for a category
 */
export function getAdaptersByCategory(category: 'kaggle' | 'colab' | 'vertex' | 'runpod'): PlatformAdapter[] {
  return adapters.filter((a) => a.id.startsWith(category))
}

export { ColabAdapter } from './colab'
export { KaggleAdapter } from './kaggle'
export { RunPodAdapter } from './runpod'
export { VertexAdapter } from './vertex'
