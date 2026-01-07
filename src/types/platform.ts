/**
 * Platform Types
 *
 * Defines compute platform profiles with resource limits.
 * Refactored from preflight/platforms.ts for broader use.
 */

/**
 * Supported platform identifiers
 */
export type PlatformId =
  | 'kaggle-p100'
  | 'kaggle-t4x2'
  | 'kaggle-cpu'
  | 'colab-free'
  | 'colab-pro'
  | 'vertex-a100'
  | 'vertex-t4'
  | 'runpod-a100'
  | 'runpod-3090'
  | 'local'

/**
 * GPU specification
 */
export interface GPUSpec {
  name: string
  vram_gb: number
  cuda_compute: number
  tensor_cores: boolean
  fp16_tflops: number
}

/**
 * Memory specification
 */
export interface MemorySpec {
  system_gb: number
  available_gb: number
}

/**
 * Disk specification
 */
export interface DiskSpec {
  total_gb: number
  working_gb: number
}

/**
 * Time limits
 */
export interface TimeSpec {
  max_hours: number
}

/**
 * Network capabilities
 */
export interface NetworkSpec {
  internet_enabled: boolean
}

/**
 * Platform resource profile
 */
export interface PlatformProfile {
  id: PlatformId
  name: string
  description: string
  category: 'kaggle' | 'colab' | 'vertex' | 'runpod' | 'local'
  gpu: GPUSpec
  memory: MemorySpec
  disk: DiskSpec
  time: TimeSpec
  network: NetworkSpec
  pricing?: {
    type: 'free' | 'quota' | 'paid'
    rate_per_hour?: number
    currency?: string
  }
}

/**
 * Platform paths for different environments
 */
export interface PlatformPaths {
  input: string
  output: string
  checkpoint: string
  model_cache: string
}

/**
 * Known platform profiles
 */
export const PLATFORMS: Record<PlatformId, PlatformProfile> = {
  'kaggle-p100': {
    id: 'kaggle-p100',
    name: 'Kaggle P100',
    description: 'Kaggle GPU notebook with Tesla P100',
    category: 'kaggle',
    gpu: {
      name: 'Tesla P100-PCIE-16GB',
      vram_gb: 16,
      cuda_compute: 6.0,
      tensor_cores: false,
      fp16_tflops: 10.6,
    },
    memory: {
      system_gb: 13,
      available_gb: 11,
    },
    disk: {
      total_gb: 73,
      working_gb: 20,
    },
    time: {
      max_hours: 9,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'quota',
    },
  },

  'kaggle-t4x2': {
    id: 'kaggle-t4x2',
    name: 'Kaggle T4 x2',
    description: 'Kaggle GPU notebook with 2x Tesla T4',
    category: 'kaggle',
    gpu: {
      name: 'Tesla T4',
      vram_gb: 15,
      cuda_compute: 7.5,
      tensor_cores: true,
      fp16_tflops: 65,
    },
    memory: {
      system_gb: 13,
      available_gb: 11,
    },
    disk: {
      total_gb: 73,
      working_gb: 20,
    },
    time: {
      max_hours: 9,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'quota',
    },
  },

  'kaggle-cpu': {
    id: 'kaggle-cpu',
    name: 'Kaggle CPU',
    description: 'Kaggle CPU-only notebook',
    category: 'kaggle',
    gpu: {
      name: 'None',
      vram_gb: 0,
      cuda_compute: 0,
      tensor_cores: false,
      fp16_tflops: 0,
    },
    memory: {
      system_gb: 13,
      available_gb: 11,
    },
    disk: {
      total_gb: 73,
      working_gb: 20,
    },
    time: {
      max_hours: 9,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'free',
    },
  },

  'colab-free': {
    id: 'colab-free',
    name: 'Colab Free',
    description: 'Google Colab free tier (T4 GPU)',
    category: 'colab',
    gpu: {
      name: 'Tesla T4',
      vram_gb: 15,
      cuda_compute: 7.5,
      tensor_cores: true,
      fp16_tflops: 65,
    },
    memory: {
      system_gb: 12.7,
      available_gb: 10,
    },
    disk: {
      total_gb: 78,
      working_gb: 50,
    },
    time: {
      max_hours: 12,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'free',
    },
  },

  'colab-pro': {
    id: 'colab-pro',
    name: 'Colab Pro',
    description: 'Google Colab Pro (A100 GPU)',
    category: 'colab',
    gpu: {
      name: 'A100-SXM4-40GB',
      vram_gb: 40,
      cuda_compute: 8.0,
      tensor_cores: true,
      fp16_tflops: 312,
    },
    memory: {
      system_gb: 83,
      available_gb: 75,
    },
    disk: {
      total_gb: 166,
      working_gb: 100,
    },
    time: {
      max_hours: 24,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'paid',
      rate_per_hour: 0.5,
      currency: 'USD',
    },
  },

  'vertex-a100': {
    id: 'vertex-a100',
    name: 'Vertex AI A100',
    description: 'Google Cloud Vertex AI with A100 GPU',
    category: 'vertex',
    gpu: {
      name: 'A100-SXM4-40GB',
      vram_gb: 40,
      cuda_compute: 8.0,
      tensor_cores: true,
      fp16_tflops: 312,
    },
    memory: {
      system_gb: 85,
      available_gb: 80,
    },
    disk: {
      total_gb: 500,
      working_gb: 400,
    },
    time: {
      max_hours: 168, // 7 days
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'paid',
      rate_per_hour: 3.67,
      currency: 'USD',
    },
  },

  'vertex-t4': {
    id: 'vertex-t4',
    name: 'Vertex AI T4',
    description: 'Google Cloud Vertex AI with T4 GPU',
    category: 'vertex',
    gpu: {
      name: 'Tesla T4',
      vram_gb: 16,
      cuda_compute: 7.5,
      tensor_cores: true,
      fp16_tflops: 65,
    },
    memory: {
      system_gb: 30,
      available_gb: 28,
    },
    disk: {
      total_gb: 200,
      working_gb: 150,
    },
    time: {
      max_hours: 168,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'paid',
      rate_per_hour: 0.35,
      currency: 'USD',
    },
  },

  'runpod-a100': {
    id: 'runpod-a100',
    name: 'RunPod A100',
    description: 'RunPod serverless with A100 GPU',
    category: 'runpod',
    gpu: {
      name: 'A100-SXM4-80GB',
      vram_gb: 80,
      cuda_compute: 8.0,
      tensor_cores: true,
      fp16_tflops: 312,
    },
    memory: {
      system_gb: 125,
      available_gb: 120,
    },
    disk: {
      total_gb: 500,
      working_gb: 400,
    },
    time: {
      max_hours: Infinity,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'paid',
      rate_per_hour: 1.99,
      currency: 'USD',
    },
  },

  'runpod-3090': {
    id: 'runpod-3090',
    name: 'RunPod RTX 3090',
    description: 'RunPod serverless with RTX 3090',
    category: 'runpod',
    gpu: {
      name: 'GeForce RTX 3090',
      vram_gb: 24,
      cuda_compute: 8.6,
      tensor_cores: true,
      fp16_tflops: 71,
    },
    memory: {
      system_gb: 64,
      available_gb: 60,
    },
    disk: {
      total_gb: 200,
      working_gb: 150,
    },
    time: {
      max_hours: Infinity,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'paid',
      rate_per_hour: 0.44,
      currency: 'USD',
    },
  },

  local: {
    id: 'local',
    name: 'Local Machine',
    description: 'Local development machine',
    category: 'local',
    gpu: {
      name: 'Variable',
      vram_gb: 24, // Default assumption
      cuda_compute: 8.6,
      tensor_cores: true,
      fp16_tflops: 71,
    },
    memory: {
      system_gb: 64,
      available_gb: 56,
    },
    disk: {
      total_gb: 1000,
      working_gb: 500,
    },
    time: {
      max_hours: Infinity,
    },
    network: {
      internet_enabled: true,
    },
    pricing: {
      type: 'free',
    },
  },
}

/**
 * Default paths for each platform category
 */
export const PLATFORM_PATHS: Record<string, PlatformPaths> = {
  kaggle: {
    input: '/kaggle/input',
    output: '/kaggle/working',
    checkpoint: '/kaggle/working/checkpoints',
    model_cache: '/kaggle/working/.cache/huggingface',
  },
  colab: {
    input: '/content/data',
    output: '/content/output',
    checkpoint: '/content/checkpoints',
    model_cache: '/root/.cache/huggingface',
  },
  vertex: {
    input: '/gcs/{bucket}/datasets',
    output: '/gcs/{bucket}/output',
    checkpoint: '/gcs/{bucket}/checkpoints',
    model_cache: '/root/.cache/huggingface',
  },
  runpod: {
    input: '/workspace/data',
    output: '/workspace/output',
    checkpoint: '/workspace/checkpoints',
    model_cache: '/workspace/.cache/huggingface',
  },
  local: {
    input: './data',
    output: './output',
    checkpoint: './checkpoints',
    model_cache: '~/.cache/huggingface',
  },
}

/**
 * Get platform profile by ID
 */
export function getPlatform(id: PlatformId): PlatformProfile {
  const platform = PLATFORMS[id]
  if (!platform) {
    throw new Error(`Unknown platform: ${id}`)
  }
  return platform
}

/**
 * Get platform paths with variable substitution
 */
export function getPlatformPaths(
  platformId: PlatformId,
  variables?: { bucket?: string; competition?: string }
): PlatformPaths {
  const platform = getPlatform(platformId)
  const basePaths = PLATFORM_PATHS[platform.category]

  if (!basePaths) {
    return PLATFORM_PATHS.local
  }

  // Substitute variables
  const substitute = (path: string): string => {
    let result = path
    if (variables?.bucket) {
      result = result.replace('{bucket}', variables.bucket)
    }
    if (variables?.competition) {
      result = result.replace('{competition}', variables.competition)
    }
    return result
  }

  return {
    input: substitute(basePaths.input),
    output: substitute(basePaths.output),
    checkpoint: substitute(basePaths.checkpoint),
    model_cache: substitute(basePaths.model_cache),
  }
}

/**
 * Get platforms by category
 */
export function getPlatformsByCategory(category: PlatformProfile['category']): PlatformProfile[] {
  return Object.values(PLATFORMS).filter((p) => p.category === category)
}

/**
 * List all platform IDs
 */
export function listPlatformIds(): PlatformId[] {
  return Object.keys(PLATFORMS) as PlatformId[]
}
