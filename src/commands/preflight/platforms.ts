import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success } from '../../lib/output'

/**
 * Platform resource profiles for pre-flight checks
 */
export interface PlatformProfile {
  name: string
  description: string
  gpu: {
    name: string
    vram_gb: number
    cuda_compute: number
    tensor_cores: boolean
    fp16_tflops: number
  }
  memory: {
    system_gb: number
    available_gb: number  // After OS/driver overhead
  }
  disk: {
    total_gb: number
    working_gb: number  // /kaggle/working or equivalent
  }
  time: {
    max_hours: number
  }
  network: {
    internet_enabled: boolean
  }
}

/**
 * Known platform profiles
 */
export const PLATFORMS: Record<string, PlatformProfile> = {
  'kaggle-p100': {
    name: 'Kaggle P100',
    description: 'Kaggle GPU notebook with Tesla P100',
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
      // IMPORTANT: /kaggle/working + HF cache share same writable partition
      // Effective limit is ~5-10GB for outputs after HF cache downloads
      // Being conservative to avoid disk space failures
      working_gb: 10,
    },
    time: {
      max_hours: 9,
    },
    network: {
      internet_enabled: true,  // Can be disabled per kernel
    },
  },

  'kaggle-t4x2': {
    name: 'Kaggle T4 x2',
    description: 'Kaggle GPU notebook with 2x Tesla T4',
    gpu: {
      name: 'Tesla T4',
      vram_gb: 15,  // Per GPU
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
      working_gb: 10,  // Conservative estimate
    },
    time: {
      max_hours: 9,
    },
    network: {
      internet_enabled: true,
    },
  },

  'kaggle-cpu': {
    name: 'Kaggle CPU',
    description: 'Kaggle CPU-only notebook',
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
      working_gb: 10,  // Conservative estimate
    },
    time: {
      max_hours: 9,
    },
    network: {
      internet_enabled: true,
    },
  },

  'colab-free': {
    name: 'Colab Free',
    description: 'Google Colab free tier (T4 GPU)',
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
      max_hours: 12,  // Can be less due to idle timeout
    },
    network: {
      internet_enabled: true,
    },
  },

  'colab-pro': {
    name: 'Colab Pro',
    description: 'Google Colab Pro (A100 GPU)',
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
  },

  'local-rtx3090': {
    name: 'Local RTX 3090',
    description: 'Local machine with RTX 3090',
    gpu: {
      name: 'GeForce RTX 3090',
      vram_gb: 24,
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
  },
}

// CLI command to list platforms
const PlatformsArgs = z.object({
  json: z.boolean().default(false).describe('Output as JSON'),
})

export const platforms: CommandDefinition<typeof PlatformsArgs> = {
  name: 'preflight platforms',
  description: 'List available platform profiles for pre-flight checks',
  help: `
Lists all known platform profiles with their resource limits.

Use these profile names with 'akk preflight check --platform <name>'.
`,
  examples: [
    'akk preflight platforms',
    'akk preflight platforms --json',
  ],
  args: PlatformsArgs,

  async run(args, ctx) {
    const platformList = Object.entries(PLATFORMS).map(([id, p]) => ({
      id,
      name: p.name,
      description: p.description,
      gpu: p.gpu.name,
      vram_gb: p.gpu.vram_gb,
      disk_gb: p.disk.working_gb,
      max_hours: p.time.max_hours,
    }))

    return success({
      platforms: platformList,
      count: platformList.length,
    })
  },
}
