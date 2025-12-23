/**
 * Kaggle Adapter Unit Tests
 *
 * Tests Kaggle platform adapter including:
 * - Setup cell generation with warning suppression
 * - Data loading templates
 * - Checkpoint saving
 * - Output saving
 */

import { describe, it, expect } from 'bun:test'
import { KaggleAdapter } from '../../src/templates/adapters/kaggle'
import type { TemplateContext } from '../../src/types/template'

describe('KaggleAdapter', () => {
  const adapter = new KaggleAdapter('kaggle-p100')

  describe('generateSetupCell', () => {
    const ctx: TemplateContext = {
      platform: 'kaggle-p100',
      model: { name: 'test-model' },
    }

    it('includes warning suppression before imports', () => {
      const setup = adapter.generateSetupCell(ctx)

      // Warning suppression should appear early in the setup
      expect(setup).toContain('PyTorch-only')  // Default framework
      expect(setup).toContain('import warnings')
      expect(setup).toContain("warnings.filterwarnings")
    })

    it('suppresses CUDA library warnings', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('TF_CPP_MIN_LOG_LEVEL')
      expect(setup).toContain('CUDA_DEVICE_ORDER')
      expect(setup).toContain('TF_ENABLE_ONEDNN_OPTS')
    })

    it('suppresses Python FutureWarning', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('FutureWarning')
    })

    it('suppresses PyTorch UserWarnings', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('UserWarning')
      expect(setup).toContain('torch.cuda.amp')
    })

    it('suppresses protobuf MessageFactory warnings', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('MessageFactory')
      expect(setup).toContain('SymbolDatabase')
    })

    it('sets transformers verbosity to error', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('TRANSFORMERS_VERBOSITY')
      expect(setup).toContain('error')
    })

    it('includes tokenizers parallelism setting', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('TOKENIZERS_PARALLELISM')
    })

    it('includes PyTorch memory optimization', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('PYTORCH_CUDA_ALLOC_CONF')
      expect(setup).toContain('expandable_segments')
    })

    it('includes environment detection', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('IS_KAGGLE')
      expect(setup).toContain('KAGGLE_KERNEL_RUN_TYPE')
    })

    it('includes GPU detection', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('nvidia-smi')
    })

    it('sets up correct paths', () => {
      const setup = adapter.generateSetupCell(ctx)

      expect(setup).toContain('/kaggle/input/')
      expect(setup).toContain('/kaggle/working')
      expect(setup).toContain('CHECKPOINT_PATH')
      expect(setup).toContain('MODEL_CACHE')
    })
  })

  describe('getPaths', () => {
    it('returns correct Kaggle paths', () => {
      const ctx: TemplateContext = {
        platform: 'kaggle-p100',
        model: { name: 'test' },
        competition: {
          competition: { slug: 'my-competition', name: 'My Comp', platform: 'kaggle', metric: 'bleu', metric_direction: 'maximize' },
          project: { name: 'test', version: '0.1.0' },
          models: {},
          kernels: {},
          submissions: { total: 0, history: [] },
          training: { default_platform: 'kaggle-p100', default_epochs: 10, default_batch_size: 2, fp16: true },
          paths: { notebooks: 'notebooks', models: 'models', submissions: 'submissions', datasets: 'datasets', artifacts: 'artifacts' },
        },
      }

      const paths = adapter.getPaths(ctx)

      expect(paths.input).toBe('/kaggle/input/my-competition')
      expect(paths.output).toBe('/kaggle/working')
      expect(paths.checkpoint).toBe('/kaggle/working/checkpoints')
      expect(paths.model_cache).toBe('/kaggle/working/.cache/huggingface')
    })
  })

  describe('generateDataLoading', () => {
    it('includes CSV loading logic', () => {
      const ctx: TemplateContext = {
        platform: 'kaggle-p100',
        model: { name: 'test' },
      }
      const dataLoading = adapter.generateDataLoading(ctx)

      expect(dataLoading).toContain('glob.glob')
      expect(dataLoading).toContain('train_df')
      expect(dataLoading).toContain('pd.read_csv')
    })
  })

  describe('generateCheckpointSave', () => {
    it('includes checkpoint saving functions', () => {
      const ctx: TemplateContext = {
        platform: 'kaggle-p100',
        model: { name: 'test' },
      }
      const checkpointSave = adapter.generateCheckpointSave(ctx)

      expect(checkpointSave).toContain('save_checkpoint')
      expect(checkpointSave).toContain('load_latest_checkpoint')
      expect(checkpointSave).toContain('save_pretrained')
    })
  })

  describe('generateOutputSave', () => {
    it('includes output saving functions', () => {
      const ctx: TemplateContext = {
        platform: 'kaggle-p100',
        model: { name: 'test' },
      }
      const outputSave = adapter.generateOutputSave(ctx)

      expect(outputSave).toContain('save_model_for_submission')
      expect(outputSave).toContain('save_predictions')
      expect(outputSave).toContain('save_training_log')
    })
  })

  describe('getRecommendedBatchSize', () => {
    it('returns appropriate batch size for model', () => {
      const batchSize = adapter.getRecommendedBatchSize('facebook/nllb-200-distilled-600M')
      expect(batchSize).toBeGreaterThan(0)
      expect(batchSize).toBeLessThanOrEqual(8)
    })
  })
})

describe('KaggleAdapter platform variations', () => {
  it('identifies P100 platform correctly', () => {
    const adapter = new KaggleAdapter('kaggle-p100')
    expect(adapter.name).toBe('Kaggle P100')
  })

  it('identifies T4x2 platform correctly', () => {
    const adapter = new KaggleAdapter('kaggle-t4x2')
    expect(adapter.name).toBe('Kaggle T4 x2')
  })
})
