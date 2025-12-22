/**
 * Template Generation Integration Tests
 *
 * Tests the full template generation pipeline and validates
 * generated notebooks against golden fixtures.
 */

import { describe, it, expect, beforeAll } from 'bun:test'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'
import { createTemplateEngine } from '../../src/templates'
import { validateNotebook, isNotebookReady, getValidationSummary } from '../../src/lib/notebook-validator'
import type { TemplateContext } from '../../src/types/template'
import type { PlatformId } from '../../src/types/platform'
import type { ToolId } from '../../src/types/tools'

// Load golden fixtures
interface GoldenFixture {
  description: string
  template: string
  platform: PlatformId
  tools: ToolId[]
  expectedStructure: {
    minCells: number
    hasConfig: boolean
    hasDataLoading: boolean
    hasTrainingLoop: boolean
    hasCheckpointSave?: boolean
    hasGCSSetup?: boolean
    hasToolDependencies?: boolean
  }
  expectedPatterns: {
    platformPaths: string[]
    configKeys: string[]
    imports: string[]
    colabSpecific?: string[]
    toolImports?: Record<string, string[]>
  }
  mustNotContain: string[]
  validation: {
    expectedScore: number
    allowedWarnings: string[]
  }
}

const fixturesDir = join(__dirname, '../fixtures/golden')

function loadFixture(name: string): GoldenFixture {
  const path = join(fixturesDir, `${name}.json`)
  return JSON.parse(readFileSync(path, 'utf-8'))
}

describe('Template Generation', () => {
  let engine: ReturnType<typeof createTemplateEngine>

  beforeAll(() => {
    engine = createTemplateEngine()
  })

  describe('Kaggle P100 Training Template', () => {
    const fixture = loadFixture('training-kaggle-p100')
    let result: ReturnType<typeof engine.render>
    let notebookCode: string

    beforeAll(() => {
      const ctx: TemplateContext = {
        platform: fixture.platform,
        model: { name: 'facebook/nllb-200-distilled-600M' },
        training: {
          epochs: 10,
          batch_size: 2,
          learning_rate: 5e-5,
        },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
      }
      result = engine.render(fixture.template, ctx)
      notebookCode = result.notebook.cells
        .filter(c => c.cell_type === 'code')
        .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
        .join('\n')
    })

    it('generates a notebook with minimum required cells', () => {
      expect(result.notebook.cells.length).toBeGreaterThanOrEqual(fixture.expectedStructure.minCells)
    })

    it('includes platform-specific paths', () => {
      for (const path of fixture.expectedPatterns.platformPaths) {
        expect(notebookCode).toContain(path)
      }
    })

    it('includes all required config keys', () => {
      for (const key of fixture.expectedPatterns.configKeys) {
        expect(notebookCode.toLowerCase()).toContain(key.toLowerCase())
      }
    })

    it('includes required imports', () => {
      for (const imp of fixture.expectedPatterns.imports) {
        expect(notebookCode).toContain(imp)
      }
    })

    it('does not contain unsubstituted template variables', () => {
      for (const pattern of fixture.mustNotContain) {
        expect(notebookCode).not.toContain(pattern)
      }
    })

    it('passes validation with high score', () => {
      const validation = validateNotebook(result.notebook, { platform: fixture.platform })
      expect(validation.score).toBeGreaterThanOrEqual(80)

      // Filter out allowed warnings
      const significantErrors = validation.errors.filter(
        e => !fixture.validation.allowedWarnings.includes(e.code)
      )
      expect(significantErrors).toHaveLength(0)
    })

    it('is marked as ready for operation', () => {
      expect(isNotebookReady(result.notebook, fixture.platform)).toBe(true)
    })
  })

  describe('Colab Pro Training Template', () => {
    const fixture = loadFixture('training-colab-pro')
    let result: ReturnType<typeof engine.render>
    let notebookCode: string

    beforeAll(() => {
      const ctx: TemplateContext = {
        platform: fixture.platform,
        model: { name: 'facebook/nllb-200-distilled-600M' },
        training: {
          epochs: 10,
          batch_size: 4,
          learning_rate: 5e-5,
        },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
        gcs: {
          bucket: 'test-bucket',
          project: 'test-project',
        },
      }
      result = engine.render(fixture.template, ctx)
      notebookCode = result.notebook.cells
        .filter(c => c.cell_type === 'code')
        .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
        .join('\n')
    })

    it('includes Colab-specific imports', () => {
      if (fixture.expectedPatterns.colabSpecific) {
        for (const pattern of fixture.expectedPatterns.colabSpecific) {
          expect(notebookCode).toContain(pattern)
        }
      }
    })

    it('includes GCS paths', () => {
      expect(notebookCode).toContain('gs://')
    })

    it('does not contain Kaggle paths', () => {
      expect(notebookCode).not.toContain('/kaggle/')
    })

    it('passes platform validation', () => {
      const validation = validateNotebook(result.notebook, { platform: fixture.platform })
      expect(validation.valid).toBe(true)
    })
  })

  describe('Colab Pro with ML Tools', () => {
    const fixture = loadFixture('training-colab-pro-with-tools')
    let result: ReturnType<typeof engine.render>
    let notebookCode: string

    beforeAll(() => {
      const ctx: TemplateContext = {
        platform: fixture.platform,
        model: { name: 'facebook/nllb-200-distilled-600M' },
        training: {
          epochs: 10,
          batch_size: 4,
          learning_rate: 5e-5,
        },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
        tools: {
          enabled: fixture.tools,
        },
      }
      result = engine.render(fixture.template, ctx)
      notebookCode = result.notebook.cells
        .filter(c => c.cell_type === 'code')
        .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
        .join('\n')
    })

    it('includes more cells with tools', () => {
      expect(result.notebook.cells.length).toBeGreaterThanOrEqual(fixture.expectedStructure.minCells)
    })

    it('includes tool-specific imports', () => {
      if (fixture.expectedPatterns.toolImports) {
        for (const [tool, imports] of Object.entries(fixture.expectedPatterns.toolImports)) {
          if (fixture.tools.includes(tool as ToolId)) {
            for (const imp of imports) {
              expect(notebookCode).toContain(imp)
            }
          }
        }
      }
    })

    it('includes sacrebleu evaluation functions', () => {
      if (fixture.tools.includes('sacrebleu')) {
        expect(notebookCode).toContain('evaluate_translations')
        expect(notebookCode).toContain('BLEU')
      }
    })

    it('includes QLoRA configuration', () => {
      if (fixture.tools.includes('qlora')) {
        expect(notebookCode).toContain('BitsAndBytesConfig')
        expect(notebookCode).toContain('LoraConfig')
      }
    })

    it('includes HPO with Optuna', () => {
      if (fixture.tools.includes('hpo')) {
        expect(notebookCode).toContain('optuna')
        expect(notebookCode).toContain('create_study')
      }
    })

    it('passes tool validation', () => {
      const validation = validateNotebook(result.notebook, { platform: fixture.platform })
      expect(validation.valid).toBe(true)
    })
  })
})

describe('All Platforms Generate Valid Notebooks', () => {
  const platforms: PlatformId[] = [
    'kaggle-p100',
    'kaggle-t4x2',
    'colab-free',
    'colab-pro',
    'vertex-a100',
    'runpod-a100',
  ]

  const engine = createTemplateEngine()

  for (const platform of platforms) {
    it(`generates valid training notebook for ${platform}`, () => {
      const ctx: TemplateContext = {
        platform,
        model: { name: 'facebook/nllb-200-distilled-600M' },
        training: { epochs: 5, batch_size: 2 },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
      }

      const result = engine.render('training', ctx)
      const validation = validateNotebook(result.notebook, { platform })

      // Should have no critical errors
      const criticalErrors = validation.errors.filter(
        e => !['MISSING_KERNELSPEC', 'PLACEHOLDER_VALUE'].includes(e.code)
      )
      expect(criticalErrors).toHaveLength(0)

      // Score should be reasonable
      expect(validation.score).toBeGreaterThanOrEqual(70)
    })

    it(`generates valid inference notebook for ${platform}`, () => {
      const ctx: TemplateContext = {
        platform,
        model: { name: 'facebook/nllb-200-distilled-600M' },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
      }

      const result = engine.render('inference', ctx)
      const validation = validateNotebook(result.notebook, { platform })

      // Should have no critical errors
      const criticalErrors = validation.errors.filter(
        e => !['MISSING_KERNELSPEC', 'PLACEHOLDER_VALUE', 'MISSING_CONFIG'].includes(e.code)
      )
      expect(criticalErrors).toHaveLength(0)
    })
  }
})

describe('All Tool Combinations Generate Valid Notebooks', () => {
  const engine = createTemplateEngine()
  const tools: ToolId[] = ['sacrebleu', 'qlora', 'accelerate', 'onnx', 'streaming', 'hpo']

  // Test individual tools
  for (const tool of tools) {
    it(`generates valid notebook with ${tool} tool`, () => {
      const ctx: TemplateContext = {
        platform: 'colab-pro',
        model: { name: 'facebook/nllb-200-distilled-600M' },
        training: { epochs: 5, batch_size: 2 },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
        tools: { enabled: [tool] },
      }

      const result = engine.render('training', ctx)
      const validation = validateNotebook(result.notebook, { platform: 'colab-pro' })

      // Should have no critical errors
      const criticalErrors = validation.errors.filter(
        e => !['MISSING_KERNELSPEC', 'PLACEHOLDER_VALUE'].includes(e.code)
      )
      expect(criticalErrors).toHaveLength(0)
    })
  }

  // Test common combinations
  const commonCombinations: ToolId[][] = [
    ['sacrebleu', 'qlora'],
    ['sacrebleu', 'hpo'],
    ['qlora', 'accelerate'],
    ['sacrebleu', 'qlora', 'hpo'],
    ['sacrebleu', 'qlora', 'accelerate', 'hpo'],
  ]

  for (const combo of commonCombinations) {
    it(`generates valid notebook with ${combo.join(', ')} tools`, () => {
      const ctx: TemplateContext = {
        platform: 'colab-pro',
        model: { name: 'facebook/nllb-200-distilled-600M' },
        training: { epochs: 5, batch_size: 2 },
        languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
        tools: { enabled: combo },
      }

      const result = engine.render('training', ctx)
      const validation = validateNotebook(result.notebook, { platform: 'colab-pro' })

      expect(validation.valid).toBe(true)
    })
  }
})

describe('Validation Summary Output', () => {
  const engine = createTemplateEngine()

  it('generates human-readable summary for valid notebook', () => {
    const ctx: TemplateContext = {
      platform: 'kaggle-p100',
      model: { name: 'facebook/nllb-200-distilled-600M' },
      training: { epochs: 5, batch_size: 2 },
      languages: { src: 'akk_Xsux', tgt: 'eng_Latn' },
    }

    const result = engine.render('training', ctx)
    const validation = validateNotebook(result.notebook, { platform: 'kaggle-p100' })
    const summary = getValidationSummary(validation)

    expect(summary).toContain('Score:')
    expect(summary).toContain('structure')
    expect(summary).toContain('syntax')
  })
})
