/**
 * Notebook Validator Unit Tests
 */

import { describe, it, expect } from 'bun:test'
import {
  validateStructure,
  validateSyntax,
  validateConfig,
  validateDependencies,
  validatePlatform,
  validateTools,
  validateSecurity,
  validateNotebook,
  isNotebookReady,
  getValidationSummary,
} from '../../src/lib/notebook-validator'
import type { NotebookContent } from '../../src/types/template'

// Helper to create a minimal valid notebook
function createNotebook(cells: Array<{ type: 'code' | 'markdown', source: string }>): NotebookContent {
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
    },
    cells: cells.map(c => ({
      cell_type: c.type,
      source: c.source.split('\n'),
      metadata: {},
      ...(c.type === 'code' ? { execution_count: null, outputs: [] } : {}),
    })),
  }
}

describe('validateStructure', () => {
  it('accepts a valid notebook structure', () => {
    const notebook = createNotebook([
      { type: 'markdown', source: '# Test Notebook' },
      { type: 'code', source: 'print("hello")' },
    ])
    const result = validateStructure(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('rejects invalid nbformat', () => {
    const notebook = createNotebook([{ type: 'code', source: 'x = 1' }])
    notebook.nbformat = 3
    const result = validateStructure(notebook)
    expect(result.errors.some(e => e.code === 'INVALID_NBFORMAT')).toBe(true)
  })

  it('rejects empty notebook', () => {
    const notebook = createNotebook([])
    const result = validateStructure(notebook)
    expect(result.errors.some(e => e.code === 'EMPTY_NOTEBOOK')).toBe(true)
  })

  it('warns on empty code cells', () => {
    const notebook = createNotebook([
      { type: 'code', source: '' },
      { type: 'code', source: 'x = 1' },
    ])
    const result = validateStructure(notebook)
    expect(result.warnings.some(w => w.code === 'EMPTY_CODE_CELL')).toBe(true)
  })

  it('warns on missing kernelspec', () => {
    const notebook = createNotebook([{ type: 'code', source: 'x = 1' }])
    delete notebook.metadata.kernelspec
    const result = validateStructure(notebook)
    expect(result.warnings.some(w => w.code === 'MISSING_KERNELSPEC')).toBe(true)
  })
})

describe('validateSyntax', () => {
  it('accepts valid Python code', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'def hello():\n    print("world")' },
    ])
    const result = validateSyntax(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('detects undefined template variables', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'model = "{{model_name}}"' },
    ])
    const result = validateSyntax(notebook)
    expect(result.errors.some(e => e.code === 'UNDEFINED_TEMPLATE_VAR')).toBe(true)
  })

  it('detects Python 2 print statements', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'print "hello"' },
    ])
    const result = validateSyntax(notebook)
    expect(result.errors.some(e => e.code === 'PYTHON2_PRINT')).toBe(true)
  })

  it('warns on mixed indentation', () => {
    const notebook = createNotebook([
      { type: 'code', source: '\t  x = 1' },
    ])
    const result = validateSyntax(notebook)
    expect(result.warnings.some(w => w.code === 'MIXED_INDENTATION')).toBe(true)
  })
})

describe('validateConfig', () => {
  it('accepts complete configuration', () => {
    const notebook = createNotebook([
      {
        type: 'code',
        source: `
CONFIG = {
    "model_name": "facebook/nllb-200-distilled-600M",
    "batch_size": 4,
}
OUTPUT_PATH = "/output"
`,
      },
    ])
    const result = validateConfig(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('warns on placeholder values', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'BUCKET = "your-bucket"' },
    ])
    const result = validateConfig(notebook)
    expect(result.warnings.some(w => w.code === 'PLACEHOLDER_VALUE')).toBe(true)
  })

  it('warns on TODO placeholders', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'MODEL = "TODO"' },
    ])
    const result = validateConfig(notebook)
    expect(result.warnings.some(w => w.code === 'PLACEHOLDER_VALUE')).toBe(true)
  })
})

describe('validateDependencies', () => {
  it('accepts notebook with proper installs', () => {
    const notebook = createNotebook([
      { type: 'code', source: '!pip install "transformers>=4.0.0"' },
      { type: 'code', source: 'from transformers import AutoModel' },
    ])
    const result = validateDependencies(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('warns on missing pip install for import', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'from peft import LoraConfig' },
    ])
    const result = validateDependencies(notebook)
    expect(result.warnings.some(w => w.code === 'MISSING_INSTALL')).toBe(true)
  })
})

describe('validatePlatform', () => {
  it('validates Kaggle paths', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'INPUT_PATH = "/kaggle/input/competition"' },
    ])
    const result = validatePlatform(notebook, 'kaggle-p100')
    expect(result.errors).toHaveLength(0)
  })

  it('warns on missing Kaggle paths', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'INPUT_PATH = "/data"' },
    ])
    const result = validatePlatform(notebook, 'kaggle-p100')
    expect(result.warnings.some(w => w.code === 'MISSING_KAGGLE_PATHS')).toBe(true)
  })

  it('validates Colab paths', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'INPUT_PATH = "/content/data"' },
    ])
    const result = validatePlatform(notebook, 'colab-pro')
    expect(result.errors).toHaveLength(0)
  })

  it('detects missing Drive mount', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'DATA = "/content/drive/MyDrive/data"' },
    ])
    const result = validatePlatform(notebook, 'colab-pro')
    expect(result.errors.some(e => e.code === 'MISSING_DRIVE_MOUNT')).toBe(true)
  })
})

describe('validateTools', () => {
  it('detects sacrebleu tool', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'from sacrebleu.metrics import BLEU, CHRF' },
    ])
    const result = validateTools(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('detects qlora tool', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'from peft import LoraConfig, get_peft_model' },
    ])
    const result = validateTools(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('fails when expected tool is missing', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'print("no tools")' },
    ])
    const result = validateTools(notebook, ['sacrebleu'])
    expect(result.errors.some(e => e.code === 'MISSING_TOOL')).toBe(true)
  })
})

describe('validateSecurity', () => {
  it('accepts safe code', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'import os\npath = os.getenv("PATH")' },
    ])
    const result = validateSecurity(notebook)
    expect(result.errors).toHaveLength(0)
  })

  it('detects hardcoded API keys', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'API_KEY = "sk-1234567890123456789012345678901234567890123456789"' },
    ])
    const result = validateSecurity(notebook)
    expect(result.errors.some(e => e.code === 'HARDCODED_SECRET')).toBe(true)
  })

  it('warns on shell injection risk', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'subprocess.call(cmd, shell=True)' },
    ])
    const result = validateSecurity(notebook)
    expect(result.warnings.some(w => w.code === 'SECURITY_RISK')).toBe(true)
  })
})

describe('validateNotebook (comprehensive)', () => {
  it('validates a complete training notebook', () => {
    const notebook = createNotebook([
      { type: 'markdown', source: '# Training Notebook' },
      {
        type: 'code',
        source: `
!pip install "transformers>=4.35.0" "datasets>=2.14.0"

CONFIG = {
    "model_name": "facebook/nllb-200-distilled-600M",
    "batch_size": 4,
    "num_epochs": 10,
}
OUTPUT_PATH = "/kaggle/working"
INPUT_PATH = "/kaggle/input/competition"
`,
      },
      {
        type: 'code',
        source: `
from transformers import AutoModelForSeq2SeqLM
import torch
model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
`,
      },
    ])

    const result = validateNotebook(notebook, { platform: 'kaggle-p100' })
    expect(result.valid).toBe(true)
    expect(result.ready).toBe(true)
    expect(result.score).toBeGreaterThanOrEqual(80)
  })

  it('returns detailed category breakdown', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'x = 1' },
    ])
    const result = validateNotebook(notebook)
    expect(result.categories).toHaveProperty('structure')
    expect(result.categories).toHaveProperty('syntax')
    expect(result.categories).toHaveProperty('config')
    expect(result.categories).toHaveProperty('security')
  })
})

describe('isNotebookReady', () => {
  it('returns true for ready notebook', () => {
    const notebook = createNotebook([
      {
        type: 'code',
        source: `
CONFIG = {"model_name": "test", "batch_size": 2}
OUTPUT_PATH = "/kaggle/working"
`,
      },
    ])
    expect(isNotebookReady(notebook, 'kaggle-p100')).toBe(true)
  })

  it('returns false for notebook with errors', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'model = "{{undefined}}"' },
    ])
    expect(isNotebookReady(notebook)).toBe(false)
  })
})

describe('getValidationSummary', () => {
  it('generates readable summary', () => {
    const notebook = createNotebook([
      { type: 'code', source: 'x = 1' },
    ])
    const result = validateNotebook(notebook)
    const summary = getValidationSummary(result)
    expect(summary).toContain('Score:')
    expect(summary).toContain('Category Breakdown:')
  })
})
