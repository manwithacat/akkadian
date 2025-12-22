/**
 * Notebook Validator
 *
 * Comprehensive validation for generated notebooks to ensure they are
 * ready for operation on target platforms.
 */

import type { NotebookContent, NotebookCell, ValidationResult, ValidationError, ValidationWarning } from '../types/template'
import type { PlatformId } from '../types/platform'

/**
 * Validation check categories
 */
export type ValidationCategory =
  | 'structure'     // Notebook format, cell structure
  | 'syntax'        // Python syntax validity
  | 'config'        // Configuration completeness
  | 'dependencies'  // Package declarations
  | 'platform'      // Platform-specific requirements
  | 'tools'         // Tool integration
  | 'security'      // Security concerns

/**
 * Extended validation result with category breakdown
 */
export interface ExtendedValidationResult extends ValidationResult {
  categories: Record<ValidationCategory, {
    passed: number
    failed: number
    warnings: number
  }>
  score: number  // 0-100 readiness score
  ready: boolean // Quick check: is it ready for operation?
}

/**
 * Validation options
 */
export interface ValidationOptions {
  platform?: PlatformId
  checkSyntax?: boolean
  checkDependencies?: boolean
  checkSecurity?: boolean
  strict?: boolean  // Treat warnings as errors
}

/**
 * Validate notebook structure
 */
export function validateStructure(notebook: NotebookContent): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // Check notebook format version
  if (notebook.nbformat !== 4) {
    errors.push({
      type: 'error',
      code: 'INVALID_NBFORMAT',
      message: `Expected nbformat 4, got ${notebook.nbformat}`,
      suggestion: 'Ensure notebook is in Jupyter Notebook format v4',
    })
  }

  // Check for cells array
  if (!Array.isArray(notebook.cells)) {
    errors.push({
      type: 'error',
      code: 'MISSING_CELLS',
      message: 'Notebook has no cells array',
      suggestion: 'Notebook must have a cells array',
    })
    return { errors, warnings }
  }

  // Check for empty notebook
  if (notebook.cells.length === 0) {
    errors.push({
      type: 'error',
      code: 'EMPTY_NOTEBOOK',
      message: 'Notebook has no cells',
      suggestion: 'Add code or markdown cells to the notebook',
    })
  }

  // Check each cell
  notebook.cells.forEach((cell, index) => {
    // Check cell_type
    if (!['code', 'markdown', 'raw'].includes(cell.cell_type)) {
      errors.push({
        type: 'error',
        code: 'INVALID_CELL_TYPE',
        message: `Cell ${index} has invalid cell_type: ${cell.cell_type}`,
        line: index,
        suggestion: 'Cell type must be "code", "markdown", or "raw"',
      })
    }

    // Check source
    if (!Array.isArray(cell.source) && typeof cell.source !== 'string') {
      errors.push({
        type: 'error',
        code: 'INVALID_CELL_SOURCE',
        message: `Cell ${index} has invalid source format`,
        line: index,
        suggestion: 'Cell source must be an array of strings or a string',
      })
    }

    // Check for empty code cells
    if (cell.cell_type === 'code') {
      const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source
      if (!source.trim()) {
        warnings.push({
          type: 'warning',
          code: 'EMPTY_CODE_CELL',
          message: `Cell ${index} is an empty code cell`,
          line: index,
          suggestion: 'Consider removing empty cells or adding code',
        })
      }
    }

    // Code cells should have outputs array
    if (cell.cell_type === 'code' && !Array.isArray(cell.outputs)) {
      warnings.push({
        type: 'warning',
        code: 'MISSING_OUTPUTS',
        message: `Code cell ${index} missing outputs array`,
        line: index,
        suggestion: 'Code cells should have an outputs array (can be empty)',
      })
    }
  })

  // Check kernelspec
  if (!notebook.metadata?.kernelspec) {
    warnings.push({
      type: 'warning',
      code: 'MISSING_KERNELSPEC',
      message: 'Notebook missing kernelspec metadata',
      suggestion: 'Add kernelspec for Python 3',
    })
  }

  return { errors, warnings }
}

/**
 * Validate Python syntax in code cells
 */
export function validateSyntax(notebook: NotebookContent): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  notebook.cells.forEach((cell, cellIndex) => {
    if (cell.cell_type !== 'code') return

    const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source

    // Skip empty cells
    if (!source.trim()) return

    // Check for common syntax issues with regex patterns
    const syntaxChecks: { pattern: RegExp; code: string; message: string; severity: 'error' | 'warning' }[] = [
      // Unclosed strings (simple check)
      { pattern: /['"][^'"]*$(?!.*['"])/m, code: 'UNCLOSED_STRING', message: 'Possible unclosed string literal', severity: 'warning' },

      // Unbalanced parentheses (simple check)
      { pattern: /\([^)]*$(?!.*\))/m, code: 'UNBALANCED_PARENS', message: 'Possible unbalanced parentheses', severity: 'warning' },

      // Triple-quoted string issues
      { pattern: /"""[^"]*$/m, code: 'UNCLOSED_DOCSTRING', message: 'Possible unclosed docstring', severity: 'warning' },

      // Invalid Python keywords as identifiers (common mistakes)
      { pattern: /\bdef\s+(?:class|def|import|from|return|yield|lambda)\s*\(/, code: 'RESERVED_KEYWORD', message: 'Using reserved keyword as function name', severity: 'error' },

      // Print statement (Python 2)
      { pattern: /^\s*print\s+[^(].*$/m, code: 'PYTHON2_PRINT', message: 'Python 2 print statement detected', severity: 'error' },

      // Tab/space mixing (potential)
      { pattern: /^\t+ +/m, code: 'MIXED_INDENTATION', message: 'Mixed tabs and spaces in indentation', severity: 'warning' },
    ]

    for (const check of syntaxChecks) {
      if (check.pattern.test(source)) {
        const item = {
          type: check.severity,
          code: check.code,
          message: `Cell ${cellIndex}: ${check.message}`,
          line: cellIndex,
          suggestion: 'Review and fix the syntax issue',
        }
        if (check.severity === 'error') {
          errors.push(item as ValidationError)
        } else {
          warnings.push(item as ValidationWarning)
        }
      }
    }

    // Check for undefined template variables
    const templateVarPattern = /\{\{(\w+)\}\}/g
    let match
    while ((match = templateVarPattern.exec(source)) !== null) {
      errors.push({
        type: 'error',
        code: 'UNDEFINED_TEMPLATE_VAR',
        message: `Cell ${cellIndex}: Undefined template variable {{${match[1]}}}`,
        line: cellIndex,
        suggestion: 'Ensure all template variables are substituted',
      })
    }
  })

  return { errors, warnings }
}

/**
 * Validate configuration completeness
 */
export function validateConfig(notebook: NotebookContent): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // Extract all code
  const allCode = notebook.cells
    .filter(c => c.cell_type === 'code')
    .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
    .join('\n')

  // Check for CONFIG dictionary
  const hasConfig = /CONFIG\s*=\s*\{/.test(allCode)
  if (!hasConfig) {
    warnings.push({
      type: 'warning',
      code: 'MISSING_CONFIG',
      message: 'No CONFIG dictionary found',
      suggestion: 'Add a CONFIG dictionary with training parameters',
    })
  }

  // Check for essential variables
  const requiredVars = [
    { pattern: /model_name|MODEL_NAME/, name: 'model_name' },
    { pattern: /batch_size|BATCH_SIZE/, name: 'batch_size' },
    { pattern: /OUTPUT_PATH|output_path|output_dir/, name: 'output path' },
  ]

  for (const { pattern, name } of requiredVars) {
    if (!pattern.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'MISSING_REQUIRED_VAR',
        message: `Missing required variable: ${name}`,
        suggestion: `Define ${name} in your configuration`,
      })
    }
  }

  // Check for placeholder values
  const placeholders = [
    { pattern: /["']your-bucket["']/, name: 'GCS bucket' },
    { pattern: /["']your-project["']/, name: 'GCS project' },
    { pattern: /["']competition-name["']/, name: 'competition name' },
    { pattern: /["']TODO["']/i, name: 'TODO placeholder' },
    { pattern: /["']REPLACE_ME["']/i, name: 'REPLACE_ME placeholder' },
  ]

  for (const { pattern, name } of placeholders) {
    if (pattern.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'PLACEHOLDER_VALUE',
        message: `Placeholder value found for ${name}`,
        suggestion: `Replace placeholder with actual value for ${name}`,
      })
    }
  }

  return { errors, warnings }
}

/**
 * Validate dependency declarations
 */
export function validateDependencies(notebook: NotebookContent): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // Extract all code
  const allCode = notebook.cells
    .filter(c => c.cell_type === 'code')
    .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
    .join('\n')

  // Find imports
  const importPattern = /^(?:from\s+(\w+)|import\s+(\w+))/gm
  const imports = new Set<string>()
  let match
  while ((match = importPattern.exec(allCode)) !== null) {
    imports.add(match[1] || match[2])
  }

  // Find pip installs
  const pipPattern = /pip\s+install[^"'\n]*["']([^"']+)["']/g
  const installedPackages = new Set<string>()
  while ((match = pipPattern.exec(allCode)) !== null) {
    // Extract package name from requirement string (e.g., "transformers>=4.0.0" -> "transformers")
    const pkgName = match[1].split(/[<>=\[\]]/)[0].toLowerCase()
    installedPackages.add(pkgName)
  }

  // Check for common packages that should be installed
  const commonPackages: Record<string, string[]> = {
    transformers: ['transformers'],
    datasets: ['datasets'],
    accelerate: ['accelerate'],
    peft: ['peft'],
    bitsandbytes: ['bitsandbytes'],
    sacrebleu: ['sacrebleu'],
    optuna: ['optuna'],
    mlflow: ['mlflow'],
    torch: ['torch', 'pytorch'],
  }

  for (const [importName, packageNames] of Object.entries(commonPackages)) {
    if (imports.has(importName)) {
      const hasInstall = packageNames.some(pkg => installedPackages.has(pkg))
      if (!hasInstall) {
        warnings.push({
          type: 'warning',
          code: 'MISSING_INSTALL',
          message: `Import '${importName}' found but no pip install for ${packageNames.join(' or ')}`,
          suggestion: `Add pip install for ${packageNames[0]}`,
        })
      }
    }
  }

  // Check for install cell near the top
  const firstCodeCellIndex = notebook.cells.findIndex(c => c.cell_type === 'code')
  if (firstCodeCellIndex >= 0) {
    const firstFewCells = notebook.cells.slice(0, Math.min(5, notebook.cells.length))
    const hasInstallInFirst5 = firstFewCells.some(cell => {
      if (cell.cell_type !== 'code') return false
      const src = Array.isArray(cell.source) ? cell.source.join('') : cell.source
      return /pip\s+install/.test(src) || /install_packages/.test(src)
    })

    if (!hasInstallInFirst5 && installedPackages.size > 0) {
      warnings.push({
        type: 'warning',
        code: 'LATE_INSTALL',
        message: 'Package installation should be near the top of the notebook',
        suggestion: 'Move pip install commands to one of the first few cells',
      })
    }
  }

  return { errors, warnings }
}

/**
 * Validate platform-specific requirements
 */
export function validatePlatform(notebook: NotebookContent, platform: PlatformId): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // Extract all code
  const allCode = notebook.cells
    .filter(c => c.cell_type === 'code')
    .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
    .join('\n')

  // Platform-specific checks
  if (platform.startsWith('kaggle')) {
    // Kaggle checks
    if (!/\/kaggle\//.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'MISSING_KAGGLE_PATHS',
        message: 'No Kaggle paths detected (/kaggle/input, /kaggle/working)',
        suggestion: 'Use Kaggle-specific paths for input and output',
      })
    }

    // Check for internet access patterns that won't work
    if (/requests\.get|urllib|httpx/.test(allCode) && !/kaggle.*internet/i.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'KAGGLE_NETWORK_ACCESS',
        message: 'Network access detected - ensure kernel has internet enabled',
        suggestion: 'Enable internet in kernel settings or use Kaggle datasets',
      })
    }
  }

  if (platform.startsWith('colab')) {
    // Colab checks
    if (!/google\.colab|\/content\//.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'MISSING_COLAB_PATHS',
        message: 'No Colab paths detected (/content/)',
        suggestion: 'Use Colab-specific paths for data storage',
      })
    }

    // Check for Drive mounting if using Drive paths
    if (/\/content\/drive/.test(allCode) && !/drive\.mount/.test(allCode)) {
      errors.push({
        type: 'error',
        code: 'MISSING_DRIVE_MOUNT',
        message: 'Google Drive path used but drive.mount() not found',
        suggestion: 'Add: from google.colab import drive; drive.mount("/content/drive")',
      })
    }
  }

  if (platform.startsWith('vertex')) {
    // Vertex AI checks
    if (!/gs:\/\//.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'MISSING_GCS_PATHS',
        message: 'No GCS paths detected (gs://)',
        suggestion: 'Use GCS paths for data and model storage',
      })
    }
  }

  // Check for injection point markers still present (should be removed after injection)
  if (/---\s*INJECTION POINT:/.test(allCode) && !/INJECTION POINT.*\n.*\n.*---\s*END INJECTION POINT/.test(allCode)) {
    warnings.push({
      type: 'warning',
      code: 'EMPTY_INJECTION_POINT',
      message: 'Empty injection point detected',
      suggestion: 'Ensure all injection points have been filled',
    })
  }

  return { errors, warnings }
}

/**
 * Validate tool integrations
 */
export function validateTools(notebook: NotebookContent, expectedTools?: string[]): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // Extract all code
  const allCode = notebook.cells
    .filter(c => c.cell_type === 'code')
    .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
    .join('\n')

  // Tool detection patterns
  const toolPatterns: Record<string, RegExp> = {
    sacrebleu: /from sacrebleu|import sacrebleu|BLEU.*CHRF|evaluate_translations/,
    qlora: /BitsAndBytesConfig|load_in_4bit|LoraConfig|get_peft_model/,
    accelerate: /from accelerate|Accelerator|accelerator\.prepare/,
    onnx: /optimum\.onnxruntime|ORTModel|export_to_onnx/,
    streaming: /streaming\s*=\s*True|IterableDataset|create_streaming_dataloader/,
    hpo: /optuna|create_study|hyperparameter_search|MLflowCallback/,
  }

  // Check for expected tools
  if (expectedTools) {
    for (const tool of expectedTools) {
      const pattern = toolPatterns[tool]
      if (pattern && !pattern.test(allCode)) {
        errors.push({
          type: 'error',
          code: 'MISSING_TOOL',
          message: `Expected tool '${tool}' not found in notebook`,
          suggestion: `Add ${tool} integration or regenerate with --tools ${tool}`,
        })
      }
    }
  }

  // Check for tool dependency installation
  const detectedTools: string[] = []
  for (const [tool, pattern] of Object.entries(toolPatterns)) {
    if (pattern.test(allCode)) {
      detectedTools.push(tool)
    }
  }

  // Tool-specific dependency checks
  const toolDeps: Record<string, RegExp> = {
    sacrebleu: /pip.*install.*sacrebleu/,
    qlora: /pip.*install.*(?:bitsandbytes|peft)/,
    accelerate: /pip.*install.*accelerate/,
    onnx: /pip.*install.*optimum/,
    hpo: /pip.*install.*optuna/,
  }

  for (const tool of detectedTools) {
    const depPattern = toolDeps[tool]
    if (depPattern && !depPattern.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'MISSING_TOOL_DEPENDENCY',
        message: `Tool '${tool}' detected but dependencies may not be installed`,
        suggestion: `Ensure ${tool} dependencies are installed in the notebook`,
      })
    }
  }

  return { errors, warnings }
}

/**
 * Validate security concerns
 */
export function validateSecurity(notebook: NotebookContent): { errors: ValidationError[], warnings: ValidationWarning[] } {
  const errors: ValidationError[] = []
  const warnings: ValidationWarning[] = []

  // Extract all code
  const allCode = notebook.cells
    .filter(c => c.cell_type === 'code')
    .map(c => Array.isArray(c.source) ? c.source.join('') : c.source)
    .join('\n')

  // Check for hardcoded secrets
  const secretPatterns = [
    { pattern: /["'][A-Za-z0-9_-]{20,}["'].*(?:key|token|secret|password|api)/i, name: 'API key or secret' },
    { pattern: /AIza[A-Za-z0-9_-]{35}/, name: 'Google API key' },
    { pattern: /ghp_[A-Za-z0-9]{36}/, name: 'GitHub personal access token' },
    { pattern: /sk-[A-Za-z0-9]{48}/, name: 'OpenAI API key' },
    { pattern: /AKIA[A-Z0-9]{16}/, name: 'AWS access key' },
  ]

  for (const { pattern, name } of secretPatterns) {
    if (pattern.test(allCode)) {
      errors.push({
        type: 'error',
        code: 'HARDCODED_SECRET',
        message: `Possible ${name} hardcoded in notebook`,
        suggestion: 'Use environment variables or secrets management instead',
      })
    }
  }

  // Check for dangerous operations
  const dangerousPatterns = [
    { pattern: /subprocess\.call.*shell\s*=\s*True/, name: 'Shell injection risk' },
    { pattern: /eval\s*\([^)]*input/, name: 'Eval with user input' },
    { pattern: /exec\s*\([^)]*input/, name: 'Exec with user input' },
    { pattern: /os\.system\s*\([^)]*\+/, name: 'Command injection risk' },
  ]

  for (const { pattern, name } of dangerousPatterns) {
    if (pattern.test(allCode)) {
      warnings.push({
        type: 'warning',
        code: 'SECURITY_RISK',
        message: `Potential security risk: ${name}`,
        suggestion: 'Review and sanitize inputs before use',
      })
    }
  }

  return { errors, warnings }
}

/**
 * Run all validations on a notebook
 */
export function validateNotebook(
  notebook: NotebookContent,
  options: ValidationOptions = {}
): ExtendedValidationResult {
  const allErrors: ValidationError[] = []
  const allWarnings: ValidationWarning[] = []
  const categories: ExtendedValidationResult['categories'] = {
    structure: { passed: 0, failed: 0, warnings: 0 },
    syntax: { passed: 0, failed: 0, warnings: 0 },
    config: { passed: 0, failed: 0, warnings: 0 },
    dependencies: { passed: 0, failed: 0, warnings: 0 },
    platform: { passed: 0, failed: 0, warnings: 0 },
    tools: { passed: 0, failed: 0, warnings: 0 },
    security: { passed: 0, failed: 0, warnings: 0 },
  }

  // Structure validation
  const structure = validateStructure(notebook)
  allErrors.push(...structure.errors)
  allWarnings.push(...structure.warnings)
  categories.structure.failed = structure.errors.length
  categories.structure.warnings = structure.warnings.length
  categories.structure.passed = structure.errors.length === 0 ? 1 : 0

  // Syntax validation
  if (options.checkSyntax !== false) {
    const syntax = validateSyntax(notebook)
    allErrors.push(...syntax.errors)
    allWarnings.push(...syntax.warnings)
    categories.syntax.failed = syntax.errors.length
    categories.syntax.warnings = syntax.warnings.length
    categories.syntax.passed = syntax.errors.length === 0 ? 1 : 0
  }

  // Config validation
  const config = validateConfig(notebook)
  allErrors.push(...config.errors)
  allWarnings.push(...config.warnings)
  categories.config.failed = config.errors.length
  categories.config.warnings = config.warnings.length
  categories.config.passed = config.errors.length === 0 ? 1 : 0

  // Dependency validation
  if (options.checkDependencies !== false) {
    const deps = validateDependencies(notebook)
    allErrors.push(...deps.errors)
    allWarnings.push(...deps.warnings)
    categories.dependencies.failed = deps.errors.length
    categories.dependencies.warnings = deps.warnings.length
    categories.dependencies.passed = deps.errors.length === 0 ? 1 : 0
  }

  // Platform validation
  if (options.platform) {
    const platform = validatePlatform(notebook, options.platform)
    allErrors.push(...platform.errors)
    allWarnings.push(...platform.warnings)
    categories.platform.failed = platform.errors.length
    categories.platform.warnings = platform.warnings.length
    categories.platform.passed = platform.errors.length === 0 ? 1 : 0
  }

  // Tool validation
  const tools = validateTools(notebook)
  allErrors.push(...tools.errors)
  allWarnings.push(...tools.warnings)
  categories.tools.failed = tools.errors.length
  categories.tools.warnings = tools.warnings.length
  categories.tools.passed = tools.errors.length === 0 ? 1 : 0

  // Security validation
  if (options.checkSecurity !== false) {
    const security = validateSecurity(notebook)
    allErrors.push(...security.errors)
    allWarnings.push(...security.warnings)
    categories.security.failed = security.errors.length
    categories.security.warnings = security.warnings.length
    categories.security.passed = security.errors.length === 0 ? 1 : 0
  }

  // Calculate score
  const totalChecks = Object.values(categories).reduce((sum, cat) => sum + cat.passed + cat.failed, 0)
  const passedChecks = Object.values(categories).reduce((sum, cat) => sum + cat.passed, 0)
  const score = totalChecks > 0 ? Math.round((passedChecks / totalChecks) * 100) : 0

  // Determine if ready
  const hasErrors = allErrors.length > 0
  const hasBlockingWarnings = options.strict && allWarnings.length > 0
  const ready = !hasErrors && !hasBlockingWarnings

  return {
    valid: allErrors.length === 0,
    errors: allErrors,
    warnings: allWarnings,
    categories,
    score,
    ready,
  }
}

/**
 * Quick validation check - returns true if notebook is ready for operation
 */
export function isNotebookReady(notebook: NotebookContent, platform?: PlatformId): boolean {
  const result = validateNotebook(notebook, { platform, strict: false })
  return result.ready
}

/**
 * Get a human-readable validation summary
 */
export function getValidationSummary(result: ExtendedValidationResult): string {
  const lines: string[] = []

  // Header
  const status = result.ready ? '✓ READY' : result.valid ? '⚠ WARNINGS' : '✗ NOT READY'
  lines.push(`${status} (Score: ${result.score}/100)`)
  lines.push('')

  // Category breakdown
  lines.push('Category Breakdown:')
  for (const [category, stats] of Object.entries(result.categories)) {
    const icon = stats.failed > 0 ? '✗' : stats.warnings > 0 ? '⚠' : '✓'
    lines.push(`  ${icon} ${category}: ${stats.failed} errors, ${stats.warnings} warnings`)
  }

  // Errors
  if (result.errors.length > 0) {
    lines.push('')
    lines.push(`Errors (${result.errors.length}):`)
    for (const err of result.errors.slice(0, 10)) {
      lines.push(`  - [${err.code}] ${err.message}`)
      if (err.suggestion) {
        lines.push(`    → ${err.suggestion}`)
      }
    }
    if (result.errors.length > 10) {
      lines.push(`  ... and ${result.errors.length - 10} more`)
    }
  }

  // Warnings
  if (result.warnings.length > 0) {
    lines.push('')
    lines.push(`Warnings (${result.warnings.length}):`)
    for (const warn of result.warnings.slice(0, 5)) {
      lines.push(`  - [${warn.code}] ${warn.message}`)
    }
    if (result.warnings.length > 5) {
      lines.push(`  ... and ${result.warnings.length - 5} more`)
    }
  }

  return lines.join('\n')
}
