/**
 * Configuration handling for Akkadian CLI
 */

import TOML from '@iarna/toml'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { dirname, join } from 'path'
import type { AkkConfig } from '../types/commands'
import {
  type CompetitionConfig,
  CompetitionConfigSchema,
  type CompetitionDirectory,
  createDefaultCompetitionConfig,
  getCompetitionPaths,
} from '../types/competition'

const CONFIG_FILE = 'akk.toml'
const COMPETITION_CONFIG_FILE = 'competition.toml'

/**
 * Find akk.toml by walking up the directory tree
 */
export async function findConfigPath(startDir: string = process.cwd()): Promise<string | null> {
  let dir = startDir

  while (dir !== '/') {
    const configPath = join(dir, CONFIG_FILE)
    if (existsSync(configPath)) {
      return configPath
    }
    dir = dirname(dir)
  }

  return null
}

/**
 * Get project root from config path
 */
export function getProjectRoot(configPath: string): string {
  return dirname(configPath)
}

/**
 * Parse TOML content using @iarna/toml
 */
function parseToml(content: string): Record<string, unknown> {
  return TOML.parse(content) as unknown as Record<string, unknown>
}

/**
 * Load and parse akk.toml
 */
export async function loadConfig(configPath: string): Promise<AkkConfig | null> {
  try {
    const content = readFileSync(configPath, 'utf-8')
    const parsed = parseToml(content) as unknown as AkkConfig
    return parsed
  } catch (_err) {
    return null
  }
}

/**
 * Load project config (akk.toml) from current directory or parents
 */
export async function loadProjectConfig(startDir: string = process.cwd()): Promise<AkkConfig | null> {
  const configPath = await findConfigPath(startDir)
  if (!configPath) return null
  return loadConfig(configPath)
}

/**
 * Default configuration
 */
export function defaultConfig(): AkkConfig {
  return {
    project: {
      name: 'akkadian-translation',
      version: '0.1.0',
    },
    kaggle: {
      username: 'manwithacat',
      competition: 'deep-past-initiative-machine-translation',
    },
    colab: {
      gcs_bucket: 'gs://akkadian-byt5-train',
      project: 'mlq425',
    },
    mlflow: {
      tracking_uri: 'sqlite:///mlflow/mlflow.db',
      artifact_location: './mlflow/artifacts',
      port: 5000,
    },
    paths: {
      competitions: 'competitions',
      notebooks: 'notebooks',
      scripts: 'scripts',
      datasets: 'datasets',
      models: 'models',
    },
  }
}

/**
 * Get the active competition directory path
 * Returns the path to competitions/<slug>/ based on akk.toml config
 */
export function getActiveCompetitionDir(projectRoot: string, config: AkkConfig): string {
  const competitionsBase = config.paths?.competitions || 'competitions'
  const slug = config.kaggle?.competition || 'default'
  return join(projectRoot, competitionsBase, slug)
}

/**
 * Load competition directory for the active competition from akk.toml
 */
export async function loadActiveCompetitionDirectory(
  projectRoot: string,
  config: AkkConfig
): Promise<CompetitionDirectory | null> {
  const competitionDir = getActiveCompetitionDir(projectRoot, config)
  return loadCompetitionDirectory(competitionDir)
}

// ============================================
// Competition Configuration
// ============================================

/**
 * Find competition.toml by walking up the directory tree
 */
export async function findCompetitionConfig(startDir: string = process.cwd()): Promise<string | null> {
  let dir = startDir

  while (dir !== '/') {
    const configPath = join(dir, COMPETITION_CONFIG_FILE)
    if (existsSync(configPath)) {
      return configPath
    }
    dir = dirname(dir)
  }

  return null
}

/**
 * Load and parse competition.toml
 */
export async function loadCompetitionConfig(configPath: string): Promise<CompetitionConfig | null> {
  try {
    const content = readFileSync(configPath, 'utf-8')
    const parsed = parseToml(content)

    // Validate with Zod schema
    const result = CompetitionConfigSchema.safeParse(parsed)
    if (!result.success) {
      console.error('Invalid competition.toml:', result.error.format())
      return null
    }

    return result.data
  } catch (_err) {
    return null
  }
}

/**
 * Load competition directory information
 */
export async function loadCompetitionDirectory(startDir: string = process.cwd()): Promise<CompetitionDirectory | null> {
  const configPath = await findCompetitionConfig(startDir)
  if (!configPath) return null

  const config = await loadCompetitionConfig(configPath)
  if (!config) return null

  const root = dirname(configPath)
  return getCompetitionPaths(root, config)
}

/**
 * Serialize config to TOML format
 */
function toToml(obj: Record<string, unknown>, prefix = ''): string {
  const lines: string[] = []
  const sections: string[] = []

  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue

    if (typeof value === 'object' && !Array.isArray(value)) {
      // Nested section
      const sectionName = prefix ? `${prefix}.${key}` : key
      sections.push(`[${sectionName}]`)
      sections.push(toToml(value as Record<string, unknown>, sectionName))
    } else if (Array.isArray(value)) {
      if (value.length > 0 && typeof value[0] === 'object') {
        // Array of tables [[section]]
        const sectionName = prefix ? `${prefix}.${key}` : key
        for (const item of value) {
          sections.push(`[[${sectionName}]]`)
          for (const [itemKey, itemValue] of Object.entries(item as Record<string, unknown>)) {
            if (itemValue !== undefined && itemValue !== null) {
              sections.push(`${itemKey} = ${formatTomlValue(itemValue)}`)
            }
          }
          sections.push('')
        }
      } else {
        // Simple array
        lines.push(`${key} = ${formatTomlValue(value)}`)
      }
    } else {
      lines.push(`${key} = ${formatTomlValue(value)}`)
    }
  }

  return [...lines, '', ...sections].join('\n')
}

/**
 * Format a value for TOML output
 */
function formatTomlValue(value: unknown): string {
  if (typeof value === 'string') {
    return `"${value.replace(/"/g, '\\"')}"`
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }
  if (typeof value === 'number') {
    return String(value)
  }
  if (Array.isArray(value)) {
    const items = value.map((v) => formatTomlValue(v))
    return `[${items.join(', ')}]`
  }
  return String(value)
}

/**
 * Save competition config to competition.toml
 */
export async function saveCompetitionConfig(config: CompetitionConfig, targetDir: string): Promise<string> {
  const configPath = join(targetDir, COMPETITION_CONFIG_FILE)

  // Generate TOML content
  const content = `# Akkadian Competition Configuration
# Generated by akk competition init

${toToml(config as unknown as Record<string, unknown>)}`

  writeFileSync(configPath, content, 'utf-8')
  return configPath
}

/**
 * Initialize a new competition directory
 */
export async function initCompetitionDirectory(
  targetDir: string,
  name: string,
  slug: string,
  username: string
): Promise<CompetitionDirectory> {
  // Create default config
  const config = createDefaultCompetitionConfig(name, slug, username)

  // Create directory structure with notebooks subdirectories
  const dirs = [
    targetDir,
    join(targetDir, config.paths.notebooks),
    join(targetDir, config.paths.notebooks, 'training'),
    join(targetDir, config.paths.notebooks, 'inference'),
    join(targetDir, config.paths.models),
    join(targetDir, config.paths.submissions),
    join(targetDir, config.paths.datasets),
    join(targetDir, config.paths.artifacts),
  ]

  for (const dir of dirs) {
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true })
    }
  }

  // Save config
  await saveCompetitionConfig(config, targetDir)

  return getCompetitionPaths(targetDir, config)
}
