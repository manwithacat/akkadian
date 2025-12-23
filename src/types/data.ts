/**
 * Data management types for Akkadian CLI
 */

import { z } from 'zod'

// Dataset source types
export const DatasetSourceTypeSchema = z.enum(['kaggle', 'etl', 'derived'])
export type DatasetSourceType = z.infer<typeof DatasetSourceTypeSchema>

// MLflow link types
export const MLflowLinkTypeSchema = z.enum(['training', 'evaluation', 'inference'])
export type MLflowLinkType = z.infer<typeof MLflowLinkTypeSchema>

// Dataset version schema
export const DatasetVersionSchema = z.object({
  id: z.string(),
  name: z.string(),
  version: z.number().int().positive(),
  sourceType: DatasetSourceTypeSchema,
  sourcePath: z.string().optional(),
  parentVersionId: z.string().optional(),
  etlPipeline: z.string().optional(),
  sqlitePath: z.string(),
  createdAt: z.string(),
  rowCount: z.number().int().optional(),
  sizeBytes: z.number().int().optional(),
  checksum: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
})

export type DatasetVersion = z.infer<typeof DatasetVersionSchema>

// MLflow link schema
export const MLflowLinkSchema = z.object({
  datasetVersionId: z.string(),
  mlflowRunId: z.string(),
  mlflowExperimentId: z.string().optional(),
  linkType: MLflowLinkTypeSchema,
  createdAt: z.string(),
})

export type MLflowLink = z.infer<typeof MLflowLinkSchema>

// Table column info
export interface ColumnInfo {
  name: string
  type: string
  position: number
}

// Table statistics
export interface TableStats {
  name: string
  rowCount: number
  columns: ColumnInfo[]
}

// CSV to SQLite conversion result
export interface ConversionResult {
  outputPath: string
  tables: TableStats[]
  totalRows: number
  sizeBytes: number
}

// Dataset config in akk.toml
export const DataConfigSchema = z.object({
  registry: z.string().default('datasets/registry.db'),
  data_dir: z.string().default('datasets'),
  datasette_port: z.number().default(8001),
  auto_sqlite: z.boolean().default(true),
})

export type DataConfig = z.infer<typeof DataConfigSchema>

// Parsed dataset reference (name:version)
export interface DatasetRef {
  name: string
  version?: number
}

/**
 * Parse a dataset reference string like "raw:1" or "v2_augmented"
 */
export function parseDatasetRef(ref: string): DatasetRef {
  const parts = ref.split(':')
  return {
    name: parts[0],
    version: parts[1] ? parseInt(parts[1], 10) : undefined,
  }
}

/**
 * Format a dataset reference
 */
export function formatDatasetRef(name: string, version?: number): string {
  return version ? `${name}:${version}` : name
}
