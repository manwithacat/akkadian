/**
 * Dataset Registry
 *
 * Manages dataset versions and MLflow linkage in a SQLite database
 */

import { Database } from 'bun:sqlite'
import { randomUUID } from 'crypto'
import type { DatasetSourceType, DatasetVersion, MLflowLink, MLflowLinkType } from '../types/data'

const SCHEMA_VERSION = 1

const INIT_SQL = `
-- Dataset versions table
CREATE TABLE IF NOT EXISTS dataset_versions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT,
    parent_version_id TEXT,
    etl_pipeline TEXT,
    sqlite_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    row_count INTEGER,
    size_bytes INTEGER,
    checksum TEXT,
    metadata TEXT,
    FOREIGN KEY (parent_version_id) REFERENCES dataset_versions(id),
    UNIQUE(name, version)
);

-- MLflow run linkage table
CREATE TABLE IF NOT EXISTS dataset_mlflow_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_version_id TEXT NOT NULL,
    mlflow_run_id TEXT NOT NULL,
    mlflow_experiment_id TEXT,
    link_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_version_id) REFERENCES dataset_versions(id),
    UNIQUE(dataset_version_id, mlflow_run_id, link_type)
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_dataset_versions_name ON dataset_versions(name);
CREATE INDEX IF NOT EXISTS idx_dataset_versions_source ON dataset_versions(source_type);
CREATE INDEX IF NOT EXISTS idx_mlflow_links_run ON dataset_mlflow_links(mlflow_run_id);
`

export interface RegisterOptions {
  name: string
  sourceType: DatasetSourceType
  sqlitePath: string
  sourcePath?: string
  parentVersionId?: string
  etlPipeline?: string
  rowCount?: number
  sizeBytes?: number
  checksum?: string
  metadata?: Record<string, unknown>
}

export interface ListFilter {
  name?: string
  sourceType?: DatasetSourceType
}

export class DatasetRegistry {
  private db: Database
  private registryPath: string

  constructor(registryPath: string) {
    this.registryPath = registryPath
    this.db = new Database(registryPath, { create: true })
    this.initSchema()
  }

  private initSchema(): void {
    this.db.exec(INIT_SQL)

    // Check/set schema version
    const versionRow = this.db.query("SELECT value FROM _schema_info WHERE key = 'version'").get() as {
      value: string
    } | null

    if (!versionRow) {
      this.db.prepare("INSERT INTO _schema_info (key, value) VALUES ('version', ?)").run(String(SCHEMA_VERSION))
    }
  }

  /**
   * Get the next version number for a dataset name
   */
  getNextVersion(name: string): number {
    const result = this.db
      .query('SELECT MAX(version) as max_version FROM dataset_versions WHERE name = ?')
      .get(name) as { max_version: number | null }

    return (result?.max_version || 0) + 1
  }

  /**
   * Register a new dataset version
   */
  register(options: RegisterOptions): DatasetVersion {
    const id = randomUUID()
    const version = this.getNextVersion(options.name)
    const createdAt = new Date().toISOString()

    const stmt = this.db.prepare(`
      INSERT INTO dataset_versions (
        id, name, version, source_type, source_path, parent_version_id,
        etl_pipeline, sqlite_path, created_at, row_count, size_bytes,
        checksum, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)

    stmt.run(
      id,
      options.name,
      version,
      options.sourceType,
      options.sourcePath || null,
      options.parentVersionId || null,
      options.etlPipeline || null,
      options.sqlitePath,
      createdAt,
      options.rowCount || null,
      options.sizeBytes || null,
      options.checksum || null,
      options.metadata ? JSON.stringify(options.metadata) : null
    )

    return {
      id,
      name: options.name,
      version,
      sourceType: options.sourceType,
      sourcePath: options.sourcePath,
      parentVersionId: options.parentVersionId,
      etlPipeline: options.etlPipeline,
      sqlitePath: options.sqlitePath,
      createdAt,
      rowCount: options.rowCount,
      sizeBytes: options.sizeBytes,
      checksum: options.checksum,
      metadata: options.metadata,
    }
  }

  /**
   * Get a specific dataset version
   */
  get(id: string): DatasetVersion | null {
    const row = this.db.query('SELECT * FROM dataset_versions WHERE id = ?').get(id) as Record<string, unknown> | null

    return row ? this.rowToDataset(row) : null
  }

  /**
   * Get latest version of a dataset by name
   */
  getLatestVersion(name: string): DatasetVersion | null {
    const row = this.db
      .query('SELECT * FROM dataset_versions WHERE name = ? ORDER BY version DESC LIMIT 1')
      .get(name) as Record<string, unknown> | null

    return row ? this.rowToDataset(row) : null
  }

  /**
   * Get a specific version of a dataset
   */
  getVersion(name: string, version: number): DatasetVersion | null {
    const row = this.db
      .query('SELECT * FROM dataset_versions WHERE name = ? AND version = ?')
      .get(name, version) as Record<string, unknown> | null

    return row ? this.rowToDataset(row) : null
  }

  /**
   * List all dataset versions with optional filtering
   */
  list(filter?: ListFilter): DatasetVersion[] {
    let sql = 'SELECT * FROM dataset_versions WHERE 1=1'
    const params: (string | number | null)[] = []

    if (filter?.name) {
      sql += ' AND name = ?'
      params.push(filter.name)
    }

    if (filter?.sourceType) {
      sql += ' AND source_type = ?'
      params.push(filter.sourceType)
    }

    sql += ' ORDER BY name, version DESC'

    const rows = this.db.query(sql).all(...params) as Array<Record<string, unknown>>
    return rows.map((row) => this.rowToDataset(row))
  }

  /**
   * Link a dataset version to an MLflow run
   */
  linkMlflowRun(datasetVersionId: string, mlflowRunId: string, linkType: MLflowLinkType, experimentId?: string): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO dataset_mlflow_links (
        dataset_version_id, mlflow_run_id, mlflow_experiment_id, link_type, created_at
      ) VALUES (?, ?, ?, ?, ?)
    `)

    stmt.run(datasetVersionId, mlflowRunId, experimentId || null, linkType, new Date().toISOString())
  }

  /**
   * Get MLflow links for a dataset version
   */
  getMlflowLinks(datasetVersionId: string): MLflowLink[] {
    const rows = this.db
      .query('SELECT * FROM dataset_mlflow_links WHERE dataset_version_id = ?')
      .all(datasetVersionId) as Array<Record<string, unknown>>

    return rows.map((row) => ({
      datasetVersionId: row.dataset_version_id as string,
      mlflowRunId: row.mlflow_run_id as string,
      mlflowExperimentId: row.mlflow_experiment_id as string | undefined,
      linkType: row.link_type as MLflowLinkType,
      createdAt: row.created_at as string,
    }))
  }

  /**
   * Get dataset versions linked to an MLflow run
   */
  getDatasetsByMlflowRun(mlflowRunId: string): DatasetVersion[] {
    const rows = this.db
      .query(
        `
      SELECT dv.* FROM dataset_versions dv
      JOIN dataset_mlflow_links ml ON dv.id = ml.dataset_version_id
      WHERE ml.mlflow_run_id = ?
    `
      )
      .all(mlflowRunId) as Array<Record<string, unknown>>

    return rows.map((row) => this.rowToDataset(row))
  }

  /**
   * Get lineage (parent chain) for a dataset
   */
  getLineage(datasetId: string): DatasetVersion[] {
    const lineage: DatasetVersion[] = []
    let current = this.get(datasetId)

    while (current) {
      lineage.push(current)
      if (current.parentVersionId) {
        current = this.get(current.parentVersionId)
      } else {
        break
      }
    }

    return lineage
  }

  /**
   * Get all unique dataset names
   */
  getDatasetNames(): string[] {
    const rows = this.db.query('SELECT DISTINCT name FROM dataset_versions ORDER BY name').all() as Array<{
      name: string
    }>

    return rows.map((row) => row.name)
  }

  /**
   * Close the database connection
   */
  close(): void {
    this.db.close()
  }

  private rowToDataset(row: Record<string, unknown>): DatasetVersion {
    return {
      id: row.id as string,
      name: row.name as string,
      version: row.version as number,
      sourceType: row.source_type as DatasetSourceType,
      sourcePath: row.source_path as string | undefined,
      parentVersionId: row.parent_version_id as string | undefined,
      etlPipeline: row.etl_pipeline as string | undefined,
      sqlitePath: row.sqlite_path as string,
      createdAt: row.created_at as string,
      rowCount: row.row_count as number | undefined,
      sizeBytes: row.size_bytes as number | undefined,
      checksum: row.checksum as string | undefined,
      metadata: row.metadata ? JSON.parse(row.metadata as string) : undefined,
    }
  }
}

/**
 * Get or create a dataset registry
 */
export function getRegistry(projectRoot: string): DatasetRegistry {
  const registryPath = `${projectRoot}/datasets/registry.db`
  return new DatasetRegistry(registryPath)
}
