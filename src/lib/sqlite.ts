/**
 * SQLite utilities for data management
 *
 * Uses Bun's built-in SQLite support for CSV-to-SQLite conversion
 */

import { Database } from 'bun:sqlite'
import { createHash } from 'crypto'
import type { ColumnInfo, TableStats, ConversionResult } from '../types/data'

/**
 * Sanitize column name for SQL compatibility
 * Preserves original name in a mapping table
 */
function sanitizeColumnName(name: string): string {
  // Trim whitespace
  let sanitized = name.trim()

  // Replace problematic characters with underscores
  sanitized = sanitized
    .replace(/[\[\](){}]/g, '') // Remove brackets/parens
    .replace(/[;:,./\\'"]/g, '_') // Replace punctuation with underscore
    .replace(/\s+/g, '_') // Replace whitespace with underscore
    .replace(/_+/g, '_') // Collapse multiple underscores
    .replace(/^_|_$/g, '') // Remove leading/trailing underscores

  // If starts with a number, prefix with underscore
  if (/^\d/.test(sanitized)) {
    sanitized = '_' + sanitized
  }

  // If empty after sanitization, use generic name
  if (!sanitized) {
    sanitized = 'column'
  }

  return sanitized
}

/**
 * Parse a CSV line handling quoted fields
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        // Escaped quote
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current)
      current = ''
    } else {
      current += char
    }
  }
  result.push(current)

  return result
}

/**
 * Infer SQL type from sample values
 */
function inferSqlType(values: string[]): string {
  const nonEmpty = values.filter((v) => v && v.trim())
  if (nonEmpty.length === 0) return 'TEXT'

  // Check if all values are integers
  const allIntegers = nonEmpty.every((v) => /^-?\d+$/.test(v.trim()))
  if (allIntegers) return 'INTEGER'

  // Check if all values are numbers
  const allNumbers = nonEmpty.every((v) => !isNaN(parseFloat(v.trim())))
  if (allNumbers) return 'REAL'

  return 'TEXT'
}

export interface CsvToSqliteOptions {
  inputPath: string
  outputPath: string
  tableName?: string
  inferTypes?: boolean
  batchSize?: number
}

/**
 * Convert a CSV file to SQLite database
 */
export async function csvToSqlite(options: CsvToSqliteOptions): Promise<ConversionResult> {
  const { inputPath, outputPath, inferTypes = true, batchSize = 1000 } = options

  // Derive table name from filename if not provided
  const tableName =
    options.tableName || inputPath.split('/').pop()?.replace(/\.csv$/i, '') || 'data'

  // Read CSV file
  const file = Bun.file(inputPath)
  const content = await file.text()
  const lines = content.trim().split('\n')

  if (lines.length < 2) {
    throw new Error('CSV file must have at least a header and one data row')
  }

  // Parse header
  const headers = parseCSVLine(lines[0])

  // Sample data for type inference (first 100 rows)
  const sampleRows: string[][] = []
  for (let i = 1; i < Math.min(lines.length, 101); i++) {
    sampleRows.push(parseCSVLine(lines[i]))
  }

  // Infer column types and sanitize names
  const columnNameMap: Record<string, string> = {} // sanitized -> original
  const usedNames = new Set<string>()

  const columns: ColumnInfo[] = headers.map((name, idx) => {
    const originalName = name.trim()
    let sanitizedName = sanitizeColumnName(originalName)

    // Handle duplicate sanitized names
    let uniqueName = sanitizedName
    let counter = 1
    while (usedNames.has(uniqueName)) {
      uniqueName = `${sanitizedName}_${counter}`
      counter++
    }
    usedNames.add(uniqueName)

    // Store mapping if name changed
    if (uniqueName !== originalName) {
      columnNameMap[uniqueName] = originalName
    }

    const sampleValues = sampleRows.map((row) => row[idx] || '')
    return {
      name: uniqueName,
      type: inferTypes ? inferSqlType(sampleValues) : 'TEXT',
      position: idx,
    }
  })

  // Create database
  const db = new Database(outputPath, { create: true })

  try {
    // Create table
    const columnDefs = columns.map((c) => `"${c.name}" ${c.type}`).join(', ')
    db.run(`CREATE TABLE IF NOT EXISTS "${tableName}" (${columnDefs})`)

    // Create metadata table
    db.run(`
      CREATE TABLE IF NOT EXISTS _metadata (
        key TEXT PRIMARY KEY,
        value TEXT
      )
    `)

    // Insert rows in batches
    const placeholders = columns.map(() => '?').join(', ')
    const insertStmt = db.prepare(`INSERT INTO "${tableName}" VALUES (${placeholders})`)

    db.run('BEGIN TRANSACTION')

    let rowCount = 0
    for (let i = 1; i < lines.length; i++) {
      const values = parseCSVLine(lines[i])

      // Convert values based on column type
      const typedValues = columns.map((col, idx) => {
        const val = values[idx]?.trim() || null
        if (val === null || val === '') return null
        if (col.type === 'INTEGER') return parseInt(val, 10)
        if (col.type === 'REAL') return parseFloat(val)
        return val
      })

      insertStmt.run(...typedValues)
      rowCount++

      // Commit batch
      if (rowCount % batchSize === 0) {
        db.run('COMMIT')
        db.run('BEGIN TRANSACTION')
      }
    }

    db.run('COMMIT')

    // Store metadata
    const metaStmt = db.prepare('INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)')
    metaStmt.run('source_file', inputPath)
    metaStmt.run('created_at', new Date().toISOString())
    metaStmt.run('row_count', String(rowCount))
    metaStmt.run('table_name', tableName)

    // Store column name mapping if any names were sanitized
    if (Object.keys(columnNameMap).length > 0) {
      db.run(`
        CREATE TABLE IF NOT EXISTS _column_names (
          table_name TEXT,
          sanitized_name TEXT,
          original_name TEXT,
          PRIMARY KEY (table_name, sanitized_name)
        )
      `)
      const colMapStmt = db.prepare('INSERT OR REPLACE INTO _column_names VALUES (?, ?, ?)')
      for (const [sanitized, original] of Object.entries(columnNameMap)) {
        colMapStmt.run(tableName, sanitized, original)
      }
    }

    // Get file size
    const outputFile = Bun.file(outputPath)
    const sizeBytes = outputFile.size

    return {
      outputPath,
      tables: [
        {
          name: tableName,
          rowCount,
          columns,
        },
      ],
      totalRows: rowCount,
      sizeBytes,
    }
  } finally {
    db.close()
  }
}

/**
 * Compute SHA256 checksum of a file
 */
export async function computeChecksum(filePath: string): Promise<string> {
  const file = Bun.file(filePath)
  const buffer = await file.arrayBuffer()
  const hash = createHash('sha256')
  hash.update(Buffer.from(buffer))
  return hash.digest('hex')
}

/**
 * Get table info from a SQLite database
 */
export function getTableInfo(
  db: Database,
  tableName: string
): Array<{ name: string; type: string; notnull: boolean; pk: boolean }> {
  const result = db.query(`PRAGMA table_info("${tableName}")`).all() as Array<{
    cid: number
    name: string
    type: string
    notnull: number
    dflt_value: unknown
    pk: number
  }>

  return result.map((row) => ({
    name: row.name,
    type: row.type,
    notnull: row.notnull === 1,
    pk: row.pk === 1,
  }))
}

/**
 * Get list of tables in a SQLite database (excludes internal tables starting with _)
 */
export function getTableList(db: Database): string[] {
  const result = db.query("SELECT name FROM sqlite_master WHERE type='table' AND substr(name, 1, 1) != '_'").all() as Array<{ name: string }>
  return result.map((row) => row.name)
}

/**
 * Get row count for a table
 */
export function getRowCount(db: Database, tableName: string): number {
  const result = db.query(`SELECT COUNT(*) as count FROM "${tableName}"`).get() as { count: number }
  return result.count
}

/**
 * Merge multiple SQLite files into one
 */
export async function mergeSqliteFiles(
  inputPaths: string[],
  outputPath: string
): Promise<ConversionResult> {
  const db = new Database(outputPath, { create: true })
  const tables: TableStats[] = []
  let totalRows = 0

  try {
    for (const inputPath of inputPaths) {
      const sourceDb = new Database(inputPath, { readonly: true })
      const tableNames = getTableList(sourceDb)

      for (const tableName of tableNames) {
        // Get schema
        const columns = getTableInfo(sourceDb, tableName)

        // Create table in output if not exists
        const columnDefs = columns.map((c) => `"${c.name}" ${c.type}`).join(', ')
        db.run(`CREATE TABLE IF NOT EXISTS "${tableName}" (${columnDefs})`)

        // Copy data
        const rows = sourceDb.query(`SELECT * FROM "${tableName}"`).all()
        if (rows.length > 0) {
          const placeholders = columns.map(() => '?').join(', ')
          const insertStmt = db.prepare(`INSERT INTO "${tableName}" VALUES (${placeholders})`)

          db.run('BEGIN TRANSACTION')
          for (const row of rows) {
            const values = columns.map((c) => (row as Record<string, unknown>)[c.name])
            insertStmt.run(...values)
          }
          db.run('COMMIT')

          totalRows += rows.length
          tables.push({
            name: tableName,
            rowCount: rows.length,
            columns: columns.map((c, i) => ({ name: c.name, type: c.type, position: i })),
          })
        }
      }

      sourceDb.close()
    }

    const outputFile = Bun.file(outputPath)
    return {
      outputPath,
      tables,
      totalRows,
      sizeBytes: outputFile.size,
    }
  } finally {
    db.close()
  }
}
