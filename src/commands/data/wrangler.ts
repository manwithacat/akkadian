/**
 * Launch Marimo data wrangler for interactive data exploration
 */

import { z } from 'zod'
import { join } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'
import { DatasetRegistry } from '../../lib/data-registry'

const WranglerArgs = z.object({
  name: z.string().optional().describe('Dataset name to explore'),
  version: z.number().optional().describe('Specific version'),
  port: z.number().default(2718).describe('Port for Marimo server'),
})

/**
 * Check if marimo is installed
 */
async function checkMarimoInstalled(): Promise<boolean> {
  const proc = Bun.spawn(['which', 'marimo'], {
    stdout: 'pipe',
    stderr: 'pipe',
  })
  return (await proc.exited) === 0
}

/**
 * Generate a Marimo notebook for data exploration
 */
function generateMarimoNotebook(dbPath: string, tables: string[]): string {
  const tableList = JSON.stringify(tables)

  return `import marimo

app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import sqlite3

    # Database path
    DB_PATH = "${dbPath}"

    # Available tables
    TABLE_LIST = ${tableList}

    return DB_PATH, TABLE_LIST, mo, pd, sqlite3

@app.cell
def __(mo):
    mo.md("""
    # Data Wrangler

    Interactive exploration of competition datasets with column statistics,
    data previews, and custom SQL queries.
    """)
    return

@app.cell
def __(DB_PATH, TABLE_LIST, mo, pd, sqlite3):
    # Connect and get table info
    conn = sqlite3.connect(DB_PATH)

    table_info = []
    for t in TABLE_LIST:
        try:
            count = pd.read_sql_query(f'SELECT COUNT(*) as cnt FROM "{t}"', conn).iloc[0]['cnt']
            cols = pd.read_sql_query(f'PRAGMA table_info("{t}")', conn)
            table_info.append({
                'Table': t,
                'Rows': f"{count:,}",
                'Columns': len(cols)
            })
        except:
            pass

    mo.md("## Tables Overview")
    return conn, table_info

@app.cell
def __(mo, pd, table_info):
    mo.ui.table(pd.DataFrame(table_info), selection=None)
    return

@app.cell
def __(TABLE_LIST, mo):
    # Table selector
    table_select = mo.ui.dropdown(
        options=TABLE_LIST,
        value=TABLE_LIST[0] if TABLE_LIST else None,
        label="Select Table"
    )
    table_select
    return table_select,

@app.cell
def __(conn, mo, pd, table_select):
    # Load selected table
    selected_table = table_select.value
    if selected_table:
        df = pd.read_sql_query(f'SELECT * FROM "{selected_table}"', conn)
        mo.md(f"### {selected_table} ({len(df):,} rows, {len(df.columns)} columns)")
    else:
        df = pd.DataFrame()
        mo.md("Select a table above")
    return df, selected_table

@app.cell
def __(df, mo, pd):
    # Column statistics
    if len(df) > 0:
        stats = []
        for col in df.columns:
            col_stats = {
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-null': int(df[col].notna().sum()),
                'Null': int(df[col].isna().sum()),
                'Unique': int(df[col].nunique()),
            }
            if df[col].dtype in ['int64', 'float64']:
                col_stats['Min'] = df[col].min()
                col_stats['Max'] = df[col].max()
                col_stats['Mean'] = round(df[col].mean(), 2)
            stats.append(col_stats)

        mo.vstack([
            mo.md("#### Column Statistics"),
            mo.ui.table(pd.DataFrame(stats), selection=None)
        ])
    else:
        mo.md("")
    return

@app.cell
def __(df, mo):
    # Data preview
    if len(df) > 0:
        mo.vstack([
            mo.md("#### Data Preview"),
            mo.ui.table(df, page_size=25, selection=None)
        ])
    else:
        mo.md("")
    return

@app.cell
def __(mo):
    mo.md("## Custom SQL Query")
    return

@app.cell
def __(mo):
    # SQL input
    sql_input = mo.ui.text_area(
        value="SELECT * FROM train LIMIT 10",
        label="Enter SQL query",
        full_width=True
    )
    sql_input
    return sql_input,

@app.cell
def __(conn, mo, pd, sql_input):
    # Execute SQL
    try:
        result = pd.read_sql_query(sql_input.value, conn)
        mo.vstack([
            mo.md(f"**{len(result)} rows returned**"),
            mo.ui.table(result, page_size=50, selection=None)
        ])
    except Exception as e:
        mo.callout(f"**SQL Error:** {e}", kind="danger")
    return

if __name__ == "__main__":
    app.run()
`
}

export const wrangler: CommandDefinition<typeof WranglerArgs> = {
  name: 'data wrangler',
  description: 'Launch Marimo data wrangler for rich dataframe exploration',
  help: `
Launches a Marimo notebook for interactive data exploration with:
- Column statistics and type info
- Interactive filtering and sorting
- Data distribution charts
- Custom SQL queries
- Missing value indicators

Requires marimo: pip install marimo

Options:
  --name      Dataset name to explore
  --version   Specific version (requires --name)
  --port      Marimo port (default: 2718)
`,
  examples: [
    'akk data wrangler',
    'akk data wrangler --name raw',
    'akk data wrangler --port 8080',
  ],
  args: WranglerArgs,

  async run(args, ctx) {
    const { name, version, port } = args

    // Check marimo is installed
    const installed = await checkMarimoInstalled()
    if (!installed) {
      return error(
        'MARIMO_NOT_INSTALLED',
        'Marimo is not installed',
        'Install with: pip install marimo',
        {}
      )
    }

    // Get registry
    const dataDir = join(ctx.cwd, 'datasets')
    const registryPath = join(dataDir, 'registry.db')

    const registryFile = Bun.file(registryPath)
    if (!(await registryFile.exists())) {
      return error(
        'NO_REGISTRY',
        'No dataset registry found',
        'Run "akk data download" first',
        {}
      )
    }

    const registry = new DatasetRegistry(registryPath)

    try {
      // Get dataset
      let dbPath: string

      if (name) {
        const dataset = version
          ? registry.getVersion(name, version)
          : registry.getLatestVersion(name)

        if (!dataset) {
          return error('DATASET_NOT_FOUND', `Dataset not found: ${name}`, '', {})
        }
        dbPath = dataset.sqlitePath
      } else {
        // Use latest dataset
        const datasets = registry.list()
        if (datasets.length === 0) {
          return error('NO_DATASETS', 'No datasets registered', '', {})
        }
        dbPath = datasets[0].sqlitePath
      }

      // Get table list from database
      const { Database } = await import('bun:sqlite')
      const db = new Database(dbPath, { readonly: true })
      const tables = db
        .query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '_%'")
        .all() as Array<{ name: string }>
      db.close()

      // Generate notebook
      const notebookContent = generateMarimoNotebook(
        dbPath,
        tables.map((t) => t.name)
      )

      // Write temporary notebook
      const notebookPath = join(dataDir, '_wrangler.py')
      await Bun.write(notebookPath, notebookContent)

      progress({ step: 'start', message: `Starting Marimo on port ${port}...` }, ctx.output)

      // Launch marimo
      const proc = Bun.spawn(['marimo', 'run', notebookPath, '--port', String(port)], {
        stdout: 'inherit',
        stderr: 'inherit',
      })

      progress({ step: 'ready', message: `Marimo running at http://localhost:${port}` }, ctx.output)

      const exitCode = await proc.exited

      return success({
        dbPath,
        tables: tables.length,
        port,
        exitCode,
      })
    } finally {
      registry.close()
    }
  },
}
