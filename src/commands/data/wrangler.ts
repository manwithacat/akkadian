/**
 * Launch Marimo data wrangler for interactive data exploration
 */

import { join } from 'path'
import { z } from 'zod'
import { DatasetRegistry } from '../../lib/data-registry'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

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
  const defaultTable = tables[0] || 'train'

  return `import marimo

app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import sqlite3
    return mo, pd, sqlite3

@app.cell
def _(sqlite3):
    DB_PATH = "${dbPath}"
    conn = sqlite3.connect(DB_PATH)
    TABLE_LIST = ${tableList}
    return DB_PATH, TABLE_LIST, conn

@app.cell
def _(mo):
    mo.md("# Data Wrangler")
    return

@app.cell
def _(TABLE_LIST, mo):
    table_dropdown = mo.ui.dropdown(
        options=TABLE_LIST,
        value="${defaultTable}",
        label="Select Table"
    )
    return table_dropdown,

@app.cell
def _(table_dropdown):
    table_dropdown
    return

@app.cell
def _(conn, mo, pd, table_dropdown):
    _name = table_dropdown.value
    if _name:
        _df = pd.read_sql_query(f'SELECT * FROM "{_name}"', conn)
        _stats = []
        for _col in _df.columns:
            _s = {'Column': _col, 'Type': str(_df[_col].dtype),
                  'Non-null': int(_df[_col].notna().sum()),
                  'Null': int(_df[_col].isna().sum()),
                  'Unique': int(_df[_col].nunique())}
            if _df[_col].dtype in ['int64', 'float64']:
                _s['Min'], _s['Max'] = _df[_col].min(), _df[_col].max()
            _stats.append(_s)
        _stats_df = pd.DataFrame(_stats)
        _output = mo.vstack([
            mo.md(f"## {_name}"),
            mo.md(f"**{len(_df):,} rows** · {len(_df.columns)} columns"),
            mo.md("### Column Info"),
            mo.ui.table(_stats_df),
            mo.md("### Data Preview"),
            mo.ui.dataframe(_df),
        ])
    else:
        _output = mo.md("No table selected")
    _output
    return

@app.cell
def _(mo):
    mo.md("---\\n## SQL Query Builder")
    return

@app.cell
def _(TABLE_LIST, mo):
    query_table = mo.ui.dropdown(options=TABLE_LIST, value="${defaultTable}", label="Table")
    limit_input = mo.ui.number(value=100, start=1, stop=10000, label="Limit")
    return limit_input, query_table,

@app.cell
def _(conn, limit_input, pd, query_table):
    # Get columns for selected table
    _cols = pd.read_sql_query(f'PRAGMA table_info("{query_table.value}")', conn)['name'].tolist()
    _col_options = ['*'] + _cols
    return _col_options,

@app.cell
def _(_col_options, mo):
    col_select = mo.ui.multiselect(options=_col_options, value=['*'], label="Columns")
    return col_select,

@app.cell
def _(col_select, limit_input, mo, query_table):
    # Build query from selections
    _cols = ', '.join(col_select.value) if col_select.value else '*'
    _query = f"SELECT {_cols} FROM {query_table.value} LIMIT {limit_input.value}"
    mo.hstack([query_table, col_select, limit_input], justify="start")
    return _query,

@app.cell
def _(_query, mo):
    sql_text = mo.ui.text_area(value=_query, label="SQL (edit to customize)", full_width=True, rows=2)
    return sql_text,

@app.cell
def _(sql_text):
    sql_text
    return

@app.cell
def _(mo):
    run_button = mo.ui.button(label="Run Query", kind="success")
    return run_button,

@app.cell
def _(run_button):
    run_button
    return

@app.cell
def _(conn, mo, pd, run_button, sql_text):
    run_button
    if sql_text.value:
        try:
            _result = pd.read_sql_query(sql_text.value, conn)
            _out = mo.vstack([
                mo.md(f"**{len(_result):,} rows**"),
                mo.ui.dataframe(_result)
            ])
        except Exception as e:
            _out = mo.callout(f"Error: {e}", kind="danger")
    else:
        _out = mo.md("Enter a query above")
    _out
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
  examples: ['akk data wrangler', 'akk data wrangler --name raw', 'akk data wrangler --port 8080'],
  args: WranglerArgs,

  async run(args, ctx) {
    const { name, version, port } = args

    // Check marimo is installed
    const installed = await checkMarimoInstalled()
    if (!installed) {
      return error('MARIMO_NOT_INSTALLED', 'Marimo is not installed', 'Install with: pip install marimo', {})
    }

    // Get registry
    const dataDir = join(ctx.cwd, 'datasets')
    const registryPath = join(dataDir, 'registry.db')

    const registryFile = Bun.file(registryPath)
    if (!(await registryFile.exists())) {
      return error('NO_REGISTRY', 'No dataset registry found', 'Run "akk data download" first', {})
    }

    const registry = new DatasetRegistry(registryPath)

    try {
      // Get dataset
      let dbPath: string

      if (name) {
        const dataset = version ? registry.getVersion(name, version) : registry.getLatestVersion(name)

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

      // Get table list from database (exclude internal tables starting with _)
      const { Database } = await import('bun:sqlite')
      const db = new Database(dbPath, { readonly: true })
      const tables = db
        .query("SELECT name FROM sqlite_master WHERE type='table' AND substr(name, 1, 1) != '_'")
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

      logStep({ step: 'start', message: `Starting Marimo on port ${port}...` }, ctx.output)

      // Launch marimo
      const proc = Bun.spawn(['marimo', 'run', notebookPath, '--port', String(port)], {
        stdout: 'inherit',
        stderr: 'inherit',
      })

      logStep({ step: 'ready', message: `Marimo running at http://localhost:${port}` }, ctx.output)

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
