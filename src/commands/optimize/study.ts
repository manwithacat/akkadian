/**
 * Manage Optuna hyperparameter studies
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const StudyArgs = z.object({
  list: z.boolean().default(false).describe('List all studies'),
  show: z.string().optional().describe('Show study details by name'),
  export: z.string().optional().describe('Export best params to TOML'),
  delete: z.string().optional().describe('Delete study by name'),
  storage: z.string().default('optimization/optuna.db').describe('Optuna storage path'),
})

export const study: CommandDefinition<typeof StudyArgs> = {
  name: 'optimize study',
  description: 'Manage Optuna hyperparameter studies',
  help: `
Manage Optuna hyperparameter optimization studies.

Operations:
  --list     List all studies in the database
  --show     Show details of a specific study
  --export   Export best parameters to TOML config
  --delete   Delete a study from the database

Studies are stored in SQLite databases (default: optimization/optuna.db).
Each study contains multiple trials with different hyperparameter combinations.

Options:
  --list      List all studies
  --show      Study name to show details
  --export    Study name to export (creates <name>_best.toml)
  --delete    Study name to delete
  --storage   Path to Optuna storage database
`,
  examples: [
    'akk optimize study --list',
    'akk optimize study --show gen-params-v1',
    'akk optimize study --export gen-params-v1',
    'akk optimize study --delete old-study',
    'akk optimize study --list --storage optimization/my-studies.db',
  ],
  args: StudyArgs,

  async run(args, ctx) {
    const { list, show, export: exportStudy, delete: deleteStudy, storage } = args

    const storagePath = storage.startsWith('/') ? storage : join(ctx.cwd, storage)
    const storageUri = `sqlite:///${storagePath}`

    // Check if storage exists
    const storageExists = await Bun.file(storagePath).exists()
    if (!storageExists && !list) {
      return error(
        'STORAGE_NOT_FOUND',
        `Optuna storage not found: ${storagePath}`,
        'Run an Optuna study first to create the database',
        {
          path: storagePath,
        }
      )
    }

    if (list) {
      logStep({ step: 'study', message: 'Listing studies...' }, ctx.output)

      const pythonCode = `
import optuna
import json

storage = "${storageUri}"
try:
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    studies = []
    for s in summaries:
        studies.append({
            "name": s.study_name,
            "direction": s.direction.name if hasattr(s.direction, 'name') else str(s.direction),
            "n_trials": s.n_trials,
            "best_value": s.best_trial.value if s.best_trial else None,
        })
    print(json.dumps(studies))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`

      const proc = Bun.spawn(['python3', '-c', pythonCode], {
        stdout: 'pipe',
        stderr: 'pipe',
        cwd: ctx.cwd,
      })

      const stdout = await new Response(proc.stdout).text()
      await proc.exited

      try {
        const result = JSON.parse(stdout.trim())
        if (result.error) {
          return error('STUDY_ERROR', result.error, 'Check Optuna storage', {
            storage: storagePath,
          })
        }
        return success({ studies: result, storage: storagePath })
      } catch {
        return success({
          studies: [],
          storage: storagePath,
          message: 'No studies found',
        })
      }
    }

    if (show) {
      logStep({ step: 'study', message: `Loading study: ${show}...` }, ctx.output)

      const pythonCode = `
import optuna
import json

storage = "${storageUri}"
study_name = "${show}"

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
    result = {
        "name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "trials_summary": [
            {"number": t.number, "value": t.value, "state": t.state.name}
            for t in study.trials[-10:]  # Last 10 trials
        ]
    }
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`

      const proc = Bun.spawn(['python3', '-c', pythonCode], {
        stdout: 'pipe',
        stderr: 'pipe',
        cwd: ctx.cwd,
      })

      const stdout = await new Response(proc.stdout).text()
      await proc.exited

      try {
        const result = JSON.parse(stdout.trim())
        if (result.error) {
          return error('STUDY_NOT_FOUND', result.error, 'Check study name with --list', { name: show })
        }
        return success(result)
      } catch {
        return error('PARSE_ERROR', 'Failed to parse study data', 'Check Optuna installation', {})
      }
    }

    if (exportStudy) {
      logStep({ step: 'study', message: `Exporting study: ${exportStudy}...` }, ctx.output)

      const pythonCode = `
import optuna
import json

storage = "${storageUri}"
study_name = "${exportStudy}"

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
    if not study.best_trial:
        print(json.dumps({"error": "No completed trials"}))
    else:
        result = {
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`

      const proc = Bun.spawn(['python3', '-c', pythonCode], {
        stdout: 'pipe',
        stderr: 'pipe',
        cwd: ctx.cwd,
      })

      const stdout = await new Response(proc.stdout).text()
      await proc.exited

      try {
        const result = JSON.parse(stdout.trim())
        if (result.error) {
          return error('EXPORT_FAILED', result.error, 'Check study has completed trials', { name: exportStudy })
        }

        // Write TOML with best params
        const tomlPath = join(ctx.cwd, 'optimization', `${exportStudy}_best.toml`)
        let toml = `# Best parameters from study: ${exportStudy}\n`
        toml += `# Best value: ${result.best_value}\n\n`
        toml += '[models.generation]\n'
        for (const [key, value] of Object.entries(result.best_params)) {
          if (typeof value === 'string') {
            toml += `${key} = "${value}"\n`
          } else {
            toml += `${key} = ${value}\n`
          }
        }

        await Bun.write(tomlPath, toml)

        return success({
          study: exportStudy,
          best_value: result.best_value,
          best_params: result.best_params,
          exported_to: tomlPath,
        })
      } catch {
        return error('PARSE_ERROR', 'Failed to parse study data', 'Check Optuna installation', {})
      }
    }

    if (deleteStudy) {
      logStep({ step: 'study', message: `Deleting study: ${deleteStudy}...` }, ctx.output)

      const pythonCode = `
import optuna
import json

storage = "${storageUri}"
study_name = "${deleteStudy}"

try:
    optuna.delete_study(study_name=study_name, storage=storage)
    print(json.dumps({"deleted": study_name}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`

      const proc = Bun.spawn(['python3', '-c', pythonCode], {
        stdout: 'pipe',
        stderr: 'pipe',
        cwd: ctx.cwd,
      })

      const stdout = await new Response(proc.stdout).text()
      await proc.exited

      try {
        const result = JSON.parse(stdout.trim())
        if (result.error) {
          return error('DELETE_FAILED', result.error, 'Check study exists with --list', { name: deleteStudy })
        }
        return success({
          deleted: deleteStudy,
          message: `Study '${deleteStudy}' deleted`,
        })
      } catch {
        return error('DELETE_ERROR', 'Failed to delete study', 'Check Optuna installation', {})
      }
    }

    return error('NO_OPERATION', 'No operation specified', 'Use --list, --show, --export, or --delete', {})
  },
}
