/**
 * Register a model in MLFlow Model Registry
 */

import { z } from 'zod'
import { join } from 'path'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'

const RegisterArgs = z.object({
  run: z.string().optional().describe('Run name to register model from'),
  runId: z.string().optional().describe('Run ID to register model from'),
  name: z.string().optional().describe('Registered model name'),
  alias: z.string().optional().describe('Alias to set (e.g., champion, baseline)'),
  experiment: z.string().default('nllb-akkadian').describe('Experiment name'),
  port: z.number().default(5001).describe('MLFlow server port'),
  list: z.boolean().default(false).describe('List registered models'),
})

export const register: CommandDefinition<typeof RegisterArgs> = {
  name: 'mlflow register',
  description: 'Register a model in MLFlow Model Registry',
  help: `
Register a model from a run in the MLFlow Model Registry.

The Model Registry provides:
- Centralized model storage
- Model versioning
- Aliases for deployment (champion, staging, etc.)
- Model lineage tracking

Options:
  --run         Run name to register model from
  --run-id      Run ID to register model from
  --name        Name for the registered model
  --alias       Alias to set (e.g., champion, baseline)
  --experiment  Experiment name (default: nllb-akkadian)
  --port        MLFlow server port (default: 5001)
  --list        List all registered models

Reserved aliases: 'latest' is reserved by MLFlow
Recommended aliases: 'champion' (best model), 'baseline', 'staging'
`,
  examples: [
    'akk mlflow register --list',
    'akk mlflow register --run nllb-v4 --name nllb-akkadian',
    'akk mlflow register --run nllb-v5 --name nllb-akkadian --alias champion',
  ],
  args: RegisterArgs,

  async run(args, ctx) {
    const { run: runName, runId, name, alias, experiment, port, list } = args
    const trackingUri = `http://localhost:${port}`

    // Build Python script
    const pythonCode = list
      ? `
import mlflow
mlflow.set_tracking_uri('${trackingUri}')
client = mlflow.MlflowClient()

print('{"models": [', end='')
first = True
for rm in client.search_registered_models():
    if not first:
        print(',', end='')
    first = False
    aliases = {a: v for a, v in rm.aliases.items()} if rm.aliases else {}
    versions = []
    for mv in client.search_model_versions(f'name="{rm.name}"'):
        versions.append({
            'version': mv.version,
            'status': mv.status,
            'run_id': mv.run_id,
        })
    import json
    print(json.dumps({
        'name': rm.name,
        'description': rm.description,
        'aliases': aliases,
        'versions': versions,
    }), end='')
print(']}')
`
      : `
import mlflow
import json
mlflow.set_tracking_uri('${trackingUri}')
client = mlflow.MlflowClient()

# Find run
exp = mlflow.get_experiment_by_name('${experiment}')
if not exp:
    print(json.dumps({'error': 'Experiment not found: ${experiment}'}))
    exit(1)

run_filter = ${runId ? `"run_id = '${runId}'"` : runName ? `"tags.mlflow.runName = '${runName}'"` : '""'}
if not run_filter:
    print(json.dumps({'error': 'Specify --run or --run-id'}))
    exit(1)

runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=run_filter)
if len(runs) == 0:
    print(json.dumps({'error': 'Run not found'}))
    exit(1)

run = runs.iloc[0]
run_id = run['run_id']
run_name = run.get('tags.mlflow.runName', run_id)

# Get artifact URI
run_info = client.get_run(run_id)
model_uri = f"{run_info.info.artifact_uri}/model"

model_name = '${name || 'nllb-akkadian'}'
alias_name = ${alias ? `'${alias}'` : 'None'}

# Check if model exists
try:
    rm = client.get_registered_model(model_name)
    exists = True
except:
    exists = False

if not exists:
    client.create_registered_model(
        name=model_name,
        description='NLLB translation model for Akkadian-English',
        tags={'language_pair': 'akk-en', 'task': 'translation'}
    )

# Create version
mv = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id,
    description=f'From run {run_name}',
    tags={'run_name': str(run_name)}
)

# Set alias
if alias_name:
    client.set_registered_model_alias(model_name, alias_name, mv.version)

print(json.dumps({
    'registered_model': model_name,
    'version': mv.version,
    'run_id': run_id,
    'run_name': str(run_name),
    'alias': alias_name,
    'url': f'${trackingUri}/#/models/{model_name}',
}))
`

    progress({ step: 'register', message: list ? 'Listing registered models...' : 'Registering model...' }, ctx.output)

    const proc = Bun.spawn(['python3', '-c', pythonCode], {
      stdout: 'pipe',
      stderr: 'pipe',
      cwd: ctx.cwd,
    })

    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      return error('REGISTER_FAILED', `Registration failed: ${stderr || stdout}`, 'Check MLFlow server is running', {
        exitCode,
        stderr,
      })
    }

    try {
      const result = JSON.parse(stdout.trim())
      if (result.error) {
        return error('REGISTER_ERROR', result.error, 'Check run name and experiment', result)
      }
      return success(result)
    } catch {
      return error('PARSE_ERROR', 'Failed to parse response', stdout, { stdout, stderr })
    }
  },
}
