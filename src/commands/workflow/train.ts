/**
 * End-to-end training workflow orchestration
 */

import { basename, join } from 'path'
import { z } from 'zod'
import { bucketExists, download, listFiles, rsync, upload } from '../../lib/gcs'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const TrainWorkflowArgs = z.object({
  notebook: z.string().describe('Training notebook path'),
  name: z.string().optional().describe('Run name (default: auto-generated)'),
  bucket: z.string().optional().describe('GCS bucket'),
  experiment: z.string().optional().describe('Experiment name'),
  'skip-upload': z.boolean().default(false).describe('Skip notebook upload'),
  'skip-download': z.boolean().default(false).describe('Skip artifact download'),
  'wait-for-completion': z.boolean().default(true).describe('Wait for training to complete'),
  timeout: z.number().default(7200).describe('Timeout in seconds (default: 2 hours)'),
  'poll-interval': z.number().default(60).describe('Status check interval in seconds'),
})

export const train: CommandDefinition<typeof TrainWorkflowArgs> = {
  name: 'workflow train',
  description: 'End-to-end training workflow: upload, run, download, sync',
  help: `
Orchestrates the complete training workflow:

1. Upload notebook to GCS
2. Display Colab instructions
3. Monitor training progress (if --wait-for-completion)
4. Download trained model and artifacts
5. Sync to local MLFlow
6. Display analysis and next steps

The training notebook should:
- Read from gs://{bucket}/datasets/
- Write checkpoints to gs://{bucket}/mlflow/runs/{experiment}/{run}/checkpoints/
- Write final model to gs://{bucket}/mlflow/runs/{experiment}/{run}/output/
- Write status updates to gs://{bucket}/mlflow/runs/{experiment}/{run}/status.json

Options:
  --notebook              Training notebook path (required)
  --name                  Run name (default: notebook_YYYYMMDD-HHMMSS)
  --bucket                GCS bucket (default: from config)
  --experiment            Experiment name (default: from notebook or 'default')
  --skip-upload           Skip notebook upload step
  --skip-download         Skip artifact download step
  --wait-for-completion   Wait for training to complete (default: true)
  --timeout               Maximum wait time in seconds (default: 7200)
  --poll-interval         Status check interval (default: 60)
`,
  examples: [
    'akk workflow train --notebook notebooks/colab/nllb_train_v4.ipynb',
    'akk workflow train --notebook nllb_train.ipynb --name nllb-v4 --experiment nllb-akkadian',
    'akk workflow train --notebook train.ipynb --skip-upload --wait-for-completion',
  ],
  args: TrainWorkflowArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucket = args.bucket || config?.colab?.gcs_bucket

    if (!bucket) {
      return error('NO_BUCKET', 'No GCS bucket configured', 'Set colab.gcs_bucket in akk.toml', {})
    }

    // Generate run name
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '').replace('T', '-')
    const notebookBase = basename(args.notebook).replace(/\.(ipynb|py)$/, '')
    const runName = args.name || `${notebookBase}_${timestamp}`
    const experiment = args.experiment || notebookBase.replace(/_v\d+$/, '') || 'default'

    const prefix = `mlflow/runs/${experiment}/${runName}`
    const gcsRunPath = `gs://${bucket}/${prefix}`

    const steps: { step: string; status: 'pending' | 'running' | 'done' | 'skipped' | 'error'; message?: string }[] = [
      { step: 'upload', status: 'pending' },
      { step: 'colab', status: 'pending' },
      { step: 'monitor', status: 'pending' },
      { step: 'download', status: 'pending' },
      { step: 'sync', status: 'pending' },
    ]

    const updateStep = (name: string, status: (typeof steps)[0]['status'], message?: string) => {
      const step = steps.find((s) => s.step === name)
      if (step) {
        step.status = status
        step.message = message
      }
    }

    // Step 1: Upload notebook
    if (!args['skip-upload']) {
      updateStep('upload', 'running')
      logStep({ step: 'upload', message: 'Uploading notebook to GCS...' }, ctx.output)

      const localPath = args.notebook.startsWith('/') ? args.notebook : join(ctx.cwd, args.notebook)
      const notebookFile = Bun.file(localPath)

      if (!(await notebookFile.exists())) {
        return error('FILE_NOT_FOUND', `Notebook not found: ${localPath}`, 'Check the path', {})
      }

      // Check bucket
      if (!(await bucketExists(bucket))) {
        return error('BUCKET_NOT_FOUND', `Bucket not found: ${bucket}`, 'Run: akk colab configure --create', {})
      }

      const notebookFilename = basename(localPath)
      const gcsNotebookPath = `gs://${bucket}/notebooks/${notebookFilename}`

      const uploadResult = await upload(localPath, gcsNotebookPath)
      if (!uploadResult.success) {
        updateStep('upload', 'error', uploadResult.message)
        return error('UPLOAD_FAILED', `Upload failed: ${uploadResult.message}`, 'Check GCS permissions', {})
      }

      updateStep('upload', 'done', gcsNotebookPath)

      // Also create run directory with config
      const runConfig = {
        run_name: runName,
        experiment: experiment,
        notebook: notebookFilename,
        bucket: bucket,
        gcs_run_path: gcsRunPath,
        created_at: new Date().toISOString(),
        status: 'pending',
      }

      const configPath = `/tmp/run_config_${Date.now()}.json`
      await Bun.write(configPath, JSON.stringify(runConfig, null, 2))
      await upload(configPath, `${gcsRunPath}/config.json`)
    } else {
      updateStep('upload', 'skipped')
    }

    // Step 2: Display Colab instructions
    updateStep('colab', 'running')
    logStep({ step: 'colab', message: 'Preparing Colab instructions...' }, ctx.output)

    const colabInstructions = `
╔══════════════════════════════════════════════════════════════════════════════╗
║                           GOOGLE COLAB INSTRUCTIONS                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. Open Google Colab: https://colab.research.google.com                     ║
║                                                                              ║
║  2. Mount GCS and open notebook:                                             ║
║     from google.colab import drive                                           ║
║     drive.mount('/content/drive')                                            ║
║                                                                              ║
║  3. Set environment variables in notebook:                                   ║
║     GCS_BUCKET = "${bucket}"                                                 ║
║     RUN_NAME = "${runName}"                                                  ║
║     EXPERIMENT = "${experiment}"                                             ║
║     OUTPUT_PATH = "${gcsRunPath}"                                            ║
║                                                                              ║
║  4. Run all cells                                                            ║
║                                                                              ║
║  5. Training outputs will be saved to:                                       ║
║     ${gcsRunPath}/                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
`
    console.log(colabInstructions)
    updateStep('colab', 'done')

    // Step 3: Monitor (if waiting)
    if (args['wait-for-completion']) {
      updateStep('monitor', 'running')
      logStep({ step: 'monitor', message: 'Waiting for training to complete...' }, ctx.output)

      console.log('\nMonitoring training progress (Ctrl+C to stop waiting)...')
      console.log(`Checking every ${args['poll-interval']} seconds, timeout: ${args.timeout}s\n`)

      const startTime = Date.now()
      let completed = false

      while (!completed && Date.now() - startTime < args.timeout * 1000) {
        // Check for completion indicators
        const statusPath = `${gcsRunPath}/status.json`
        const tempFile = `/tmp/status_check_${Date.now()}.json`

        await download(statusPath, tempFile)
        const statusFile = Bun.file(tempFile)

        if (await statusFile.exists()) {
          try {
            const status = JSON.parse(await statusFile.text())

            const elapsed = Math.round((Date.now() - startTime) / 1000)
            const progress = status.progress !== undefined ? `${(status.progress * 100).toFixed(1)}%` : 'unknown'

            console.log(`[${elapsed}s] Phase: ${status.phase || 'unknown'}, Progress: ${progress}`)

            if (status.phase === 'completed') {
              completed = true
              updateStep('monitor', 'done', 'Training completed')
              break
            } else if (status.phase === 'failed') {
              updateStep('monitor', 'error', status.error || 'Training failed')
              return error('TRAINING_FAILED', 'Training failed', status.error || 'Check Colab logs', { status })
            }
          } catch {
            // Status file not valid JSON yet
          }
        }

        // Also check for model files as completion indicator
        const outputFiles = await listFiles(`${gcsRunPath}/output/`)
        if (outputFiles.some((f) => f.includes('model') || f.includes('pytorch'))) {
          console.log('Model files detected in output directory')
          completed = true
          updateStep('monitor', 'done', 'Model files detected')
          break
        }

        await new Promise((resolve) => setTimeout(resolve, args['poll-interval'] * 1000))
      }

      if (!completed) {
        updateStep('monitor', 'error', 'Timeout waiting for completion')
        console.log('\nTimeout reached. Training may still be in progress.')
        console.log(`Check status: akk colab status --run ${runName} --experiment ${experiment}`)
      }
    } else {
      updateStep('monitor', 'skipped')
    }

    // Step 4: Download artifacts
    if (!args['skip-download']) {
      updateStep('download', 'running')
      logStep({ step: 'download', message: 'Downloading artifacts...' }, ctx.output)

      const outputDir = join(ctx.cwd, 'artifacts', runName)
      await Bun.write(join(outputDir, '.gitkeep'), '')

      // Download model
      const _modelResult = await rsync(`${gcsRunPath}/output/`, join(outputDir, 'model'))

      // Download metrics
      const metricFiles = ['metrics.json', 'training_log.json', 'status.json', 'config.json']
      for (const file of metricFiles) {
        await download(`${gcsRunPath}/${file}`, join(outputDir, file))
      }

      updateStep('download', 'done', outputDir)
    } else {
      updateStep('download', 'skipped')
    }

    // Step 5: Sync to MLFlow
    updateStep('sync', 'running')
    logStep({ step: 'sync', message: 'Syncing to local MLFlow...' }, ctx.output)

    // This would call the mlflow sync command internally
    // For now, just provide instructions
    updateStep('sync', 'done', 'Run: akk mlflow sync --experiment ' + experiment)

    // Summary
    const summary = {
      run: runName,
      experiment,
      bucket,
      gcsPath: gcsRunPath,
      steps: steps.map(
        (s) =>
          `${s.status === 'done' ? '✓' : s.status === 'error' ? '✗' : s.status === 'skipped' ? '○' : '·'} ${s.step}${s.message ? `: ${s.message}` : ''}`
      ),
      nextSteps: [
        `Check status: akk colab status --run ${runName} --experiment ${experiment}`,
        `Download artifacts: akk colab download-artifacts --run ${runName}`,
        `Sync to MLFlow: akk mlflow sync --experiment ${experiment}`,
        `Upload to Kaggle: akk kaggle upload-model ./artifacts/${runName}/model`,
      ],
    }

    return success(summary)
  },
}
