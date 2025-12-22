/**
 * Configure GCS bucket and auth for Colab
 */

import { z } from 'zod'
import type { CommandDefinition } from '../../types/commands'
import { success, error, progress } from '../../lib/output'
import { checkAuth, getCurrentProject, setProject, bucketExists, createBucket, listBuckets } from '../../lib/gcs'

const ConfigureArgs = z.object({
  bucket: z.string().optional().describe('GCS bucket name'),
  project: z.string().optional().describe('GCP project ID'),
  create: z.boolean().default(false).describe('Create bucket if not exists'),
  location: z.string().default('us-central1').describe('Bucket location (for creation)'),
})

export const configure: CommandDefinition<typeof ConfigureArgs> = {
  name: 'colab configure',
  description: 'Set up GCS bucket and auth for Colab',
  help: `
Configures Google Cloud Storage for use with Colab.
Checks authentication, project settings, and bucket access.

Options:
  --bucket    GCS bucket name (default: from akk.toml)
  --project   GCP project ID (default: from akk.toml or gcloud config)
  --create    Create bucket if it doesn't exist
  --location  Bucket location for creation (default: us-central1)
`,
  examples: [
    'akk colab configure',
    'akk colab configure --bucket akkadian-models --project my-project',
    'akk colab configure --bucket new-bucket --create --location europe-west1',
  ],
  args: ConfigureArgs,

  async run(args, ctx) {
    const config = ctx.config
    const bucketName = args.bucket || config?.colab?.gcs_bucket
    const projectId = args.project || config?.colab?.project

    const checks: { name: string; status: 'ok' | 'warning' | 'error'; message: string }[] = []

    // Check authentication
    progress({ step: 'auth', message: 'Checking GCloud authentication...' }, ctx.output)
    const auth = await checkAuth()

    if (!auth.authenticated) {
      checks.push({
        name: 'Authentication',
        status: 'error',
        message: 'Not authenticated. Run: gcloud auth login',
      })
    } else {
      checks.push({
        name: 'Authentication',
        status: 'ok',
        message: `Authenticated as ${auth.account}`,
      })
    }

    // Check/set project
    progress({ step: 'project', message: 'Checking GCP project...' }, ctx.output)
    let currentProject = await getCurrentProject()

    if (projectId && currentProject !== projectId) {
      const result = await setProject(projectId)
      if (result.success) {
        currentProject = projectId
        checks.push({
          name: 'Project',
          status: 'ok',
          message: `Set to ${projectId}`,
        })
      } else {
        checks.push({
          name: 'Project',
          status: 'error',
          message: result.message,
        })
      }
    } else if (currentProject) {
      checks.push({
        name: 'Project',
        status: 'ok',
        message: currentProject,
      })
    } else {
      checks.push({
        name: 'Project',
        status: 'warning',
        message: 'No project set. Run: gcloud config set project YOUR_PROJECT',
      })
    }

    // Check bucket
    if (bucketName) {
      progress({ step: 'bucket', message: `Checking bucket: ${bucketName}...` }, ctx.output)
      const exists = await bucketExists(bucketName)

      if (exists) {
        checks.push({
          name: 'Bucket',
          status: 'ok',
          message: `gs://${bucketName} exists`,
        })
      } else if (args.create) {
        progress({ step: 'create', message: `Creating bucket: ${bucketName}...` }, ctx.output)
        const result = await createBucket(bucketName, args.location)

        if (result.success) {
          checks.push({
            name: 'Bucket',
            status: 'ok',
            message: `Created gs://${bucketName} in ${args.location}`,
          })
        } else {
          checks.push({
            name: 'Bucket',
            status: 'error',
            message: `Failed to create: ${result.message}`,
          })
        }
      } else {
        checks.push({
          name: 'Bucket',
          status: 'error',
          message: `Bucket gs://${bucketName} does not exist. Use --create to create it.`,
        })
      }
    } else {
      // List available buckets
      const buckets = await listBuckets()
      checks.push({
        name: 'Bucket',
        status: 'warning',
        message: `No bucket configured. Available: ${buckets.slice(0, 5).join(', ')}${buckets.length > 5 ? '...' : ''}`,
      })
    }

    // Check for errors
    const errors = checks.filter((c) => c.status === 'error')
    if (errors.length > 0) {
      return error(
        'CONFIG_INCOMPLETE',
        `Configuration incomplete: ${errors.map((e) => e.name).join(', ')} need attention`,
        'Fix the issues above and run again',
        { checks }
      )
    }

    return success({
      account: auth.account,
      project: currentProject,
      bucket: bucketName,
      checks: checks.map((c) => `${c.status === 'ok' ? '✓' : '!'} ${c.name}: ${c.message}`),
    })
  },
}
