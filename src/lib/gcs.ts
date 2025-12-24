/**
 * Google Cloud Storage (gsutil) Wrapper
 */

import { type CommandResult, cli } from './process'

export interface GCSConfig {
  bucket: string
  project: string
}

/**
 * Run gsutil command (using shared process utility)
 */
async function runGsutil(args: string[]): Promise<CommandResult> {
  return cli.gsutil(args)
}

/**
 * Run gcloud command (using shared process utility)
 */
async function runGcloud(args: string[]): Promise<CommandResult> {
  return cli.gcloud(args)
}

/**
 * Check if authenticated with gcloud
 */
export async function checkAuth(): Promise<{ authenticated: boolean; account?: string }> {
  const { stdout, exitCode } = await runGcloud(['auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'])

  if (exitCode !== 0 || !stdout.trim()) {
    return { authenticated: false }
  }

  return { authenticated: true, account: stdout.trim() }
}

/**
 * Get current project
 */
export async function getCurrentProject(): Promise<string | null> {
  const { stdout, exitCode } = await runGcloud(['config', 'get-value', 'project'])

  if (exitCode !== 0 || !stdout.trim()) {
    return null
  }

  return stdout.trim()
}

/**
 * Set project
 */
export async function setProject(project: string): Promise<{ success: boolean; message: string }> {
  const { stdout, stderr, exitCode } = await runGcloud(['config', 'set', 'project', project])

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: `Project set to ${project}` }
}

/**
 * List buckets
 */
export async function listBuckets(): Promise<string[]> {
  const { stdout, exitCode } = await runGsutil(['ls'])

  if (exitCode !== 0) {
    return []
  }

  return stdout
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => line.replace('gs://', '').replace('/', ''))
}

/**
 * Check if bucket exists
 */
export async function bucketExists(bucket: string): Promise<boolean> {
  const { exitCode } = await runGsutil(['ls', '-b', `gs://${bucket}`])
  return exitCode === 0
}

/**
 * Create bucket
 */
export async function createBucket(bucket: string, location?: string): Promise<{ success: boolean; message: string }> {
  const args = ['mb']
  if (location) {
    args.push('-l', location)
  }
  args.push(`gs://${bucket}`)

  const { stdout, stderr, exitCode } = await runGsutil(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * List files in GCS path
 */
export async function listFiles(gcsPath: string): Promise<string[]> {
  const { stdout, exitCode } = await runGsutil(['ls', gcsPath])

  if (exitCode !== 0) {
    return []
  }

  return stdout.trim().split('\n').filter(Boolean)
}

/**
 * Download file or directory from GCS
 */
export async function download(
  gcsPath: string,
  localPath: string,
  options: { recursive?: boolean } = {}
): Promise<{ success: boolean; message: string }> {
  const args = ['-m', 'cp']
  if (options.recursive) {
    args.push('-r')
  }
  args.push(gcsPath, localPath)

  const { stdout, stderr, exitCode } = await runGsutil(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Upload file or directory to GCS
 */
export async function upload(
  localPath: string,
  gcsPath: string,
  options: { recursive?: boolean } = {}
): Promise<{ success: boolean; message: string }> {
  const args = ['-m', 'cp']
  if (options.recursive) {
    args.push('-r')
  }
  args.push(localPath, gcsPath)

  const { stdout, stderr, exitCode } = await runGsutil(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Rsync from GCS to local
 */
export async function rsync(
  gcsPath: string,
  localPath: string,
  options: { delete?: boolean } = {}
): Promise<{ success: boolean; message: string }> {
  const args = ['-m', 'rsync', '-r']
  if (options.delete) {
    args.push('-d')
  }
  args.push(gcsPath, localPath)

  const { stdout, stderr, exitCode } = await runGsutil(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout }
}

/**
 * Get file size in GCS
 */
export async function getSize(gcsPath: string): Promise<number | null> {
  const { stdout, exitCode } = await runGsutil(['du', '-s', gcsPath])

  if (exitCode !== 0) {
    return null
  }

  const match = stdout.match(/^(\d+)/)
  return match ? parseInt(match[1], 10) : null
}

/**
 * Check if a file or directory exists in GCS
 */
export async function exists(gcsPath: string): Promise<boolean> {
  const { exitCode } = await runGsutil(['ls', gcsPath])
  return exitCode === 0
}

/**
 * Delete a file or directory from GCS
 */
export async function deleteFile(
  gcsPath: string,
  options: { recursive?: boolean } = {}
): Promise<{ success: boolean; message: string }> {
  const args = ['rm']
  if (options.recursive) {
    args.push('-r')
  }
  args.push(gcsPath)

  const { stdout, stderr, exitCode } = await runGsutil(args)

  if (exitCode !== 0) {
    return { success: false, message: stderr || stdout }
  }

  return { success: true, message: stdout || 'Deleted successfully' }
}
