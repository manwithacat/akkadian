/**
 * Akkadian MCP Server
 *
 * Minimal, token-efficient MCP server that provides:
 * - Single "akk" tool for CLI operations
 * - Domain knowledge resource for on-demand loading
 * - Project status resource for current state
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { spawn } from 'child_process'
import { join } from 'path'
import { z } from 'zod'
import {
  findCompetitionConfig,
  findConfigPath,
  getActiveCompetitionDir,
  getProjectRoot,
  loadCompetitionConfig,
  loadConfig,
} from '../lib/config'
import { listRegisteredKernels } from '../lib/kernel-registry'
import {
  formatQuickReference,
  getCommandHelp,
  getErrorFix,
  getKnowledge,
  getPattern,
  getPlatformInfo,
  getQuickReference,
  getWorkflow,
  searchKnowledge,
} from './knowledge'

/**
 * Run akk CLI command and return output
 */
async function runAkkCommand(command: string): Promise<{
  success: boolean
  stdout: string
  stderr: string
  exitCode: number
}> {
  return new Promise((resolve) => {
    // Find akk binary - try dist first, then bun run
    const akkPath = join(__dirname, '../../dist/akk')

    const proc = spawn(akkPath, command.split(/\s+/).filter(Boolean), {
      cwd: process.cwd(),
      env: process.env,
    })

    let stdout = ''
    let stderr = ''

    proc.stdout?.on('data', (data) => {
      stdout += data.toString()
    })

    proc.stderr?.on('data', (data) => {
      stderr += data.toString()
    })

    proc.on('close', (code) => {
      resolve({
        success: code === 0,
        stdout,
        stderr,
        exitCode: code ?? 1,
      })
    })

    proc.on('error', (err) => {
      resolve({
        success: false,
        stdout: '',
        stderr: err.message,
        exitCode: 1,
      })
    })
  })
}

/**
 * Fetch recent submissions from Kaggle API
 */
async function fetchKaggleSubmissions(competition?: string): Promise<unknown[] | null> {
  try {
    const slug = competition || 'deep-past-initiative-machine-translation'
    const result = await runAkkCommand(`kaggle submissions --competition ${slug}`)
    if (result.success && result.stdout) {
      const data = JSON.parse(result.stdout)
      return data?.data?.submissions?.slice(0, 5) || []
    }
  } catch {
    // Ignore errors - this is best-effort
  }
  return null
}

/**
 * Get current project status
 * Enhanced to provide useful context even without competition.toml
 */
async function getProjectStatus(): Promise<Record<string, unknown>> {
  // Try to load project config (akk.toml) first for project root
  const akkConfigPath = await findConfigPath()
  const projectConfig = akkConfigPath ? await loadConfig(akkConfigPath) : null

  // Try to find competition.toml from cwd first
  let configPath = await findCompetitionConfig()

  // If not found from cwd, try the active competition directory from akk.toml
  if (!configPath && akkConfigPath && projectConfig) {
    const projectRoot = getProjectRoot(akkConfigPath)
    const activeCompDir = getActiveCompetitionDir(projectRoot, projectConfig)
    configPath = await findCompetitionConfig(activeCompDir)
  }

  // Even without competition.toml, try to get useful info
  if (!configPath) {
    // Fetch recent submissions from Kaggle API as fallback
    const slug = projectConfig?.kaggle?.competition
    const submissions = await fetchKaggleSubmissions(slug)

    return {
      hasCompetition: false,
      project: projectConfig
        ? {
            name: projectConfig.project?.name,
            hasConfig: true,
            competition: projectConfig.kaggle?.competition,
          }
        : null,
      recentSubmissions: submissions,
      hint: 'Run "akk competition init <slug>" to set up competition tracking',
    }
  }

  const config = await loadCompetitionConfig(configPath)
  if (!config) {
    return {
      hasCompetition: false,
      error: 'Failed to load competition.toml',
    }
  }

  const kernels = await listRegisteredKernels()

  return {
    hasCompetition: true,
    competition: {
      name: config.competition.name,
      slug: config.competition.slug,
      platform: config.competition.platform,
      metric: config.competition.metric,
    },
    activeModel: config.active_model
      ? {
          name: config.active_model.name,
          bestScore: config.active_model.best_score,
        }
      : null,
    submissions: {
      total: config.submissions.total,
      bestScore: config.submissions.best_score,
    },
    kernels: kernels.map((k) => ({
      name: k.name,
      version: k.currentVersion,
      lastStatus: k.lastStatus,
    })),
    training: {
      platform: config.training.default_platform,
      batchSize: config.training.default_batch_size,
    },
  }
}

/**
 * Create and start the MCP server
 */
export async function createServer(): Promise<McpServer> {
  const server = new McpServer({
    name: 'akkadian',
    version: '1.0.0',
  })

  // ============================================
  // Single Tool: akk
  // ============================================

  server.tool(
    'akk',
    'Run Akkadian CLI commands. Use "help" for command list.',
    {
      command: z.string().describe('CLI command to run (e.g., "doctor", "kaggle list-kernels", "help preflight")'),
    },
    async ({ command }) => {
      // Handle help commands specially for token efficiency
      if (command === 'help' || command === '') {
        return {
          content: [
            {
              type: 'text' as const,
              text: formatQuickReference(),
            },
          ],
        }
      }

      // Handle help for specific command
      if (command.startsWith('help ')) {
        const topic = command.slice(5).trim()

        // Check if it's a command
        const cmdHelp = getCommandHelp(topic)
        if (cmdHelp) {
          return {
            content: [{ type: 'text' as const, text: cmdHelp }],
          }
        }

        // Check if it's a workflow
        const wfHelp = getWorkflow(topic)
        if (wfHelp) {
          return {
            content: [{ type: 'text' as const, text: wfHelp }],
          }
        }

        // Check if it's a pattern
        const patternHelp = getPattern(topic)
        if (patternHelp) {
          return {
            content: [{ type: 'text' as const, text: patternHelp }],
          }
        }

        // Check if it's a platform
        const platHelp = getPlatformInfo(topic)
        if (platHelp) {
          return {
            content: [{ type: 'text' as const, text: platHelp }],
          }
        }

        // Check if it's an error code
        const errHelp = getErrorFix(topic)
        if (errHelp) {
          return {
            content: [{ type: 'text' as const, text: errHelp }],
          }
        }

        // Search for topic
        const results = searchKnowledge(topic)
        if (results.length > 0) {
          return {
            content: [
              {
                type: 'text' as const,
                text: `Found matches for "${topic}":\n${results.map((r) => `- ${r}`).join('\n')}\n\nUse "help <match>" for details.`,
              },
            ],
          }
        }

        return {
          content: [
            {
              type: 'text' as const,
              text: `No help found for "${topic}". Try "help" for available commands.`,
            },
          ],
        }
      }

      // Run actual CLI command
      const result = await runAkkCommand(command)

      if (result.success) {
        return {
          content: [
            {
              type: 'text' as const,
              text: result.stdout || 'Command completed successfully.',
            },
          ],
        }
      } else {
        return {
          content: [
            {
              type: 'text' as const,
              text: `Error (exit ${result.exitCode}):\n${result.stderr || result.stdout}`,
            },
          ],
          isError: true,
        }
      }
    }
  )

  // ============================================
  // Resources
  // ============================================

  // Domain knowledge - quick reference (minimal tokens)
  server.resource(
    'Domain Knowledge (Quick)',
    'akk://knowledge/quick',
    {
      description: 'Quick command reference for Akkadian CLI',
      mimeType: 'application/json',
    },
    async () => ({
      contents: [
        {
          uri: 'akk://knowledge/quick',
          mimeType: 'application/json',
          text: JSON.stringify(getQuickReference(), null, 2),
        },
      ],
    })
  )

  // Domain knowledge - full (on-demand)
  server.resource(
    'Domain Knowledge (Full)',
    'akk://knowledge/full',
    {
      description: 'Complete domain knowledge including workflows, patterns, and troubleshooting',
      mimeType: 'application/json',
    },
    async () => ({
      contents: [
        {
          uri: 'akk://knowledge/full',
          mimeType: 'application/json',
          text: JSON.stringify(getKnowledge('full'), null, 2),
        },
      ],
    })
  )

  // Project status
  server.resource(
    'Project Status',
    'akk://status',
    {
      description: 'Current competition, model, and kernel status',
      mimeType: 'application/json',
    },
    async () => ({
      contents: [
        {
          uri: 'akk://status',
          mimeType: 'application/json',
          text: JSON.stringify(await getProjectStatus(), null, 2),
        },
      ],
    })
  )

  return server
}

/**
 * Main entry point
 */
export async function main(): Promise<void> {
  const server = await createServer()
  const transport = new StdioServerTransport()

  await server.connect(transport)

  // Handle shutdown
  process.on('SIGINT', async () => {
    await server.close()
    process.exit(0)
  })

  process.on('SIGTERM', async () => {
    await server.close()
    process.exit(0)
  })
}

// Run if executed directly
if (import.meta.main) {
  main().catch((err) => {
    console.error('MCP Server error:', err)
    process.exit(1)
  })
}
