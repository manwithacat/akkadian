/**
 * MCP Server Command
 *
 * Starts the MCP server for LLM agent integration.
 */

import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import { main as startMcpServer } from '../../mcp/server'
import type { CommandDefinition } from '../../types/commands'

const McpServeArgs = z.object({
  // No args needed - server runs on stdio
})

export const mcpServe: CommandDefinition<typeof McpServeArgs> = {
  name: 'mcp serve',
  description: 'Start MCP server for LLM agent integration',
  help: `
Starts the Model Context Protocol (MCP) server on stdio.

The server provides:
- Single "akk" tool for running CLI commands
- "akk://knowledge/quick" resource for command reference
- "akk://knowledge/full" resource for complete domain knowledge
- "akk://status" resource for project state

Configuration in Claude Desktop:
{
  "mcpServers": {
    "akkadian": {
      "command": "/path/to/akk",
      "args": ["mcp", "serve"]
    }
  }
}

Token Efficiency:
- Only the tool schema (~100 tokens) is always loaded
- Domain knowledge is fetched on-demand via resources
- Use "akk help" for minimal command reference
- Use "akk help <topic>" for specific documentation
`,
  examples: ['akk mcp serve'],
  args: McpServeArgs,

  async run(_args, ctx) {
    // Note: This command doesn't return - it runs the MCP server
    // The server communicates via stdio, so we can't use normal output
    try {
      await startMcpServer()
      // This won't be reached unless server exits
      return success({ message: 'MCP server stopped' })
    } catch (err) {
      return error(
        'MCP_SERVER_ERROR',
        err instanceof Error ? err.message : String(err),
        'Check server logs for details'
      )
    }
  },
}
