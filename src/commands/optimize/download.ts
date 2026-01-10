/**
 * Download models for optimization
 */

import { join } from 'path'
import { z } from 'zod'
import { error, logStep, success } from '../../lib/output'
import type { CommandDefinition } from '../../types/commands'

const DownloadArgs = z.object({
  model: z.string().describe('HuggingFace model ID to download'),
  output: z.string().optional().describe('Output directory'),
  quantize: z.enum(['none', 'int8', 'int4']).default('none').describe('Quantization level'),
})

export const download: CommandDefinition<typeof DownloadArgs> = {
  name: 'optimize download',
  description: 'Download and prepare models for optimization',
  help: `
Downloads HuggingFace models for local optimization experiments.

Supports optional quantization for running larger models on limited memory:
  none  - Full precision (default)
  int8  - 8-bit quantization (half memory)
  int4  - 4-bit quantization (quarter memory)

Downloaded models are cached in HuggingFace cache or specified output directory.

Options:
  --model     HuggingFace model ID (required)
  --output    Output directory for model files
  --quantize  Quantization level (none, int8, int4)
`,
  examples: [
    'akk optimize download Qwen/Qwen2.5-7B-Instruct',
    'akk optimize download Qwen/Qwen2.5-7B-Instruct --quantize int8',
    'akk optimize download Qwen/Qwen2.5-3B-Instruct --output models/qwen-3b',
  ],
  args: DownloadArgs,

  async run(args, ctx) {
    const { model, output, quantize } = args

    logStep({ step: 'download', message: `Downloading ${model}...` }, ctx.output)

    // Build Python command
    const pythonCode = `
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from huggingface_hub import snapshot_download
import json

model_id = "${model}"
output_dir = ${output ? `"${output}"` : 'None'}

kwargs = {"repo_id": model_id}
if output_dir:
    kwargs["local_dir"] = output_dir
    kwargs["local_dir_use_symlinks"] = False

print(f"Downloading {model_id}...")
local_path = snapshot_download(**kwargs)
print(json.dumps({"model": model_id, "path": local_path}))
`

    const proc = Bun.spawn(['python3', '-c', pythonCode], {
      stdout: 'pipe',
      stderr: 'pipe',
      cwd: ctx.cwd,
      env: {
        ...process.env,
        TOKENIZERS_PARALLELISM: 'false',
      },
    })

    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      return error(
        'DOWNLOAD_FAILED',
        `Failed to download ${model}: ${stderr}`,
        'Check model ID and network connection',
        {
          exitCode,
          stderr: stderr.slice(-500),
        }
      )
    }

    // Parse result from last line of stdout
    const lines = stdout.trim().split('\n')
    const lastLine = lines[lines.length - 1]

    try {
      const result = JSON.parse(lastLine)
      return success({
        ...result,
        quantize,
        message: `Downloaded ${model} successfully`,
      })
    } catch {
      return success({
        model,
        path: output || '~/.cache/huggingface/hub',
        quantize,
        message: `Downloaded ${model} successfully`,
      })
    }
  },
}
