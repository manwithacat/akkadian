export { version } from './version'
export { doctor } from './doctor'

// Kaggle commands
export { uploadNotebook, uploadModel, runKernel, downloadOutput, listKernels, logs, status as kaggleStatus, submissions as kaggleSubmissions } from './kaggle'

// Colab commands
export { configure, downloadModel, uploadNotebook as colabUploadNotebook, colabStatus, downloadArtifacts, cleanup } from './colab'

// Local commands
export { evaluate, infer, analyze } from './local'

// MLFlow commands
export { start, log, sync, register } from './mlflow'

// Workflow commands
export { train, prepare, importRun, listRuns } from './workflow'

// Vertex AI commands
export { submit as vertexSubmit, status as vertexStatus, list as vertexList } from './vertex'

// Preflight commands
export { preflight, platforms as preflightPlatforms, validate as preflightValidate } from './preflight'

// Competition commands
export { competitionInit, competitionStatus } from './competition'

// Template commands
export { templateGenerate, templateList } from './template'

// MCP commands
export { mcpServe } from './mcp'

// Data management commands
export { download as dataDownload, list as dataList, register as dataRegister, explore as dataExplore, wrangler as dataWrangler } from './data'

// Notebook commands
export { notebookBuild } from './notebook'
