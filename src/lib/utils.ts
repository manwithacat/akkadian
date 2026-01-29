/**
 * Shared utilities for Akkadian CLI
 */

/**
 * Convert a name to a URL-safe slug
 */
export function toSlug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-')
}

/**
 * Extract dataset paths referenced in notebook code.
 * Detects /kaggle/input/dataset-slug patterns in Python source.
 */
export function extractDatasetReferences(content: string): string[] {
  const datasets = new Set<string>()

  // Pattern 1: Direct path strings like /kaggle/input/dataset-slug/...
  const directPathPattern = /["']\/kaggle\/input\/([a-z0-9-]+)(?:\/[^"']*)?["']/gi
  let match = directPathPattern.exec(content)
  while (match !== null) {
    datasets.add(match[1])
    match = directPathPattern.exec(content)
  }

  // Pattern 2: KAGGLE_INPUT / "dataset-slug/file.csv" (pathlib style)
  const pathlibPattern = /KAGGLE_INPUT\s*\/\s*["']([a-z0-9-]+)(?:\/[^"']*)?["']/gi
  match = pathlibPattern.exec(content)
  while (match !== null) {
    datasets.add(match[1])
    match = pathlibPattern.exec(content)
  }

  // Pattern 3: Path("/kaggle/input/dataset-slug/...")
  const pathPattern = /Path\s*\(\s*["']\/kaggle\/input\/([a-z0-9-]+)/gi
  match = pathPattern.exec(content)
  while (match !== null) {
    datasets.add(match[1])
    match = pathPattern.exec(content)
  }

  return Array.from(datasets)
}
