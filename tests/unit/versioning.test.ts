/**
 * Versioning Unit Tests
 *
 * Tests kernel versioning functionality including:
 * - Semver, timestamp, and experiment strategies
 * - Version incrementing
 * - Kernel ID generation with embedded versions
 */

import { describe, it, expect } from 'bun:test'
import {
  NotebookVersioningSchema,
  KernelMetadataSchema,
  generateVersionedKernelId,
  incrementVersion,
} from '../../src/types/competition'

describe('NotebookVersioningSchema', () => {
  it('parses valid versioning config with defaults', () => {
    const result = NotebookVersioningSchema.parse({})
    expect(result.use_kaggle_versioning).toBe(true)
    expect(result.strategy).toBe('semver')
    expect(result.current_version).toBe('1.0.0')
  })

  it('parses config with custom values', () => {
    const result = NotebookVersioningSchema.parse({
      use_kaggle_versioning: false,
      strategy: 'timestamp',
      current_version: '2.0.0',
      base_name: 'my-kernel',
    })
    expect(result.use_kaggle_versioning).toBe(false)
    expect(result.strategy).toBe('timestamp')
    expect(result.current_version).toBe('2.0.0')
    expect(result.base_name).toBe('my-kernel')
  })

  it('accepts numeric versions', () => {
    const result = NotebookVersioningSchema.parse({
      current_version: 5,
    })
    expect(result.current_version).toBe(5)
  })
})

describe('KernelMetadataSchema', () => {
  it('parses metadata with versioning', () => {
    const result = KernelMetadataSchema.parse({
      id: 'user/kernel',
      title: 'Test Kernel',
      code_file: 'test.py',
      versioning: {
        use_kaggle_versioning: false,
        strategy: 'semver',
        current_version: '1.0.3',
        base_name: 'nllb-inference',
      },
    })
    expect(result.versioning).toBeDefined()
    expect(result.versioning?.use_kaggle_versioning).toBe(false)
    expect(result.versioning?.current_version).toBe('1.0.3')
  })

  it('parses metadata without versioning', () => {
    const result = KernelMetadataSchema.parse({
      id: 'user/kernel',
      title: 'Test Kernel',
      code_file: 'test.py',
    })
    expect(result.versioning).toBeUndefined()
  })

  it('parses model_sources as string array', () => {
    const result = KernelMetadataSchema.parse({
      id: 'user/kernel',
      title: 'Test Kernel',
      code_file: 'test.py',
      model_sources: ['user/model/transformers/v2'],
    })
    expect(result.model_sources).toEqual(['user/model/transformers/v2'])
  })
})

describe('generateVersionedKernelId', () => {
  describe('with Kaggle versioning enabled', () => {
    it('returns simple slug without version suffix', () => {
      const versioning = NotebookVersioningSchema.parse({
        use_kaggle_versioning: true,
        current_version: '1.0.0',
      })
      const result = generateVersionedKernelId('user', 'my-kernel', versioning)
      expect(result.id).toBe('user/my-kernel')
      expect(result.slug).toBe('my-kernel')
    })
  })

  describe('with semver strategy', () => {
    it('embeds version in kernel ID', () => {
      const versioning = NotebookVersioningSchema.parse({
        use_kaggle_versioning: false,
        strategy: 'semver',
        current_version: '1.0.0',
      })
      const result = generateVersionedKernelId('manwithacat', 'nllb-inference', versioning)
      expect(result.id).toBe('manwithacat/nllb-inference-v1-0-0')
      expect(result.slug).toBe('nllb-inference-v1-0-0')
    })

    it('handles different version numbers', () => {
      const versioning = NotebookVersioningSchema.parse({
        use_kaggle_versioning: false,
        strategy: 'semver',
        current_version: '2.5.3',
      })
      const result = generateVersionedKernelId('user', 'kernel', versioning)
      expect(result.slug).toBe('kernel-v2-5-3')
    })
  })

  describe('with experiment strategy', () => {
    it('creates exp-NN format', () => {
      const versioning = NotebookVersioningSchema.parse({
        use_kaggle_versioning: false,
        strategy: 'experiment',
        current_version: 5,
      })
      const result = generateVersionedKernelId('user', 'train', versioning)
      expect(result.slug).toBe('train-exp-05')
    })

    it('pads single digit experiment numbers', () => {
      const versioning = NotebookVersioningSchema.parse({
        use_kaggle_versioning: false,
        strategy: 'experiment',
        current_version: 1,
      })
      const result = generateVersionedKernelId('user', 'train', versioning)
      expect(result.slug).toBe('train-exp-01')
    })
  })

  describe('with timestamp strategy', () => {
    it('creates timestamp format', () => {
      const versioning = NotebookVersioningSchema.parse({
        use_kaggle_versioning: false,
        strategy: 'timestamp',
        current_version: '1',
      })
      const result = generateVersionedKernelId('user', 'train', versioning)
      // Should match pattern: train-YYYYMMDD-HHMMSS
      expect(result.slug).toMatch(/^train-\d{8}-\d{6}$/)
    })
  })

  it('normalizes base name to valid slug', () => {
    const versioning = NotebookVersioningSchema.parse({
      use_kaggle_versioning: false,
      strategy: 'semver',
      current_version: '1.0.0',
    })
    const result = generateVersionedKernelId('user', 'My_Awesome Kernel!!!', versioning)
    expect(result.slug).toBe('my-awesome-kernel-v1-0-0')
  })
})

describe('incrementVersion', () => {
  describe('semver strategy', () => {
    it('increments patch by default', () => {
      expect(incrementVersion('1.0.0', 'semver')).toBe('1.0.1')
      expect(incrementVersion('1.2.3', 'semver')).toBe('1.2.4')
    })

    it('increments minor version', () => {
      expect(incrementVersion('1.0.0', 'semver', 'minor')).toBe('1.1.0')
      expect(incrementVersion('2.5.9', 'semver', 'minor')).toBe('2.6.0')
    })

    it('increments major version', () => {
      expect(incrementVersion('1.0.0', 'semver', 'major')).toBe('2.0.0')
      expect(incrementVersion('5.3.2', 'semver', 'major')).toBe('6.0.0')
    })

    it('handles partial version strings', () => {
      expect(incrementVersion('1', 'semver', 'patch')).toBe('1.0.1')
      expect(incrementVersion('1.2', 'semver', 'patch')).toBe('1.2.1')
    })
  })

  describe('timestamp strategy', () => {
    it('increments numeric value', () => {
      expect(incrementVersion(1, 'timestamp')).toBe(2)
      expect(incrementVersion(10, 'timestamp')).toBe(11)
    })
  })

  describe('experiment strategy', () => {
    it('increments numeric value', () => {
      expect(incrementVersion(1, 'experiment')).toBe(2)
      expect(incrementVersion(5, 'experiment')).toBe(6)
    })

    it('parses string to number', () => {
      expect(incrementVersion('3', 'experiment')).toBe(4)
    })
  })
})
