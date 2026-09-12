/** Presentation helpers. No business logic and no derived workflow state. */

import type { ExperimentStatus, LeakageRiskLevel, ModelRunStatus, Severity } from '../api/types';

export function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return '—';
  return `${value.toFixed(digits)}%`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

export const STATUS_COLORS: Record<ExperimentStatus, string> = {
  CREATED: 'default',
  EDA_RUNNING: 'processing',
  EDA_COMPLETED: 'cyan',
  FEATURE_REVIEW: 'blue',
  READY_FOR_TRAINING: 'blue',
  PREPARING: 'processing',
  TRAINING: 'processing',
  EVALUATING: 'processing',
  COMPLETED: 'success',
  COMPLETED_WITH_WARNINGS: 'warning',
  FAILED: 'error',
};

export const STATUS_LABELS: Record<ExperimentStatus, string> = {
  CREATED: 'Created',
  EDA_RUNNING: 'Running EDA',
  EDA_COMPLETED: 'EDA complete',
  FEATURE_REVIEW: 'Awaiting feature review',
  READY_FOR_TRAINING: 'Ready for training',
  PREPARING: 'Preparing data',
  TRAINING: 'Training',
  EVALUATING: 'Evaluating',
  COMPLETED: 'Completed',
  COMPLETED_WITH_WARNINGS: 'Completed with warnings',
  FAILED: 'Failed',
};

export const MODEL_STATUS_COLORS: Record<ModelRunStatus, string> = {
  QUEUED: 'default',
  RUNNING: 'processing',
  COMPLETED: 'success',
  FAILED: 'error',
  SKIPPED: 'default',
};

export const SEVERITY_COLORS: Record<Severity, string> = {
  info: 'default',
  low: 'blue',
  medium: 'orange',
  high: 'volcano',
  critical: 'red',
};

export const RISK_COLORS: Record<LeakageRiskLevel, string> = {
  none: 'default',
  requires_review: 'gold',
  potential_leakage: 'volcano',
  confirmed_duplicate: 'red',
};

export const RISK_LABELS: Record<LeakageRiskLevel, string> = {
  none: 'No finding',
  requires_review: 'Requires review',
  potential_leakage: 'Potential leakage',
  confirmed_duplicate: 'Duplicates the target',
};

export function humanize(value: string): string {
  return value.replace(/_/g, ' ').replace(/^./, (character) => character.toUpperCase());
}
