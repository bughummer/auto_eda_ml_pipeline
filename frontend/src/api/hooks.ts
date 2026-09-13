/**
 * TanStack Query hooks.
 *
 * Polling is driven by the status the backend reports — the frontend never derives or
 * advances workflow state, it only decides how often to ask again.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query';

import { ApiError, api } from './client';
import type {
  ComparisonReport,
  ExperimentRecord,
  DataDictionary,
  EdaReport,
  ExperimentListResponse,
  ExperimentStatus,
  FeatureProposalReport,
  FeatureReviewResponse,
  HealthResponse,
  LeakageReport,
  ModelDescriptor,
  ModelMetadata,
  ReasoningReport,
  TrainingConfigResponse,
  TrainingRequest,
  TrainingStatusResponse,
} from './types';

const POLL_INTERVAL_MS = 3000;

const ACTIVE_STATUSES: ExperimentStatus[] = [
  'CREATED',
  'EDA_RUNNING',
  'EDA_COMPLETED',
  'READY_FOR_TRAINING',
  'PREPARING',
  'TRAINING',
  'EVALUATING',
];

export function isActive(status: ExperimentStatus | undefined): boolean {
  return status !== undefined && ACTIVE_STATUSES.includes(status);
}

/**
 * An artifact bucket to read from. Undefined means the platform's own bucket; a value means
 * "show me what is in that approved bucket", which is how previous experiments produced by
 * another environment are browsed. It is part of every query key, so switching buckets never
 * shows a cached result from the previous one.
 */
export type ArtifactRoot = string | undefined;

function withRoot(path: string, root: ArtifactRoot): string {
  return root ? `${path}${path.includes('?') ? '&' : '?'}root=${encodeURIComponent(root)}` : path;
}

export const keys = {
  health: ['health'] as const,
  models: ['models'] as const,
  experiments: (root: ArtifactRoot) => ['experiments', root ?? 'default'] as const,
  experiment: (id: string, root: ArtifactRoot) => ['experiment', root ?? 'default', id] as const,
  artifact: (id: string, root: ArtifactRoot, kind: string) =>
    ['experiment', root ?? 'default', id, kind] as const,
};

/** Artifacts that do not exist yet are an expected state, not an error to retry. */
const retryUnlessMissing = (failureCount: number, error: unknown) => {
  if (error instanceof ApiError && (error.status === 404 || error.status < 500)) return false;
  return failureCount < 2;
};

export function useHealth() {
  return useQuery({ queryKey: keys.health, queryFn: () => api.get<HealthResponse>('/health') });
}

export function useModelCatalogue() {
  return useQuery({
    queryKey: keys.models,
    queryFn: () => api.get<ModelDescriptor[]>('/models'),
    staleTime: 5 * 60 * 1000,
  });
}

export function useExperiments(root: ArtifactRoot = undefined) {
  return useQuery({
    queryKey: keys.experiments(root),
    queryFn: () => api.get<ExperimentListResponse>(withRoot('/experiments', root)),
    refetchInterval: POLL_INTERVAL_MS * 2,
  });
}

export function useExperiment(id: string, root: ArtifactRoot = undefined) {
  return useQuery({
    queryKey: keys.experiment(id, root),
    queryFn: () => api.get<ExperimentRecord>(withRoot(`/experiments/${id}`, root)),
    // Keep polling while the backend says work is in flight; stop as soon as it is terminal.
    refetchInterval: (query) =>
      isActive(query.state.data?.status) ? POLL_INTERVAL_MS : false,
  });
}

function artifactQuery<T>(key: readonly unknown[], path: string, enabled: boolean) {
  return {
    queryKey: key,
    queryFn: () => api.get<T>(path),
    enabled,
    retry: retryUnlessMissing,
    // An artifact that does not exist yet appears once its stage finishes, so keep asking
    // until it does — and stop as soon as we have it.
    refetchInterval: (query: { state: { data?: T } }) =>
      query.state.data === undefined ? POLL_INTERVAL_MS : false,
  } satisfies UseQueryOptions<T, ApiError>;
}

export function useEda(id: string, root: ArtifactRoot = undefined, enabled = true) {
  return useQuery(artifactQuery<EdaReport>(keys.artifact(id, root, 'eda'), withRoot(`/experiments/${id}/eda`, root), enabled));
}

export function useLeakage(id: string, root: ArtifactRoot = undefined, enabled = true) {
  return useQuery(
    artifactQuery<LeakageReport>(keys.artifact(id, root, 'leakage'), withRoot(`/experiments/${id}/leakage`, root), enabled),
  );
}

export function useFeatureReview(id: string, root: ArtifactRoot = undefined, enabled = true) {
  return useQuery(
    artifactQuery<FeatureReviewResponse>(keys.artifact(id, root, 'features'), withRoot(`/experiments/${id}/features`, root), enabled),
  );
}

export function useTrainingConfig(id: string, root: ArtifactRoot = undefined, enabled = true) {
  return useQuery(
    artifactQuery<TrainingConfigResponse>(
      keys.artifact(id, root, 'training-config'),
      withRoot(`/experiments/${id}/training-config`, root),
      enabled,
    ),
  );
}

export function useTrainingStatus(id: string, root: ArtifactRoot = undefined, enabled = true) {
  return useQuery({
    ...artifactQuery<TrainingStatusResponse>(
      keys.artifact(id, root, 'training-status'),
      withRoot(`/experiments/${id}/training-status`, root),
      enabled,
    ),
    refetchInterval: (query) => (isActive(query.state.data?.status) ? POLL_INTERVAL_MS : false),
  });
}

export function useComparison(id: string, root: ArtifactRoot = undefined, enabled = true) {
  return useQuery(
    artifactQuery<ComparisonReport>(keys.artifact(id, root, 'comparison'), withRoot(`/experiments/${id}/comparison`, root), enabled),
  );
}

export function useModelMetadata(id: string, name: string | null, root: ArtifactRoot = undefined) {
  return useQuery(
    artifactQuery<ModelMetadata>(
      keys.artifact(id, root, `model:${name ?? ''}`),
      withRoot(`/experiments/${id}/models/${name}`, root),
      Boolean(name),
    ),
  );
}

export function useDataDictionary(id: string) {
  return useQuery(
    artifactQuery<DataDictionary>(keys.artifact(id, undefined, 'data-dictionary'), `/experiments/${id}/data-dictionary`, true),
  );
}

export function useReasoning(id: string, enabled: boolean) {
  return useQuery(
    artifactQuery<ReasoningReport>(keys.artifact(id, undefined, 'reasoning'), `/experiments/${id}/reasoning`, enabled),
  );
}

/**
 * Proposed derived features. Unlike the other artifact queries this one does not poll: a
 * proposal set only appears because somebody asked for one, so waiting for it is pointless.
 */
export function useFeatureProposals(id: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.artifact(id, undefined, 'feature-proposals'),
    queryFn: () => api.get<FeatureProposalReport>(`/experiments/${id}/feature-proposals`),
    enabled,
    retry: retryUnlessMissing,
  });
}

export function useProposeFeatures(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: { target_definition?: string; prediction_timing?: string }) =>
      api.post<FeatureProposalReport>(`/experiments/${id}/feature-proposals`, body),
    onSuccess: (report) =>
      queryClient.setQueryData(keys.artifact(id, undefined, 'feature-proposals'), report),
  });
}

export function useApproveProposals(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (approved_names: string[]) =>
      api.put<FeatureProposalReport>(`/experiments/${id}/feature-proposals`, { approved_names }),
    onSuccess: (report) =>
      queryClient.setQueryData(keys.artifact(id, undefined, 'feature-proposals'), report),
  });
}

export function useCreateExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      name: string;
      dataset_uri: string;
      target_column: string;
      problem_type?: string;
    }) => api.post<{ experiment_id: string; status: ExperimentStatus }>('/experiments', body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });
}

export function useSaveFeatures(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: { selected_features: string[]; exclusion_reasons: Record<string, string> }) =>
      api.put(`/experiments/${id}/features`, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiment'] });
    },
  });
}

export function useStartTraining(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: TrainingRequest) => api.post(`/experiments/${id}/training`, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiment'] });
    },
  });
}

export function useUploadDictionary(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) =>
      api.upload<DataDictionary>(`/experiments/${id}/data-dictionary`, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiment'] });
    },
  });
}

export function useRunReasoning(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: { target_definition?: string; prediction_timing?: string; question?: string }) =>
      api.post<ReasoningReport>(`/experiments/${id}/reasoning`, body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.artifact(id, undefined, 'reasoning') }),
  });
}

export function useDeleteExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete(`/experiments/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });
}
