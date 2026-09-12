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
  DataDictionary,
  EdaReport,
  ExperimentRecord,
  ExperimentStatus,
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

export const keys = {
  health: ['health'] as const,
  models: ['models'] as const,
  experiments: ['experiments'] as const,
  experiment: (id: string) => ['experiment', id] as const,
  eda: (id: string) => ['experiment', id, 'eda'] as const,
  leakage: (id: string) => ['experiment', id, 'leakage'] as const,
  features: (id: string) => ['experiment', id, 'features'] as const,
  trainingConfig: (id: string) => ['experiment', id, 'training-config'] as const,
  trainingStatus: (id: string) => ['experiment', id, 'training-status'] as const,
  comparison: (id: string) => ['experiment', id, 'comparison'] as const,
  model: (id: string, name: string) => ['experiment', id, 'model', name] as const,
  dictionary: (id: string) => ['experiment', id, 'data-dictionary'] as const,
  reasoning: (id: string) => ['experiment', id, 'reasoning'] as const,
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

export function useExperiments() {
  return useQuery({
    queryKey: keys.experiments,
    queryFn: () => api.get<{ experiments: ExperimentRecord[]; count: number }>('/experiments'),
    refetchInterval: POLL_INTERVAL_MS * 2,
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: keys.experiment(id),
    queryFn: () => api.get<ExperimentRecord>(`/experiments/${id}`),
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

export function useEda(id: string, enabled = true) {
  return useQuery(artifactQuery<EdaReport>(keys.eda(id), `/experiments/${id}/eda`, enabled));
}

export function useLeakage(id: string, enabled = true) {
  return useQuery(
    artifactQuery<LeakageReport>(keys.leakage(id), `/experiments/${id}/leakage`, enabled),
  );
}

export function useFeatureReview(id: string, enabled = true) {
  return useQuery(
    artifactQuery<FeatureReviewResponse>(keys.features(id), `/experiments/${id}/features`, enabled),
  );
}

export function useTrainingConfig(id: string, enabled = true) {
  return useQuery(
    artifactQuery<TrainingConfigResponse>(
      keys.trainingConfig(id),
      `/experiments/${id}/training-config`,
      enabled,
    ),
  );
}

export function useTrainingStatus(id: string, enabled = true) {
  return useQuery({
    ...artifactQuery<TrainingStatusResponse>(
      keys.trainingStatus(id),
      `/experiments/${id}/training-status`,
      enabled,
    ),
    refetchInterval: (query) => (isActive(query.state.data?.status) ? POLL_INTERVAL_MS : false),
  });
}

export function useComparison(id: string, enabled = true) {
  return useQuery(
    artifactQuery<ComparisonReport>(keys.comparison(id), `/experiments/${id}/comparison`, enabled),
  );
}

export function useModelMetadata(id: string, name: string | null) {
  return useQuery(
    artifactQuery<ModelMetadata>(
      keys.model(id, name ?? ''),
      `/experiments/${id}/models/${name}`,
      Boolean(name),
    ),
  );
}

export function useDataDictionary(id: string) {
  return useQuery(
    artifactQuery<DataDictionary>(keys.dictionary(id), `/experiments/${id}/data-dictionary`, true),
  );
}

export function useReasoning(id: string, enabled: boolean) {
  return useQuery(
    artifactQuery<ReasoningReport>(keys.reasoning(id), `/experiments/${id}/reasoning`, enabled),
  );
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
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.experiments }),
  });
}

export function useSaveFeatures(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: { selected_features: string[]; exclusion_reasons: Record<string, string> }) =>
      api.put(`/experiments/${id}/features`, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.features(id) });
      queryClient.invalidateQueries({ queryKey: keys.experiment(id) });
    },
  });
}

export function useStartTraining(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: TrainingRequest) => api.post(`/experiments/${id}/training`, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.experiment(id) });
      queryClient.invalidateQueries({ queryKey: keys.trainingStatus(id) });
    },
  });
}

export function useUploadDictionary(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) =>
      api.upload<DataDictionary>(`/experiments/${id}/data-dictionary`, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: keys.dictionary(id) });
      queryClient.invalidateQueries({ queryKey: keys.features(id) });
    },
  });
}

export function useRunReasoning(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: { target_definition?: string; prediction_timing?: string; question?: string }) =>
      api.post<ReasoningReport>(`/experiments/${id}/reasoning`, body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.reasoning(id) }),
  });
}

export function useDeleteExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete(`/experiments/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.experiments }),
  });
}
