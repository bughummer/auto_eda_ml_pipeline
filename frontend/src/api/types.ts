/**
 * Mirrors the Pydantic contracts in `ml_engine/contracts`.
 *
 * These types are the frontend's half of the API contract: when a contract changes, this
 * file changes in the same commit. Run `make openapi` to regenerate the reference document.
 */

export type ExperimentStatus =
  | 'CREATED'
  | 'EDA_RUNNING'
  | 'EDA_COMPLETED'
  | 'FEATURE_REVIEW'
  | 'READY_FOR_TRAINING'
  | 'PREPARING'
  | 'TRAINING'
  | 'EVALUATING'
  | 'COMPLETED'
  | 'COMPLETED_WITH_WARNINGS'
  | 'FAILED';

export type ModelRunStatus = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'SKIPPED';
export type ProblemType = 'binary_classification' | 'multiclass_classification' | 'regression';
export type RequestedProblemType = 'auto' | ProblemType;
export type Severity = 'info' | 'low' | 'medium' | 'high' | 'critical';
export type LeakageRiskLevel =
  | 'none'
  | 'requires_review'
  | 'potential_leakage'
  | 'confirmed_duplicate';
export type RecommendedAction =
  | 'keep'
  | 'review'
  | 'consider_excluding'
  | 'strongly_consider_excluding'
  | 'excluded_automatically';
export type WarningCategory =
  | 'data_quality'
  | 'leakage'
  | 'anomaly'
  | 'schema'
  | 'target'
  | 'feature'
  | 'preprocessing'
  | 'metric'
  | 'model';
export type SemanticType =
  | 'numeric_continuous'
  | 'numeric_discrete'
  | 'boolean'
  | 'categorical'
  | 'high_cardinality_categorical'
  | 'text'
  | 'datetime'
  | 'identifier'
  | 'constant'
  | 'empty'
  | 'unknown';

export interface AnalysisWarning {
  rule: string;
  category: WarningCategory;
  severity: Severity;
  message: string;
  column: string | null;
  recommended_action: RecommendedAction;
  details: Record<string, string | number | boolean | null>;
}

export interface DatasetReference {
  uri: string;
  file_format: string;
  version_id: string | null;
  etag: string | null;
  size_bytes: number | null;
  last_modified: string | null;
}

export interface ExperimentRecord {
  experiment_id: string;
  name: string;
  created_by: string | null;
  created_at: string;
  updated_at: string;
  dataset: DatasetReference;
  target_column: string;
  requested_problem_type: RequestedProblemType;
  status: ExperimentStatus;
  current_stage: string;
  artifact_prefix: string;
  best_model: string | null;
  best_score: number | null;
  primary_metric: string | null;
  failure_code: string | null;
  failure_message: string | null;
  model_statuses: Record<string, ModelRunStatus>;
}

export interface NumericStats {
  min: number | null;
  max: number | null;
  mean: number | null;
  median: number | null;
  std: number | null;
  q01: number | null;
  q05: number | null;
  q25: number | null;
  q75: number | null;
  q95: number | null;
  q99: number | null;
  zero_count: number;
  zero_percentage: number;
  negative_count: number;
  skewness: number | null;
}

export interface CategoryFrequency {
  value: string;
  count: number;
  percentage: number;
}

export interface ColumnProfile {
  name: string;
  semantic_type: SemanticType;
  dtype: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  is_constant: boolean;
  is_high_cardinality: boolean;
  is_likely_id: boolean;
  is_text_like: boolean;
  numeric: NumericStats | null;
  categorical: { top_values: CategoryFrequency[]; truncated: boolean } | null;
  datetime_stats: { min: string | null; max: string | null; range_days: number | null } | null;
  sample_values: string[];
}

export interface EdaReport {
  experiment_id: string;
  generated_at: string;
  duration_seconds: number;
  dataset: {
    row_count: number;
    column_count: number;
    duplicate_row_count: number;
    duplicate_row_percentage: number;
    memory_usage_bytes: number;
    file_format: string;
    source_uri: string;
    sampled: boolean;
    sample_row_count: number | null;
  };
  target: {
    column: string;
    exists: boolean;
    inferred_problem_type: ProblemType | null;
    missing_count: number;
    missing_percentage: number;
    unique_count: number;
    class_distribution: { label: string; count: number; percentage: number }[] | null;
    imbalance_ratio: number | null;
    positive_class: string | null;
    numeric: NumericStats | null;
  };
  columns: ColumnProfile[];
  warnings: AnalysisWarning[];
}

export interface FeatureReviewItem {
  feature: string;
  selected: boolean;
  semantic_type: SemanticType;
  dtype: string;
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  is_constant: boolean;
  is_high_cardinality: boolean;
  is_likely_id: boolean;
  leakage_risk: LeakageRiskLevel;
  max_severity: Severity | null;
  recommended_action: RecommendedAction;
  reasons: string[];
  rules: string[];
  warnings: AnalysisWarning[];
  documentation: string | null;
  available_at_prediction_time: boolean | null;
}

export interface FeatureReviewResponse {
  experiment_id: string;
  target_column: string;
  problem_type: ProblemType | null;
  features: FeatureReviewItem[];
  selected_count: number;
  excluded_count: number;
  decided: boolean;
}

export interface TrainingConfigResponse {
  experiment_id: string;
  problem_type: ProblemType;
  requested_problem_type: RequestedProblemType;
  primary_metric: string;
  available_metrics: string[];
  available_models: string[];
  selected_models: string[];
  validation_fraction: number;
  random_seed: number;
  class_weighting: 'auto' | 'none';
  selected_feature_count: number;
}

export interface TrainingRequest {
  problem_type?: RequestedProblemType;
  primary_metric?: string | null;
  models?: string[];
  validation_fraction?: number;
  random_seed?: number;
  class_weighting?: 'auto' | 'none';
}

export interface ModelRunState {
  model_name: string;
  display_name: string;
  status: ModelRunStatus;
  primary_score: number | null;
  failure_message: string | null;
}

export interface TrainingStatusResponse {
  experiment_id: string;
  status: ExperimentStatus;
  current_stage: string;
  models: ModelRunState[];
  best_model: string | null;
  best_score: number | null;
  primary_metric: string | null;
  failure_message: string | null;
  updated_at: string;
}

export interface ModelComparisonEntry {
  model_name: string;
  display_name: string;
  status: ModelRunStatus;
  primary_score: number | null;
  metrics: Record<string, number>;
  training_duration_seconds: number | null;
  feature_count: number | null;
  rank: number | null;
  warning_count: number;
  failure_message: string | null;
}

export interface ComparisonReport {
  experiment_id: string;
  generated_at: string;
  problem_type: ProblemType;
  primary_metric: string;
  direction: 'maximize' | 'minimize';
  best_model: string | null;
  best_score: number | null;
  models: ModelComparisonEntry[];
  succeeded_count: number;
  failed_count: number;
  warnings: AnalysisWarning[];
}

export interface FeatureImportanceEntry {
  feature: string;
  importance: number;
  raw_value: number;
  rank: number;
}

export interface ModelMetadata {
  experiment_id: string;
  model_name: string;
  display_name: string;
  status: ModelRunStatus;
  problem_type: ProblemType;
  library: string;
  library_version: string;
  hyperparameters: Record<string, string | number | boolean | null>;
  features_used: string[];
  feature_count: number;
  training_row_count: number;
  validation_row_count: number;
  training_duration_seconds: number | null;
  primary_metric: string | null;
  primary_score: number | null;
  metrics: {
    values: Record<string, number>;
    confusion_matrix: { labels: string[]; matrix: number[][] } | null;
    warnings: AnalysisWarning[];
  };
  train_metrics: { values: Record<string, number> } | null;
  feature_importance: {
    method: string;
    is_signed: boolean;
    entries: FeatureImportanceEntry[];
    note: string | null;
  } | null;
  preprocessing: {
    numeric_columns: string[];
    categorical_columns: string[];
    boolean_columns: string[];
    datetime_columns: string[];
    dropped_columns: Record<string, string>;
    output_feature_count: number;
    strategy: string;
  } | null;
  artifacts: { model_uri: string | null; metadata_uri: string | null };
  package_versions: Record<string, string>;
  warnings: AnalysisWarning[];
}

export interface LeakageFinding {
  feature: string;
  rule: string;
  severity: Severity;
  risk_level: LeakageRiskLevel;
  explanation: string;
  recommended_action: RecommendedAction;
}

export interface LeakageReport {
  experiment_id: string;
  target_column: string;
  findings: LeakageFinding[];
  checks_executed: string[];
  checks_skipped: Record<string, string>;
}

export interface ModelDescriptor {
  name: string;
  display_name: string;
  library: string;
  library_version: string | null;
  supported_problem_types: ProblemType[];
  supports_native_categorical: boolean;
  supports_class_weighting: boolean;
  available: boolean;
  unavailable_reason: string | null;
  description: string;
}

export interface DataDictionary {
  source_format: string;
  source_name: string | null;
  uploaded_at: string;
  columns: {
    column: string;
    business_definition: string | null;
    source_system: string | null;
    collection_timing: string | null;
    update_frequency: string | null;
    available_at_prediction_time: boolean | null;
    owner: string | null;
  }[];
  unmatched_columns: string[];
  undocumented_columns: string[];
}

export interface ReasoningReport {
  experiment_id: string;
  generated_at: string;
  model_id: string;
  input_artifacts: string[];
  executive_summary: string;
  semantic_leakage: {
    feature: string;
    severity: Severity;
    confidence: string;
    reasoning: string;
    recommended_action: RecommendedAction;
    evidence_basis: string[];
  }[];
  feature_interpretations: { feature: string; interpretation: string; concern: string | null }[];
  data_quality_concerns: {
    topic: string;
    severity: Severity;
    explanation: string;
    affected_columns: string[];
  }[];
  model_behaviour_narrative: string | null;
  assumptions: string[];
  limitations: string[];
  proposed_experiments: {
    title: string;
    hypothesis: string;
    proposed_changes: string[];
    expected_insight: string;
    priority: string;
  }[];
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  mode: string;
  orchestrator: string;
  artifact_root: string;
  bedrock_enabled: boolean;
  configuration_problems: string[];
  available_models: string[];
}

export interface ApiErrorBody {
  error: { code: string; message: string; details: Record<string, unknown>; request_id: string | null };
}
