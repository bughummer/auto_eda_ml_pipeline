/** Training configuration and live per-model progress. */

import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  InputNumber,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  App as AntApp,
} from 'antd';

import { ApiError } from '../api/client';
import {
  useComparison,
  useExperiment,
  useModelCatalogue,
  useStartTraining,
  useTrainingConfig,
  useTrainingStatus,
} from '../api/hooks';
import type { ModelRunState, TrainingRequest } from '../api/types';
import { QueryState, StatusTag } from '../components/common';
import { MODEL_STATUS_COLORS, formatNumber, humanize } from '../lib/format';

export function TrainingTab({ experimentId, root }: { experimentId: string; root?: string }) {
  const { message } = AntApp.useApp();
  const experiment = useExperiment(experimentId, root);
  const config = useTrainingConfig(experimentId, root);
  const catalogue = useModelCatalogue();
  const status = useTrainingStatus(experimentId, root);
  const comparison = useComparison(experimentId, root);
  const isReadOnly = Boolean(root);
  const start = useStartTraining(experimentId);
  const [form] = Form.useForm<TrainingRequest>();

  const hasStarted = (status.data?.models.length ?? 0) > 0;
  const isRunning = ['PREPARING', 'TRAINING', 'EVALUATING'].includes(
    experiment.data?.status ?? '',
  );

  const onFinish = (values: TrainingRequest) => {
    start.mutate(values, {
      onSuccess: () => message.success('Training started'),
      onError: (error) =>
        message.error(error instanceof ApiError ? error.message : 'Could not start training'),
    });
  };

  return (
    <QueryState
      isLoading={config.isLoading}
      error={config.error}
      pendingMessage="Complete EDA before configuring training"
    >
      {config.data && (
        <Row gutter={16}>
          <Col span={10}>
            <Card title="Training configuration" size="small">
              <Descriptions size="small" column={1} style={{ marginBottom: 16 }}>
                <Descriptions.Item label="Problem type">
                  <Tag color="blue">{humanize(config.data.problem_type)}</Tag>
                  {config.data.requested_problem_type === 'auto' && (
                    <Typography.Text type="secondary">detected from the target</Typography.Text>
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="Selected features">
                  {config.data.selected_feature_count}
                </Descriptions.Item>
              </Descriptions>

              <Form<TrainingRequest>
                form={form}
                layout="vertical"
                onFinish={onFinish}
                initialValues={{
                  primary_metric: config.data.primary_metric,
                  models: config.data.selected_models,
                  validation_fraction: config.data.validation_fraction,
                  random_seed: config.data.random_seed,
                  class_weighting: config.data.class_weighting,
                }}
              >
                <Form.Item
                  name="models"
                  label="Models"
                  extra="Each model trains in its own job; one failure does not stop the others."
                  rules={[{ required: true, message: 'Choose at least one model' }]}
                >
                  <Select
                    mode="multiple"
                    options={config.data.available_models.map((name) => {
                      const descriptor = catalogue.data?.find((item) => item.name === name);
                      return {
                        value: name,
                        label: descriptor?.display_name ?? name,
                        title: descriptor?.description,
                      };
                    })}
                  />
                </Form.Item>

                <Form.Item
                  name="primary_metric"
                  label="Primary metric"
                  extra="Used to rank models. The optimization direction is fixed per metric."
                >
                  <Select options={config.data.available_metrics.map((m) => ({ value: m, label: m }))} />
                </Form.Item>

                <Form.Item name="validation_fraction" label="Validation fraction">
                  <InputNumber min={0.05} max={0.5} step={0.05} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item name="random_seed" label="Random seed" extra="Recorded for reproducibility.">
                  <InputNumber min={0} max={2 ** 31 - 1} style={{ width: '100%' }} />
                </Form.Item>

                <Form.Item
                  name="class_weighting"
                  label="Class weighting"
                  extra="Applied by every model that supports it. No resampling is performed."
                >
                  <Select
                    options={[
                      { value: 'auto', label: 'Automatic (balanced)' },
                      { value: 'none', label: 'None' },
                    ]}
                    disabled={config.data.problem_type === 'regression'}
                  />
                </Form.Item>

                {start.error && (
                  <Alert
                    type="error"
                    showIcon
                    style={{ marginBottom: 12 }}
                    message={(start.error as Error).message}
                  />
                )}

                <Space>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={start.isPending}
                    disabled={isRunning || isReadOnly}
                  >
                    {hasStarted ? 'Run again' : 'Start training'}
                  </Button>
                  {isRunning && <Typography.Text type="secondary">Training is running…</Typography.Text>}
                </Space>
              </Form>
            </Card>
          </Col>

          <Col span={14}>
            <Card
              title="Progress"
              size="small"
              extra={experiment.data && <StatusTag status={experiment.data.status} />}
            >
              {!hasStarted && <Typography.Text type="secondary">No training run yet.</Typography.Text>}
              {hasStarted && (
                <Table<ModelRunState>
                  size="small"
                  rowKey="model_name"
                  pagination={false}
                  dataSource={status.data?.models ?? []}
                  columns={[
                    { title: 'Model', dataIndex: 'display_name' },
                    {
                      title: 'Status',
                      dataIndex: 'status',
                      width: 130,
                      render: (value: ModelRunState['status']) => (
                        <Tag color={MODEL_STATUS_COLORS[value]}>{value}</Tag>
                      ),
                    },
                    {
                      title: status.data?.primary_metric ?? 'Score',
                      dataIndex: 'primary_score',
                      width: 130,
                      render: (value: number | null) => formatNumber(value),
                    },
                    {
                      title: 'Detail',
                      dataIndex: 'failure_message',
                      render: (value: string | null) =>
                        value ? <Typography.Text type="danger">{value}</Typography.Text> : '—',
                    },
                  ]}
                />
              )}
              {status.data?.failure_message && (
                <Alert
                  type="error"
                  showIcon
                  style={{ marginTop: 12 }}
                  message="The experiment failed"
                  description={status.data.failure_message}
                />
              )}
              {comparison.data?.warnings?.length ? (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginTop: 12 }}
                  message="Completed with warnings"
                  description={
                    <ul style={{ marginBottom: 0 }}>
                      {comparison.data.warnings.map((warning) => (
                        <li key={warning.rule}>{warning.message}</li>
                      ))}
                    </ul>
                  }
                />
              ) : null}
            </Card>
          </Col>
        </Row>
      )}
    </QueryState>
  );
}
