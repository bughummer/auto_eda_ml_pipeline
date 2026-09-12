/** Model comparison, with the per-model detail drawer. */

import { Alert, Card, Descriptions, Drawer, Table, Tag, Typography } from 'antd';
import { useState } from 'react';

import { useComparison, useModelMetadata } from '../api/hooks';
import type { ModelComparisonEntry } from '../api/types';
import { QueryState } from '../components/common';
import { MODEL_STATUS_COLORS, formatDuration, formatNumber, humanize } from '../lib/format';

export function ComparisonTab({ experimentId }: { experimentId: string }) {
  const { data, isLoading, error } = useComparison(experimentId);
  const [openModel, setOpenModel] = useState<string | null>(null);

  return (
    <QueryState isLoading={isLoading} error={error} pendingMessage="Training has not finished yet">
      {data && (
        <>
          <Descriptions bordered size="small" column={4} style={{ marginBottom: 16 }}>
            <Descriptions.Item label="Primary metric">{data.primary_metric}</Descriptions.Item>
            <Descriptions.Item label="Direction">{data.direction}</Descriptions.Item>
            <Descriptions.Item label="Best model">{data.best_model ?? '—'}</Descriptions.Item>
            <Descriptions.Item label="Best score">{formatNumber(data.best_score)}</Descriptions.Item>
          </Descriptions>

          {data.failed_count > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message={`${data.failed_count} model(s) failed; the remaining results are still valid`}
            />
          )}

          <Table<ModelComparisonEntry>
            size="small"
            rowKey="model_name"
            pagination={false}
            dataSource={data.models}
            onRow={(record) => ({
              onClick: () => record.status === 'COMPLETED' && setOpenModel(record.model_name),
              style: { cursor: record.status === 'COMPLETED' ? 'pointer' : 'default' },
            })}
            rowClassName={(record) => (record.model_name === data.best_model ? 'best-model-row' : '')}
            columns={[
              { title: '#', dataIndex: 'rank', width: 60, render: (value: number | null) => value ?? '—' },
              {
                title: 'Model',
                dataIndex: 'display_name',
                render: (value: string, record) => (
                  <>
                    <Typography.Text strong={record.model_name === data.best_model}>{value}</Typography.Text>
                    {record.model_name === data.best_model && (
                      <Tag color="green" style={{ marginLeft: 8 }}>
                        best
                      </Tag>
                    )}
                  </>
                ),
              },
              {
                title: 'Status',
                dataIndex: 'status',
                width: 120,
                render: (value: ModelComparisonEntry['status']) => (
                  <Tag color={MODEL_STATUS_COLORS[value]}>{value}</Tag>
                ),
              },
              {
                title: data.primary_metric,
                dataIndex: 'primary_score',
                width: 130,
                render: (value: number | null) => formatNumber(value),
              },
              {
                title: 'Other metrics',
                dataIndex: 'metrics',
                render: (metrics: Record<string, number>, record) => (
                  <>
                    {Object.entries(metrics)
                      .filter(([name]) => name !== data.primary_metric)
                      .slice(0, 4)
                      .map(([name, value]) => (
                        <Tag key={`${record.model_name}-${name}`}>
                          {name}: {formatNumber(value, 3)}
                        </Tag>
                      ))}
                  </>
                ),
              },
              {
                title: 'Features',
                dataIndex: 'feature_count',
                width: 100,
                render: (value: number | null) => value ?? '—',
              },
              {
                title: 'Time',
                dataIndex: 'training_duration_seconds',
                width: 100,
                render: (value: number | null) => formatDuration(value),
              },
              {
                title: 'Warnings',
                dataIndex: 'warning_count',
                width: 100,
                render: (value: number, record) =>
                  record.failure_message ? (
                    <Typography.Text type="danger">failed</Typography.Text>
                  ) : (
                    value
                  ),
              },
            ]}
          />

          <Drawer
            width={760}
            open={Boolean(openModel)}
            onClose={() => setOpenModel(null)}
            title={openModel ? `Model detail — ${openModel}` : ''}
            destroyOnClose
          >
            {openModel && <ModelDetail experimentId={experimentId} modelName={openModel} />}
          </Drawer>
        </>
      )}
    </QueryState>
  );
}

function ModelDetail({ experimentId, modelName }: { experimentId: string; modelName: string }) {
  const { data, isLoading, error } = useModelMetadata(experimentId, modelName);

  return (
    <QueryState isLoading={isLoading} error={error}>
      {data && (
        <>
          <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
            <Descriptions.Item label="Library">
              {data.library} {data.library_version}
            </Descriptions.Item>
            <Descriptions.Item label="Problem type">{humanize(data.problem_type)}</Descriptions.Item>
            <Descriptions.Item label="Training rows">{data.training_row_count}</Descriptions.Item>
            <Descriptions.Item label="Validation rows">{data.validation_row_count}</Descriptions.Item>
            <Descriptions.Item label="Features after preprocessing">{data.feature_count}</Descriptions.Item>
            <Descriptions.Item label="Training time">
              {formatDuration(data.training_duration_seconds)}
            </Descriptions.Item>
          </Descriptions>

          <Card size="small" title="Validation metrics" style={{ marginBottom: 16 }}>
            <Table
              size="small"
              pagination={false}
              rowKey="metric"
              dataSource={Object.entries(data.metrics.values).map(([metric, value]) => ({
                metric,
                value,
                train: data.train_metrics?.values?.[metric] ?? null,
              }))}
              columns={[
                { title: 'Metric', dataIndex: 'metric' },
                {
                  title: 'Validation',
                  dataIndex: 'value',
                  render: (value: number) => formatNumber(value),
                },
                {
                  title: 'Training',
                  dataIndex: 'train',
                  render: (value: number | null) => formatNumber(value),
                },
              ]}
            />
          </Card>

          {data.metrics.confusion_matrix && (
            <Card size="small" title="Confusion matrix (rows = actual)" style={{ marginBottom: 16 }}>
              <Table
                size="small"
                pagination={false}
                rowKey="label"
                dataSource={data.metrics.confusion_matrix.matrix.map((row, index) => ({
                  label: data.metrics.confusion_matrix!.labels[index],
                  row,
                }))}
                columns={[
                  { title: 'actual \\ predicted', dataIndex: 'label' },
                  ...data.metrics.confusion_matrix.labels.map((label, index) => ({
                    title: label,
                    key: label,
                    render: (_value: unknown, record: { row: number[] }) => record.row[index],
                  })),
                ]}
              />
            </Card>
          )}

          {data.feature_importance && (
            <Card
              size="small"
              title={`Feature importance (${data.feature_importance.method})`}
              style={{ marginBottom: 16 }}
            >
              <Table
                size="small"
                rowKey="feature"
                pagination={{ pageSize: 10, hideOnSinglePage: true }}
                dataSource={data.feature_importance.entries}
                columns={[
                  { title: '#', dataIndex: 'rank', width: 60 },
                  { title: 'Feature', dataIndex: 'feature' },
                  {
                    title: 'Share',
                    dataIndex: 'importance',
                    width: 120,
                    render: (value: number) => `${(value * 100).toFixed(2)}%`,
                  },
                  {
                    title: 'Raw',
                    dataIndex: 'raw_value',
                    width: 120,
                    render: (value: number) => formatNumber(value, 4),
                  },
                ]}
              />
              {data.feature_importance.note && (
                <Typography.Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                  {data.feature_importance.note}
                </Typography.Paragraph>
              )}
            </Card>
          )}

          <Card size="small" title="Hyperparameters and environment">
            <Descriptions size="small" column={2} bordered>
              {Object.entries(data.hyperparameters).map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>
                  {String(value)}
                </Descriptions.Item>
              ))}
              {Object.entries(data.package_versions).map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>
                  {value}
                </Descriptions.Item>
              ))}
            </Descriptions>
          </Card>

          {data.warnings.length > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginTop: 16 }}
              message="Model warnings"
              description={
                <ul style={{ marginBottom: 0 }}>
                  {data.warnings.map((warning, index) => (
                    <li key={`${warning.rule}-${index}`}>{warning.message}</li>
                  ))}
                </ul>
              }
            />
          )}
        </>
      )}
    </QueryState>
  );
}
