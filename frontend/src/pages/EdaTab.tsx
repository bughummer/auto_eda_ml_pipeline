/** Automated EDA: dataset summary, target analysis, warnings and the column profile table. */

import { Alert, Card, Col, Descriptions, Row, Statistic, Table, Tabs, Tag, Typography } from 'antd';

import { useEda } from '../api/hooks';
import type { ColumnProfile } from '../api/types';
import { QueryState, WarningsTable } from '../components/common';
import { formatBytes, formatNumber, formatPercent, humanize } from '../lib/format';

export function EdaTab({ experimentId, root }: { experimentId: string; root?: string }) {
  const { data, isLoading, error } = useEda(experimentId, root);

  return (
    <QueryState isLoading={isLoading} error={error} pendingMessage="EDA has not finished yet">
      {data && (
        <>
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Rows" value={data.dataset.row_count} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="Columns" value={data.dataset.column_count} />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic
                  title="Duplicate rows"
                  value={data.dataset.duplicate_row_count}
                  suffix={`(${formatPercent(data.dataset.duplicate_row_percentage)})`}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card size="small">
                <Statistic title="In memory" value={formatBytes(data.dataset.memory_usage_bytes)} />
              </Card>
            </Col>
          </Row>

          {data.dataset.sampled && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message={`Profiled a sample of ${data.dataset.sample_row_count} rows`}
              description="Statistics describe the sample, not the complete dataset."
            />
          )}

          <Tabs
            defaultActiveKey="target"
            items={[
              {
                key: 'target',
                label: 'Target',
                children: <TargetPanel eda={data} />,
              },
              {
                key: 'warnings',
                label: `Warnings (${data.warnings.length})`,
                children: <WarningsTable warnings={data.warnings} />,
              },
              {
                key: 'columns',
                label: `Columns (${data.columns.length})`,
                children: <ColumnTable columns={data.columns} targetColumn={data.target.column} />,
              },
            ]}
          />
        </>
      )}
    </QueryState>
  );
}

function TargetPanel({ eda }: { eda: NonNullable<ReturnType<typeof useEda>['data']> }) {
  const { target } = eda;
  if (!target.exists) {
    return (
      <Alert
        type="error"
        showIcon
        message={`Target column '${target.column}' is not present in the dataset`}
      />
    );
  }
  return (
    <>
      <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
        <Descriptions.Item label="Column">{target.column}</Descriptions.Item>
        <Descriptions.Item label="Detected problem type">
          {target.inferred_problem_type ? (
            <Tag color="blue">{humanize(target.inferred_problem_type)}</Tag>
          ) : (
            <Tag color="red">Not learnable as it stands</Tag>
          )}
        </Descriptions.Item>
        <Descriptions.Item label="Distinct values">{target.unique_count}</Descriptions.Item>
        <Descriptions.Item label="Missing">
          {target.missing_count} ({formatPercent(target.missing_percentage)})
        </Descriptions.Item>
        {target.imbalance_ratio !== null && (
          <Descriptions.Item label="Imbalance ratio">
            {formatNumber(target.imbalance_ratio, 2)} : 1
          </Descriptions.Item>
        )}
        {target.positive_class && (
          <Descriptions.Item label="Positive class">{target.positive_class}</Descriptions.Item>
        )}
      </Descriptions>

      {target.class_distribution && (
        <Table
          size="small"
          rowKey="label"
          pagination={false}
          dataSource={target.class_distribution}
          columns={[
            { title: 'Class', dataIndex: 'label' },
            { title: 'Rows', dataIndex: 'count', width: 140 },
            {
              title: 'Share',
              dataIndex: 'percentage',
              width: 160,
              render: (value: number) => formatPercent(value),
            },
          ]}
        />
      )}

      {target.numeric && (
        <Descriptions bordered size="small" column={4} title="Target distribution">
          <Descriptions.Item label="min">{formatNumber(target.numeric.min)}</Descriptions.Item>
          <Descriptions.Item label="q25">{formatNumber(target.numeric.q25)}</Descriptions.Item>
          <Descriptions.Item label="median">{formatNumber(target.numeric.median)}</Descriptions.Item>
          <Descriptions.Item label="mean">{formatNumber(target.numeric.mean)}</Descriptions.Item>
          <Descriptions.Item label="q75">{formatNumber(target.numeric.q75)}</Descriptions.Item>
          <Descriptions.Item label="max">{formatNumber(target.numeric.max)}</Descriptions.Item>
          <Descriptions.Item label="std">{formatNumber(target.numeric.std)}</Descriptions.Item>
          <Descriptions.Item label="zeros">{target.numeric.zero_count}</Descriptions.Item>
        </Descriptions>
      )}
    </>
  );
}

function ColumnTable({
  columns,
  targetColumn,
}: {
  columns: ColumnProfile[];
  targetColumn: string;
}) {
  return (
    <Table<ColumnProfile>
      size="small"
      rowKey="name"
      dataSource={columns}
      pagination={{ pageSize: 25, hideOnSinglePage: true }}
      expandable={{
        expandedRowRender: (record) => <ColumnDetail profile={record} />,
      }}
      columns={[
        {
          title: 'Column',
          dataIndex: 'name',
          render: (name: string) => (
            <>
              <Typography.Text strong>{name}</Typography.Text>
              {name === targetColumn && (
                <Tag color="blue" style={{ marginLeft: 8 }}>
                  target
                </Tag>
              )}
            </>
          ),
          sorter: (a, b) => a.name.localeCompare(b.name),
        },
        {
          title: 'Type',
          dataIndex: 'semantic_type',
          width: 200,
          render: (value: string) => <Tag>{humanize(value)}</Tag>,
        },
        { title: 'dtype', dataIndex: 'dtype', width: 110 },
        {
          title: 'Missing',
          dataIndex: 'missing_percentage',
          width: 120,
          render: (value: number) => formatPercent(value),
          sorter: (a, b) => a.missing_percentage - b.missing_percentage,
        },
        {
          title: 'Distinct',
          dataIndex: 'unique_count',
          width: 120,
          sorter: (a, b) => a.unique_count - b.unique_count,
        },
        {
          title: 'Flags',
          width: 240,
          render: (_value, record) => (
            <>
              {record.is_constant && <Tag color="orange">constant</Tag>}
              {record.is_likely_id && <Tag color="volcano">identifier</Tag>}
              {record.is_high_cardinality && <Tag color="gold">high cardinality</Tag>}
              {record.is_text_like && <Tag>text</Tag>}
            </>
          ),
        },
      ]}
    />
  );
}

function ColumnDetail({ profile }: { profile: ColumnProfile }) {
  if (profile.numeric) {
    const numeric = profile.numeric;
    return (
      <Descriptions size="small" column={6} bordered>
        <Descriptions.Item label="min">{formatNumber(numeric.min)}</Descriptions.Item>
        <Descriptions.Item label="q01">{formatNumber(numeric.q01)}</Descriptions.Item>
        <Descriptions.Item label="q25">{formatNumber(numeric.q25)}</Descriptions.Item>
        <Descriptions.Item label="median">{formatNumber(numeric.median)}</Descriptions.Item>
        <Descriptions.Item label="q75">{formatNumber(numeric.q75)}</Descriptions.Item>
        <Descriptions.Item label="q99">{formatNumber(numeric.q99)}</Descriptions.Item>
        <Descriptions.Item label="mean">{formatNumber(numeric.mean)}</Descriptions.Item>
        <Descriptions.Item label="std">{formatNumber(numeric.std)}</Descriptions.Item>
        <Descriptions.Item label="max">{formatNumber(numeric.max)}</Descriptions.Item>
        <Descriptions.Item label="zeros">
          {numeric.zero_count} ({formatPercent(numeric.zero_percentage)})
        </Descriptions.Item>
        <Descriptions.Item label="negatives">{numeric.negative_count}</Descriptions.Item>
        <Descriptions.Item label="skew">{formatNumber(numeric.skewness, 3)}</Descriptions.Item>
      </Descriptions>
    );
  }
  if (profile.categorical) {
    return (
      <Table
        size="small"
        rowKey="value"
        pagination={false}
        dataSource={profile.categorical.top_values}
        columns={[
          { title: 'Value', dataIndex: 'value' },
          { title: 'Rows', dataIndex: 'count', width: 120 },
          {
            title: 'Share',
            dataIndex: 'percentage',
            width: 140,
            render: (value: number) => formatPercent(value),
          },
        ]}
        footer={
          profile.categorical.truncated
            ? () => 'Only the most frequent values are shown.'
            : undefined
        }
      />
    );
  }
  if (profile.datetime_stats) {
    return (
      <Descriptions size="small" column={3} bordered>
        <Descriptions.Item label="earliest">{profile.datetime_stats.min ?? '—'}</Descriptions.Item>
        <Descriptions.Item label="latest">{profile.datetime_stats.max ?? '—'}</Descriptions.Item>
        <Descriptions.Item label="range (days)">
          {formatNumber(profile.datetime_stats.range_days, 1)}
        </Descriptions.Item>
      </Descriptions>
    );
  }
  return <Typography.Text type="secondary">Sample: {profile.sample_values.join(', ')}</Typography.Text>;
}
