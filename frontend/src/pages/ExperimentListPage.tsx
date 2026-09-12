/** Experiment list: the entry point. Status comes from the backend, never inferred here. */

import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';
import { Button, Card, Popconfirm, Space, Table, Typography } from 'antd';
import { Link, useNavigate } from 'react-router-dom';

import { useDeleteExperiment, useExperiments } from '../api/hooks';
import type { ExperimentRecord } from '../api/types';
import { QueryState, StatusTag } from '../components/common';
import { formatDateTime, formatNumber } from '../lib/format';

export function ExperimentListPage() {
  const navigate = useNavigate();
  const { data, isLoading, error, refetch, isFetching } = useExperiments();
  const remove = useDeleteExperiment();

  return (
    <Card
      title="Experiments"
      extra={
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => refetch()} loading={isFetching}>
            Refresh
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>
            New experiment
          </Button>
        </Space>
      }
    >
      <QueryState isLoading={isLoading} error={error}>
        <Table<ExperimentRecord>
          rowKey="experiment_id"
          dataSource={data?.experiments ?? []}
          pagination={{ pageSize: 20, hideOnSinglePage: true }}
          columns={[
            {
              title: 'Name',
              dataIndex: 'name',
              render: (name: string, record) => (
                <Link to={`/experiments/${record.experiment_id}`}>{name}</Link>
              ),
            },
            { title: 'Target', dataIndex: 'target_column', width: 160 },
            {
              title: 'Dataset',
              dataIndex: ['dataset', 'uri'],
              ellipsis: true,
              render: (uri: string) => <Typography.Text code>{uri}</Typography.Text>,
            },
            {
              title: 'Status',
              dataIndex: 'status',
              width: 220,
              render: (_value, record) => <StatusTag status={record.status} />,
            },
            {
              title: 'Best model',
              dataIndex: 'best_model',
              width: 200,
              render: (model: string | null, record) =>
                model ? `${model} (${record.primary_metric}: ${formatNumber(record.best_score)})` : '—',
            },
            {
              title: 'Created',
              dataIndex: 'created_at',
              width: 180,
              render: (value: string) => formatDateTime(value),
              defaultSortOrder: 'descend',
              sorter: (a, b) => a.created_at.localeCompare(b.created_at),
            },
            {
              title: '',
              width: 90,
              render: (_value, record) => (
                <Popconfirm
                  title="Remove this experiment from the list?"
                  description="The artifacts in object storage are kept for audit."
                  onConfirm={() => remove.mutate(record.experiment_id)}
                >
                  <Button size="small" danger type="text">
                    Remove
                  </Button>
                </Popconfirm>
              ),
            },
          ]}
        />
      </QueryState>
    </Card>
  );
}
