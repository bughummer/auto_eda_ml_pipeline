/**
 * Experiment list: the entry point.
 *
 * Experiments are read from an artifact bucket, so picking a different approved bucket shows
 * the experiments that were run against it — including ones this instance never started.
 * Status always comes from the backend; nothing is inferred here.
 */

import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';
import { Button, Card, Popconfirm, Select, Space, Table, Tooltip, Typography } from 'antd';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';

import { useDeleteExperiment, useExperiments, useHealth } from '../api/hooks';
import type { ExperimentRecord } from '../api/types';
import { QueryState, StatusTag } from '../components/common';
import { formatDateTime, formatNumber } from '../lib/format';

export function ExperimentListPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { data: health } = useHealth();
  // The selected bucket lives in the URL, so a link to a browsed result is shareable.
  const selectedRoot = searchParams.get('root') ?? undefined;
  const { data, isLoading, error, refetch, isFetching } = useExperiments(selectedRoot);

  const roots = data?.available_roots ?? health?.artifact_roots ?? [];
  const activeRoot = selectedRoot ?? data?.root ?? health?.artifact_root ?? '';
  const isBrowsingAnotherBucket = Boolean(
    selectedRoot && health?.artifact_root && selectedRoot !== health.artifact_root,
  );
  const remove = useDeleteExperiment();

  return (
    <Card
      title="Experiments"
      extra={
        <Space>
          {/* One bucket is the normal case, and a one-item dropdown is just noise. The
              selector appears only when another artifact bucket has been configured. */}
          {roots.length > 1 && (
            <Tooltip title="Experiment results are read from this artifact bucket.">
              <Select
                value={activeRoot || undefined}
                style={{ minWidth: 320 }}
                placeholder="Artifact bucket"
                options={roots.map((root) => ({
                  value: root,
                  label: root === health?.artifact_root ? `${root} (this platform)` : root,
                }))}
                onChange={(root) =>
                  setSearchParams(root === health?.artifact_root ? {} : { root }, { replace: true })
                }
              />
            </Tooltip>
          )}
          <Button icon={<ReloadOutlined />} onClick={() => refetch()} loading={isFetching}>
            Refresh
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>
            New experiment
          </Button>
        </Space>
      }
    >
      {isBrowsingAnotherBucket && (
        <Typography.Paragraph type="secondary">
          Showing experiments stored in <Typography.Text code>{activeRoot}</Typography.Text>. This
          is a read-only view of another environment's results; new experiments are always
          written to this platform's own bucket.
        </Typography.Paragraph>
      )}
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
                <Link
                  to={{
                    pathname: `/experiments/${record.experiment_id}`,
                    search: selectedRoot ? `?root=${encodeURIComponent(selectedRoot)}` : '',
                  }}
                >
                  {name}
                </Link>
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
              render: (_value, record) =>
                isBrowsingAnotherBucket ? null : (
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
