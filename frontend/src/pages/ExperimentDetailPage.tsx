/** One experiment: status header plus the workflow tabs, in the order the work happens. */

import { ArrowLeftOutlined } from '@ant-design/icons';
import { Alert, Button, Card, Descriptions, Space, Tabs, Typography } from 'antd';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';

import { useExperiment } from '../api/hooks';
import { QueryState, StatusTag } from '../components/common';
import { formatDateTime, formatNumber } from '../lib/format';
import { ComparisonTab } from './ComparisonTab';
import { EdaTab } from './EdaTab';
import { FeatureReviewTab } from './FeatureReviewTab';
import { ReasoningTab } from './ReasoningTab';
import { TrainingTab } from './TrainingTab';

export function ExperimentDetailPage() {
  const { experimentId = '' } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  // Present when the user opened this from another artifact bucket; everything below reads
  // from that bucket instead of this platform's own.
  const root = searchParams.get('root') ?? undefined;
  const { data, isLoading, error } = useExperiment(experimentId, root);
  const isReadOnly = Boolean(root);

  return (
    <QueryState isLoading={isLoading} error={error}>
      {data && (
        <>
          <Card size="small" style={{ marginBottom: 16 }}>
            <Space align="start" style={{ width: '100%', justifyContent: 'space-between' }}>
              <Space direction="vertical" size={4}>
                <Space>
                  <Button
                    icon={<ArrowLeftOutlined />}
                    size="small"
                    onClick={() =>
                      navigate(root ? `/?root=${encodeURIComponent(root)}` : '/')
                    }
                  >
                    Experiments
                  </Button>
                  <Typography.Title level={4} style={{ margin: 0 }}>
                    {data.name}
                  </Typography.Title>
                  <StatusTag status={data.status} />
                </Space>
                <Descriptions size="small" column={4}>
                  <Descriptions.Item label="Target">{data.target_column}</Descriptions.Item>
                  <Descriptions.Item label="Dataset">
                    <Typography.Text code>{data.dataset.uri}</Typography.Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="Stage">{data.current_stage}</Descriptions.Item>
                  <Descriptions.Item label="Updated">{formatDateTime(data.updated_at)}</Descriptions.Item>
                  {data.best_model && (
                    <Descriptions.Item label="Best model">
                      {data.best_model} ({data.primary_metric}: {formatNumber(data.best_score)})
                    </Descriptions.Item>
                  )}
                </Descriptions>
              </Space>
            </Space>
          </Card>

          {isReadOnly && (
            <Alert
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
              message="Viewing an experiment from another artifact bucket"
              description="Its results are shown read-only. Feature selection and training act on this platform's own bucket."
            />
          )}

          {data.status === 'FAILED' && (
            <Alert
              type="error"
              showIcon
              style={{ marginBottom: 16 }}
              message={`Experiment failed${data.failure_code ? ` (${data.failure_code})` : ''}`}
              description={data.failure_message}
            />
          )}

          <Tabs
            defaultActiveKey="eda"
            items={[
              {
                key: 'eda',
                label: 'EDA',
                children: <EdaTab experimentId={experimentId} root={root} />,
              },
              {
                key: 'features',
                label: 'Feature review',
                children: <FeatureReviewTab experimentId={experimentId} root={root} />,
              },
              {
                key: 'training',
                label: 'Training',
                children: <TrainingTab experimentId={experimentId} root={root} />,
              },
              {
                key: 'comparison',
                label: 'Comparison',
                children: <ComparisonTab experimentId={experimentId} root={root} />,
              },
              {
                key: 'reasoning',
                label: 'Semantic analysis',
                children: <ReasoningTab experimentId={experimentId} />,
              },
            ]}
          />
        </>
      )}
    </QueryState>
  );
}
