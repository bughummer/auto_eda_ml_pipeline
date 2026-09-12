/** Create an experiment: a validated dataset URI, a target, and an optional problem type. */

import { Alert, Button, Card, Form, Input, Select, Space, Typography } from 'antd';
import { useNavigate } from 'react-router-dom';

import { ApiError } from '../api/client';
import { useCreateExperiment } from '../api/hooks';

interface FormValues {
  name: string;
  dataset_uri: string;
  target_column: string;
  problem_type: string;
}

export function CreateExperimentPage() {
  const navigate = useNavigate();
  const create = useCreateExperiment();

  const onFinish = (values: FormValues) => {
    create.mutate(values, {
      onSuccess: (response) => navigate(`/experiments/${response.experiment_id}`),
    });
  };

  return (
    <Card title="New experiment" style={{ maxWidth: 760 }}>
      <Form<FormValues>
        layout="vertical"
        initialValues={{ problem_type: 'auto' }}
        onFinish={onFinish}
        requiredMark
      >
        <Form.Item
          name="name"
          label="Experiment name"
          rules={[{ required: true, message: 'Give the experiment a name' }]}
        >
          <Input placeholder="Customer churn baseline" maxLength={120} />
        </Form.Item>

        <Form.Item
          name="dataset_uri"
          label="Dataset"
          extra="S3 URI of an approved dataset, for example s3://approved-data/curated/churn.parquet"
          rules={[
            { required: true, message: 'Enter the dataset location' },
            {
              validator: (_rule, value: string) =>
                !value || value.startsWith('s3://')
                  ? Promise.resolve()
                  : Promise.reject(new Error('The dataset URI must start with s3://')),
            },
          ]}
        >
          <Input placeholder="s3://approved-data/curated/churn.parquet" />
        </Form.Item>

        <Form.Item
          name="target_column"
          label="Target column"
          rules={[{ required: true, message: 'Name the column to predict' }]}
        >
          <Input placeholder="churned" />
        </Form.Item>

        <Form.Item
          name="problem_type"
          label="Problem type"
          extra="Automatic detection reads the target column during EDA. Override it if you know better."
        >
          <Select
            options={[
              { value: 'auto', label: 'Detect automatically' },
              { value: 'binary_classification', label: 'Binary classification' },
              { value: 'multiclass_classification', label: 'Multiclass classification' },
              { value: 'regression', label: 'Regression' },
            ]}
          />
        </Form.Item>

        {create.error && (
          <Alert
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
            message={create.error instanceof ApiError ? create.error.code : 'Request failed'}
            description={(create.error as Error).message}
          />
        )}

        <Space>
          <Button type="primary" htmlType="submit" loading={create.isPending}>
            Create and run EDA
          </Button>
          <Button onClick={() => navigate('/')}>Cancel</Button>
        </Space>
        <Typography.Paragraph type="secondary" style={{ marginTop: 16, marginBottom: 0 }}>
          Profiling starts immediately and runs on ephemeral compute. This page does not wait for it.
        </Typography.Paragraph>
      </Form>
    </Card>
  );
}
