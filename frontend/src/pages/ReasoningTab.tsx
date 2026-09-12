/**
 * Semantic analysis.
 *
 * Everything here is an interpretation of deterministic results. No number on this tab was
 * produced by the language model.
 */

import { Alert, App as AntApp, Button, Card, Form, Input, List, Space, Tag, Typography } from 'antd';

import { ApiError } from '../api/client';
import { useHealth, useReasoning, useRunReasoning } from '../api/hooks';
import { QueryState, SeverityTag } from '../components/common';
import { humanize } from '../lib/format';

interface ReasoningFormValues {
  target_definition?: string;
  prediction_timing?: string;
  question?: string;
}

export function ReasoningTab({ experimentId }: { experimentId: string }) {
  const { message } = AntApp.useApp();
  const { data: health } = useHealth();
  const enabled = Boolean(health?.bedrock_enabled);
  const { data, isLoading, error } = useReasoning(experimentId, enabled);
  const run = useRunReasoning(experimentId);

  if (!enabled) {
    return (
      <Alert
        type="info"
        showIcon
        message="Semantic analysis is not enabled in this environment"
        description="It requires a configured Bedrock model. Deterministic EDA, leakage screening and model results are unaffected."
      />
    );
  }

  return (
    <>
      <Card size="small" title="Business context" style={{ marginBottom: 16 }}>
        <Form<ReasoningFormValues>
          layout="vertical"
          onFinish={(values) =>
            run.mutate(values, {
              onSuccess: () => message.success('Analysis complete'),
              onError: (runError) =>
                message.error(
                  runError instanceof ApiError ? runError.message : 'The analysis failed',
                ),
            })
          }
        >
          <Form.Item
            name="target_definition"
            label="What does the target actually mean?"
            extra="Business definition, not a statistic."
          >
            <Input.TextArea rows={2} maxLength={2000} />
          </Form.Item>
          <Form.Item
            name="prediction_timing"
            label="When is the prediction made relative to the outcome?"
            extra="This is what makes post-outcome features identifiable."
          >
            <Input.TextArea rows={2} maxLength={2000} />
          </Form.Item>
          <Form.Item name="question" label="Specific question (optional)">
            <Input.TextArea rows={2} maxLength={2000} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={run.isPending}>
            Run semantic analysis
          </Button>
        </Form>
      </Card>

      <QueryState
        isLoading={isLoading}
        error={error}
        pendingMessage="No semantic analysis has been generated yet"
      >
        {data && (
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <Card size="small" title="Executive summary">
              <Typography.Paragraph>{data.executive_summary}</Typography.Paragraph>
              <Typography.Text type="secondary">
                Model: {data.model_id} · Inputs: {data.input_artifacts.join(', ')}
              </Typography.Text>
            </Card>

            {data.semantic_leakage.length > 0 && (
              <Card size="small" title="Possible semantic leakage">
                <List
                  dataSource={data.semantic_leakage}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={
                          <Space>
                            <Typography.Text strong>{item.feature}</Typography.Text>
                            <SeverityTag severity={item.severity} />
                            <Tag>confidence: {item.confidence}</Tag>
                            <Tag color="volcano">{humanize(item.recommended_action)}</Tag>
                          </Space>
                        }
                        description={item.reasoning}
                      />
                    </List.Item>
                  )}
                />
              </Card>
            )}

            {data.data_quality_concerns.length > 0 && (
              <Card size="small" title="Data quality concerns">
                <List
                  dataSource={data.data_quality_concerns}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={
                          <Space>
                            <Typography.Text strong>{item.topic}</Typography.Text>
                            <SeverityTag severity={item.severity} />
                          </Space>
                        }
                        description={
                          <>
                            <div>{item.explanation}</div>
                            {item.affected_columns.length > 0 && (
                              <div style={{ marginTop: 4 }}>
                                {item.affected_columns.map((column) => (
                                  <Tag key={column}>{column}</Tag>
                                ))}
                              </div>
                            )}
                          </>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            )}

            {data.model_behaviour_narrative && (
              <Card size="small" title="Model behaviour">
                <Typography.Paragraph>{data.model_behaviour_narrative}</Typography.Paragraph>
              </Card>
            )}

            {data.proposed_experiments.length > 0 && (
              <Card size="small" title="Proposed follow-up experiments">
                <List
                  dataSource={data.proposed_experiments}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={
                          <Space>
                            <Typography.Text strong>{item.title}</Typography.Text>
                            <Tag>{item.priority}</Tag>
                          </Space>
                        }
                        description={
                          <>
                            <div>{item.hypothesis}</div>
                            <ul style={{ marginBottom: 0 }}>
                              {item.proposed_changes.map((change) => (
                                <li key={change}>{change}</li>
                              ))}
                            </ul>
                            <Typography.Text type="secondary">{item.expected_insight}</Typography.Text>
                          </>
                        }
                      />
                    </List.Item>
                  )}
                />
                <Typography.Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
                  Proposals only — a person starts the next experiment.
                </Typography.Paragraph>
              </Card>
            )}

            {(data.assumptions.length > 0 || data.limitations.length > 0) && (
              <Card size="small" title="Assumptions and limitations">
                <ul>
                  {data.assumptions.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                  {data.limitations.map((item) => (
                    <li key={item}>
                      <Typography.Text type="warning">{item}</Typography.Text>
                    </li>
                  ))}
                </ul>
              </Card>
            )}

            <Alert type="info" showIcon message={data.disclaimer} />
          </Space>
        )}
      </QueryState>
    </>
  );
}
