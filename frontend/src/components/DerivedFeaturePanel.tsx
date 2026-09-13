/**
 * Proposed derived features.
 *
 * A model suggests specifications; the backend refuses everything that would not be valid
 * before this component ever sees it. What is left is a list of candidates, each with the
 * exact formula it would compute and the reasoning behind it, and none of them is used until
 * somebody here ticks it. Approving nothing is a normal outcome, not a failure.
 */

import { Alert, App as AntApp, Button, Card, Checkbox, Collapse, Form, Input, Space, Table, Tag, Tooltip, Typography } from 'antd';
import { useEffect, useState } from 'react';

import { ApiError } from '../api/client';
import { useApproveProposals, useFeatureProposals, useProposeFeatures } from '../api/hooks';
import type { FeatureProposal, ProposalRejection, RejectedProposal } from '../api/types';
import { formatDateTime, humanize } from '../lib/format';

/** The formula a specification computes, in the notation a data scientist would write. */
export function formulaOf(proposal: FeatureProposal): string {
  switch (proposal.op) {
    case 'ratio':
      return `${proposal.numerator} / ${proposal.denominator}`;
    case 'difference':
      return `${proposal.left} - ${proposal.right}`;
    case 'date_difference':
      return `${proposal.end} - ${proposal.start} (in ${proposal.unit})`;
    case 'map_categories': {
      const pairs = Object.entries(proposal.mapping);
      const shown = pairs
        .slice(0, 4)
        .map(([from, to]) => `${from} → ${to}`)
        .join(', ');
      const rest = pairs.length > 4 ? `, +${pairs.length - 4} more` : '';
      const fallback = proposal.default === null ? '' : `, else ${proposal.default}`;
      return `${proposal.column}: ${shown}${rest}${fallback}`;
    }
    case 'is_missing':
      return `${proposal.column} is missing`;
  }
}

/** Which source columns a specification reads. Mirrors `source_columns` in the engine. */
export function sourcesOf(proposal: FeatureProposal): string[] {
  switch (proposal.op) {
    case 'ratio':
      return [proposal.numerator, proposal.denominator];
    case 'difference':
      return [proposal.left, proposal.right];
    case 'date_difference':
      return [proposal.start, proposal.end];
    case 'map_categories':
    case 'is_missing':
      return [proposal.column];
  }
}

const REFUSAL_LABELS: Record<ProposalRejection, string> = {
  malformed_specification: 'Not a supported operation',
  invalid_name: 'Unusable column name',
  duplicate_name: 'Name already claimed',
  name_already_in_dataset: 'Name already in the dataset',
  unknown_column: 'Column does not exist',
  target_reference: 'Reads the target',
  wrong_column_type: 'Wrong column type',
  empty_mapping: 'Empty mapping',
  limit_exceeded: 'Over the per-experiment limit',
};

export function DerivedFeaturePanel({
  experimentId,
  enabled,
  readOnly,
}: {
  experimentId: string;
  enabled: boolean;
  readOnly: boolean;
}) {
  const { message } = AntApp.useApp();
  const { data } = useFeatureProposals(experimentId, enabled);
  const propose = useProposeFeatures(experimentId);
  const approve = useApproveProposals(experimentId);
  const [approved, setApproved] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (data) setApproved(new Set(data.approved_names));
  }, [data]);

  if (!enabled) return null;

  const toggle = (name: string) =>
    setApproved((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  const candidates = data?.candidates ?? [];
  const rejected = data?.rejected ?? [];

  const columns = [
    {
      title: '',
      key: 'approved',
      width: 48,
      render: (_: unknown, proposal: FeatureProposal) => (
        <Checkbox
          disabled={readOnly}
          checked={approved.has(proposal.name)}
          onChange={() => toggle(proposal.name)}
        />
      ),
    },
    {
      title: 'Feature',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, proposal: FeatureProposal) => (
        <Space direction="vertical" size={0}>
          <Typography.Text strong>{name}</Typography.Text>
          <Tag>{humanize(proposal.op)}</Tag>
        </Space>
      ),
    },
    {
      title: 'Computes',
      key: 'formula',
      render: (_: unknown, proposal: FeatureProposal) => (
        <Typography.Text code>{formulaOf(proposal)}</Typography.Text>
      ),
    },
    {
      title: 'From',
      key: 'sources',
      render: (_: unknown, proposal: FeatureProposal) => (
        <Space size={4} wrap>
          {sourcesOf(proposal).map((column) => (
            <Tag key={column}>{column}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: 'Why',
      dataIndex: 'rationale',
      key: 'rationale',
      render: (rationale: string) => <Typography.Text>{rationale}</Typography.Text>,
    },
    {
      title: 'Known at prediction time',
      key: 'available',
      width: 190,
      render: (_: unknown, proposal: FeatureProposal) =>
        proposal.available_at_prediction_time ? (
          <Tag color="green">Claimed available</Tag>
        ) : (
          <Tooltip title="Every input must be known when a prediction is actually made. This one is not, by the proposer's own account.">
            <Tag color="red">Not available</Tag>
          </Tooltip>
        ),
    },
  ];

  return (
    <Card
      size="small"
      title="Proposed derived features"
      style={{ marginBottom: 16 }}
      extra={
        data?.decided_at ? (
          <Typography.Text type="secondary">
            {data.approved_names.length} approved by {data.decided_by ?? 'someone'} on{' '}
            {formatDateTime(data.decided_at)}
          </Typography.Text>
        ) : null
      }
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 12 }}
        message="Suggestions, not decisions"
        description="A model reads the profile and any attached documentation and proposes features from a fixed vocabulary. It writes no code, and nothing here is computed until you approve it. Approving none is a normal answer."
      />

      <Form<{ target_definition?: string; prediction_timing?: string }>
        layout="vertical"
        disabled={readOnly}
        onFinish={(values) =>
          propose.mutate(values, {
            onSuccess: (report) =>
              message.success(
                report.candidates.length > 0
                  ? `${report.candidates.length} feature(s) proposed`
                  : 'No usable features were proposed',
              ),
            onError: (proposeError) =>
              message.error(
                proposeError instanceof ApiError
                  ? proposeError.message
                  : 'Could not generate proposals',
              ),
          })
        }
      >
        <Form.Item
          name="prediction_timing"
          label="When is the prediction actually made?"
          extra="This is what lets the proposer judge whether a feature would be known at that moment. Without it, treat every availability claim as unverified."
        >
          <Input.TextArea rows={2} maxLength={2000} />
        </Form.Item>
        <Form.Item name="target_definition" label="What does the target mean, in business terms?">
          <Input.TextArea rows={2} maxLength={2000} />
        </Form.Item>
        <Button type="primary" htmlType="submit" loading={propose.isPending} disabled={readOnly}>
          {data ? 'Propose again' : 'Propose features'}
        </Button>
      </Form>

      {candidates.length > 0 && (
        <>
          <Table<FeatureProposal>
            style={{ marginTop: 16 }}
            rowKey="name"
            size="small"
            pagination={false}
            dataSource={candidates}
            columns={columns}
          />
          <Space style={{ marginTop: 12 }}>
            <Button
              type="primary"
              loading={approve.isPending}
              disabled={readOnly}
              onClick={() =>
                approve.mutate([...approved], {
                  onSuccess: (report) =>
                    message.success(
                      report.approved_names.length > 0
                        ? `${report.approved_names.length} derived feature(s) will be built`
                        : 'No derived features will be built',
                    ),
                  onError: (approveError) =>
                    message.error(
                      approveError instanceof ApiError
                        ? approveError.message
                        : 'Could not save the decision',
                    ),
                })
              }
            >
              Save decision
            </Button>
            <Typography.Text type="secondary">
              {approved.size} of {candidates.length} selected
            </Typography.Text>
          </Space>
        </>
      )}

      {data && candidates.length === 0 && (
        <Alert
          type="warning"
          showIcon
          style={{ marginTop: 16 }}
          message="Nothing usable was proposed"
          description="Either the model had no suggestion worth making, or everything it suggested was refused. A dataset with no documentation gives it very little to work with."
        />
      )}

      {rejected.length > 0 && (
        <Collapse
          ghost
          style={{ marginTop: 8 }}
          items={[
            {
              key: 'rejected',
              label: `${rejected.length} suggestion(s) were refused before reaching you`,
              children: (
                <Table<RejectedProposal>
                  rowKey={(item, index) => `${item.name}-${index}`}
                  size="small"
                  pagination={false}
                  dataSource={rejected}
                  columns={[
                    { title: 'Name', dataIndex: 'name', key: 'name' },
                    {
                      title: 'Reason',
                      dataIndex: 'reason',
                      key: 'reason',
                      render: (reason: ProposalRejection) => (
                        <Tag color={reason === 'target_reference' ? 'red' : 'default'}>
                          {REFUSAL_LABELS[reason] ?? humanize(reason)}
                        </Tag>
                      ),
                    },
                    { title: 'Detail', dataIndex: 'message', key: 'message' },
                  ]}
                />
              ),
            },
          ]}
        />
      )}
    </Card>
  );
}
