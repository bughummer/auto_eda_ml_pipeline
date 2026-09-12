/** Small shared presentational pieces. */

import { Alert, Empty, Spin, Table, Tag, Tooltip } from 'antd';
import type { ReactNode } from 'react';

import { ApiError } from '../api/client';
import type { AnalysisWarning, ExperimentStatus, Severity } from '../api/types';
import { SEVERITY_COLORS, STATUS_COLORS, STATUS_LABELS, humanize } from '../lib/format';

export function StatusTag({ status }: { status: ExperimentStatus }) {
  return <Tag color={STATUS_COLORS[status]}>{STATUS_LABELS[status]}</Tag>;
}

export function SeverityTag({ severity }: { severity: Severity }) {
  return <Tag color={SEVERITY_COLORS[severity]}>{severity}</Tag>;
}

export function QueryState({
  isLoading,
  error,
  pendingMessage,
  children,
}: {
  isLoading: boolean;
  error: unknown;
  pendingMessage?: string;
  children: ReactNode;
}) {
  if (isLoading) {
    return (
      <div style={{ padding: 48, textAlign: 'center' }}>
        <Spin />
      </div>
    );
  }
  if (error instanceof ApiError && error.isNotReady) {
    return (
      <Alert
        type="info"
        showIcon
        message={pendingMessage ?? 'Not available yet'}
        description={error.message}
      />
    );
  }
  if (error) {
    const apiError = error instanceof ApiError ? error : null;
    return (
      <Alert
        type="error"
        showIcon
        message={apiError ? `${apiError.code}` : 'Request failed'}
        description={
          <>
            <div>{(error as Error).message}</div>
            {apiError?.requestId && (
              <div style={{ marginTop: 8, fontSize: 12, opacity: 0.7 }}>
                Request id: {apiError.requestId}
              </div>
            )}
          </>
        }
      />
    );
  }
  return <>{children}</>;
}

export function WarningsTable({ warnings }: { warnings: AnalysisWarning[] }) {
  if (warnings.length === 0) {
    return <Empty description="No warnings were raised" />;
  }
  return (
    <Table<AnalysisWarning>
      size="small"
      rowKey={(warning, index) => `${warning.rule}-${warning.column ?? ''}-${index}`}
      dataSource={warnings}
      pagination={warnings.length > 20 ? { pageSize: 20 } : false}
      columns={[
        {
          title: 'Severity',
          dataIndex: 'severity',
          width: 110,
          render: (severity: Severity) => <SeverityTag severity={severity} />,
          filters: (['critical', 'high', 'medium', 'low', 'info'] as Severity[]).map((value) => ({
            text: value,
            value,
          })),
          onFilter: (value, record) => record.severity === value,
        },
        {
          title: 'Category',
          dataIndex: 'category',
          width: 130,
          render: (category: string) => <Tag>{humanize(category)}</Tag>,
        },
        { title: 'Column', dataIndex: 'column', width: 180, render: (value) => value ?? '—' },
        {
          title: 'Finding',
          dataIndex: 'message',
          render: (message: string, record) => (
            <Tooltip title={`rule: ${record.rule}`}>
              <span>{message}</span>
            </Tooltip>
          ),
        },
      ]}
    />
  );
}
