/**
 * Feature review: the point where a human decides what the model may see.
 *
 * Nothing is excluded automatically except the target. The platform shows its evidence and
 * its recommendation; the selection that gets saved is exactly what the user chose.
 */

import { InboxOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  Input,
  Popconfirm,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
  Upload,
  App as AntApp,
} from 'antd';
import { useEffect, useMemo, useState } from 'react';

import { ApiError } from '../api/client';
import { useFeatureReview, useSaveFeatures, useUploadDictionary } from '../api/hooks';
import type { FeatureReviewItem, LeakageRiskLevel } from '../api/types';
import { QueryState } from '../components/common';
import { RISK_COLORS, RISK_LABELS, formatPercent, humanize } from '../lib/format';

const EXCLUSION_ACTIONS = new Set(['consider_excluding', 'strongly_consider_excluding']);

export function FeatureReviewTab({
  experimentId,
  root,
}: {
  experimentId: string;
  root?: string;
}) {
  const { message } = AntApp.useApp();
  const { data, isLoading, error } = useFeatureReview(experimentId, root);
  const isReadOnly = Boolean(root);
  const save = useSaveFeatures(experimentId);
  const uploadDictionary = useUploadDictionary(experimentId);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState<LeakageRiskLevel | 'all'>('all');

  useEffect(() => {
    if (data) {
      setSelected(new Set(data.features.filter((item) => item.selected).map((i) => i.feature)));
    }
  }, [data]);

  const visible = useMemo(() => {
    const features = data?.features ?? [];
    return features.filter((item) => {
      const matchesSearch = item.feature.toLowerCase().includes(search.trim().toLowerCase());
      const matchesRisk = riskFilter === 'all' || item.leakage_risk === riskFilter;
      return matchesSearch && matchesRisk;
    });
  }, [data, search, riskFilter]);

  const flaggedSelected = useMemo(
    () =>
      (data?.features ?? []).filter(
        (item) => selected.has(item.feature) && EXCLUSION_ACTIONS.has(item.recommended_action),
      ),
    [data, selected],
  );

  const toggle = (feature: string, include: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (include) next.add(feature);
      else next.delete(feature);
      return next;
    });
  };

  const applyToVisible = (include: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      visible.forEach((item) => (include ? next.add(item.feature) : next.delete(item.feature)));
      return next;
    });
  };

  const onSave = () => {
    if (!data) return;
    const exclusionReasons: Record<string, string> = {};
    data.features
      .filter((item) => !selected.has(item.feature))
      .forEach((item) => {
        exclusionReasons[item.feature] = item.reasons[0] ?? 'Excluded during feature review';
      });
    save.mutate(
      { selected_features: Array.from(selected), exclusion_reasons: exclusionReasons },
      {
        onSuccess: () => message.success('Feature selection saved'),
        onError: (saveError) =>
          message.error(
            saveError instanceof ApiError ? saveError.message : 'Could not save the selection',
          ),
      },
    );
  };

  return (
    <QueryState isLoading={isLoading} error={error} pendingMessage="EDA has not finished yet">
      {data && (
        <>
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={5}>
              <Card size="small">
                <Statistic title="Selected" value={selected.size} />
              </Card>
            </Col>
            <Col span={5}>
              <Card size="small">
                <Statistic title="Excluded" value={data.features.length - selected.size} />
              </Card>
            </Col>
            <Col span={14}>
              <Upload.Dragger
                accept=".json,.csv,.xlsx,.xls"
                showUploadList={false}
                beforeUpload={(file) => {
                  uploadDictionary.mutate(file as File, {
                    onSuccess: (dictionary) =>
                      message.success(
                        `Documentation loaded for ${dictionary.columns.length} columns`,
                      ),
                    onError: (uploadError) => message.error((uploadError as Error).message),
                  });
                  return false;
                }}
                style={{ padding: 8 }}
              >
                <p style={{ margin: 0 }}>
                  <InboxOutlined style={{ marginRight: 8 }} />
                  Attach a data dictionary (JSON, CSV or XLSX) to add business context
                </p>
              </Upload.Dragger>
            </Col>
          </Row>

          {flaggedSelected.length > 0 && (
            <Alert
              type="warning"
              showIcon
              style={{ marginBottom: 16 }}
              message={`${flaggedSelected.length} selected feature(s) are recommended for exclusion`}
              description={flaggedSelected.map((item) => item.feature).join(', ')}
            />
          )}

          <Space style={{ marginBottom: 12 }} wrap>
            <Input.Search
              allowClear
              placeholder="Filter by column name"
              style={{ width: 260 }}
              onChange={(event) => setSearch(event.target.value)}
            />
            <Select<LeakageRiskLevel | 'all'>
              value={riskFilter}
              style={{ width: 220 }}
              onChange={setRiskFilter}
              options={[
                { value: 'all', label: 'All risk levels' },
                { value: 'confirmed_duplicate', label: RISK_LABELS.confirmed_duplicate },
                { value: 'potential_leakage', label: RISK_LABELS.potential_leakage },
                { value: 'requires_review', label: RISK_LABELS.requires_review },
                { value: 'none', label: RISK_LABELS.none },
              ]}
            />
            <Popconfirm
              title={`Include all ${visible.length} visible features?`}
              description="Bulk actions apply only to the rows currently visible."
              onConfirm={() => applyToVisible(true)}
            >
              <Button size="small">Include visible</Button>
            </Popconfirm>
            <Popconfirm
              title={`Exclude all ${visible.length} visible features?`}
              description="Bulk actions apply only to the rows currently visible."
              onConfirm={() => applyToVisible(false)}
            >
              <Button size="small">Exclude visible</Button>
            </Popconfirm>
            <Button
              type="primary"
              onClick={onSave}
              loading={save.isPending}
              disabled={selected.size === 0 || isReadOnly}
            >
              Save selection
            </Button>
          </Space>

          <Table<FeatureReviewItem>
            size="small"
            rowKey="feature"
            dataSource={visible}
            pagination={{ pageSize: 25, hideOnSinglePage: true }}
            rowSelection={{
              selectedRowKeys: Array.from(selected),
              onSelect: (record, include) => toggle(record.feature, include),
              onSelectAll: (include, _rows, changed) =>
                changed.forEach((row) => toggle(row.feature, include)),
            }}
            expandable={{
              rowExpandable: (record) => record.reasons.length > 0 || Boolean(record.documentation),
              expandedRowRender: (record) => (
                <div>
                  {record.documentation && (
                    <Typography.Paragraph>
                      <Typography.Text strong>Business definition: </Typography.Text>
                      {record.documentation}
                    </Typography.Paragraph>
                  )}
                  {record.available_at_prediction_time === false && (
                    <Alert
                      type="warning"
                      showIcon
                      style={{ marginBottom: 8 }}
                      message="Documented as unavailable at prediction time"
                    />
                  )}
                  <ul style={{ marginBottom: 0 }}>
                    {record.reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </div>
              ),
            }}
            columns={[
              { title: 'Feature', dataIndex: 'feature', sorter: (a, b) => a.feature.localeCompare(b.feature) },
              {
                title: 'Type',
                dataIndex: 'semantic_type',
                width: 180,
                render: (value: string) => <Tag>{humanize(value)}</Tag>,
              },
              {
                title: 'Missing',
                dataIndex: 'missing_percentage',
                width: 110,
                render: (value: number) => formatPercent(value),
                sorter: (a, b) => a.missing_percentage - b.missing_percentage,
              },
              {
                title: 'Distinct',
                dataIndex: 'unique_count',
                width: 110,
                sorter: (a, b) => a.unique_count - b.unique_count,
              },
              {
                title: 'Leakage risk',
                dataIndex: 'leakage_risk',
                width: 190,
                render: (risk: LeakageRiskLevel) => (
                  <Tag color={RISK_COLORS[risk]}>{RISK_LABELS[risk]}</Tag>
                ),
                sorter: (a, b) => a.leakage_risk.localeCompare(b.leakage_risk),
              },
              {
                title: 'Recommendation',
                dataIndex: 'recommended_action',
                width: 230,
                render: (action: string, record) => (
                  <Tooltip title={record.reasons[0] ?? ''}>
                    <Tag color={EXCLUSION_ACTIONS.has(action) ? 'volcano' : 'default'}>
                      {humanize(action)}
                    </Tag>
                  </Tooltip>
                ),
              },
            ]}
          />
          <Typography.Paragraph type="secondary" style={{ marginTop: 12 }}>
            The platform never removes a feature on its own. Recommendations are evidence for your
            decision, and the selection you save is stored with the experiment.
          </Typography.Paragraph>
        </>
      )}
    </QueryState>
  );
}
