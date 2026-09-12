/** Application shell and routing. */

import { Alert, Layout, Typography } from 'antd';
import { Navigate, Route, Routes } from 'react-router-dom';

import { useHealth } from './api/hooks';
import { CreateExperimentPage } from './pages/CreateExperimentPage';
import { ExperimentDetailPage } from './pages/ExperimentDetailPage';
import { ExperimentListPage } from './pages/ExperimentListPage';

export function App() {
  const { data: health } = useHealth();

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Header style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
          ML Factory
        </Typography.Title>
        {health && (
          <Typography.Text style={{ color: 'rgba(255,255,255,0.65)' }}>
            {health.mode} mode · {health.orchestrator} orchestrator
          </Typography.Text>
        )}
      </Layout.Header>
      <Layout.Content style={{ padding: 24, maxWidth: 1600, width: '100%', margin: '0 auto' }}>
        {health?.configuration_problems?.length ? (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
            message="The platform is not fully configured"
            description={
              <ul style={{ marginBottom: 0 }}>
                {health.configuration_problems.map((problem) => (
                  <li key={problem}>{problem}</li>
                ))}
              </ul>
            }
          />
        ) : null}
        <Routes>
          <Route path="/" element={<ExperimentListPage />} />
          <Route path="/experiments/new" element={<CreateExperimentPage />} />
          <Route path="/experiments/:experimentId" element={<ExperimentDetailPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout.Content>
      <Layout.Footer style={{ textAlign: 'center' }}>
        <Typography.Text type="secondary">
          Every statistic and metric shown here is computed deterministically. Language-model
          output is confined to the semantic analysis tab and never changes a result.
        </Typography.Text>
      </Layout.Footer>
    </Layout>
  );
}
