import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import DashboardPage from './pages/DashboardPage';
import DatasetsPage from './pages/DatasetsPage';
import TrainingJobsPage from './pages/TrainingJobsPage';
import ProjectsPage from './pages/ProjectsPage';
import ProjectDetailPage from './pages/ProjectDetailPage';
import UserGuidePage from './pages/UserGuidePage';
import SettingsPage from './pages/SettingsPage';
import ModelsPage from './pages/ModelsPage';

/**
 * Main application component with routing.
 */
function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="training" element={<TrainingJobsPage />} />
        <Route path="projects" element={<ProjectsPage />} />
        <Route path="projects/:id" element={<ProjectDetailPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="guide" element={<UserGuidePage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}

export default App;

