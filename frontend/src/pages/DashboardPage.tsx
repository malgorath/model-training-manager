import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Play,
  CheckCircle2,
  Clock,
  TrendingUp,
  Loader2,
  FolderGit2,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { datasetApi, trainingJobApi, workerApi, projectApi } from '../services/api';
import WorkerDashboard from '../components/WorkerDashboard';
import JobMonitor from '../components/JobMonitor';
import type { ProjectStatus } from '../types';

const statusConfig: Record<ProjectStatus, { icon: React.ReactNode; color: string; bgColor: string }> = {
  pending: { icon: <Clock className="h-4 w-4" />, color: 'text-yellow-400', bgColor: 'bg-yellow-900/20' },
  running: { icon: <Loader2 className="h-4 w-4 animate-spin" />, color: 'text-blue-400', bgColor: 'bg-blue-900/20' },
  completed: { icon: <CheckCircle2 className="h-4 w-4" />, color: 'text-green-400', bgColor: 'bg-green-900/20' },
  failed: { icon: <CheckCircle2 className="h-4 w-4" />, color: 'text-red-400', bgColor: 'bg-red-900/20' },
  cancelled: { icon: <Clock className="h-4 w-4" />, color: 'text-gray-400', bgColor: 'bg-gray-900/20' },
};

/**
 * Dashboard page component.
 */
export default function DashboardPage() {
  const navigate = useNavigate();
  
  const { data: projects, isLoading: loadingProjects } = useQuery({
    queryKey: ['projects', 1],
    queryFn: () => projectApi.list(1, 5),
  });

  const { data: datasets, isLoading: loadingDatasets } = useQuery({
    queryKey: ['datasets', 1],
    queryFn: () => datasetApi.list(1, 5),
  });

  const { data: jobs, isLoading: loadingJobs } = useQuery({
    queryKey: ['jobs', 1],
    queryFn: () => trainingJobApi.list(1, 10),
  });

  const { isLoading: loadingWorkers } = useQuery({
    queryKey: ['workers'],
    queryFn: workerApi.getStatus,
    refetchInterval: 5000,
  });

  const completedJobs = jobs?.items.filter((j) => j.status === 'completed').length ?? 0;
  const runningJobs = jobs?.items.filter((j) => j.status === 'running').length ?? 0;
  const pendingJobs = jobs?.items.filter((j) => j.status === 'pending' || j.status === 'queued').length ?? 0;

  const stats = [
    {
      label: 'Projects',
      value: projects?.total ?? 0,
      icon: FolderGit2,
      color: 'text-primary-400',
      bgColor: 'bg-primary-500/10',
      link: '/projects',
    },
    {
      label: 'Datasets',
      value: datasets?.total ?? 0,
      icon: Database,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10',
      link: '/datasets',
    },
    {
      label: 'Completed',
      value: completedJobs,
      icon: CheckCircle2,
      color: 'text-primary-400',
      bgColor: 'bg-primary-500/10',
      link: '/training',
    },
    {
      label: 'Running',
      value: runningJobs,
      icon: Play,
      color: 'text-amber-400',
      bgColor: 'bg-amber-500/10',
      link: '/training',
    },
  ];

  const isLoading = loadingProjects || loadingDatasets || loadingJobs || loadingWorkers;

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <p className="mt-2 text-surface-400">
          Monitor your training jobs and manage workers
        </p>
      </div>

      {/* Stats */}
      <div className="mb-8 grid grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Link
            key={stat.label}
            to={stat.link}
            className="card group p-6 transition-all hover:border-surface-700 hover:shadow-lg"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-surface-400">{stat.label}</p>
                <p className="mt-1 text-3xl font-bold text-white">
                  {isLoading ? (
                    <Loader2 className="h-8 w-8 animate-spin text-surface-500" />
                  ) : (
                    stat.value
                  )}
                </p>
              </div>
              <div className={`flex h-12 w-12 items-center justify-center rounded-xl ${stat.bgColor}`}>
                <stat.icon className={`h-6 w-6 ${stat.color}`} />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Projects Section - First thing visible */}
      <div className="mb-8">
        <div className="card">
          <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/10">
                  <FolderGit2 className="h-5 w-5 text-primary-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Recent Projects</h3>
                  <p className="text-sm text-surface-400">Your training projects</p>
                </div>
              </div>
              <Link to="/projects" className="btn-ghost text-sm">
                View all
              </Link>
            </div>
          </div>
          <div className="p-6">
            {loadingProjects ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
              </div>
            ) : projects && projects.items.length > 0 ? (
              <div className="space-y-4">
                {projects.items.map((project) => {
                  const statusInfo = statusConfig[project.status];
                  return (
                    <div
                      key={project.id}
                      className="p-4 rounded-lg border border-surface-800 hover:border-surface-700 hover:bg-surface-800/50 cursor-pointer transition-all"
                      onClick={() => navigate(`/projects/${project.id}`)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h4 className="text-base font-medium text-white">{project.name}</h4>
                            <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded ${statusInfo.bgColor} ${statusInfo.color}`}>
                              {statusInfo.icon}
                              <span className="text-xs font-medium capitalize">{project.status}</span>
                            </div>
                          </div>
                          {project.description && (
                            <p className="text-sm text-surface-400 mb-2">{project.description}</p>
                          )}
                          <div className="flex items-center gap-4 text-xs text-surface-500">
                            <span>Model: {project.base_model}</span>
                            <span>Type: {project.training_type}</span>
                            <span>Traits: {project.traits.length}</span>
                          </div>
                          {project.status === 'running' && (
                            <div className="mt-3">
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-surface-400">Progress</span>
                                <span className="text-surface-300">{project.progress.toFixed(1)}%</span>
                              </div>
                              <div className="h-1.5 bg-surface-800 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary-500 transition-all"
                                  style={{ width: `${project.progress}%` }}
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-12">
                <FolderGit2 className="h-12 w-12 text-surface-600 mx-auto mb-4" />
                <p className="text-surface-400">No projects yet</p>
                <Link to="/projects" className="btn-primary mt-4 inline-flex items-center gap-2">
                  Create Project
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-2 gap-8">
        {/* Worker Dashboard */}
        <div>
          <WorkerDashboard />
        </div>

        {/* Recent Jobs */}
        <div className="card">
          <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-500/10">
                  <TrendingUp className="h-5 w-5 text-accent-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Recent Jobs</h3>
                  <p className="text-sm text-surface-400">Latest training activity</p>
                </div>
              </div>
              <Link to="/training" className="btn-ghost text-sm">
                View all
              </Link>
            </div>
          </div>
          <div className="p-6">
            <JobMonitor 
              showFilters={true}
              onSelectJob={(job) => {
                // Navigate to training jobs page with job selected
                navigate(`/training?jobId=${job.id}`);
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

