import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Play,
  CheckCircle2,
  Clock,
  TrendingUp,
  Loader2,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { datasetApi, trainingJobApi, workerApi } from '../services/api';
import WorkerDashboard from '../components/WorkerDashboard';
import JobMonitor from '../components/JobMonitor';

/**
 * Dashboard page component.
 */
export default function DashboardPage() {
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
    {
      label: 'Pending',
      value: pendingJobs,
      icon: Clock,
      color: 'text-surface-400',
      bgColor: 'bg-surface-500/10',
      link: '/training',
    },
  ];

  const isLoading = loadingDatasets || loadingJobs || loadingWorkers;

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
            <JobMonitor />
          </div>
        </div>
      </div>
    </div>
  );
}

