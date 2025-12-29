import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play,
  Square,
  RefreshCw,
  Cpu,
  Circle,
  Loader2,
  Power,
} from 'lucide-react';
import { clsx } from 'clsx';
import { workerApi, configApi } from '../services/api';
import type { WorkerStatus } from '../types';

/**
 * Get status badge color based on worker status.
 */
const getStatusColor = (status: WorkerStatus): string => {
  switch (status) {
    case 'idle':
      return 'text-emerald-400';
    case 'busy':
      return 'text-amber-400';
    case 'stopped':
      return 'text-surface-400';
    case 'error':
      return 'text-red-400';
    default:
      return 'text-surface-400';
  }
};

/**
 * Worker dashboard component for managing training workers.
 */
export default function WorkerDashboard() {
  const queryClient = useQueryClient();

  const { data: poolStatus, isLoading } = useQuery({
    queryKey: ['workers'],
    queryFn: workerApi.getStatus,
    refetchInterval: 3000,
  });

  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: configApi.get,
  });

  const startMutation = useMutation({
    mutationFn: (count: number) => workerApi.start(count),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['workers'] }),
  });

  const stopMutation = useMutation({
    mutationFn: workerApi.stop,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['workers'] }),
  });

  const restartMutation = useMutation({
    mutationFn: workerApi.restart,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['workers'] }),
  });

  const autoStartMutation = useMutation({
    mutationFn: (enabled: boolean) => configApi.update({ auto_start_workers: enabled }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['config'] }),
  });

  const isProcessing = startMutation.isPending || stopMutation.isPending || restartMutation.isPending;

  if (isLoading) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      {/* Header */}
      <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/10">
                <Cpu className="h-5 w-5 text-primary-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Worker Pool</h3>
                <p className="text-sm text-surface-400">
                  Manage training workers
                </p>
              </div>
            </div>

            {/* Auto-Start Toggle */}
            <div className="flex items-center gap-3 border-l border-surface-700 pl-6">
              <Power className={clsx(
                'h-4 w-4',
                config?.auto_start_workers ? 'text-emerald-400' : 'text-surface-500'
              )} />
              <span className="text-sm text-surface-300">Auto-Start</span>
              <button
                onClick={() => autoStartMutation.mutate(!config?.auto_start_workers)}
                disabled={autoStartMutation.isPending}
                className={clsx(
                  'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-surface-900',
                  config?.auto_start_workers ? 'bg-emerald-500' : 'bg-surface-700'
                )}
                role="switch"
                aria-checked={config?.auto_start_workers ?? false}
              >
                <span
                  className={clsx(
                    'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                    config?.auto_start_workers ? 'translate-x-6' : 'translate-x-1'
                  )}
                />
              </button>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => startMutation.mutate(1)}
              disabled={isProcessing || (poolStatus?.active_workers ?? 0) >= (poolStatus?.max_workers ?? 4)}
              className="btn-primary"
            >
              {startMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              Start Worker
            </button>
            <button
              onClick={() => stopMutation.mutate()}
              disabled={isProcessing || (poolStatus?.active_workers ?? 0) === 0}
              className="btn-danger"
            >
              {stopMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Square className="h-4 w-4" />
              )}
              Stop All
            </button>
            <button
              onClick={() => restartMutation.mutate()}
              disabled={isProcessing || (poolStatus?.active_workers ?? 0) === 0}
              className="btn-secondary"
            >
              {restartMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Restart
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-px bg-surface-800">
        <div className="bg-surface-900 px-6 py-4">
          <p className="text-sm text-surface-400">Total Workers</p>
          <p className="mt-1 text-2xl font-semibold text-white">
            {poolStatus?.total_workers ?? 0}
          </p>
        </div>
        <div className="bg-surface-900 px-6 py-4">
          <p className="text-sm text-surface-400">Idle</p>
          <p className="mt-1 text-2xl font-semibold text-emerald-400">
            {poolStatus?.idle_workers ?? 0}
          </p>
        </div>
        <div className="bg-surface-900 px-6 py-4">
          <p className="text-sm text-surface-400">Busy</p>
          <p className="mt-1 text-2xl font-semibold text-amber-400">
            {poolStatus?.busy_workers ?? 0}
          </p>
        </div>
        <div className="bg-surface-900 px-6 py-4">
          <p className="text-sm text-surface-400">Jobs Queued</p>
          <p className="mt-1 text-2xl font-semibold text-primary-400">
            {poolStatus?.jobs_in_queue ?? 0}
          </p>
        </div>
      </div>

      {/* Worker List */}
      <div className="p-6">
        {poolStatus?.workers && poolStatus.workers.length > 0 ? (
          <div className="space-y-3">
            {poolStatus.workers.map((worker) => (
              <div
                key={worker.id}
                className="flex items-center justify-between rounded-lg border border-surface-800 bg-surface-800/30 px-4 py-3"
              >
                <div className="flex items-center gap-3">
                  <Circle
                    className={clsx(
                      'h-3 w-3 fill-current',
                      getStatusColor(worker.status)
                    )}
                  />
                  <div>
                    <p className="font-medium text-white">{worker.id}</p>
                    <p className="text-sm text-surface-400 capitalize">
                      {worker.status}
                      {worker.current_job_id && (
                        <span className="ml-2">
                          → Job #{worker.current_job_id}
                        </span>
                      )}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-surface-300">
                    {worker.jobs_completed} jobs completed
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-8 text-center">
            <Cpu className="mx-auto h-12 w-12 text-surface-600" />
            <p className="mt-3 text-surface-400">No active workers</p>
            <p className="text-sm text-surface-500">
              Start a worker to begin processing training jobs
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

