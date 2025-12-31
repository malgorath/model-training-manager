import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play,
  Clock,
  CheckCircle2,
  XCircle,
  AlertCircle,
  StopCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
  Download,
} from 'lucide-react';
import { clsx } from 'clsx';
import { trainingJobApi } from '../services/api';
import { downloadBlob } from '../utils/download';
import type { TrainingJob, TrainingStatus } from '../types';

/**
 * Get status badge styling.
 */
const getStatusBadge = (status: TrainingStatus) => {
  switch (status) {
    case 'completed':
      return { icon: CheckCircle2, className: 'badge-success' };
    case 'running':
      return { icon: Play, className: 'badge-info' };
    case 'pending':
    case 'queued':
      return { icon: Clock, className: 'badge-warning' };
    case 'failed':
      return { icon: XCircle, className: 'badge-error' };
    case 'cancelled':
      return { icon: StopCircle, className: 'badge-neutral' };
    default:
      return { icon: AlertCircle, className: 'badge-neutral' };
  }
};

interface JobMonitorProps {
  onSelectJob?: (job: TrainingJob) => void;
  showFilters?: boolean;
}

/**
 * Training job monitor component.
 */
export default function JobMonitor({ onSelectJob, showFilters = true }: JobMonitorProps) {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<TrainingStatus | undefined>();
  const pageSize = 10;

  const { data, isLoading } = useQuery({
    queryKey: ['jobs', page, statusFilter],
    queryFn: () => trainingJobApi.list(page, pageSize, statusFilter),
    refetchInterval: 5000,
    // Ensure we always fetch jobs, even if statusFilter is undefined
    enabled: true,
  });

  const startMutation = useMutation({
    mutationFn: trainingJobApi.start,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
  });

  const cancelMutation = useMutation({
    mutationFn: trainingJobApi.cancel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const formatDuration = (startedAt: string | null, completedAt: string | null): string => {
    if (!startedAt) return '-';
    const start = new Date(startedAt).getTime();
    const end = completedAt ? new Date(completedAt).getTime() : Date.now();
    const duration = Math.floor((end - start) / 1000);
    
    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  return (
    <div>
      {/* Filters */}
      {showFilters && (
        <div className="mb-4 flex items-center gap-2">
          <span className="text-sm text-surface-400">Filter:</span>
          {(['pending', 'running', 'completed', 'failed'] as TrainingStatus[]).map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(statusFilter === status ? undefined : status)}
              className={clsx(
                'rounded-lg px-3 py-1.5 text-sm capitalize transition-colors',
                statusFilter === status
                  ? 'bg-primary-500/20 text-primary-400'
                  : 'bg-surface-800 text-surface-400 hover:bg-surface-700'
              )}
            >
              {status}
            </button>
          ))}
        </div>
      )}

      {/* Job List */}
      {!data?.items.length ? (
        <div className="py-12 text-center">
          <Play className="mx-auto h-12 w-12 text-surface-600" />
          <p className="mt-3 text-surface-400">No training jobs yet</p>
          <p className="text-sm text-surface-500">
            Create a training job to get started
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {data.items.map((job) => {
            const statusInfo = getStatusBadge(job.status);
            const StatusIcon = statusInfo.icon;

            return (
              <div
                key={job.id}
                onClick={() => onSelectJob?.(job)}
                className={clsx(
                  'card cursor-pointer overflow-hidden transition-all hover:border-surface-700 hover:shadow-lg',
                  job.status === 'running' && 'border-primary-500/30'
                )}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface-800">
                        {job.status === 'running' ? (
                          <Loader2 className="h-5 w-5 animate-spin text-primary-400" />
                        ) : (
                          <StatusIcon className="h-5 w-5 text-surface-400" />
                        )}
                      </div>
                      <div>
                        <h4 className="font-medium text-white">{job.name}</h4>
                        <p className="text-sm text-surface-400">
                          {job.training_type.toUpperCase()} • {job.epochs} epochs
                          {job.worker_id && (
                            <span className="ml-2 text-primary-400">
                              • Worker: {job.worker_id}
                            </span>
                          )}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className={statusInfo.className}>
                        {job.status}
                      </span>

                      {(job.status === 'pending' || (job.status === 'queued' && job.progress === 0)) && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            startMutation.mutate(job.id);
                          }}
                          disabled={startMutation.isPending || cancelMutation.isPending}
                          className="rounded-lg p-2 text-surface-400 hover:bg-primary-500/10 hover:text-primary-400 transition-colors"
                          title="Start job"
                        >
                          <Play className="h-4 w-4" />
                        </button>
                      )}

                      {(job.status === 'running' || job.status === 'queued') && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            cancelMutation.mutate(job.id);
                          }}
                          disabled={cancelMutation.isPending || startMutation.isPending}
                          className="rounded-lg p-2 text-surface-400 hover:bg-red-500/10 hover:text-red-400 transition-colors"
                          title="Cancel job"
                        >
                          <StopCircle className="h-4 w-4" />
                        </button>
                      )}

                      {job.status === 'completed' && job.model_path && (
                        <DownloadButton jobId={job.id} jobName={job.name} />
                      )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {(job.status === 'running' || job.status === 'completed') && (
                    <div className="mt-4">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-surface-400">
                          Epoch {job.current_epoch} / {job.epochs}
                        </span>
                        <span className="text-surface-300">
                          {job.progress.toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-800">
                        <div
                          className={clsx(
                            'h-full rounded-full transition-all duration-500',
                            job.status === 'completed'
                              ? 'bg-emerald-500'
                              : 'bg-gradient-to-r from-primary-500 to-accent-500'
                          )}
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs text-surface-500">
                        <span>
                          {job.current_loss !== null && `Loss: ${job.current_loss.toFixed(4)}`}
                        </span>
                        <span>
                          Duration: {formatDuration(job.started_at, job.completed_at)}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {job.status === 'failed' && job.error_message && (
                    <div className="mt-4 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-400">
                      {job.error_message}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination */}
      {data && data.pages > 1 && (
        <div className="mt-4 flex items-center justify-between">
          <p className="text-sm text-surface-400">
            Showing {(page - 1) * pageSize + 1} to{' '}
            {Math.min(page * pageSize, data.total)} of {data.total} jobs
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="btn-ghost"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </button>
            <button
              onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
              disabled={page === data.pages}
              className="btn-ghost"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Download button component for completed jobs in the list view.
 */
function DownloadButton({ jobId, jobName }: { jobId: number; jobName: string }) {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDownloading(true);
    
    try {
      const blob = await trainingJobApi.download(jobId);
      const contentType = blob.type;
      let filename: string;
      
      if (contentType === 'application/gzip') {
        filename = `model_${jobName.replace(/\s+/g, '_')}_${jobId}.tar.gz`;
      } else {
        filename = `model_${jobName.replace(/\s+/g, '_')}_${jobId}`;
      }
      
      downloadBlob(blob, filename);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <button
      onClick={handleDownload}
      disabled={isDownloading}
      className="rounded-lg p-2 text-surface-400 hover:bg-emerald-500/10 hover:text-emerald-400 transition-colors disabled:opacity-50"
      title="Download model"
    >
      {isDownloading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <Download className="h-4 w-4" />
      )}
    </button>
  );
}