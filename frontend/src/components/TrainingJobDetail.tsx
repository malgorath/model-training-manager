import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  CheckCircle2,
  XCircle,
  Clock,
  Play,
  StopCircle,
  AlertCircle,
  Loader2,
  X,
  FileText,
  Calendar,
  Timer,
  TrendingDown,
  HardDrive,
  Download,
} from 'lucide-react';
import { clsx } from 'clsx';
import { trainingJobApi } from '../services/api';
import { downloadBlob } from '../utils/download';
import type { TrainingStatus } from '../types';

interface TrainingJobDetailProps {
  jobId: number;
  onClose: () => void;
}

/**
 * Get status badge styling.
 */
const getStatusBadge = (status: TrainingStatus) => {
  switch (status) {
    case 'completed':
      return { icon: CheckCircle2, className: 'badge-success', color: 'text-emerald-400' };
    case 'running':
      return { icon: Play, className: 'badge-info', color: 'text-primary-400' };
    case 'pending':
    case 'queued':
      return { icon: Clock, className: 'badge-warning', color: 'text-amber-400' };
    case 'failed':
      return { icon: XCircle, className: 'badge-error', color: 'text-red-400' };
    case 'cancelled':
      return { icon: StopCircle, className: 'badge-neutral', color: 'text-surface-400' };
    default:
      return { icon: AlertCircle, className: 'badge-neutral', color: 'text-surface-400' };
  }
};

/**
 * Format duration in human-readable format.
 */
const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
};

/**
 * Format date/time.
 */
const formatDateTime = (dateString: string | null): string => {
  if (!dateString) return '-';
  const date = new Date(dateString);
  return date.toLocaleString();
};

/**
 * Calculate elapsed time.
 */
const getElapsedTime = (startedAt: string | null): number => {
  if (!startedAt) return 0;
  const start = new Date(startedAt).getTime();
  return Math.floor((Date.now() - start) / 1000);
};

/**
 * Estimate remaining time based on progress.
 */
const getEstimatedTimeRemaining = (
  startedAt: string | null,
  progress: number
): number | null => {
  if (!startedAt || progress <= 0) return null;
  const elapsed = getElapsedTime(startedAt);
  const estimated = Math.floor((elapsed / progress) * (100 - progress));
  return estimated;
};

/**
 * Training job detail component with logs and stats.
 */
export default function TrainingJobDetail({ jobId, onClose }: TrainingJobDetailProps) {
  const logRef = useRef<HTMLTextAreaElement>(null);

  const { data: job, isLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => trainingJobApi.get(jobId),
    refetchInterval: (query) => {
      const jobData = query.state.data;
      return jobData?.status === 'running' ? 2000 : false;
    },
  });

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logRef.current && job?.log) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [job?.log]);

  if (isLoading || !job) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="card max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
          </div>
        </div>
      </div>
    );
  }

  const statusInfo = getStatusBadge(job.status);
  const StatusIcon = statusInfo.icon;
  const elapsed = getElapsedTime(job.started_at);
  const estimatedRemaining = getEstimatedTimeRemaining(job.started_at, job.progress);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="card max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={clsx('flex h-12 w-12 items-center justify-center rounded-lg', statusInfo.className.replace('badge-', 'bg-').replace('-', '-500/10'))}>
              {job.status === 'running' ? (
                <Loader2 className="h-6 w-6 animate-spin text-primary-400" />
              ) : (
                <StatusIcon className={clsx('h-6 w-6', statusInfo.color)} />
              )}
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">{job.name}</h2>
              <p className="text-sm text-surface-400">
                {job.description || 'No description'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-surface-400 hover:bg-surface-700 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Status */}
            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-1">
                <AlertCircle className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Status</span>
              </div>
              <span className={statusInfo.className}>{job.status}</span>
            </div>

            {/* Progress */}
            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-1">
                <TrendingDown className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Progress</span>
              </div>
              <p className="text-lg font-semibold text-white">{job.progress.toFixed(1)}%</p>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-surface-900">
                <div
                  className={clsx(
                    'h-full rounded-full transition-all',
                    job.status === 'completed'
                      ? 'bg-emerald-500'
                      : 'bg-gradient-to-r from-primary-500 to-accent-500'
                  )}
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            </div>

            {/* Loss */}
            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-1">
                <TrendingDown className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Loss</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {job.current_loss !== null ? job.current_loss.toFixed(4) : '-'}
              </p>
            </div>

            {/* Epochs */}
            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-1">
                <Play className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wide">Epochs</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {job.current_epoch} / {job.epochs}
              </p>
            </div>
          </div>

          {/* Time Information */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-2">
                <Calendar className="h-4 w-4" />
                <span className="text-sm font-medium">Started</span>
              </div>
              <p className="text-white">{formatDateTime(job.started_at)}</p>
            </div>

            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-2">
                <Timer className="h-4 w-4" />
                <span className="text-sm font-medium">Elapsed</span>
              </div>
              <p className="text-white">
                {elapsed > 0 ? formatDuration(elapsed) : '-'}
              </p>
            </div>

            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 text-surface-400 mb-2">
                <Clock className="h-4 w-4" />
                <span className="text-sm font-medium">Estimated Remaining</span>
              </div>
              <p className="text-white">
                {estimatedRemaining !== null && job.status === 'running'
                  ? formatDuration(estimatedRemaining)
                  : job.status === 'completed' ? 'Completed' : '-'}
              </p>
            </div>
          </div>

          {/* Training Parameters */}
          <div className="bg-surface-800/50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Training Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-surface-400">Type:</span>
                <p className="text-white font-medium mt-1">{job.training_type.toUpperCase()}</p>
              </div>
              <div>
                <span className="text-surface-400">Model:</span>
                <p className="text-white font-medium mt-1">{job.model_name}</p>
              </div>
              <div>
                <span className="text-surface-400">Batch Size:</span>
                <p className="text-white font-medium mt-1">{job.batch_size}</p>
              </div>
              <div>
                <span className="text-surface-400">Learning Rate:</span>
                <p className="text-white font-medium mt-1">{job.learning_rate}</p>
              </div>
              <div>
                <span className="text-surface-400">LoRA R:</span>
                <p className="text-white font-medium mt-1">{job.lora_r}</p>
              </div>
              <div>
                <span className="text-surface-400">LoRA Alpha:</span>
                <p className="text-white font-medium mt-1">{job.lora_alpha}</p>
              </div>
              <div>
                <span className="text-surface-400">LoRA Dropout:</span>
                <p className="text-white font-medium mt-1">{job.lora_dropout}</p>
              </div>
              <div>
                <span className="text-surface-400">Worker:</span>
                <p className="text-white font-medium mt-1">{job.worker_id || '-'}</p>
              </div>
            </div>
          </div>

          {/* Model Path & Download */}
          {job.model_path && (
            <div className="bg-surface-800/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-surface-400">
                  <HardDrive className="h-4 w-4" />
                  <span className="text-sm font-medium">Model Output</span>
                </div>
                {job.status === 'completed' && (
                  <DownloadButton jobId={job.id} jobName={job.name} />
                )}
              </div>
              <p className="text-white font-mono text-sm break-all">{job.model_path}</p>
            </div>
          )}

          {/* Training Logs */}
          <div className="bg-surface-800/50 rounded-lg p-4">
            <div className="flex items-center gap-2 text-surface-400 mb-4">
              <FileText className="h-4 w-4" />
              <span className="text-sm font-medium">Training Log</span>
            </div>
            <textarea
              ref={logRef}
              readOnly
              value={job.log || 'No logs available yet...'}
              className="w-full h-64 bg-surface-900 text-surface-200 font-mono text-sm p-4 rounded-lg border border-surface-700 resize-none focus:outline-none focus:ring-2 focus:ring-primary-500/50"
              style={{ scrollBehavior: 'smooth' }}
            />
          </div>

          {/* Error Message */}
          {job.status === 'failed' && job.error_message && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <div className="flex items-center gap-2 text-red-400 mb-2">
                <XCircle className="h-4 w-4" />
                <span className="text-sm font-medium">Error</span>
              </div>
              <p className="text-red-400">{job.error_message}</p>
            </div>
          )}

          {/* Completion Info */}
          {job.status === 'completed' && job.completed_at && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2 text-emerald-400 mb-2">
                    <CheckCircle2 className="h-4 w-4" />
                    <span className="text-sm font-medium">Completed</span>
                  </div>
                  <p className="text-emerald-400">{formatDateTime(job.completed_at)}</p>
                </div>
                {job.model_path && (
                  <DownloadButton jobId={job.id} jobName={job.name} variant="large" />
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Download button component for completed jobs.
 */
function DownloadButton({ 
  jobId, 
  jobName, 
  variant = 'default' 
}: { 
  jobId: number; 
  jobName: string;
  variant?: 'default' | 'large';
}) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    setIsDownloading(true);
    setError(null);
    
    try {
      const blob = await trainingJobApi.download(jobId);
      
      // Get filename from content-disposition or generate one
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
      setError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setIsDownloading(false);
    }
  };

  if (variant === 'large') {
    return (
      <div className="flex flex-col items-end">
        <button
          onClick={handleDownload}
          disabled={isDownloading}
          className="btn-primary flex items-center gap-2"
        >
          {isDownloading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Download className="h-4 w-4" />
          )}
          {isDownloading ? 'Downloading...' : 'Download Model'}
        </button>
        {error && (
          <p className="text-red-400 text-xs mt-1">{error}</p>
        )}
      </div>
    );
  }

  return (
    <button
      onClick={handleDownload}
      disabled={isDownloading}
      className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-primary-400 hover:text-primary-300 bg-primary-500/10 hover:bg-primary-500/20 rounded-lg transition-colors disabled:opacity-50"
      title={error || 'Download model'}
    >
      {isDownloading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <Download className="h-4 w-4" />
      )}
      <span>{isDownloading ? 'Downloading...' : 'Download'}</span>
    </button>
  );
}

