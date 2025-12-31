import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, CheckCircle2, XCircle, Clock, Loader2, AlertCircle, Download, RotateCw, Gauge, StopCircle } from 'lucide-react';
import { projectApi } from '../services/api';
import type { ProjectStatus } from '../types';

const statusConfig: Record<ProjectStatus, { icon: React.ReactNode; color: string; bgColor: string }> = {
  pending: { icon: <Clock className="h-5 w-5" />, color: 'text-yellow-400', bgColor: 'bg-yellow-900/20' },
  running: { icon: <Loader2 className="h-5 w-5 animate-spin" />, color: 'text-blue-400', bgColor: 'bg-blue-900/20' },
  completed: { icon: <CheckCircle2 className="h-5 w-5" />, color: 'text-green-400', bgColor: 'bg-green-900/20' },
  failed: { icon: <XCircle className="h-5 w-5" />, color: 'text-red-400', bgColor: 'bg-red-900/20' },
  cancelled: { icon: <AlertCircle className="h-5 w-5" />, color: 'text-gray-400', bgColor: 'bg-gray-900/20' },
};

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: project, isLoading } = useQuery({
    queryKey: ['projects', id],
    queryFn: async () => {
      if (!id) return null;
      try {
        const result = await projectApi.get(parseInt(id));
        return result || null;
      } catch (error) {
        console.error('Error fetching project:', error);
        return null;
      }
    },
    enabled: !!id,
    retry: false,
  });

  const startMutation = useMutation({
    mutationFn: () => projectApi.start(parseInt(id!)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => projectApi.cancel(parseInt(id!)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.invalidateQueries({ queryKey: ['projects', id] });
    },
  });

  const retryMutation = useMutation({
    mutationFn: () => projectApi.retry(parseInt(id!)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.invalidateQueries({ queryKey: ['projects', id] });
    },
  });

  const validateMutation = useMutation({
    mutationFn: () => projectApi.validate(parseInt(id!)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects', id] });
    },
  });

  if (isLoading || !project) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary-400" />
        </div>
      </div>
    );
  }

  // Calculate speed stats from project data
  // Only calculate if project exists and has started_at
  const calculateSpeedStats = (): {
    elapsed: number;
    progressPerSecond: number;
    estimatedTotalTime: number;
    estimatedRemainingTime: number;
    samplesPerSecond: number | null;
  } | null => {
    // Double-check project exists (defensive programming)
    if (!project) return null;
    if (!project.started_at) return null;
    
    try {
      const now = new Date();
      const startTime = new Date(project.started_at);
      const elapsed = (now.getTime() - startTime.getTime()) / 1000; // seconds
      
      if (elapsed <= 0 || project.progress <= 0) return null;
      
      // Calculate rates
      const progressPerSecond = project.progress / elapsed;
      const estimatedTotalTime = elapsed / (project.progress / 100);
      const estimatedRemainingTime = estimatedTotalTime - elapsed;
      
      // Calculate samples per second (if we have epoch info)
      // This is an approximation - actual calculation would need batch size and dataset size
      const samplesPerSecond = progressPerSecond > 0 ? (progressPerSecond * 1000) : null; // Rough estimate
      
      return {
        elapsed,
        progressPerSecond,
        estimatedTotalTime,
        estimatedRemainingTime,
        samplesPerSecond,
      };
    } catch (error) {
      // If any error occurs, return null to prevent crashes
      console.error('Error calculating speed stats:', error);
      return null;
    }
  };

  const speedStats = calculateSpeedStats();

  const statusInfo = statusConfig[project.status];

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="border-b border-surface-800 px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/projects')}
              className="p-2 hover:bg-surface-800 rounded-lg transition-colors"
            >
              <ArrowLeft className="h-5 w-5 text-surface-400" />
            </button>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-white">{project.name}</h1>
              {project.description && (
                <p className="text-surface-400 mt-1">{project.description}</p>
              )}
            </div>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${statusInfo.bgColor} ${statusInfo.color}`}>
              {statusInfo.icon}
              <span className="font-medium capitalize">{project.status}</span>
            </div>
            {project.status === 'pending' && (
              <button
                onClick={() => startMutation.mutate()}
                className="btn-primary flex items-center gap-2"
                disabled={startMutation.isPending}
              >
                <Play className="h-4 w-4" />
                Start Training
              </button>
            )}
            {(project.status === 'running' || project.status === 'pending') && (
              <button
                onClick={() => cancelMutation.mutate()}
                className="btn-secondary flex items-center gap-2 text-red-400 hover:text-red-300"
                disabled={cancelMutation.isPending}
              >
                <StopCircle className="h-4 w-4" />
                {cancelMutation.isPending ? 'Cancelling...' : 'Cancel'}
              </button>
            )}
            {project.status === 'failed' && (
              <button
                onClick={() => retryMutation.mutate()}
                className="btn-primary flex items-center gap-2"
                disabled={retryMutation.isPending}
              >
                <RotateCw className="h-4 w-4" />
                {retryMutation.isPending ? 'Retrying...' : 'Retry'}
              </button>
            )}
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Project Info */}
          <div className="grid grid-cols-2 gap-6">
            <div>
              <label className="label">Base Model</label>
              <p className="text-surface-300">{project.base_model}</p>
            </div>
            <div>
              <label className="label">Training Type</label>
              <p className="text-surface-300 capitalize">{project.training_type}</p>
            </div>
            <div>
              <label className="label">Max Rows</label>
              <p className="text-surface-300">{project.max_rows != null ? project.max_rows.toLocaleString() : 'N/A'}</p>
            </div>
            <div>
              <label className="label">Output Directory</label>
              <p className="text-surface-300 font-mono text-sm">{project.output_directory}</p>
            </div>
          </div>

          {/* Progress */}
          {project.status === 'running' && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="label">Training Progress</label>
                <span className="text-surface-300">{project.progress.toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-surface-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 transition-all"
                  style={{ width: `${project.progress}%` }}
                />
              </div>
              {project.current_loss != null && (
                <p className="text-sm text-surface-400 mt-2">Loss: {project.current_loss.toFixed(4)}</p>
              )}
            </div>
          )}

          {/* Traits */}
          <div>
            <label className="label mb-4">Traits & Dataset Allocations</label>
            <div className="space-y-4">
              {project.traits.map((trait) => (
                <div key={trait.id} className="border border-surface-700 rounded-lg p-4">
                  <h4 className="font-semibold text-white capitalize mb-3">{trait.trait_type.replace('_', '/')}</h4>
                  <div className="space-y-2">
                    {trait.datasets.map((alloc, idx) => (
                      <div key={idx} className="flex items-center justify-between text-sm">
                        <span className="text-surface-300">Dataset {alloc.dataset_id}</span>
                        <span className="text-surface-400">{alloc.percentage}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Error Message */}
          {project.error_message && (
            <div className="p-4 bg-red-900/20 border border-red-800 rounded-lg">
              <p className="text-sm font-medium text-red-400 mb-1">Error</p>
              <p className="text-sm text-surface-300">{project.error_message}</p>
            </div>
          )}

          {/* Completed Actions */}
          {project.status === 'completed' && project.model_path && (
            <div className="border-t border-surface-800 pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <label className="label">Model Path</label>
                  <p className="text-surface-300 font-mono text-sm">{project.model_path}</p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => validateMutation.mutate()}
                    className="btn-secondary"
                    disabled={validateMutation.isPending}
                  >
                    Validate Model
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Speed Stats */}
          {speedStats && (project.status === 'running' || project.status === 'completed') && (
            <div>
              <label className="label mb-2 flex items-center gap-2">
                <Gauge className="h-4 w-4" />
                Training Speed Stats
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-surface-800/50 rounded-lg p-3 border border-surface-700">
                  <p className="text-xs text-surface-400 mb-1">Elapsed Time</p>
                  <p className="text-sm font-semibold text-white">
                    {speedStats.elapsed >= 3600
                      ? `${Math.floor(speedStats.elapsed / 3600)}h ${Math.floor((speedStats.elapsed % 3600) / 60)}m`
                      : speedStats.elapsed >= 60
                      ? `${Math.floor(speedStats.elapsed / 60)}m ${Math.floor(speedStats.elapsed % 60)}s`
                      : `${Math.floor(speedStats.elapsed)}s`}
                  </p>
                </div>
                <div className="bg-surface-800/50 rounded-lg p-3 border border-surface-700">
                  <p className="text-xs text-surface-400 mb-1">Progress Rate</p>
                  <p className="text-sm font-semibold text-white">
                    {speedStats.progressPerSecond.toFixed(3)}%/s
                  </p>
                </div>
                {project.status === 'running' && speedStats.estimatedRemainingTime > 0 && (
                  <div className="bg-surface-800/50 rounded-lg p-3 border border-surface-700">
                    <p className="text-xs text-surface-400 mb-1">Est. Remaining</p>
                    <p className="text-sm font-semibold text-white">
                      {speedStats.estimatedRemainingTime >= 3600
                        ? `${Math.floor(speedStats.estimatedRemainingTime / 3600)}h ${Math.floor((speedStats.estimatedRemainingTime % 3600) / 60)}m`
                        : speedStats.estimatedRemainingTime >= 60
                        ? `${Math.floor(speedStats.estimatedRemainingTime / 60)}m ${Math.floor(speedStats.estimatedRemainingTime % 60)}s`
                        : `${Math.floor(speedStats.estimatedRemainingTime)}s`}
                    </p>
                  </div>
                )}
                {project.status === 'completed' && project.completed_at && project.started_at && (
                  <div className="bg-surface-800/50 rounded-lg p-3 border border-surface-700">
                    <p className="text-xs text-surface-400 mb-1">Total Duration</p>
                    <p className="text-sm font-semibold text-white">
                      {(() => {
                        const start = new Date(project.started_at);
                        const end = new Date(project.completed_at!);
                        const duration = (end.getTime() - start.getTime()) / 1000;
                        if (duration >= 3600) {
                          return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
                        } else if (duration >= 60) {
                          return `${Math.floor(duration / 60)}m ${Math.floor(duration % 60)}s`;
                        } else {
                          return `${Math.floor(duration)}s`;
                        }
                      })()}
                    </p>
                  </div>
                )}
                {speedStats.samplesPerSecond && (
                  <div className="bg-surface-800/50 rounded-lg p-3 border border-surface-700">
                    <p className="text-xs text-surface-400 mb-1">Est. Samples/s</p>
                    <p className="text-sm font-semibold text-white">
                      {speedStats.samplesPerSecond.toFixed(1)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Logs */}
          {project.log && (
            <div>
              <label className="label mb-2">Training Logs</label>
              <div className="bg-surface-950 border border-surface-700 rounded-lg p-4 max-h-96 overflow-y-auto">
                <pre className="text-xs text-surface-400 font-mono whitespace-pre-wrap">{project.log}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
