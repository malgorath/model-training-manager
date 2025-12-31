import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Play, Clock, CheckCircle2, XCircle, Loader2, AlertCircle, Trash2, StopCircle } from 'lucide-react';
import { projectApi } from '../services/api';
import type { ProjectStatus } from '../types';

const statusConfig: Record<ProjectStatus, { icon: React.ReactNode; color: string; bgColor: string }> = {
  pending: { icon: <Clock className="h-4 w-4" />, color: 'text-yellow-400', bgColor: 'bg-yellow-900/20' },
  running: { icon: <Loader2 className="h-4 w-4 animate-spin" />, color: 'text-blue-400', bgColor: 'bg-blue-900/20' },
  completed: { icon: <CheckCircle2 className="h-4 w-4" />, color: 'text-green-400', bgColor: 'bg-green-900/20' },
  failed: { icon: <XCircle className="h-4 w-4" />, color: 'text-red-400', bgColor: 'bg-red-900/20' },
  cancelled: { icon: <AlertCircle className="h-4 w-4" />, color: 'text-gray-400', bgColor: 'bg-gray-900/20' },
};

export default function ProjectList() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ['projects', 1],
    queryFn: () => projectApi.list(1, 50),
    refetchOnMount: true,
    refetchOnWindowFocus: false,
  });

  const deleteMutation = useMutation({
    mutationFn: projectApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });

  const startMutation = useMutation({
    mutationFn: projectApi.start,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });

  const cancelMutation = useMutation({
    mutationFn: projectApi.cancel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });

  const handleDelete = (e: React.MouseEvent, projectId: number, projectName: string) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to delete project "${projectName}"? This action cannot be undone.`)) {
      deleteMutation.mutate(projectId);
    }
  };

  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary-400" />
        </div>
      </div>
    );
  }

  const projects = data?.items || [];

  return (
    <div className="card">
      <div className="border-b border-surface-800 px-6 py-4">
        <h2 className="text-xl font-semibold text-white">Projects</h2>
        <p className="text-sm text-surface-400 mt-1">Manage training projects with traits and dataset allocations</p>
      </div>

      {projects.length === 0 ? (
        <div className="p-12 text-center">
          <p className="text-surface-400">No projects yet. Create your first project to get started.</p>
        </div>
      ) : (
        <div className="divide-y divide-surface-800">
          {projects.map((project) => {
            const statusInfo = statusConfig[project.status];
            return (
              <div
                key={project.id}
                className="px-6 py-4 hover:bg-surface-800/50 cursor-pointer transition-colors"
                onClick={() => navigate(`/projects/${project.id}`)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <h3 className="text-lg font-medium text-white">{project.name}</h3>
                      <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded ${statusInfo.bgColor} ${statusInfo.color}`}>
                        {statusInfo.icon}
                        <span className="text-xs font-medium capitalize">{project.status}</span>
                      </div>
                    </div>
                    {project.description && (
                      <p className="text-sm text-surface-400 mt-1">{project.description}</p>
                    )}
                    <div className="flex items-center gap-4 mt-2 text-sm text-surface-500">
                      <span>Model: {project.base_model}</span>
                      <span>Type: {project.training_type}</span>
                      <span>Max Rows: {project.max_rows != null ? project.max_rows.toLocaleString() : 'N/A'}</span>
                      <span>Traits: {project.traits.length}</span>
                    </div>
                    {project.status === 'running' && (
                      <div className="mt-3">
                        <div className="flex items-center justify-between text-sm mb-1">
                          <span className="text-surface-400">Progress</span>
                          <span className="text-surface-300">{project.progress.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary-500 transition-all"
                            style={{ width: `${project.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {project.error_message && (
                      <p className="text-sm text-red-400 mt-2">{project.error_message}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {project.status === 'pending' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          startMutation.mutate(project.id);
                        }}
                        disabled={startMutation.isPending}
                        className="btn-primary flex items-center gap-2"
                      >
                        {startMutation.isPending ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Starting...
                          </>
                        ) : (
                          <>
                            <Play className="h-4 w-4" />
                            Start
                          </>
                        )}
                      </button>
                    )}
                    {(project.status === 'running' || project.status === 'pending') && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          cancelMutation.mutate(project.id);
                        }}
                        disabled={cancelMutation.isPending}
                        className="btn-secondary flex items-center gap-2 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                        title="Cancel project"
                      >
                        {cancelMutation.isPending ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Cancelling...
                          </>
                        ) : (
                          <>
                            <StopCircle className="h-4 w-4" />
                            Cancel
                          </>
                        )}
                      </button>
                    )}
                    {project.status !== 'running' && project.status !== 'pending' && (
                      <button
                        onClick={(e) => handleDelete(e, project.id, project.name)}
                        disabled={deleteMutation.isPending}
                        className="btn-secondary flex items-center gap-2 text-red-400 hover:text-red-300 hover:bg-red-500/10"
                        title="Delete project"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
