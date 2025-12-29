import { useQuery } from '@tanstack/react-query';
import { Server, Loader2 } from 'lucide-react';
import TrainingConfig from '../components/TrainingConfig';
import { healthApi } from '../services/api';

/**
 * Settings page component.
 */
export default function SettingsPage() {
  const { data: health, isLoading: loadingHealth } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    retry: false,
  });

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="mt-2 text-surface-400">
          Configure training parameters and system settings
        </p>
      </div>

      {/* Content */}
      <div className="grid grid-cols-3 gap-8">
        {/* Training Config */}
        <div className="col-span-2">
          <TrainingConfig />
        </div>

        {/* System Info */}
        <div className="space-y-6">
          {/* Health Status */}
          <div className="card">
            <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-500/10">
                  <Server className="h-5 w-5 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">System Status</h3>
                  <p className="text-sm text-surface-400">Backend service health</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              {loadingHealth ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-6 w-6 animate-spin text-primary-500" />
                </div>
              ) : health ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-surface-400">Status</span>
                    <span className="badge-success">Healthy</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-surface-400">Version</span>
                    <span className="text-white">{health.version}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-surface-400">Environment</span>
                    <span className="text-white capitalize">{health.environment}</span>
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <p className="text-red-400">Unable to connect to backend</p>
                </div>
              )}
            </div>
          </div>

          {/* About */}
          <div className="card p-6">
            <h4 className="text-lg font-semibold text-white">About</h4>
            <p className="mt-2 text-sm text-surface-400">
              Model Training Manager is a professional-grade application for
              managing model training with support for QLoRA, RAG, and standard
              training methods.
            </p>
            <div className="mt-4 rounded-lg bg-surface-800/50 p-4">
              <p className="text-xs text-surface-500">
                Powered by Ollama • llama3.2:3b
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

