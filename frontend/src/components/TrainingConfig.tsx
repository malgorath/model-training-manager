import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Settings, Save, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { configApi } from '../services/api';
import type { TrainingConfigUpdate, TrainingType, ModelProvider } from '../types';

/**
 * Training configuration component.
 */
export default function TrainingConfig() {
  const queryClient = useQueryClient();

  const { data: config, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: configApi.get,
  });

  const [formData, setFormData] = useState<TrainingConfigUpdate>({});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (config) {
      setFormData({
        max_concurrent_workers: config.max_concurrent_workers,
        default_model: config.default_model,
        default_training_type: config.default_training_type,
        default_batch_size: config.default_batch_size,
        default_learning_rate: config.default_learning_rate,
        default_epochs: config.default_epochs,
        default_lora_r: config.default_lora_r,
        default_lora_alpha: config.default_lora_alpha,
        default_lora_dropout: config.default_lora_dropout,
        auto_start_workers: config.auto_start_workers,
        model_provider: config.model_provider,
        model_api_url: config.model_api_url,
        output_directory_base: config.output_directory_base,
        model_cache_path: config.model_cache_path,
      });
    }
  }, [config]);

  const updateMutation = useMutation({
    mutationFn: configApi.update,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
      setHasChanges(false);
    },
  });

  const updateField = <K extends keyof TrainingConfigUpdate>(
    field: K,
    value: TrainingConfigUpdate[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setHasChanges(true);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateMutation.mutate(formData);
  };

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
    <form onSubmit={handleSubmit} className="card">
      <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/10">
            <Settings className="h-5 w-5 text-primary-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Training Configuration</h3>
            <p className="text-sm text-surface-400">
              Default settings for training jobs
            </p>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="space-y-6">
          {/* Worker Settings */}
          <div>
            <h4 className="mb-4 text-sm font-medium uppercase tracking-wider text-surface-400">
              Worker Settings
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="max_workers" className="label">
                  Max Concurrent Workers
                </label>
                <input
                  type="number"
                  id="max_workers"
                  value={formData.max_concurrent_workers ?? 4}
                  onChange={(e) =>
                    updateField('max_concurrent_workers', Number(e.target.value))
                  }
                  min={1}
                  max={32}
                  className="input"
                />
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={formData.auto_start_workers ?? false}
                    onChange={(e) => updateField('auto_start_workers', e.target.checked)}
                    className="h-4 w-4 rounded border-surface-600 bg-surface-800 text-primary-500 focus:ring-primary-500"
                  />
                  <span className="text-sm text-surface-300">
                    Auto-start workers on startup
                  </span>
                </label>
              </div>
            </div>
          </div>

          {/* Model Settings */}
          <div>
            <h4 className="mb-4 text-sm font-medium uppercase tracking-wider text-surface-400">
              Model Settings
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="default_model" className="label">
                  Default Model
                </label>
                <input
                  type="text"
                  id="default_model"
                  value={formData.default_model ?? 'llama3.2:3b'}
                  onChange={(e) => updateField('default_model', e.target.value)}
                  className="input"
                />
              </div>
              <div>
                <label htmlFor="default_type" className="label">
                  Default Training Type
                </label>
                <select
                  id="default_type"
                  value={formData.default_training_type ?? 'qlora'}
                  onChange={(e) =>
                    updateField('default_training_type', e.target.value as TrainingType)
                  }
                  className="input"
                >
                  <option value="qlora">QLoRA</option>
                  <option value="rag">RAG</option>
                  <option value="standard">Standard</option>
                </select>
              </div>
            </div>
          </div>

          {/* Model API Settings */}
          <div>
            <h4 className="mb-4 text-sm font-medium uppercase tracking-wider text-surface-400">
              Model API Settings
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="model_provider" className="label">
                  Model Provider
                </label>
                <select
                  id="model_provider"
                  value={formData.model_provider ?? 'ollama'}
                  onChange={(e) => {
                    const provider = e.target.value as ModelProvider;
                    updateField('model_provider', provider);
                    // Update default URL when provider changes
                    if (provider === 'ollama' && (!formData.model_api_url || formData.model_api_url.includes('1234'))) {
                      updateField('model_api_url', 'http://localhost:11434');
                    } else if (provider === 'lm_studio' && (!formData.model_api_url || formData.model_api_url.includes('11434'))) {
                      updateField('model_api_url', 'http://localhost:1234');
                    }
                  }}
                  className="input"
                >
                  <option value="ollama">Ollama</option>
                  <option value="lm_studio">LM Studio</option>
                </select>
              </div>
              <div>
                <label htmlFor="model_api_url" className="label">
                  Model API URL
                </label>
                <input
                  type="text"
                  id="model_api_url"
                  value={formData.model_api_url ?? (formData.model_provider === 'lm_studio' ? 'http://localhost:1234' : 'http://localhost:11434')}
                  onChange={(e) => updateField('model_api_url', e.target.value)}
                  placeholder={
                    formData.model_provider === 'lm_studio'
                      ? 'http://localhost:1234'
                      : 'http://localhost:11434'
                  }
                  className="input"
                />
                <p className="mt-1 text-xs text-surface-500">
                  {formData.model_provider === 'lm_studio'
                    ? 'LM Studio typically runs on port 1234'
                    : 'Ollama typically runs on port 11434'}
                </p>
              </div>
            </div>
          </div>

          {/* Training Parameters */}
          <div>
            <h4 className="mb-4 text-sm font-medium uppercase tracking-wider text-surface-400">
              Default Training Parameters
            </h4>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label htmlFor="default_batch_size" className="label">
                  Batch Size
                </label>
                <input
                  type="number"
                  id="default_batch_size"
                  value={formData.default_batch_size ?? 4}
                  onChange={(e) =>
                    updateField('default_batch_size', Number(e.target.value))
                  }
                  min={1}
                  max={64}
                  className="input"
                />
              </div>
              <div>
                <label htmlFor="default_learning_rate" className="label">
                  Learning Rate
                </label>
                <input
                  type="number"
                  id="default_learning_rate"
                  value={formData.default_learning_rate ?? 0.0002}
                  onChange={(e) =>
                    updateField('default_learning_rate', Number(e.target.value))
                  }
                  step={0.0001}
                  min={0.000001}
                  max={1}
                  className="input"
                />
              </div>
              <div>
                <label htmlFor="default_epochs" className="label">
                  Epochs
                </label>
                <input
                  type="number"
                  id="default_epochs"
                  value={formData.default_epochs ?? 3}
                  onChange={(e) =>
                    updateField('default_epochs', Number(e.target.value))
                  }
                  min={1}
                  max={100}
                  className="input"
                />
              </div>
            </div>
          </div>

          {/* LoRA Parameters */}
          <div>
            <h4 className="mb-4 text-sm font-medium uppercase tracking-wider text-surface-400">
              Default LoRA Parameters
            </h4>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label htmlFor="default_lora_r" className="label">
                  LoRA Rank (r)
                </label>
                <input
                  type="number"
                  id="default_lora_r"
                  value={formData.default_lora_r ?? 16}
                  onChange={(e) =>
                    updateField('default_lora_r', Number(e.target.value))
                  }
                  min={1}
                  max={256}
                  className="input"
                />
              </div>
              <div>
                <label htmlFor="default_lora_alpha" className="label">
                  LoRA Alpha
                </label>
                <input
                  type="number"
                  id="default_lora_alpha"
                  value={formData.default_lora_alpha ?? 32}
                  onChange={(e) =>
                    updateField('default_lora_alpha', Number(e.target.value))
                  }
                  min={1}
                  max={512}
                  className="input"
                />
              </div>
              <div>
                <label htmlFor="default_lora_dropout" className="label">
                  LoRA Dropout
                </label>
                <input
                  type="number"
                  id="default_lora_dropout"
                  value={formData.default_lora_dropout ?? 0.05}
                  onChange={(e) =>
                    updateField('default_lora_dropout', Number(e.target.value))
                  }
                  step={0.01}
                  min={0}
                  max={1}
                  className="input"
                />
              </div>
            </div>
            </div>
          </div>

          {/* Directory Settings */}
          <div>
            <h4 className="mb-4 text-sm font-medium uppercase tracking-wider text-surface-400">
              Directory Settings
            </h4>
            <div className="space-y-4">
              <div>
                <label htmlFor="output_directory_base" className="label">
                  Output Directory Base
                </label>
                <input
                  type="text"
                  id="output_directory_base"
                  value={formData.output_directory_base ?? ''}
                  onChange={(e) => updateField('output_directory_base', e.target.value || undefined)}
                  placeholder="/output (base directory for project outputs)"
                  className="input"
                />
                <p className="mt-1 text-xs text-surface-500">
                  Base directory where trained models will be saved. Individual projects will create subdirectories here.
                </p>
              </div>
              <div>
                <label htmlFor="model_cache_path" className="label">
                  Model Cache Path
                </label>
                <input
                  type="text"
                  id="model_cache_path"
                  value={formData.model_cache_path ?? ''}
                  onChange={(e) => updateField('model_cache_path', e.target.value || undefined)}
                  placeholder="~/.cache/huggingface/hub (leave empty for default)"
                  className="input"
                />
                <p className="mt-1 text-xs text-surface-500">
                  Base path for HuggingFace model cache. Leave empty to use default location (~/.cache/huggingface/hub).
                </p>
              </div>
            </div>
          </div>

          {/* Messages */}
        {updateMutation.isError && (
          <div className="mt-6 flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
            <AlertCircle className="h-5 w-5" />
            {updateMutation.error?.message || 'Failed to save configuration'}
          </div>
        )}

        {updateMutation.isSuccess && (
          <div className="mt-6 flex items-center gap-2 rounded-lg bg-emerald-500/10 px-4 py-3 text-sm text-emerald-400">
            <CheckCircle2 className="h-5 w-5" />
            Configuration saved successfully!
          </div>
        )}

        {/* Submit Button */}
        <div className="mt-6 flex justify-end">
          <button
            type="submit"
            disabled={!hasChanges || updateMutation.isPending}
            className="btn-primary"
          >
            {updateMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4" />
                Save Changes
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}

