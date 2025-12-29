import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Play, X, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { datasetApi, trainingJobApi, configApi } from '../services/api';
import type { TrainingType } from '../types';

interface TrainingJobFormProps {
  onSuccess?: () => void;
  onClose?: () => void;
}

/**
 * Training job creation form component.
 */
export default function TrainingJobForm({ onSuccess, onClose }: TrainingJobFormProps) {
  const queryClient = useQueryClient();

  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: configApi.get,
  });

  const { data: datasets } = useQuery({
    queryKey: ['datasets', 1],
    queryFn: () => datasetApi.list(1, 100),
  });

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    dataset_id: 0,
    training_type: 'qlora' as TrainingType,
    batch_size: config?.default_batch_size ?? 4,
    learning_rate: config?.default_learning_rate ?? 0.0002,
    epochs: config?.default_epochs ?? 3,
    lora_r: config?.default_lora_r ?? 16,
    lora_alpha: config?.default_lora_alpha ?? 32,
    lora_dropout: config?.default_lora_dropout ?? 0.05,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const createMutation = useMutation({
    mutationFn: trainingJobApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      onSuccess?.();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate(formData);
  };

  const updateField = <K extends keyof typeof formData>(
    field: K,
    value: (typeof formData)[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="card max-w-lg">
      <div className="border-b border-surface-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Create Training Job</h3>
          {onClose && (
            <button
              onClick={onClose}
              className="rounded-lg p-2 text-surface-400 hover:bg-surface-800 hover:text-surface-100"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="p-6">
        <div className="space-y-4">
          {/* Job Name */}
          <div>
            <label htmlFor="name" className="label">
              Job Name
            </label>
            <input
              type="text"
              id="name"
              value={formData.name}
              onChange={(e) => updateField('name', e.target.value)}
              placeholder="Enter job name"
              className="input"
              required
            />
          </div>

          {/* Description */}
          <div>
            <label htmlFor="description" className="label">
              Description (optional)
            </label>
            <textarea
              id="description"
              value={formData.description}
              onChange={(e) => updateField('description', e.target.value)}
              placeholder="Enter job description"
              rows={2}
              className="input resize-none"
            />
          </div>

          {/* Dataset Selection */}
          <div>
            <label htmlFor="dataset" className="label">
              Training Dataset
            </label>
            <select
              id="dataset"
              value={formData.dataset_id}
              onChange={(e) => updateField('dataset_id', Number(e.target.value))}
              className="input"
              required
            >
              <option value={0}>Select a dataset</option>
              {datasets?.items.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.row_count} rows)
                </option>
              ))}
            </select>
          </div>

          {/* Training Type */}
          <div>
            <label htmlFor="type" className="label">
              Training Type
            </label>
            <select
              id="type"
              value={formData.training_type}
              onChange={(e) => updateField('training_type', e.target.value as TrainingType)}
              className="input"
            >
              <option value="qlora">QLoRA (Quantized LoRA)</option>
              <option value="unsloth">Unsloth (Optimized LoRA)</option>
              <option value="rag">RAG (Retrieval-Augmented)</option>
              <option value="standard">Standard Fine-tuning</option>
            </select>
          </div>

          {/* Basic Parameters */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="batch_size" className="label">
                Batch Size
              </label>
              <input
                type="number"
                id="batch_size"
                value={formData.batch_size}
                onChange={(e) => updateField('batch_size', Number(e.target.value))}
                min={1}
                max={64}
                className="input"
              />
            </div>
            <div>
              <label htmlFor="epochs" className="label">
                Epochs
              </label>
              <input
                type="number"
                id="epochs"
                value={formData.epochs}
                onChange={(e) => updateField('epochs', Number(e.target.value))}
                min={1}
                max={100}
                className="input"
              />
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-primary-400 hover:text-primary-300"
          >
            {showAdvanced ? 'Hide' : 'Show'} advanced settings
          </button>

          {/* Advanced Parameters */}
          {showAdvanced && (
            <div className="space-y-4 rounded-lg border border-surface-800 bg-surface-800/30 p-4">
              <div>
                <label htmlFor="learning_rate" className="label">
                  Learning Rate
                </label>
                <input
                  type="number"
                  id="learning_rate"
                  value={formData.learning_rate}
                  onChange={(e) => updateField('learning_rate', Number(e.target.value))}
                  step={0.0001}
                  min={0.000001}
                  max={1}
                  className="input"
                />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label htmlFor="lora_r" className="label">
                    LoRA Rank
                  </label>
                  <input
                    type="number"
                    id="lora_r"
                    value={formData.lora_r}
                    onChange={(e) => updateField('lora_r', Number(e.target.value))}
                    min={1}
                    max={256}
                    className="input"
                  />
                </div>
                <div>
                  <label htmlFor="lora_alpha" className="label">
                    LoRA Alpha
                  </label>
                  <input
                    type="number"
                    id="lora_alpha"
                    value={formData.lora_alpha}
                    onChange={(e) => updateField('lora_alpha', Number(e.target.value))}
                    min={1}
                    max={512}
                    className="input"
                  />
                </div>
                <div>
                  <label htmlFor="lora_dropout" className="label">
                    LoRA Dropout
                  </label>
                  <input
                    type="number"
                    id="lora_dropout"
                    value={formData.lora_dropout}
                    onChange={(e) => updateField('lora_dropout', Number(e.target.value))}
                    step={0.01}
                    min={0}
                    max={1}
                    className="input"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Error Message */}
        {createMutation.isError && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
            <AlertCircle className="h-5 w-5" />
            {createMutation.error?.message || 'Failed to create job'}
          </div>
        )}

        {/* Success Message */}
        {createMutation.isSuccess && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-emerald-500/10 px-4 py-3 text-sm text-emerald-400">
            <CheckCircle2 className="h-5 w-5" />
            Training job created successfully!
          </div>
        )}

        {/* Submit Button */}
        <div className="mt-6 flex justify-end gap-3">
          {onClose && (
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
          )}
          <button
            type="submit"
            disabled={!formData.name || !formData.dataset_id || createMutation.isPending}
            className="btn-primary"
          >
            {createMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Create Job
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

