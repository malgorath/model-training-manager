import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { X, Loader2, AlertCircle, CheckCircle2, Info, Plus, Trash2, HelpCircle } from 'lucide-react';
import { projectApi, datasetApi, configApi } from '../services/api';
import type { ProjectCreate, TraitType, DatasetAllocation, TrainingType } from '../types';
import { HelpTooltip } from './Tutorial';

interface ProjectFormProps {
  onSuccess?: () => void;
  onClose?: () => void;
}

type TraitConfig = {
  trait_type: TraitType;
  datasets: DatasetAllocation[];
};

/**
 * Multi-step project creation form with trait configuration.
 */
export default function ProjectForm({ onSuccess, onClose }: ProjectFormProps) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState<Partial<ProjectCreate>>({
    name: '',
    description: '',
    base_model: '',
    training_type: 'qlora',
    max_rows: 50000,
    output_directory: '',
    traits: [],
  });

  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: configApi.get,
  });

  const { data: datasetsData } = useQuery({
    queryKey: ['datasets', 1],
    queryFn: () => datasetApi.list(1, 100),
  });

  const datasets = datasetsData?.items || [];

  const [traits, setTraits] = useState<TraitConfig[]>([]);
  const [outputDirValid, setOutputDirValid] = useState<boolean | null>(null);
  const [modelValid, setModelValid] = useState<boolean | null>(null);
  const [validatingDir, setValidatingDir] = useState(false);
  const [validatingModel, setValidatingModel] = useState(false);

  useEffect(() => {
    if (config?.default_model) {
      setFormData(prev => ({ ...prev, base_model: config.default_model }));
    }
  }, [config]);

  const createMutation = useMutation({
    mutationFn: projectApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      onSuccess?.();
    },
  });

  const validateOutputDir = async (path: string) => {
    if (!path) return;
    setValidatingDir(true);
    try {
      const result = await projectApi.validateOutputDir(path);
      setOutputDirValid(result.valid && result.writable);
    } catch {
      setOutputDirValid(false);
    } finally {
      setValidatingDir(false);
    }
  };

  const validateModel = async (modelName: string) => {
    if (!modelName) return;
    setValidatingModel(true);
    try {
      const result = await projectApi.validateModel(modelName);
      setModelValid(result.available);
    } catch {
      setModelValid(false);
    } finally {
      setValidatingModel(false);
    }
  };

  const addTrait = (traitType: TraitType) => {
    if (traits.find(t => t.trait_type === traitType)) return;
    setTraits([...traits, { trait_type: traitType, datasets: [] }]);
  };

  const removeTrait = (index: number) => {
    setTraits(traits.filter((_, i) => i !== index));
  };

  const addDatasetToTrait = (traitIndex: number, datasetId: number, percentage: number) => {
    const updated = [...traits];
    const trait = updated[traitIndex];
    
    // Check if dataset already used in this trait
    if (trait.datasets.find(d => d.dataset_id === datasetId)) return;
    
    trait.datasets.push({ dataset_id: datasetId, percentage });
    setTraits(updated);
  };

  const removeDatasetFromTrait = (traitIndex: number, datasetIndex: number) => {
    const updated = [...traits];
    updated[traitIndex].datasets.splice(datasetIndex, 1);
    setTraits(updated);
  };

  const updatePercentage = (traitIndex: number, datasetIndex: number, percentage: number) => {
    const updated = [...traits];
    updated[traitIndex].datasets[datasetIndex].percentage = percentage;
    setTraits(updated);
  };

  const getTotalPercentage = (traitIndex: number): number => {
    return traits[traitIndex]?.datasets.reduce((sum, d) => sum + d.percentage, 0) || 0;
  };

  const canProceedToNext = () => {
    if (step === 1) {
      return formData.name && formData.base_model && formData.training_type && formData.max_rows;
    }
    if (step === 2) {
      const reasoningTrait = traits.find(t => t.trait_type === 'reasoning');
      return reasoningTrait && reasoningTrait.datasets.length === 1 && reasoningTrait.datasets[0].percentage === 100;
    }
    if (step === 3) {
      const codingTrait = traits.find(t => t.trait_type === 'coding');
      return !codingTrait || (codingTrait.datasets.length === 1 && codingTrait.datasets[0].percentage === 100);
    }
    if (step === 4) {
      const generalTrait = traits.find(t => t.trait_type === 'general_tools');
      if (!generalTrait) return true;
      const total = getTotalPercentage(traits.indexOf(generalTrait));
      return Math.abs(total - 100) < 0.01;
    }
    if (step === 5) {
      return formData.output_directory && outputDirValid === true && modelValid === true;
    }
    return false;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const projectData: ProjectCreate = {
      name: formData.name!,
      description: formData.description,
      base_model: formData.base_model!,
      training_type: formData.training_type as TrainingType,
      max_rows: formData.max_rows!,
      output_directory: formData.output_directory!,
      traits: traits.map(t => ({
        trait_type: t.trait_type,
        datasets: t.datasets,
      })),
    };
    createMutation.mutate(projectData);
  };

  return (
    <div className="card max-w-4xl max-h-[90vh] overflow-y-auto">
      <div className="border-b border-surface-800 px-6 py-4 sticky top-0 bg-surface-900 z-10">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white">Create Project</h3>
            <p className="text-sm text-surface-400">Step {step} of 5</p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="rounded-lg p-2 text-surface-400 hover:bg-surface-800 hover:text-surface-100"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>
        <div className="flex gap-2 mt-4">
          {[1, 2, 3, 4, 5].map((s) => (
            <div
              key={s}
              className={`h-2 flex-1 rounded ${
                s === step ? 'bg-primary-500' : s < step ? 'bg-primary-600' : 'bg-surface-700'
              }`}
            />
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="p-6">
        {/* Step 1: Basic Info */}
        {step === 1 && (
          <div className="space-y-4">
            <div>
              <label htmlFor="name" className="label">
                <HelpTooltip content="Give your project a descriptive name. This will be used to identify your trained model.">
                  Project Name *
                </HelpTooltip>
              </label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="input"
                required
              />
            </div>
            <div>
              <label htmlFor="description" className="label">
                Description
              </label>
              <textarea
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="input"
                rows={3}
              />
            </div>
            <div>
              <label htmlFor="base_model" className="label">
                <HelpTooltip content="The base model to fine-tune. Must be available in your HuggingFace cache or configured local paths. Examples: meta-llama/Llama-3.2-3B-Instruct, microsoft/Phi-3-mini-4k-instruct">
                  Base Model *
                </HelpTooltip>
              </label>
              <input
                type="text"
                id="base_model"
                value={formData.base_model}
                onChange={(e) => {
                  setFormData({ ...formData, base_model: e.target.value });
                  setModelValid(null);
                }}
                onBlur={() => validateModel(formData.base_model || '')}
                className="input"
                required
              />
              {validatingModel && <p className="text-sm text-surface-400 mt-1">Validating...</p>}
              {modelValid === true && <p className="text-sm text-green-400 mt-1">✓ Model available</p>}
              {modelValid === false && <p className="text-sm text-red-400 mt-1">✗ Model not found</p>}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="training_type" className="label">
                  Training Type *
                </label>
                <select
                  id="training_type"
                  value={formData.training_type}
                  onChange={(e) => setFormData({ ...formData, training_type: e.target.value as TrainingType })}
                  className="input"
                  required
                >
                  <option value="qlora">QLoRA</option>
                  <option value="unsloth">Unsloth</option>
                  <option value="rag">RAG</option>
                  <option value="standard">Standard</option>
                </select>
              </div>
              <div>
                <label htmlFor="max_rows" className="label">
                  <HelpTooltip content="Maximum number of training rows to use. The system will sample from your datasets to reach this limit based on percentage allocations.">
                    Max Rows *
                  </HelpTooltip>
                </label>
                <select
                  id="max_rows"
                  value={formData.max_rows}
                  onChange={(e) => setFormData({ ...formData, max_rows: parseInt(e.target.value) as any })}
                  className="input"
                  required
                >
                  <option value={50000}>50K</option>
                  <option value={100000}>100K</option>
                  <option value={250000}>250K</option>
                  <option value={500000}>500K</option>
                  <option value={1000000}>1M</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Reasoning Trait */}
        {step === 2 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <Info className="h-5 w-5 text-primary-400" />
              <h4 className="text-lg font-semibold">Reasoning Trait</h4>
            </div>
            <p className="text-sm text-surface-400 mb-4">
              Select exactly one dataset for reasoning training (100% allocation).
            </p>
            {(() => {
              const trait = traits.find(t => t.trait_type === 'reasoning') || { trait_type: 'reasoning' as TraitType, datasets: [] };
              const traitIndex = traits.findIndex(t => t.trait_type === 'reasoning');
              
              return (
                <div>
                  {trait.datasets.length === 0 ? (
                    <select
                      className="input"
                      onChange={(e) => {
                        const datasetId = parseInt(e.target.value);
                        if (datasetId) {
                          if (traitIndex === -1) {
                            addTrait('reasoning');
                            setTimeout(() => addDatasetToTrait(traits.length, datasetId, 100), 0);
                          } else {
                            addDatasetToTrait(traitIndex, datasetId, 100);
                          }
                        }
                      }}
                    >
                      <option value="">Select dataset...</option>
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="border border-surface-700 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <span>{datasets.find(d => d.id === trait.datasets[0].dataset_id)?.name}</span>
                        <button
                          type="button"
                          onClick={() => removeDatasetFromTrait(traitIndex, 0)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                      <div className="mt-2">
                        <label className="label text-sm">Percentage: 100%</label>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {/* Step 3: Coding Trait */}
        {step === 3 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <Info className="h-5 w-5 text-primary-400" />
              <h4 className="text-lg font-semibold">Coding Trait (Optional)</h4>
            </div>
            <p className="text-sm text-surface-400 mb-4">
              Optionally select one dataset for coding training (100% allocation).
            </p>
            {(() => {
              const trait = traits.find(t => t.trait_type === 'coding');
              const traitIndex = traits.findIndex(t => t.trait_type === 'coding');
              
              if (!trait) {
                return (
                  <button
                    type="button"
                    onClick={() => addTrait('coding')}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Plus className="h-4 w-4" />
                    Add Coding Trait
                  </button>
                );
              }
              
              return (
                <div>
                  {trait.datasets.length === 0 ? (
                    <select
                      className="input"
                      onChange={(e) => {
                        const datasetId = parseInt(e.target.value);
                        if (datasetId) addDatasetToTrait(traitIndex, datasetId, 100);
                      }}
                    >
                      <option value="">Select dataset...</option>
                      {datasets.filter(d => !traits.some(t => t.datasets.some(ds => ds.dataset_id === d.id))).map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="border border-surface-700 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <span>{datasets.find(d => d.id === trait.datasets[0].dataset_id)?.name}</span>
                        <button
                          type="button"
                          onClick={() => removeTrait(traitIndex)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {/* Step 4: General/Tools Trait */}
        {step === 4 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <Info className="h-5 w-5 text-primary-400" />
              <h4 className="text-lg font-semibold">General/Tools Trait (Optional)</h4>
            </div>
            <p className="text-sm text-surface-400 mb-4">
              Add one or more datasets with percentages that sum to 100%.
            </p>
            {(() => {
              const trait = traits.find(t => t.trait_type === 'general_tools');
              const traitIndex = traits.findIndex(t => t.trait_type === 'general_tools');
              
              if (!trait) {
                return (
                  <button
                    type="button"
                    onClick={() => addTrait('general_tools')}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Plus className="h-4 w-4" />
                    Add General/Tools Trait
                  </button>
                );
              }
              
              const total = getTotalPercentage(traitIndex);
              const availableDatasets = datasets.filter(d => !traits.some(t => t.trait_type !== 'general_tools' && t.datasets.some(ds => ds.dataset_id === d.id)));
              
              return (
                <div className="space-y-4">
                  {trait.datasets.map((alloc, idx) => {
                    const dataset = datasets.find(d => d.id === alloc.dataset_id);
                    return (
                      <div key={idx} className="border border-surface-700 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span>{dataset?.name}</span>
                          <button
                            type="button"
                            onClick={() => removeDatasetFromTrait(traitIndex, idx)}
                            className="text-red-400 hover:text-red-300"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                        <label className="label text-sm">Percentage</label>
                        <input
                          type="number"
                          min="0"
                          max="100"
                          step="0.01"
                          value={alloc.percentage}
                          onChange={(e) => updatePercentage(traitIndex, idx, parseFloat(e.target.value) || 0)}
                          className="input"
                        />
                      </div>
                    );
                  })}
                  {availableDatasets.length > 0 && (
                    <select
                      className="input"
                      onChange={(e) => {
                        const datasetId = parseInt(e.target.value);
                        if (datasetId) {
                          const remaining = 100 - total;
                          addDatasetToTrait(traitIndex, datasetId, remaining > 0 ? remaining : 0);
                        }
                      }}
                    >
                      <option value="">Add dataset...</option>
                      {availableDatasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} ({d.row_count} rows)
                        </option>
                      ))}
                    </select>
                  )}
                  <div className={`text-sm font-semibold ${Math.abs(total - 100) < 0.01 ? 'text-green-400' : 'text-red-400'}`}>
                    Total: {total.toFixed(2)}% {Math.abs(total - 100) < 0.01 ? '✓' : '(must equal 100%)'}
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* Step 5: Output Directory */}
        {step === 5 && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <Info className="h-5 w-5 text-primary-400" />
              <h4 className="text-lg font-semibold">Output Directory</h4>
            </div>
            <p className="text-sm text-surface-400 mb-4">
              Specify where the trained model will be saved. The directory must be writable.
            </p>
            <div>
              <label htmlFor="output_directory" className="label">
                Output Directory *
              </label>
              <input
                type="text"
                id="output_directory"
                value={formData.output_directory}
                onChange={(e) => {
                  setFormData({ ...formData, output_directory: e.target.value });
                  setOutputDirValid(null);
                }}
                onBlur={() => validateOutputDir(formData.output_directory || '')}
                className="input"
                placeholder={config?.output_directory_base || "/output/project-name"}
                required
              />
              {validatingDir && <p className="text-sm text-surface-400 mt-1">Validating...</p>}
              {outputDirValid === true && <p className="text-sm text-green-400 mt-1">✓ Directory is writable</p>}
              {outputDirValid === false && <p className="text-sm text-red-400 mt-1">✗ Directory is not writable or doesn't exist</p>}
            </div>
          </div>
        )}

        {createMutation.error && (
          <div className="mt-4 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-400 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-400">Error</p>
              <p className="text-sm text-surface-300">{createMutation.error.message}</p>
            </div>
          </div>
        )}

        <div className="flex justify-between mt-6 pt-4 border-t border-surface-800">
          <button
            type="button"
            onClick={() => setStep(Math.max(1, step - 1))}
            className="btn-secondary"
            disabled={step === 1}
          >
            Previous
          </button>
          {step < 5 ? (
            <button
              type="button"
              onClick={() => setStep(step + 1)}
              className="btn-primary"
              disabled={!canProceedToNext()}
            >
              Next
            </button>
          ) : (
            <button
              type="submit"
              className="btn-primary"
              disabled={!canProceedToNext() || createMutation.isPending}
            >
              {createMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Creating...
                </>
              ) : (
                'Create Project'
              )}
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
