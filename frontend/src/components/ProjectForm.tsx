import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { X, Loader2, AlertCircle, Info, Plus, Trash2 } from 'lucide-react';
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

  const { data: availableModels, isLoading: loadingModels } = useQuery({
    queryKey: ['availableModels'],
    queryFn: () => projectApi.listAvailableModels(),
  });

  const [traits, setTraits] = useState<TraitConfig[]>([]);
  const [outputDirValid, setOutputDirValid] = useState<boolean | null>(null);
  const [modelValid, setModelValid] = useState<boolean | null>(null);
  const [validatingDir, setValidatingDir] = useState(false);
  const [validatingModel, setValidatingModel] = useState(false);
  const [modelTypes, setModelTypes] = useState<{
    model_type: string;
    available_types: string[];
    recommended: string | null;
  } | null>(null);
  const [loadingModelTypes, setLoadingModelTypes] = useState(false);

  // Helper function to sanitize project name for filesystem
  const sanitizeProjectName = (name: string): string => {
    return name
      .replace(/[<>:"/\\|?*]/g, '_') // Replace invalid filesystem characters
      .replace(/\s+/g, '-') // Replace spaces with hyphens
      .replace(/\.+$/, '') // Remove trailing dots
      .toLowerCase()
      .trim() || 'project';
  };

  // Track if user manually edited output_directory
  const [outputDirManuallyEdited, setOutputDirManuallyEdited] = useState(false);

  // Set default model when config loads
  useEffect(() => {
    if (config?.default_model && !formData.base_model) {
      setFormData(prev => ({ ...prev, base_model: config.default_model }));
    }
  }, [config, formData.base_model]);

  // Fetch model types when base_model changes
  useEffect(() => {
    if (formData.base_model && availableModels && availableModels.includes(formData.base_model)) {
      setLoadingModelTypes(true);
      projectApi.getModelTypes(formData.base_model)
        .then((data) => {
          setModelTypes(data);
          // Auto-set recommended model_type if available
          if (data.recommended) {
            setFormData(prev => ({ ...prev, model_type: data.recommended || undefined }));
          } else if (data.model_type) {
            // Fallback to detected model_type if recommended not available
            setFormData(prev => ({ ...prev, model_type: data.model_type || undefined }));
          } else if (data.available_types && data.available_types.length > 0) {
            // Fallback to first available type if neither recommended nor detected
            setFormData(prev => ({ ...prev, model_type: data.available_types[0] || undefined }));
          }
        })
        .catch((err) => {
          console.error('Failed to fetch model types:', err);
          setModelTypes(null);
        })
        .finally(() => {
          setLoadingModelTypes(false);
        });
    } else {
      setModelTypes(null);
      setFormData(prev => ({ ...prev, model_type: undefined }));
    }
  }, [formData.base_model, availableModels]);

  // Auto-generate output directory from base + project name
  useEffect(() => {
    // Use config value or default to './output'
    const baseDir = (config?.output_directory_base || './output').trim();
    
    // Skip if manually edited
    if (outputDirManuallyEdited) {
      return;
    }
    
    if (formData.name && formData.name.trim()) {
      // Project name exists - generate full path: base/project-name
      const sanitizedName = sanitizeProjectName(formData.name);
      const generatedPath = baseDir.endsWith('/') 
        ? `${baseDir}${sanitizedName}`
        : `${baseDir}/${sanitizedName}`;
      
      // Always update when name changes
      setFormData(prev => {
        const currentPath = prev.output_directory || '';
        // Update if empty, matches base, or matches the generated pattern
        if (!currentPath || 
            currentPath === baseDir || 
            currentPath.startsWith(baseDir + '/')) {
          return { ...prev, output_directory: generatedPath };
        }
        return prev;
      });
    } else if (!formData.output_directory) {
      // No project name yet - just set the base directory
      setFormData(prev => ({ ...prev, output_directory: baseDir }));
    }
  }, [config?.output_directory_base, formData.name, outputDirManuallyEdited]);

  const createMutation = useMutation({
    mutationFn: projectApi.create,
    onSuccess: async () => {
      // Invalidate and refetch all project-related queries to refresh the list and counts
      await queryClient.invalidateQueries({ queryKey: ['projects'] });
      await queryClient.refetchQueries({ queryKey: ['projects'] });
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
    // Skip validation if model is in the available models list
    if (availableModels && availableModels.includes(modelName)) {
      setModelValid(true);
      return;
    }
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
    if (traitIndex < 0 || traitIndex >= traits.length) {
      console.error(`Invalid trait index: ${traitIndex}`);
      return;
    }
    const updated = [...traits];
    const trait = updated[traitIndex];
    
    if (!trait || !trait.datasets) {
      console.error(`Trait at index ${traitIndex} is invalid`);
      return;
    }
    
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

  const calculateTotalRows = (): number => {
    let total = 0;
    traits.forEach(trait => {
      trait.datasets.forEach(alloc => {
        const dataset = datasets.find(d => d.id === alloc.dataset_id);
        if (dataset) {
          total += Math.floor((dataset.row_count * alloc.percentage) / 100);
        }
      });
    });
    return total;
  };

  const canProceedToNext = () => {
    if (step === 1) {
      // Basic validation - model_type will be set in onClick handler if needed
      return formData.name && formData.base_model && formData.training_type && modelValid === true;
    }
    if (step === 2) {
      const reasoningTrait = traits.find(t => t.trait_type === 'reasoning');
      return reasoningTrait && reasoningTrait.datasets.length === 1;
    }
    if (step === 3) {
      const codingTrait = traits.find(t => t.trait_type === 'coding');
      return !codingTrait || codingTrait.datasets.length === 1;
    }
    if (step === 4) {
      // General tools trait is optional, but if it exists, must have at least one dataset
      const generalTrait = traits.find(t => t.trait_type === 'general_tools');
      return !generalTrait || generalTrait.datasets.length > 0;
    }
    if (step === 5) {
      // Only require output directory and model to be set, not validated
      // Backend will create the directory if it doesn't exist
      return formData.output_directory && modelValid === true;
    }
    return false;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // Ensure model_type is set before submission (workaround for browser automation)
    let finalModelType = formData.model_type;
    if (!finalModelType && modelTypes) {
      finalModelType = modelTypes.recommended || modelTypes.model_type || (modelTypes.available_types && modelTypes.available_types[0]);
    }
    const projectData: ProjectCreate = {
      name: formData.name!,
      description: formData.description,
      base_model: formData.base_model!,
      model_type: finalModelType,
      training_type: formData.training_type as TrainingType,
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
                value={formData.name || ''}
                onChange={(e) => {
                  const newName = e.target.value;
                  setFormData(prev => {
                    const updated = { ...prev, name: newName };
                    // If output directory was auto-generated, allow it to update
                    if (outputDirManuallyEdited && config?.output_directory_base) {
                      const baseDir = config.output_directory_base.trim();
                      const currentPath = prev.output_directory || '';
                      // If current path matches the pattern, reset manual edit flag
                      if (currentPath.startsWith(baseDir + '/') || currentPath === baseDir) {
                        setOutputDirManuallyEdited(false);
                      }
                    }
                    return updated;
                  });
                }}
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
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                className="input"
                rows={3}
              />
            </div>
            <div>
              <label htmlFor="base_model" className="label">
                <HelpTooltip content="The base model to fine-tune. Select from available models in your HuggingFace cache or configured local paths.">
                  Base Model *
                </HelpTooltip>
              </label>
              {loadingModels ? (
                <div className="input flex items-center gap-2 text-surface-400">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading models...
                </div>
              ) : (
                <select
                  id="base_model"
                  value={formData.base_model}
                  onChange={(e) => {
                    const selectedModel = e.target.value;
                    setFormData(prev => ({ ...prev, base_model: selectedModel }));
                    // Auto-validate if model is in available list
                    if (selectedModel && availableModels && availableModels.includes(selectedModel)) {
                      setModelValid(true);
                    } else {
                      setModelValid(null);
                      if (selectedModel) {
                        validateModel(selectedModel);
                      }
                    }
                  }}
                  className="input"
                  required
                >
                  <option value="">Select a model...</option>
                  {availableModels && availableModels.length > 0 ? (
                    availableModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))
                  ) : (
                    <option value="" disabled>
                      No models available
                    </option>
                  )}
                </select>
              )}
              {!loadingModels && formData.base_model && modelValid === true && (
                <p className="text-sm text-green-400 mt-1">✓ Model available</p>
              )}
              {!loadingModels && formData.base_model && modelValid === false && (
                <p className="text-sm text-red-400 mt-1">✗ Model not found</p>
              )}
            </div>
            {formData.base_model && modelValid === true && (
              <div>
                <label htmlFor="model_type" className="label">
                  <HelpTooltip content="The specific model architecture type (e.g., 'llama', 'bert'). Auto-detected from model config.json.">
                    Model Type *
                  </HelpTooltip>
                </label>
                {loadingModelTypes ? (
                  <div className="input flex items-center gap-2 text-surface-400">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Detecting model types...
                  </div>
                ) : (
                  <select
                    id="model_type"
                    value={formData.model_type || ''}
                    onChange={(e) => setFormData(prev => ({ ...prev, model_type: e.target.value }))}
                    className="input"
                    required
                    disabled={!modelTypes?.available_types?.length}
                  >
                    <option value="">Select Model Type</option>
                    {modelTypes?.available_types?.map((type) => (
                      <option key={type} value={type}>
                        {type} {type === modelTypes.recommended && '(Recommended)'}
                      </option>
                    ))}
                  </select>
                )}
                {modelTypes?.model_type && !formData.model_type && (
                  <p className="text-sm text-surface-400 mt-1">
                    Detected: <span className="font-semibold">{modelTypes.model_type}</span>. Select from dropdown.
                  </p>
                )}
                {modelTypes?.model_type && formData.model_type && formData.model_type !== modelTypes.model_type && (
                  <p className="text-sm text-yellow-400 mt-1">
                    Warning: Detected model type is '{modelTypes.model_type}', but selected '{formData.model_type}'.
                  </p>
                )}
              </div>
            )}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="training_type" className="label">
                  Training Type *
                </label>
                <select
                  id="training_type"
                  value={formData.training_type}
                  onChange={(e) => setFormData(prev => ({ ...prev, training_type: e.target.value as TrainingType }))}
                  className="input"
                  required
                >
                  <option value="qlora">QLoRA</option>
                  <option value="unsloth">Unsloth</option>
                  <option value="rag">RAG</option>
                  <option value="standard">Standard</option>
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
              Select exactly one dataset for reasoning training. Set the percentage of the dataset file to use.
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
                            // Add trait with dataset in one operation, default to 100%
                            setTraits([...traits, { trait_type: 'reasoning' as TraitType, datasets: [{ dataset_id: datasetId, percentage: 100 }] }]);
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
                    <div className="border border-surface-700 rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{datasets.find(d => d.id === trait.datasets[0].dataset_id)?.name}</span>
                        <button
                          type="button"
                          onClick={() => removeDatasetFromTrait(traitIndex, 0)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                      {(() => {
                        const dataset = datasets.find(d => d.id === trait.datasets[0].dataset_id);
                        const rowsUsed = dataset ? Math.floor((dataset.row_count * trait.datasets[0].percentage) / 100) : 0;
                        return (
                          <>
                            <div>
                              <label className="label text-sm">Percentage of dataset file to use</label>
                              <input
                                type="number"
                                min="0"
                                max="100"
                                step="0.01"
                                value={trait.datasets[0].percentage}
                                onChange={(e) => updatePercentage(traitIndex, 0, parseFloat(e.target.value) || 0)}
                                className="input"
                              />
                            </div>
                            {dataset && (
                              <div className="text-sm text-surface-400">
                                {dataset.row_count} rows × {trait.datasets[0].percentage}% = {rowsUsed} rows used
                              </div>
                            )}
                          </>
                        );
                      })()}
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
              Optionally select one dataset for coding training. Set the percentage of the dataset file to use.
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
                    <div className="border border-surface-700 rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{datasets.find(d => d.id === trait.datasets[0].dataset_id)?.name}</span>
                        <button
                          type="button"
                          onClick={() => removeTrait(traitIndex)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                      {(() => {
                        const dataset = datasets.find(d => d.id === trait.datasets[0].dataset_id);
                        const rowsUsed = dataset ? Math.floor((dataset.row_count * trait.datasets[0].percentage) / 100) : 0;
                        return (
                          <>
                            <div>
                              <label className="label text-sm">Percentage of dataset file to use</label>
                              <input
                                type="number"
                                min="0"
                                max="100"
                                step="0.01"
                                value={trait.datasets[0].percentage}
                                onChange={(e) => updatePercentage(traitIndex, 0, parseFloat(e.target.value) || 0)}
                                className="input"
                              />
                            </div>
                            {dataset && (
                              <div className="text-sm text-surface-400">
                                {dataset.row_count} rows × {trait.datasets[0].percentage}% = {rowsUsed} rows used
                              </div>
                            )}
                          </>
                        );
                      })()}
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
              Add one or more datasets. Set the percentage of each dataset file to use (0-100%).
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
              
              const availableDatasets = datasets.filter(d => !traits.some(t => t.trait_type !== 'general_tools' && t.datasets.some(ds => ds.dataset_id === d.id)));
              const totalRows = calculateTotalRows();
              
              return (
                <div className="space-y-4">
                  {trait.datasets.map((alloc, idx) => {
                    const dataset = datasets.find(d => d.id === alloc.dataset_id);
                    const rowsUsed = dataset ? Math.floor((dataset.row_count * alloc.percentage) / 100) : 0;
                    return (
                      <div key={idx} className="border border-surface-700 rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{dataset?.name}</span>
                          <button
                            type="button"
                            onClick={() => removeDatasetFromTrait(traitIndex, idx)}
                            className="text-red-400 hover:text-red-300"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                        <div>
                          <label className="label text-sm">Percentage of dataset file to use</label>
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
                        {dataset && (
                          <div className="text-sm text-surface-400">
                            {dataset.row_count} rows × {alloc.percentage}% = {rowsUsed} rows used
                          </div>
                        )}
                      </div>
                    );
                  })}
                  {availableDatasets.length > 0 && (
                    <select
                      className="input"
                      onChange={(e) => {
                        const datasetId = parseInt(e.target.value);
                        if (datasetId) {
                          addDatasetToTrait(traitIndex, datasetId, 100);
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
                  <div className="text-sm font-semibold text-green-400">
                    Total rows: {totalRows.toLocaleString()}
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
              <h4 className="text-lg font-semibold">Output Directory & Model Validation</h4>
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
                value={formData.output_directory || ''}
                onChange={(e) => {
                  setFormData(prev => ({ ...prev, output_directory: e.target.value }));
                  setOutputDirValid(null);
                  setOutputDirManuallyEdited(true);
                }}
                onBlur={() => {
                  // Optional validation - doesn't block creation
                  // Backend will create directory if it doesn't exist
                  if (formData.output_directory) {
                    validateOutputDir(formData.output_directory);
                  }
                }}
                className="input"
                placeholder={config?.output_directory_base ? `${config.output_directory_base}/project-name` : "/output/project-name"}
                required
              />
              {validatingDir && <p className="text-sm text-surface-400 mt-1">Validating...</p>}
              {outputDirValid === true && <p className="text-sm text-green-400 mt-1">✓ Directory is writable</p>}
              {outputDirValid === false && (
                <p className="text-sm text-amber-400 mt-1">
                  ⚠ Directory will be created automatically if it doesn't exist
                </p>
              )}
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
              onClick={() => {
                // Ensure model_type is set before advancing (workaround for browser automation)
                if (step === 1) {
                  // Set model_type if available but not set
                  if (!formData.model_type && modelTypes) {
                    const modelTypeToSet = modelTypes.recommended || modelTypes.model_type || (modelTypes.available_types && modelTypes.available_types[0]);
                    if (modelTypeToSet) {
                      setFormData(prev => ({ ...prev, model_type: modelTypeToSet }));
                    }
                  }
                  // Validate basic fields before advancing
                  if (formData.name && formData.base_model && formData.training_type && modelValid) {
                    setStep(step + 1);
                  }
                  return;
                }
                setStep(step + 1);
              }}
              className="btn-primary"
              disabled={step === 1 ? (!formData.name || !formData.base_model || !formData.training_type || !modelValid) : !canProceedToNext()}
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
