import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { projectApi } from '../services/api';

interface ModelSelectorProps {
  value?: string;
  onChange: (modelName: string) => void;
  onValidate?: (isValid: boolean) => void;
}

/**
 * Model selector component with local model detection and validation.
 */
export default function ModelSelector({ value = '', onChange, onValidate }: ModelSelectorProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [isValid, setIsValid] = useState<boolean | null>(null);

  const { data: availableModels, isLoading } = useQuery({
    queryKey: ['available-models'],
    queryFn: () => projectApi.listAvailableModels(),
  });

  useEffect(() => {
    if (value) {
      validateModel(value);
    }
  }, [value]);

  const validateModel = async (modelName: string) => {
    if (!modelName) {
      setIsValid(null);
      onValidate?.(false);
      return;
    }

    setIsValidating(true);
    try {
      const result = await projectApi.validateModel(modelName);
      setIsValid(result.available);
      onValidate?.(result.available);
    } catch {
      setIsValid(false);
      onValidate?.(false);
    } finally {
      setIsValidating(false);
    }
  };

  const filteredModels = availableModels?.filter((model) =>
    model.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  return (
    <div className="space-y-2">
      <div className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setIsValid(null);
          }}
          onBlur={() => validateModel(value)}
          placeholder="Enter model name (e.g., meta-llama/Llama-3.2-3B-Instruct)"
          className="input pr-10"
        />
        {isValidating && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <Loader2 className="h-4 w-4 animate-spin text-surface-400" />
          </div>
        )}
        {!isValidating && isValid === true && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <CheckCircle2 className="h-4 w-4 text-green-400" />
          </div>
        )}
        {!isValidating && isValid === false && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <XCircle className="h-4 w-4 text-red-400" />
          </div>
        )}
      </div>

      {isValid === false && value && (
        <p className="text-sm text-red-400">
          Model not found. Please ensure it's downloaded to HuggingFace cache.
        </p>
      )}

      {availableModels && availableModels.length > 0 && (
        <div className="border border-surface-700 rounded-lg max-h-48 overflow-y-auto">
          <div className="sticky top-0 bg-surface-900 border-b border-surface-700 p-2">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search available models..."
                className="input pl-8 text-sm"
              />
            </div>
          </div>
          {isLoading ? (
            <div className="p-4 text-center">
              <Loader2 className="h-5 w-5 animate-spin text-surface-400 mx-auto" />
            </div>
          ) : filteredModels.length > 0 ? (
            <div className="divide-y divide-surface-700">
              {filteredModels.map((model) => (
                <button
                  key={model}
                  type="button"
                  onClick={() => onChange(model)}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-surface-800 transition-colors"
                >
                  {model}
                </button>
              ))}
            </div>
          ) : (
            <div className="p-4 text-center text-sm text-surface-400">
              {searchTerm ? 'No models found' : 'No local models available'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
