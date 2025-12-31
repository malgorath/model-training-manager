import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Search,
  Download,
  X,
  ExternalLink,
  Heart,
  ArrowDownToLine,
  Loader2,
  Database,
  Tag,
} from 'lucide-react';
import { huggingfaceApi } from '../services/api';
import type { HFDataset, HFDownloadRequest } from '../types';
import LoadingModal from './LoadingModal';

interface HuggingFaceImportProps {
  onSuccess: () => void;
  onClose: () => void;
}

/**
 * Component for searching and importing datasets from Hugging Face Hub.
 * 
 * Note: This component is specifically for datasets. For models, use the ModelsPage search interface.
 */
export default function HuggingFaceImport({ onSuccess, onClose }: HuggingFaceImportProps) {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<HFDataset[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<HFDataset | null>(null);
  const [downloadOptions, setDownloadOptions] = useState({
    name: '',
    split: 'train',
    config: '',
    maxRows: 10000,
  });
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setError(null);
    
    try {
      const response = await huggingfaceApi.search(query, 20, 0);
      setResults(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setIsSearching(false);
    }
  };

  const downloadMutation = useMutation({
    mutationFn: (request: HFDownloadRequest) => huggingfaceApi.download(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      onSuccess();
    },
    onError: (err: Error) => {
      setError(err.message);
    },
  });

  const handleDownload = () => {
    if (!selectedDataset) return;

    downloadMutation.mutate({
      dataset_id: selectedDataset.id,
      name: downloadOptions.name || undefined,
      split: downloadOptions.split,
      config: downloadOptions.config || undefined,
      max_rows: downloadOptions.maxRows,
    });
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <>
      <LoadingModal
        isOpen={downloadMutation.isPending}
        message="Downloading Dataset"
        subtitle={`Downloading ${selectedDataset?.id || 'dataset'} from HuggingFace... This may take a while.`}
      />
      <div className="w-[95vw] max-w-[1400px] max-h-[90vh] flex flex-col overflow-hidden rounded-xl border border-surface-700 bg-surface-900 shadow-2xl">
      {/* Header */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-surface-800 bg-surface-800/50 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-500/10">
            <Database className="h-5 w-5 text-amber-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Import from Hugging Face</h2>
            <p className="text-sm text-surface-400">Search and download datasets</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-2 text-surface-400 hover:bg-surface-800 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Content */}
      <div className="flex min-h-0 flex-1">
        {/* Left: Search Results */}
        <div className="flex min-w-0 flex-1 flex-col border-r border-surface-800">
          {/* Search Bar */}
          <div className="flex-shrink-0 border-b border-surface-800 p-4">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-surface-500" />
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Search datasets (e.g., 'sentiment', 'qa', 'code')"
                  className="input w-full pl-10"
                />
              </div>
              <button
                onClick={handleSearch}
                disabled={isSearching || !query.trim()}
                className="btn-primary"
              >
                {isSearching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Search'
                )}
              </button>
            </div>
          </div>

          {/* Results */}
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            {error && (
              <div className="mb-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-400">
                {error}
              </div>
            )}

            {results.length === 0 && !isSearching && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Database className="h-12 w-12 text-surface-600" />
                <p className="mt-4 text-surface-400">
                  Search for datasets on Hugging Face Hub
                </p>
                <p className="mt-1 text-sm text-surface-500">
                  Try searching for &quot;sentiment&quot;, &quot;summarization&quot;, or &quot;code&quot;
                </p>
              </div>
            )}

            <div className="space-y-2">
              {results.map((dataset) => (
                <button
                  key={dataset.id}
                  onClick={() => {
                    setSelectedDataset(dataset);
                    setDownloadOptions((prev) => ({
                      ...prev,
                      name: dataset.name,
                    }));
                  }}
                  className={`w-full rounded-lg border p-3 text-left transition-all ${
                    selectedDataset?.id === dataset.id
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-surface-700 bg-surface-800/50 hover:border-surface-600'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium text-white">{dataset.id}</p>
                      {dataset.description && (
                        <p className="mt-1 line-clamp-1 text-sm text-surface-400">
                          {dataset.description}
                        </p>
                      )}
                    </div>
                    <a
                      href={`https://huggingface.co/datasets/${dataset.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="flex-shrink-0 rounded p-1 text-surface-500 hover:bg-surface-700 hover:text-white"
                    >
                      <ExternalLink className="h-4 w-4" />
                    </a>
                  </div>
                  <div className="mt-2 flex items-center gap-4 text-xs text-surface-500">
                    <span className="flex items-center gap-1">
                      <ArrowDownToLine className="h-3 w-3" />
                      {formatNumber(dataset.downloads)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Heart className="h-3 w-3" />
                      {formatNumber(dataset.likes)}
                    </span>
                  </div>
                  {dataset.tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {dataset.tags.slice(0, 4).map((tag) => (
                        <span
                          key={tag}
                          className="inline-flex items-center gap-1 rounded bg-surface-700 px-1.5 py-0.5 text-xs text-surface-400"
                        >
                          <Tag className="h-2.5 w-2.5" />
                          {tag}
                        </span>
                      ))}
                      {dataset.tags.length > 4 && (
                        <span className="text-xs text-surface-500">
                          +{dataset.tags.length - 4} more
                        </span>
                      )}
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Download Options */}
        <div className="flex w-[420px] flex-shrink-0 flex-col border-l border-surface-800">
          {selectedDataset ? (
            <div className="flex min-h-0 flex-1 flex-col">
              {/* Scrollable content */}
              <div className="min-h-0 flex-1 overflow-y-auto p-6">
                <div className="space-y-5">
                  <div>
                    <h3 className="truncate font-medium text-white" title={selectedDataset.id}>
                      {selectedDataset.id}
                    </h3>
                    <p className="mt-1 text-sm text-surface-400">Configure download options</p>
                    {selectedDataset.description && (
                      <p className="mt-2 line-clamp-3 text-xs text-surface-500">
                        {selectedDataset.description}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-surface-300">
                      Dataset Name
                    </label>
                    <input
                      type="text"
                      value={downloadOptions.name}
                      onChange={(e) =>
                        setDownloadOptions((prev) => ({ ...prev, name: e.target.value }))
                      }
                      placeholder={selectedDataset.name}
                      className="input mt-1.5 w-full"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-surface-300">
                      Split
                    </label>
                    <select
                      value={downloadOptions.split}
                      onChange={(e) =>
                        setDownloadOptions((prev) => ({ ...prev, split: e.target.value }))
                      }
                      className="input mt-1.5 w-full"
                    >
                      <option value="train">train</option>
                      <option value="test">test</option>
                      <option value="validation">validation</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-surface-300">
                      Configuration (optional)
                    </label>
                    <input
                      type="text"
                      value={downloadOptions.config}
                      onChange={(e) =>
                        setDownloadOptions((prev) => ({ ...prev, config: e.target.value }))
                      }
                      placeholder="e.g., en, default"
                      className="input mt-1.5 w-full"
                    />
                    <p className="mt-1.5 text-xs text-surface-500">
                      Some datasets have multiple configs (languages, subsets)
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-surface-300">
                      Max Rows
                    </label>
                    <input
                      type="number"
                      value={downloadOptions.maxRows}
                      onChange={(e) =>
                        setDownloadOptions((prev) => ({
                          ...prev,
                          maxRows: parseInt(e.target.value) || 10000,
                        }))
                      }
                      min={100}
                      max={1000000}
                      className="input mt-1.5 w-full"
                    />
                    <p className="mt-1.5 text-xs text-surface-500">
                      Limit rows for large datasets (100 - 1,000,000)
                    </p>
                  </div>
                </div>
              </div>

              {/* Fixed footer with download button */}
              <div className="flex-shrink-0 border-t border-surface-800 bg-surface-900/95 p-6">
                <button
                  onClick={handleDownload}
                  disabled={downloadMutation.isPending}
                  className="btn-primary w-full"
                >
                  {downloadMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="h-4 w-4" />
                      Download Dataset
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center px-6 py-12 text-center">
              <Download className="h-12 w-12 text-surface-600" />
              <p className="mt-4 text-surface-400">Select a dataset</p>
              <p className="mt-1 text-sm text-surface-500">
                Choose a dataset from the search results to configure download options
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
    </>
  );
}

