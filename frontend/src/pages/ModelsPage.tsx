import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Search, Package, Download, Loader2, AlertCircle, X, RefreshCw } from 'lucide-react';
import { modelsApi } from '../services/api';
import ModelCard from '../components/ModelCard';
import LoadingModal from '../components/LoadingModal';
import type { HuggingFaceModel, LocalModel } from '../types';

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'search' | 'local'>('search');
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);

  // Search models query
  const {
    data: searchResults,
    isLoading: searching,
    refetch: searchModels,
  } = useQuery({
    queryKey: ['models', 'search', searchQuery],
    queryFn: () => modelsApi.search(searchQuery, 20, 0),
    enabled: false, // Only search on button click
  });

  // Local models query
  const {
    data: localModels,
    isLoading: loadingLocal,
    refetch: refetchLocal,
  } = useQuery({
    queryKey: ['models', 'local'],
    queryFn: modelsApi.listLocal,
  });

  // Download mutation
  const downloadMutation = useMutation({
    mutationFn: modelsApi.download,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models', 'local'] });
      setDownloadingModel(null);
    },
    onError: () => {
      setDownloadingModel(null);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: modelsApi.deleteLocal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models', 'local'] });
    },
  });

  // Scan mutation
  const scanMutation = useMutation({
    mutationFn: modelsApi.scan,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models', 'local'] });
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      searchModels();
    }
  };

  const handleDownload = (modelId: string) => {
    setDownloadingModel(modelId);
    downloadMutation.mutate(modelId);
  };

  const handleDelete = (modelId: string) => {
    if (window.confirm(`Are you sure you want to delete model ${modelId}?`)) {
      deleteMutation.mutate(modelId);
    }
  };

  const downloadingModelName = downloadingModel
    ? searchResults?.items.find((m) => m.id === downloadingModel)?.name || downloadingModel
    : null;

  return (
    <>
      <LoadingModal
        isOpen={downloadMutation.isPending && !!downloadingModel}
        message="Downloading Model"
        subtitle={`Downloading ${downloadingModelName || 'model'} from HuggingFace... This may take a while.`}
      />
      <div className="space-y-6">
      <div className="card">
        <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/10">
              <Package className="h-5 w-5 text-primary-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Models</h1>
              <p className="text-sm text-surface-400">
                Search and download models from HuggingFace Hub
              </p>
            </div>
          </div>
        </div>

        <div className="p-6">
          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-surface-800">
            <button
              onClick={() => setActiveTab('search')}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === 'search'
                  ? 'text-primary-400 border-b-2 border-primary-400'
                  : 'text-surface-400 hover:text-surface-300'
              }`}
            >
              Search HuggingFace
            </button>
            <button
              onClick={() => setActiveTab('local')}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === 'local'
                  ? 'text-primary-400 border-b-2 border-primary-400'
                  : 'text-surface-400 hover:text-surface-300'
              }`}
            >
              Local Models ({localModels?.length || 0})
            </button>
          </div>

          {/* Search Tab */}
          {activeTab === 'search' && (
            <div className="space-y-6">
              <form onSubmit={handleSearch} className="flex gap-2">
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-surface-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search for models..."
                    className="input pl-10"
                  />
                </div>
                <button
                  type="submit"
                  disabled={searching || !searchQuery.trim()}
                  className="btn btn-primary flex items-center gap-2"
                >
                  {searching ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <Search className="h-4 w-4" />
                      Search
                    </>
                  )}
                </button>
              </form>

              {searchResults && (
                <div>
                  <p className="text-sm text-surface-400 mb-4">
                    Found {searchResults.items.length} results for "{searchResults.query}"
                  </p>
                  {searchResults.items.some(m => m.id.toLowerCase().includes('gguf')) && (
                    <div className="mb-4 rounded-lg border border-yellow-500/50 bg-yellow-500/10 p-4">
                      <div className="flex items-start gap-3">
                        <AlertCircle className="h-5 w-5 text-yellow-400 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                          <p className="text-sm font-medium text-yellow-400 mb-1">
                            GGUF Models Warning
                          </p>
                          <p className="text-sm text-yellow-300">
                            GGUF models (quantized inference models) cannot be used for training. 
                            Only HuggingFace format models (with config.json and tokenizer files) are suitable for training.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {searchResults.items.map((model) => {
                      const isGGUF = model.id.toLowerCase().includes('gguf');
                      return (
                        <div key={model.id} className={isGGUF ? 'opacity-60' : ''}>
                          <ModelCard
                            model={model}
                            isLocal={false}
                            onDownload={handleDownload}
                            downloading={downloadingModel === model.id}
                          />
                          {isGGUF && (
                            <p className="mt-2 text-xs text-yellow-400 text-center">
                              ⚠️ GGUF model - not suitable for training
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {downloadMutation.isError && (
                <div className="card bg-red-500/10 border-red-500/50">
                  <div className="p-4 flex items-center gap-3">
                    <AlertCircle className="h-5 w-5 text-red-400" />
                    <div className="flex-1">
                      <p className="text-sm text-red-400 font-medium">Download failed</p>
                      <p className="text-xs text-red-300 mt-1">
                        {downloadMutation.error instanceof Error
                          ? downloadMutation.error.message
                          : 'An error occurred while downloading the model'}
                      </p>
                    </div>
                    <button
                      onClick={() => downloadMutation.reset()}
                      className="text-red-400 hover:text-red-300"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}

              {downloadMutation.isSuccess && (
                <div className="card bg-green-500/10 border-green-500/50">
                  <div className="p-4 flex items-center gap-3">
                    <Download className="h-5 w-5 text-green-400" />
                    <p className="text-sm text-green-400 font-medium">
                      Model downloaded successfully!
                    </p>
                    <button
                      onClick={() => {
                        downloadMutation.reset();
                        setActiveTab('local');
                      }}
                      className="ml-auto text-green-400 hover:text-green-300"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Local Models Tab */}
          {activeTab === 'local' && (
            <div className="space-y-6">
              <div className="flex items-center justify-end">
                <button
                  onClick={() => scanMutation.mutate()}
                  disabled={scanMutation.isPending}
                  className="btn-ghost flex items-center gap-2"
                  title="Scan directories and refresh model list"
                >
                  {scanMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  Refresh
                </button>
              </div>
              {loadingLocal ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
                </div>
              ) : localModels && localModels.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {localModels.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      isLocal={true}
                      onDelete={handleDelete}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Package className="h-12 w-12 text-surface-600 mx-auto mb-4" />
                  <p className="text-surface-400">No local models found</p>
                  <p className="text-sm text-surface-500 mt-2">
                    Search and download models from HuggingFace Hub to get started
                  </p>
                </div>
              )}

              {deleteMutation.isError && (
                <div className="card bg-red-500/10 border-red-500/50">
                  <div className="p-4 flex items-center gap-3">
                    <AlertCircle className="h-5 w-5 text-red-400" />
                    <div className="flex-1">
                      <p className="text-sm text-red-400 font-medium">Delete failed</p>
                      <p className="text-xs text-red-300 mt-1">
                        {deleteMutation.error instanceof Error
                          ? deleteMutation.error.message
                          : 'An error occurred while deleting the model'}
                      </p>
                    </div>
                    <button
                      onClick={() => deleteMutation.reset()}
                      className="text-red-400 hover:text-red-300"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
    </>
  );
}
