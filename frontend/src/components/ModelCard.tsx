import { Download, Trash2, Package } from 'lucide-react';
import type { HuggingFaceModel, LocalModel } from '../types';

interface ModelCardProps {
  model: HuggingFaceModel | LocalModel;
  isLocal?: boolean;
  onDownload?: (modelId: string) => void;
  onDelete?: (modelId: string) => void;
  downloading?: boolean;
}

export default function ModelCard({ model, isLocal = false, onDownload, onDelete, downloading = false }: ModelCardProps) {
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <div className="card hover:bg-surface-800/50 transition-colors">
      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Package className="h-5 w-5 text-primary-400" />
              <h3 className="text-lg font-semibold text-white">{model.name}</h3>
              {'is_private' in model && model.is_private && (
                <span className="px-2 py-1 text-xs font-medium rounded bg-yellow-500/20 text-yellow-400">
                  Private
                </span>
              )}
            </div>
            <p className="text-sm text-surface-400 mb-2">
              by <span className="text-surface-300 font-medium">{model.author}</span>
            </p>
            <p className="text-sm text-surface-300 line-clamp-2">
              {model.description || 'No description available'}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          {'downloads' in model && (
            <span className="px-2 py-1 text-xs rounded bg-surface-700 text-surface-300">
              {formatNumber(model.downloads)} downloads
            </span>
          )}
          {'likes' in model && (
            <span className="px-2 py-1 text-xs rounded bg-surface-700 text-surface-300">
              {formatNumber(model.likes)} likes
            </span>
          )}
          {'file_size' in model && (
            <span className="px-2 py-1 text-xs rounded bg-surface-700 text-surface-300">
              {formatFileSize(model.file_size)}
            </span>
          )}
          {'model_type' in model && model.model_type && (
            <span className="px-2 py-1 text-xs rounded bg-primary-500/20 text-primary-400">
              {model.model_type}
            </span>
          )}
        </div>

        {'tags' in model && model.tags && model.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-4">
            {model.tags.slice(0, 5).map((tag, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 text-xs rounded bg-surface-700/50 text-surface-400"
              >
                {tag}
              </span>
            ))}
            {model.tags.length > 5 && (
              <span className="px-2 py-0.5 text-xs text-surface-500">
                +{model.tags.length - 5} more
              </span>
            )}
          </div>
        )}

        <div className="flex gap-2">
          {!isLocal && onDownload && (
            <button
              onClick={() => onDownload('model_id' in model ? model.model_id : model.id)}
              disabled={downloading}
              className="flex-1 btn btn-primary flex items-center justify-center gap-2"
            >
              <Download className="h-4 w-4" />
              {downloading ? 'Downloading...' : 'Download'}
            </button>
          )}
          {isLocal && onDelete && (
            <button
              onClick={() => onDelete('model_id' in model ? model.model_id : model.id)}
              className="flex-1 btn btn-danger flex items-center justify-center gap-2"
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
