import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Database,
  Trash2,
  FileText,
  FileJson,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { clsx } from 'clsx';
import { datasetApi } from '../services/api';
import type { Dataset } from '../types';

interface DatasetListProps {
  onSelect?: (dataset: Dataset) => void;
  selectedId?: number;
}

/**
 * Dataset list component with pagination.
 */
export default function DatasetList({ onSelect, selectedId }: DatasetListProps) {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const { data, isLoading } = useQuery({
    queryKey: ['datasets', page],
    queryFn: () => datasetApi.list(page, pageSize),
  });

  const deleteMutation = useMutation({
    mutationFn: datasetApi.delete,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['datasets'] }),
  });

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const formatDate = (dateStr: string): string => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
      </div>
    );
  }

  if (!data?.items.length) {
    return (
      <div className="py-12 text-center">
        <Database className="mx-auto h-12 w-12 text-surface-600" />
        <p className="mt-3 text-surface-400">No datasets yet</p>
        <p className="text-sm text-surface-500">
          Upload a CSV or JSON file to get started
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="overflow-hidden rounded-xl border border-surface-800">
        <table className="w-full">
          <thead>
            <tr className="border-b border-surface-800 bg-surface-800/30">
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-surface-400">
                Dataset
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-surface-400">
                Type
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-surface-400">
                Rows
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-surface-400">
                Size
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-surface-400">
                Created
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider text-surface-400">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-800">
            {data.items.map((dataset) => (
              <tr
                key={dataset.id}
                onClick={() => onSelect?.(dataset)}
                className={clsx(
                  'transition-colors',
                  onSelect && 'cursor-pointer hover:bg-surface-800/50',
                  selectedId === dataset.id && 'bg-primary-500/5'
                )}
              >
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-surface-800">
                      {dataset.file_type === 'csv' ? (
                        <FileText className="h-5 w-5 text-emerald-400" />
                      ) : (
                        <FileJson className="h-5 w-5 text-amber-400" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium text-white">{dataset.name}</p>
                      <p className="text-sm text-surface-400">{dataset.filename}</p>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className="badge-info uppercase">{dataset.file_type}</span>
                </td>
                <td className="px-6 py-4 text-surface-300">
                  {dataset.row_count.toLocaleString()}
                </td>
                <td className="px-6 py-4 text-surface-300">
                  {formatFileSize(dataset.file_size)}
                </td>
                <td className="px-6 py-4 text-surface-300">
                  {formatDate(dataset.created_at)}
                </td>
                <td className="px-6 py-4 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm('Are you sure you want to delete this dataset?')) {
                        deleteMutation.mutate(dataset.id);
                      }
                    }}
                    disabled={deleteMutation.isPending}
                    className="rounded-lg p-2 text-surface-400 hover:bg-red-500/10 hover:text-red-400"
                  >
                    {deleteMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {data.pages > 1 && (
        <div className="mt-4 flex items-center justify-between">
          <p className="text-sm text-surface-400">
            Showing {(page - 1) * pageSize + 1} to{' '}
            {Math.min(page * pageSize, data.total)} of {data.total} datasets
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="btn-ghost"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </button>
            <button
              onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
              disabled={page === data.pages}
              className="btn-ghost"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

