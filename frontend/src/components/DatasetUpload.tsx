import { useState, useCallback } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Upload, File, X, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { datasetApi } from '../services/api';

interface DatasetUploadProps {
  onSuccess?: () => void;
  onClose?: () => void;
}

/**
 * Dataset upload component with drag and drop support.
 */
export default function DatasetUpload({ onSuccess, onClose }: DatasetUploadProps) {
  const queryClient = useQueryClient();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      datasetApi.upload(file, { name: name || file.name, description }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      onSuccess?.();
    },
  });

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      if (file.name.endsWith('.csv') || file.name.endsWith('.json')) {
        setSelectedFile(file);
        if (!name) setName(file.name.replace(/\.(csv|json)$/, ''));
      }
    }
  }, [name]);

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files[0]) {
        setSelectedFile(files[0]);
        if (!name) setName(files[0].name.replace(/\.(csv|json)$/, ''));
      }
    },
    [name]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="card max-w-lg">
      <div className="border-b border-surface-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Upload Dataset</h3>
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
        {/* Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={clsx(
            'relative rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200',
            dragActive
              ? 'border-primary-500 bg-primary-500/5'
              : 'border-surface-700 hover:border-surface-600',
            selectedFile && 'border-emerald-500/50 bg-emerald-500/5'
          )}
        >
          <input
            type="file"
            accept=".csv,.json"
            onChange={handleFileSelect}
            className="absolute inset-0 cursor-pointer opacity-0"
          />

          {selectedFile ? (
            <div className="flex flex-col items-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/10">
                <File className="h-6 w-6 text-emerald-400" />
              </div>
              <p className="mt-3 font-medium text-white">{selectedFile.name}</p>
              <p className="text-sm text-surface-400">
                {formatFileSize(selectedFile.size)}
              </p>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedFile(null);
                }}
                className="mt-2 text-sm text-surface-400 hover:text-surface-100"
              >
                Choose different file
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-800">
                <Upload className="h-6 w-6 text-surface-400" />
              </div>
              <p className="mt-3 font-medium text-white">
                Drop your file here, or click to browse
              </p>
              <p className="mt-1 text-sm text-surface-400">
                Supports CSV and JSON files
              </p>
            </div>
          )}
        </div>

        {/* Form Fields */}
        <div className="mt-6 space-y-4">
          <div>
            <label htmlFor="name" className="label">
              Dataset Name
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter dataset name"
              className="input"
              required
            />
          </div>

          <div>
            <label htmlFor="description" className="label">
              Description (optional)
            </label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter dataset description"
              rows={3}
              className="input resize-none"
            />
          </div>
        </div>

        {/* Error Message */}
        {uploadMutation.isError && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400">
            <AlertCircle className="h-5 w-5" />
            {uploadMutation.error?.message || 'Upload failed'}
          </div>
        )}

        {/* Success Message */}
        {uploadMutation.isSuccess && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-emerald-500/10 px-4 py-3 text-sm text-emerald-400">
            <CheckCircle2 className="h-5 w-5" />
            Dataset uploaded successfully!
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
            disabled={!selectedFile || !name || uploadMutation.isPending}
            className="btn-primary"
          >
            {uploadMutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" />
                Upload Dataset
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

