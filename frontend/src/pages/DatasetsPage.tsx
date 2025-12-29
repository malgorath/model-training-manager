import { useState } from 'react';
import { Plus, Download } from 'lucide-react';
import DatasetList from '../components/DatasetList';
import DatasetUpload from '../components/DatasetUpload';
import HuggingFaceImport from '../components/HuggingFaceImport';

/**
 * Datasets page component.
 */
export default function DatasetsPage() {
  const [showUpload, setShowUpload] = useState(false);
  const [showHuggingFace, setShowHuggingFace] = useState(false);

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Datasets</h1>
          <p className="mt-2 text-surface-400">
            Manage your training datasets
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowHuggingFace(true)}
            className="btn-secondary"
          >
            <Download className="h-4 w-4" />
            Import from Hugging Face
          </button>
          <button
            onClick={() => setShowUpload(true)}
            className="btn-primary"
          >
            <Plus className="h-4 w-4" />
            Upload Dataset
          </button>
        </div>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <DatasetUpload
            onSuccess={() => setShowUpload(false)}
            onClose={() => setShowUpload(false)}
          />
        </div>
      )}

      {/* Hugging Face Import Modal */}
      {showHuggingFace && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <HuggingFaceImport
            onSuccess={() => setShowHuggingFace(false)}
            onClose={() => setShowHuggingFace(false)}
          />
        </div>
      )}

      {/* Dataset List */}
      <div className="card p-6">
        <DatasetList />
      </div>
    </div>
  );
}

