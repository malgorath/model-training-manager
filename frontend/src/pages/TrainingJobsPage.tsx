import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Plus, Play } from 'lucide-react';
import JobMonitor from '../components/JobMonitor';
import TrainingJobForm from '../components/TrainingJobForm';
import TrainingJobDetail from '../components/TrainingJobDetail';
import WorkerDashboard from '../components/WorkerDashboard';

/**
 * Training jobs page component.
 */
export default function TrainingJobsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(() => {
    const jobIdParam = searchParams.get('jobId');
    return jobIdParam ? parseInt(jobIdParam, 10) : null;
  });

  // Update URL when job is selected
  useEffect(() => {
    if (selectedJobId) {
      setSearchParams({ jobId: selectedJobId.toString() });
    } else {
      setSearchParams({});
    }
  }, [selectedJobId, setSearchParams]);

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Training Jobs</h1>
          <p className="mt-2 text-surface-400">
            Create and monitor training jobs
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="btn-primary"
        >
          <Plus className="h-4 w-4" />
          New Training Job
        </button>
      </div>

      {/* Create Job Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <TrainingJobForm
            onSuccess={() => setShowCreateForm(false)}
            onClose={() => setShowCreateForm(false)}
          />
        </div>
      )}

      {/* Content */}
      <div className="grid grid-cols-3 gap-8">
        {/* Job List */}
        <div className="col-span-2">
          <div className="card">
            <div className="border-b border-surface-800 bg-surface-800/30 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/10">
                  <Play className="h-5 w-5 text-primary-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Training Jobs</h3>
                  <p className="text-sm text-surface-400">All training jobs</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              <JobMonitor onSelectJob={(job) => setSelectedJobId(job.id)} />
            </div>
          </div>
        </div>

        {/* Worker Dashboard */}
        <div>
          <WorkerDashboard />
        </div>
      </div>

      {/* Job Detail Modal */}
      {selectedJobId !== null && (
        <TrainingJobDetail
          jobId={selectedJobId}
          onClose={() => setSelectedJobId(null)}
        />
      )}
    </div>
  );
}

