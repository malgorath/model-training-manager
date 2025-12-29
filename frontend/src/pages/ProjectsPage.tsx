import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus } from 'lucide-react';
import ProjectList from '../components/ProjectList';
import ProjectForm from '../components/ProjectForm';

export default function ProjectsPage() {
  const navigate = useNavigate();
  const [showForm, setShowForm] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Projects</h1>
          <p className="text-surface-400 mt-1">Manage training projects with traits and dataset allocations</p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="h-5 w-5" />
          New Project
        </button>
      </div>

      {showForm ? (
        <ProjectForm
          onSuccess={() => {
            setShowForm(false);
          }}
          onClose={() => setShowForm(false)}
        />
      ) : (
        <ProjectList />
      )}
    </div>
  );
}
