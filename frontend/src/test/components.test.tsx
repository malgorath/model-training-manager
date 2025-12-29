/**
 * Component tests for the Model Training Manager frontend.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

// Mock API modules
vi.mock('../services/api', () => ({
  datasetApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Dataset',
          filename: 'test.csv',
          file_type: 'csv',
          file_size: 1024,
          row_count: 100,
          column_count: 2,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
      pages: 1,
    }),
    upload: vi.fn().mockResolvedValue({ id: 1, name: 'New Dataset' }),
    delete: vi.fn().mockResolvedValue(undefined),
  },
  trainingJobApi: {
    list: vi.fn().mockResolvedValue({
      items: [
        {
          id: 1,
          name: 'Test Job',
          status: 'running',
          training_type: 'qlora',
          progress: 50,
          current_epoch: 2,
          epochs: 4,
          current_loss: 0.5,
          created_at: '2024-01-01T00:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 10,
      pages: 1,
    }),
    create: vi.fn().mockResolvedValue({ id: 1, name: 'New Job' }),
    cancel: vi.fn().mockResolvedValue({ id: 1, status: 'cancelled' }),
  },
  workerApi: {
    getStatus: vi.fn().mockResolvedValue({
      total_workers: 2,
      active_workers: 2,
      idle_workers: 1,
      busy_workers: 1,
      max_workers: 8,
      workers: [
        { id: 'worker-1', status: 'idle', jobs_completed: 5, started_at: '2024-01-01T00:00:00Z' },
        { id: 'worker-2', status: 'busy', current_job_id: 1, jobs_completed: 3, started_at: '2024-01-01T00:00:00Z' },
      ],
      jobs_in_queue: 0,
    }),
    start: vi.fn().mockResolvedValue({}),
    stop: vi.fn().mockResolvedValue({}),
    restart: vi.fn().mockResolvedValue({}),
  },
  configApi: {
    get: vi.fn().mockResolvedValue({
      id: 1,
      max_concurrent_workers: 4,
      active_workers: 2,
      default_model: 'llama3.2:3b',
      default_training_type: 'qlora',
      default_batch_size: 4,
      default_learning_rate: 0.0002,
      default_epochs: 3,
      default_lora_r: 16,
      default_lora_alpha: 32,
      default_lora_dropout: 0.05,
      auto_start_workers: false,
    }),
    update: vi.fn().mockResolvedValue({}),
  },
  healthApi: {
    check: vi.fn().mockResolvedValue({
      status: 'healthy',
      version: '1.0.0',
      environment: 'development',
    }),
  },
}));

// Import components after mocks
import Layout from '../components/Layout';
import DatasetList from '../components/DatasetList';
import WorkerDashboard from '../components/WorkerDashboard';
import JobMonitor from '../components/JobMonitor';
import TrainingConfig from '../components/TrainingConfig';

/**
 * Create a wrapper with providers for testing.
 */
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe('Layout Component', () => {
  it('renders navigation links', () => {
    render(<Layout />, { wrapper: createWrapper() });

    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Datasets')).toBeInTheDocument();
    expect(screen.getByText('Training Jobs')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders application title', () => {
    render(<Layout />, { wrapper: createWrapper() });

    expect(screen.getByText('Trainers')).toBeInTheDocument();
    expect(screen.getByText('Model Training Manager')).toBeInTheDocument();
  });
});

describe('DatasetList Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders dataset list', async () => {
    render(<DatasetList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Dataset')).toBeInTheDocument();
    });
  });

  it('displays file type badge', async () => {
    render(<DatasetList />, { wrapper: createWrapper() });

    await waitFor(() => {
      // The badge shows lowercase 'csv' with uppercase CSS styling
      expect(screen.getByText('csv')).toBeInTheDocument();
    });
  });

  it('shows row count', async () => {
    render(<DatasetList />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('100')).toBeInTheDocument();
    });
  });
});

describe('WorkerDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders worker pool header', async () => {
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Worker Pool')).toBeInTheDocument();
    });
  });

  it('displays worker stats', async () => {
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Total Workers')).toBeInTheDocument();
      expect(screen.getByText('Idle')).toBeInTheDocument();
      expect(screen.getByText('Busy')).toBeInTheDocument();
    });
  });

  it('shows worker list', async () => {
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('worker-1')).toBeInTheDocument();
      expect(screen.getByText('worker-2')).toBeInTheDocument();
    });
  });

  it('has control buttons', async () => {
    render(<WorkerDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Start Worker')).toBeInTheDocument();
      expect(screen.getByText('Stop All')).toBeInTheDocument();
      expect(screen.getByText('Restart')).toBeInTheDocument();
    });
  });
});

describe('JobMonitor Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders job list', async () => {
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Test Job')).toBeInTheDocument();
    });
  });

  it('displays job status', async () => {
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      // There's both a filter button and a badge with 'running' - use getAllByText
      const runningElements = screen.getAllByText('running');
      expect(runningElements.length).toBeGreaterThan(0);
    });
  });

  it('shows progress bar', async () => {
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });
  });

  it('has status filter buttons', async () => {
    render(<JobMonitor />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('pending')).toBeInTheDocument();
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('failed')).toBeInTheDocument();
    });
  });
});

describe('TrainingConfig Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders configuration form', async () => {
    render(<TrainingConfig />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Training Configuration')).toBeInTheDocument();
    });
  });

  it('displays worker settings section', async () => {
    render(<TrainingConfig />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Worker Settings')).toBeInTheDocument();
      expect(screen.getByText('Max Concurrent Workers')).toBeInTheDocument();
    });
  });

  it('displays model settings section', async () => {
    render(<TrainingConfig />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Model Settings')).toBeInTheDocument();
      expect(screen.getByText('Default Model')).toBeInTheDocument();
    });
  });

  it('has save button', async () => {
    render(<TrainingConfig />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByText('Save Changes')).toBeInTheDocument();
    });
  });
});

